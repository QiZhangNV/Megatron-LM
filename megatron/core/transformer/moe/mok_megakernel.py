# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal Mixture-of-Kittens adapter for MCore MoE training experiments.

This module intentionally keeps the integration narrow: MCore owns routing,
parameters, DDP, and the optimizer, while MoK replaces dispatch, routed/shared
expert computation, and combine. Checkpoint conversion remains out of scope.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

import torch
from torch import nn

_MOK_MODULE_INDICES = itertools.count()
_MOK_HIGH_PRECISION_INIT_ATTR = "_mok_high_precision_init_val"


def _debug_record(
    stage: str,
    param: torch.Tensor,
    *,
    tensors: dict[str, torch.Tensor | None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record an opt-in lifecycle snapshot without importing debug code normally."""
    from megatron.core.mok_param_lifecycle_debug import enabled, record

    if enabled():
        record(stage, param, tensors=tensors, metadata=metadata)


def _debug_tag(param: nn.Parameter, name: str) -> None:
    from megatron.core.mok_param_lifecycle_debug import enabled, tag_parameter

    if enabled():
        tag_parameter(param, name)


def _copy_parameter_attributes(dst: nn.Parameter, src: torch.Tensor, *, allreduce: bool) -> None:
    """Copy the parameter metadata MCore uses for optimizer/DDP classification."""
    dst.allreduce = allreduce
    for name in (
        "sequence_parallel",
        "tensor_model_parallel",
        "partition_dim",
        "partition_stride",
        "shared",
    ):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def _dequantize_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a logical TE/plain parameter as an ordinary BF16 tensor."""
    tensor = tensor.detach()
    dequantize = getattr(tensor, "dequantize", None)
    if callable(dequantize):
        tensor = dequantize()
    return tensor.to(dtype=torch.bfloat16)


def _materialize_parameter_init(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return the logical initialization value and whether TE preserved it losslessly."""
    get_high_precision_init_val = getattr(tensor, "get_high_precision_init_val", None)
    if callable(get_high_precision_init_val):
        init_val = get_high_precision_init_val().detach()
        tensor.clear_high_precision_init_val()
        return init_val, True
    return _dequantize_bf16(tensor), False


def _attach_high_precision_init(param: nn.Parameter, init_val: torch.Tensor | None) -> None:
    """Preserve a reordered pre-quantization init until MCore creates its FP32 master."""
    if init_val is None:
        return
    if init_val.shape != param.shape:
        raise RuntimeError(
            "MOK high-precision initialization shape mismatch: "
            f"init={tuple(init_val.shape)}, param={tuple(param.shape)}"
        )
    setattr(param, _MOK_HIGH_PRECISION_INIT_ATTR, init_val.detach().contiguous())


def _indexed_grouped_weight(linear: nn.Module, index: int, num_experts: int) -> torch.Tensor:
    """Return one logical expert weight from either TE grouped layout."""
    if not getattr(linear, "single_grouped_weight", False):
        return getattr(linear, f"weight{index}")

    weight = linear.weight
    split_quantized = getattr(weight, "split_into_quantized_tensors", None)
    if callable(split_quantized):
        return split_quantized()[index]
    if weight.ndim >= 3 and weight.shape[0] == num_experts:
        return weight[index]
    if weight.shape[0] % num_experts != 0:
        raise RuntimeError(
            f"Cannot split grouped weight with shape {tuple(weight.shape)} "
            f"into {num_experts} experts"
        )
    return weight.narrow(
        0, index * (weight.shape[0] // num_experts), weight.shape[0] // num_experts
    )


def _new_bf16_parameter(
    shape: Iterable[int], reference: torch.Tensor, *, allreduce: bool, zero: bool = False
) -> nn.Parameter:
    data = torch.empty(tuple(shape), dtype=torch.bfloat16, device=reference.device)
    if zero:
        data.zero_()
    param = nn.Parameter(data)
    _copy_parameter_attributes(param, reference, allreduce=allreduce)
    return param


def _dummy_weight_gradient(param: nn.Parameter) -> torch.Tensor:
    """Return a storage-free gradient sentinel used to trigger MCore DDP hooks.

    MOK has already accumulated the numerical gradient into ``main_grad``. The
    autograd return only needs the parameter's shape and dtype so that MCore's
    post-accumulate hook runs; the hook does not read it when
    ``grad_added_to_main_grad`` is true. A detached parameter view therefore
    avoids allocating a full-sized dummy gradient.
    """
    return param.detach()


def _main_grad_buffer(param: nn.Parameter) -> torch.Tensor:
    """Return and validate the optimizer-visible FP32 gradient buffer."""
    main_grad = getattr(param, "main_grad", None)
    if main_grad is None:
        raise RuntimeError(
            "MOK gradient accumulation fusion requires DDP to assign param.main_grad"
        )
    if main_grad.shape != param.shape:
        raise RuntimeError(
            "MOK weight-gradient shape mismatch: "
            f"main_grad={tuple(main_grad.shape)}, param={tuple(param.shape)}"
        )
    if main_grad.dtype != torch.float32 or not main_grad.is_contiguous():
        raise RuntimeError("MOK direct accumulation requires contiguous FP32 main_grad")
    if getattr(param, "zero_out_wgrad", False):
        raise RuntimeError("MOK does not support zero_out_wgrad parameters")
    if main_grad.device != param.device:
        raise RuntimeError("MOK main_grad must be on the parameter device")
    return main_grad


def _finish_weight_gradient(param: nn.Parameter) -> torch.Tensor:
    """Mark an in-kernel accumulation complete and return a DDP hook-only grad."""
    param.grad_added_to_main_grad = True
    return _dummy_weight_gradient(param)


def _storage_view(
    storage: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Return a zero-copy dense view over a TE grouped backing tensor."""
    if not storage.is_cuda or not storage.is_contiguous():
        raise RuntimeError(f"MOK {name} storage must be contiguous CUDA storage")
    if storage.dtype != dtype:
        if storage.dtype == torch.uint8 and dtype == torch.float8_e4m3fn:
            storage = storage.view(dtype)
        else:
            raise RuntimeError(
                f"MOK {name} storage has dtype {storage.dtype}, expected {dtype}"
            )
    expected_numel = 1
    for dim in shape:
        expected_numel *= dim
    if storage.numel() != expected_numel:
        raise RuntimeError(
            f"MOK {name} storage size mismatch: got {storage.numel()}, "
            f"expected {expected_numel} for {shape}"
        )
    return storage.view(shape)


def _grouped_mxfp8_scale_view(
    param: nn.Parameter,
    member_attr: str,
    shape: tuple[int, ...],
    *,
    name: str,
) -> torch.Tensor:
    """Expose all experts' contiguous TE MXFP8 scale storage through member zero."""
    from megatron.core.fp8_utils import get_grouped_quantized_members

    members = get_grouped_quantized_members(param)
    if not members:
        raise RuntimeError(f"MOK {name} grouped parameter has no quantized members")
    first = getattr(members[0], member_attr, None)
    if first is None or first.dtype != torch.uint8 or not first.is_contiguous():
        raise RuntimeError(
            f"MOK {name} requires contiguous uint8 TE member storage {member_attr}"
        )
    expected_numel = 1
    for dim in shape:
        expected_numel *= dim
    storage_numel = first.untyped_storage().nbytes() // first.element_size()
    available_numel = storage_numel - first.storage_offset()
    if available_numel < expected_numel:
        raise RuntimeError(
            f"MOK {name} scale storage is too small: available={available_numel}, "
            f"expected={expected_numel}"
        )
    flat = torch.as_strided(
        first,
        (expected_numel,),
        (1,),
        storage_offset=first.storage_offset(),
    )
    return flat.view(shape)


def _swizzle_mxfp8_scale(
    logical_scale: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Convert TE's logical E8M0 matrix to MOK's tcgen05 scale layout.

    TE stores rowwise scales as ``[E, M, K / 32]``. MOK consumes the
    tcgen05 1x scale-factor layout ``[E * M / 128, K / 128, 32, 16]``.
    Only the scale bytes are copied; the much larger FP8 payload stays in
    native TE storage.
    """
    if rows % 128 != 0 or columns % 128 != 0:
        raise RuntimeError("MOK MXFP8 scale dimensions must be divisible by 128")
    if logical_scale.dtype != torch.uint8 or logical_scale.ndim != 3:
        raise RuntimeError("MOK requires logical uint8 MXFP8 scales shaped [E, M, K/32]")
    num_experts = logical_scale.shape[0]
    expected_shape = (num_experts, rows, columns // 32)
    if tuple(logical_scale.shape) != expected_shape:
        raise RuntimeError(
            f"MOK logical scale shape mismatch: got {tuple(logical_scale.shape)}, "
            f"expected {expected_shape}"
        )
    return (
        logical_scale.reshape(
            num_experts, rows // 128, 128, columns // 128, 4
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(num_experts, rows // 128, columns // 128, 4, 32, 4)
        .transpose(-3, -2)
        .reshape(num_experts * rows // 128, columns // 128, 32, 16)
        .contiguous()
    )


def _native_single_grouped_weight_views(
    fc1: nn.Parameter,
    fc2: nn.Parameter,
    *,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    use_mxfp8: bool,
):
    """Build MOK gate/up/down views directly over native TE grouped parameters."""
    e, i, h = num_experts, intermediate_size, hidden_size
    if tuple(fc1.shape) != (e, 2 * i, h) or tuple(fc2.shape) != (e, h, i):
        raise RuntimeError(
            "MOK requires native single-grouped FC1/FC2 shapes "
            f"{(e, 2 * i, h)} and {(e, h, i)}, got "
            f"{tuple(fc1.shape)} and {tuple(fc2.shape)}"
        )

    if not use_mxfp8:
        if fc1.dtype != torch.bfloat16 or fc2.dtype != torch.bfloat16:
            raise RuntimeError("MOK BF16 requires native BF16 grouped parameters")
        if not fc1.is_contiguous() or not fc2.is_contiguous():
            raise RuntimeError("MOK BF16 requires contiguous grouped parameters")
        return fc1, fc1, fc2

    from megatron.core.fp8_utils import is_grouped_mxfp8tensor

    if not is_grouped_mxfp8tensor(fc1) or not is_grouped_mxfp8tensor(fc2):
        raise RuntimeError("MOK MXFP8 requires native TE grouped MXFP8 parameters")

    fc1_row = _storage_view(
        fc1.rowwise_data, (e, 2 * i, h), dtype=torch.float8_e4m3fn, name="FC1 rowwise"
    )
    fc1_col = _storage_view(
        fc1.columnwise_data,
        (e, 2 * i, h),
        dtype=torch.float8_e4m3fn,
        name="FC1 columnwise",
    )
    fc2_row = _storage_view(
        fc2.rowwise_data, (e, h, i), dtype=torch.float8_e4m3fn, name="FC2 rowwise"
    )
    fc2_col = _storage_view(
        fc2.columnwise_data,
        (e, h, i),
        dtype=torch.float8_e4m3fn,
        name="FC2 columnwise",
    )
    # TE keeps E8M0 scales in logical order. Rowwise member storage is
    # [M, K/32]; columnwise member storage is [M/32, K] for the original
    # matrix, so transpose the latter into logical order for the transposed
    # FP8 payload. These are all zero-copy views.
    fc1_row_sc = _grouped_mxfp8_scale_view(
        fc1, "_rowwise_scale_inv", (e, 2 * i, h // 32), name="FC1 rowwise"
    )
    fc1_col_sc = _grouped_mxfp8_scale_view(
        fc1,
        "_columnwise_scale_inv",
        (e, 2 * i // 32, h),
        name="FC1 columnwise",
    ).transpose(-2, -1)
    fc2_row_sc = _grouped_mxfp8_scale_view(
        fc2, "_rowwise_scale_inv", (e, h, i // 32), name="FC2 rowwise"
    )
    fc2_col_sc = _grouped_mxfp8_scale_view(
        fc2,
        "_columnwise_scale_inv",
        (e, h // 32, i),
        name="FC2 columnwise",
    ).transpose(-2, -1)
    # The final flag tells MOK that ``columnwise_data`` is TE's native
    # columnwise-quantized storage in the original [E, M, K] tensor shape.
    # MOK can then consume it directly for dgrad instead of materializing an
    # explicit [E, K, M] transpose on every backward.
    fc1_views = (fc1_row, fc1_row_sc, fc1_col, fc1_col_sc, True)
    fc2_views = (fc2_row, fc2_row_sc, fc2_col, fc2_col_sc, True)
    return fc1_views, fc1_views, fc2_views


@torch.no_grad()
def _accumulate_weight_gradient(param: nn.Parameter, grad: torch.Tensor) -> torch.Tensor:
    """Accumulate a materialized wgrad; fallback for non-fused MOK precisions."""
    main_grad = _main_grad_buffer(param)
    if main_grad.shape != grad.shape:
        raise RuntimeError(
            "MOK weight-gradient shape mismatch: "
            f"main_grad={tuple(main_grad.shape)}, grad={tuple(grad.shape)}"
        )
    main_grad.add_(grad)
    return _finish_weight_gradient(param)


def _mok_mxfp8_backward_weight_views(
    native_weight: tuple[torch.Tensor, ...],
    *,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Prepare zero-copy payloads and compact scale layouts for forward/backward."""
    row_data, row_scale, column_data, column_scale, native_columnwise = native_weight
    if native_columnwise is not True:
        raise RuntimeError("MOK MCore integration requires native TE columnwise weights")
    return (
        row_data,
        _swizzle_mxfp8_scale(row_scale, rows=rows, columns=columns),
        column_data,
        _swizzle_mxfp8_scale(column_scale, rows=columns, columns=rows),
        True,
    )

class _MoKAutograd(torch.autograd.Function):
    """Autograd bridge from MCore parameters to MoK's functional API."""

    @staticmethod
    def forward(
        ctx,
        module: "MoKMegakernel",
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        routed_fc1: torch.Tensor,
        routed_down: torch.Tensor,
        shared_gate: torch.Tensor,
        shared_up: torch.Tensor,
        shared_down: torch.Tensor,
    ) -> torch.Tensor:
        from mok import functional

        workspace = functional.get_workspace(
            module.mok_config,
            module.ep_group,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=top_experts.shape[1],
        )
        schedule = functional.build_schedule(
            workspace, module.mok_config, top_experts, num_local_experts=module.num_local_experts
        )
        trace_param_lifecycle = module.is_first_microbatch
        if trace_param_lifecycle:
            _debug_record(
                "forward.after_param_sync_before_quantize",
                module.routed_fc1_weight,
                tensors={"main_grad": getattr(module.routed_fc1_weight, "main_grad", None)},
            )
        prepared_gate, prepared_up, prepared_down = module.quantized_routed_weights()
        if module.use_mxfp8_weights:
            gate_forward = prepared_gate[:2]
            up_forward = prepared_up[:2]
            down_forward = prepared_down[:2]
        else:
            gate_forward = prepared_gate
            up_forward = prepared_up
            down_forward = prepared_down
        if trace_param_lifecycle and module.use_mxfp8_weights:
            _debug_record(
                "forward.mok_quantized_weight_cache",
                module.routed_fc1_weight,
                tensors={
                    "native_rowwise_data": prepared_gate[0],
                    "prepared_rowwise_scale": prepared_gate[1],
                    "native_columnwise_data": prepared_gate[2],
                    "prepared_columnwise_scale": prepared_gate[3],
                    "actual_forward_data": gate_forward[0],
                    "actual_forward_scale": gate_forward[1],
                },
            )
        output, forward_context = functional.forward(
            module.mok_config,
            workspace,
            schedule,
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_down,
            gate_forward,
            up_forward,
            down_forward,
            swiglu_limit=module.swiglu_limit,
        )

        ctx.module = module
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.forward_context = forward_context
        ctx.quantized_weights = (prepared_gate, prepared_up, prepared_down)
        ctx.trace_param_lifecycle = trace_param_lifecycle
        ctx.save_for_backward(
            x,
            router_weights,
            routed_fc1,
            routed_down,
            shared_gate,
            shared_up,
            shared_down,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from mok import functional

        (
            x,
            router_weights,
            routed_fc1,
            routed_down,
            shared_gate,
            shared_up,
            shared_down,
        ) = ctx.saved_tensors
        prepared_gate, prepared_up, prepared_down = ctx.quantized_weights
        if ctx.module.use_mxfp8_weights:
            backward_gate = prepared_gate
            backward_up = prepared_up
            backward_down = prepared_down[2:]
        else:
            backward_gate = prepared_gate
            backward_up = prepared_up
            backward_down = prepared_down
        direct_wgrad_accumulation = ctx.module.fuse_wgrad_accumulation
        main_grads = None
        if direct_wgrad_accumulation:
            main_grads = (
                _main_grad_buffer(ctx.module.shared_gate_weight),
                _main_grad_buffer(ctx.module.routed_fc1_weight),
                _main_grad_buffer(ctx.module.shared_up_weight),
                _main_grad_buffer(ctx.module.routed_fc1_weight),
                _main_grad_buffer(ctx.module.shared_down_weight),
                _main_grad_buffer(ctx.module.routed_down_weight),
            )
        if ctx.trace_param_lifecycle:
            backward_debug_tensors = {
                "main_grad_before": getattr(ctx.module.routed_fc1_weight, "main_grad", None)
            }
            if isinstance(backward_gate, tuple):
                backward_debug_tensors.update(
                    {
                        "actual_backward_rowwise_data": backward_gate[0],
                        "actual_backward_rowwise_scale": backward_gate[1],
                        "actual_backward_columnwise_data": backward_gate[2],
                        "actual_backward_columnwise_scale": backward_gate[3],
                    }
                )
            _debug_record(
                "backward.before_mok_kernel",
                ctx.module.routed_fc1_weight,
                tensors=backward_debug_tensors,
            )
        (
            d_x,
            d_router_weights,
            _d_routed_gate,
            _d_routed_up,
            d_routed_down,
            d_shared_gate,
            d_shared_up,
            d_shared_down,
        ) = functional.backward(
            ctx.module.mok_config,
            ctx.workspace,
            ctx.schedule,
            ctx.forward_context,
            grad_output.contiguous(),
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_down,
            backward_gate,
            backward_up,
            backward_down,
            swiglu_limit=ctx.module.swiglu_limit,
            main_grads=main_grads,
        )

        if ctx.trace_param_lifecycle:
            _debug_record(
                "backward.after_mok_kernel",
                ctx.module.routed_fc1_weight,
                tensors={
                    "main_grad_after": getattr(ctx.module.routed_fc1_weight, "main_grad", None)
                },
            )
        if ctx.module.fuse_wgrad_accumulation:
            d_routed_fc1 = _finish_weight_gradient(ctx.module.routed_fc1_weight)
            d_routed_down = _finish_weight_gradient(ctx.module.routed_down_weight)
            d_shared_gate = _finish_weight_gradient(ctx.module.shared_gate_weight)
            d_shared_up = _finish_weight_gradient(ctx.module.shared_up_weight)
            d_shared_down = _finish_weight_gradient(ctx.module.shared_down_weight)

        ctx.module = None
        ctx.workspace = None
        ctx.schedule = None
        ctx.forward_context = None
        ctx.quantized_weights = None
        return (
            None,
            d_x,
            d_router_weights,
            None,
            d_routed_fc1,
            d_routed_down,
            d_shared_gate,
            d_shared_up,
            d_shared_down,
        )


class MoKMegakernel(nn.Module):
    """Own MCore trainable weights in the layouts consumed by the MoK kernel."""

    def __init__(
        self,
        config,
        ep_group,
        routed_experts: nn.Module,
        shared_experts: nn.Module,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        try:
            from mok.functional import MoKConfig
        except ImportError as exc:
            raise ImportError(
                "--use-mok-megakernel requires the latest mixture-of-kittens package "
                "on PYTHONPATH"
            ) from exc

        if not config.moe_single_grouped_weight:
            raise ValueError("MOK integration requires moe_single_grouped_weight=True")
        if not config.gradient_accumulation_fusion:
            raise ValueError(
                "MOK native grouped weights require gradient_accumulation_fusion=True"
            )
        if config.moe_mlp_glu_interleave_size is not None:
            raise ValueError("MoK weight import requires non-interleaved MCore routed FC1 weights")
        if config.moe_shared_expert_glu_interleave_size is not None:
            raise ValueError("MoK weight import requires non-interleaved shared FC1 weights")
        if config.moe_shared_expert_gate:
            raise ValueError("MoK does not support MCore's optional shared-expert output gate")

        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size
        self.shared_intermediate_size = config.moe_shared_expert_intermediate_size
        self.topk = config.moe_router_topk
        self.swiglu_limit = config.activation_func_clamp_value
        self.use_mxfp8_weights = config.mok_use_mxfp8_weights
        self.fuse_wgrad_accumulation = config.gradient_accumulation_fusion
        self._debug_module_index = next(_MOK_MODULE_INDICES)
        self.mok_config = MoKConfig(
            fwd_num_comm_sms=config.mok_fwd_num_comm_sms,
            bwd_num_comm_sms=config.mok_bwd_num_comm_sms,
            minibatch_size=config.mok_minibatch_size,
            macrobatch_size=config.mok_macrobatch_size,
            schedule_capacity_multiplier=config.mok_schedule_capacity_multiplier,
            all_gather_top_experts_chunk_bytes=config.mok_all_gather_top_experts_chunk_bytes,
            scale_router_before_fc2=config.mok_scale_router_before_fc2,
        )

        fc1 = routed_experts.linear_fc1
        fc2 = routed_experts.linear_fc2
        if not getattr(fc1, "single_grouped_weight", False) or not getattr(
            fc2, "single_grouped_weight", False
        ):
            raise ValueError("MOK requires single-grouped weights for routed FC1 and FC2")
        # Register aliases of the same Parameter objects so DDP's MOK forward
        # pre-hook waits for their overlap-param-gather buckets. named_parameters
        # deduplicates them; no additional payload storage is allocated.
        self.register_parameter("routed_fc1_weight", fc1.weight)
        self.register_parameter("routed_down_weight", fc2.weight)
        _debug_tag(self.routed_fc1_weight, f"module{self._debug_module_index}.routed_fc1_weight")
        _debug_tag(self.routed_down_weight, f"module{self._debug_module_index}.routed_down_weight")
        self._import_shared_weights(shared_experts)

        # MegatronModule.set_is_first_microbatch discovers this attribute and resets it
        # once per optimizer iteration, matching TE's weight-cache lifecycle.
        self.is_first_microbatch = True
        self._prepared_routed_weight_cache = None

    @torch.no_grad()
    def _import_routed_weights(self, experts: nn.Module) -> None:
        fc1 = experts.linear_fc1
        fc2 = experts.linear_fc2
        fc1_ref = _indexed_grouped_weight(fc1, 0, self.num_local_experts)
        fc2_ref = _indexed_grouped_weight(fc2, 0, self.num_local_experts)
        i, h, e = self.intermediate_size, self.hidden_size, self.num_local_experts

        self.routed_gate_weight = _new_bf16_parameter((e, i, h), fc1_ref, allreduce=False)
        self.routed_up_weight = _new_bf16_parameter((e, i, h), fc1_ref, allreduce=False)
        self.routed_down_weight = _new_bf16_parameter((e, h, i), fc2_ref, allreduce=False)
        _debug_tag(self.routed_gate_weight, f"module{self._debug_module_index}.routed_gate_weight")
        _debug_tag(self.routed_up_weight, f"module{self._debug_module_index}.routed_up_weight")
        _debug_tag(self.routed_down_weight, f"module{self._debug_module_index}.routed_down_weight")

        routed_gate_init = None
        routed_up_init = None
        routed_down_init = None
        fc1_has_preserved_init = None
        fc2_has_preserved_init = None

        for expert_idx in range(e):
            source_fc1, current_fc1_has_preserved_init = _materialize_parameter_init(
                _indexed_grouped_weight(fc1, expert_idx, self.num_local_experts)
            )
            source_fc2, current_fc2_has_preserved_init = _materialize_parameter_init(
                _indexed_grouped_weight(fc2, expert_idx, self.num_local_experts)
            )
            source_fc1 = source_fc1.reshape(2 * i, h)
            source_fc2 = source_fc2.reshape(h, i)

            if fc1_has_preserved_init is None:
                fc1_has_preserved_init = current_fc1_has_preserved_init
                if fc1_has_preserved_init:
                    routed_gate_init = torch.empty(
                        (e, i, h), dtype=source_fc1.dtype, device=source_fc1.device
                    )
                    routed_up_init = torch.empty_like(routed_gate_init)
            elif fc1_has_preserved_init != current_fc1_has_preserved_init:
                raise RuntimeError("MOK routed FC1 weights have inconsistent initialization state")

            if fc2_has_preserved_init is None:
                fc2_has_preserved_init = current_fc2_has_preserved_init
                if fc2_has_preserved_init:
                    routed_down_init = torch.empty(
                        (e, h, i), dtype=source_fc2.dtype, device=source_fc2.device
                    )
            elif fc2_has_preserved_init != current_fc2_has_preserved_init:
                raise RuntimeError("MOK routed FC2 weights have inconsistent initialization state")

            self.routed_gate_weight[expert_idx].copy_(source_fc1[:i].to(torch.bfloat16))
            self.routed_up_weight[expert_idx].copy_(source_fc1[i:].to(torch.bfloat16))
            self.routed_down_weight[expert_idx].copy_(source_fc2.to(torch.bfloat16))
            if routed_gate_init is not None:
                routed_gate_init[expert_idx].copy_(source_fc1[:i])
                routed_up_init[expert_idx].copy_(source_fc1[i:])
            if routed_down_init is not None:
                routed_down_init[expert_idx].copy_(source_fc2)

        _attach_high_precision_init(self.routed_gate_weight, routed_gate_init)
        _attach_high_precision_init(self.routed_up_weight, routed_up_init)
        _attach_high_precision_init(self.routed_down_weight, routed_down_init)

    @torch.no_grad()
    def _import_shared_weights(self, shared: nn.Module) -> None:
        fc1_ref = shared.linear_fc1.weight
        fc2_ref = shared.linear_fc2.weight
        routed_i = self.intermediate_size
        shared_i = self.shared_intermediate_size
        h = self.hidden_size

        # Upstream MoK currently has one intermediate-size template parameter for
        # both routed and shared experts. Zero-padding is mathematically inert and
        # keeps the DSv4-Pro shared MLP (2048) equivalent inside a routed-I=3072
        # kernel. The extra shared compute is reported as a known POC overhead.
        self.shared_gate_weight = _new_bf16_parameter(
            (routed_i, h), fc1_ref, allreduce=True, zero=True
        )
        self.shared_up_weight = _new_bf16_parameter(
            (routed_i, h), fc1_ref, allreduce=True, zero=True
        )
        self.shared_down_weight = _new_bf16_parameter(
            (h, routed_i), fc2_ref, allreduce=True, zero=True
        )
        _debug_tag(self.shared_gate_weight, f"module{self._debug_module_index}.shared_gate_weight")
        _debug_tag(self.shared_up_weight, f"module{self._debug_module_index}.shared_up_weight")
        _debug_tag(self.shared_down_weight, f"module{self._debug_module_index}.shared_down_weight")
        source_fc1, fc1_has_preserved_init = _materialize_parameter_init(fc1_ref)
        source_fc2, fc2_has_preserved_init = _materialize_parameter_init(fc2_ref)
        source_fc1 = source_fc1.reshape(2 * shared_i, h)
        source_fc2 = source_fc2.reshape(h, shared_i)
        self.shared_gate_weight[:shared_i].copy_(source_fc1[:shared_i].to(torch.bfloat16))
        self.shared_up_weight[:shared_i].copy_(source_fc1[shared_i:].to(torch.bfloat16))
        self.shared_down_weight[:, :shared_i].copy_(source_fc2.to(torch.bfloat16))

        shared_gate_init = None
        shared_up_init = None
        shared_down_init = None
        if fc1_has_preserved_init:
            shared_gate_init = torch.zeros(
                (routed_i, h), dtype=source_fc1.dtype, device=source_fc1.device
            )
            shared_up_init = torch.zeros_like(shared_gate_init)
            shared_gate_init[:shared_i].copy_(source_fc1[:shared_i])
            shared_up_init[:shared_i].copy_(source_fc1[shared_i:])
        if fc2_has_preserved_init:
            shared_down_init = torch.zeros(
                (h, routed_i), dtype=source_fc2.dtype, device=source_fc2.device
            )
            shared_down_init[:, :shared_i].copy_(source_fc2)

        _attach_high_precision_init(self.shared_gate_weight, shared_gate_init)
        _attach_high_precision_init(self.shared_up_weight, shared_up_init)
        _attach_high_precision_init(self.shared_down_weight, shared_down_init)

    @torch.no_grad()
    def quantized_routed_weights(self):
        """Expose native TE grouped parameters with scale layouts cached per iteration."""
        if not self.use_mxfp8_weights:
            self.is_first_microbatch = False
            return _native_single_grouped_weight_views(
                self.routed_fc1_weight,
                self.routed_down_weight,
                num_experts=self.num_local_experts,
                intermediate_size=self.intermediate_size,
                hidden_size=self.hidden_size,
                use_mxfp8=False,
            )

        if self._prepared_routed_weight_cache is None or self.is_first_microbatch:
            native_gate, _, native_down = _native_single_grouped_weight_views(
                self.routed_fc1_weight,
                self.routed_down_weight,
                num_experts=self.num_local_experts,
                intermediate_size=self.intermediate_size,
                hidden_size=self.hidden_size,
                use_mxfp8=True,
            )
            prepared_gate = _mok_mxfp8_backward_weight_views(
                native_gate,
                rows=2 * self.intermediate_size,
                columns=self.hidden_size,
            )
            prepared_down = _mok_mxfp8_backward_weight_views(
                native_down,
                rows=self.hidden_size,
                columns=self.intermediate_size,
            )
            # Only the compact scale layouts allocate storage. FP8 row/column
            # payloads remain zero-copy views of the current TE gather buffer.
            self._prepared_routed_weight_cache = (
                prepared_gate,
                prepared_gate,
                prepared_down,
            )

        self.is_first_microbatch = False
        return self._prepared_routed_weight_cache

    def forward(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> torch.Tensor:
        del routing_map  # Router side effects/losses are already attached to probs.
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        router_weights, top_experts = torch.topk(
            probs.reshape(x.shape[0], -1), self.topk, dim=-1, sorted=False
        )
        router_weights = router_weights.to(dtype=torch.float32).contiguous()
        top_experts = top_experts.to(dtype=torch.int64).contiguous()

        output = _MoKAutograd.apply(
            self,
            x,
            router_weights,
            top_experts,
            self.routed_fc1_weight,
            self.routed_down_weight,
            self.shared_gate_weight,
            self.shared_up_weight,
            self.shared_down_weight,
        )
        return output.view(original_shape)
