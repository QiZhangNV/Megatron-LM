# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MCore-owned functional runtime around the current FK MegaMoE runners.

The external FK tree remains unmodified. This module turns its tester-oriented
entry points into one shared, versioned runtime per MCore EP group and model
shape, scopes NVSHMEM to that EP group, and supplies real MCore tensors.
"""

from __future__ import annotations

import atexit
import gc
import math
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from megatron.core.transformer.moe.megakernel.fk.route_padding import (
    build_route_padding_tensors,
    calculate_local_route_capacity,
)
from megatron.core.transformer.moe.megakernel.fk.weights import FkWeightView

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass(frozen=True)
class FkRuntimeConfig:
    """Static FK runtime and kernel choices supported by the first integration."""

    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_local_experts: int
    topk: int
    capacity_factor: float
    swiglu_limit: float | None
    fwd_group_hint: int
    fwd_col_quant_num_ctas: int
    bwd_token_back_mode: str
    bwd_epi_flag_batch: tuple[int, int]

    @classmethod
    def from_transformer_config(
        cls, config: "TransformerConfig", *, num_local_experts: int
    ) -> "FkRuntimeConfig":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_ffn_hidden_size,
            num_experts=config.num_moe_experts,
            num_local_experts=num_local_experts,
            topk=config.moe_router_topk,
            capacity_factor=config.fk_expert_rank_capacity_factor,
            swiglu_limit=config.activation_func_clamp_value,
            fwd_group_hint=config.fk_fwd_group_hint,
            fwd_col_quant_num_ctas=config.fk_fwd_col_quant_num_ctas,
            bwd_token_back_mode=config.fk_bwd_token_back_mode,
            bwd_epi_flag_batch=tuple(config.fk_bwd_epi_flag_batch),
        )


@dataclass
class _SystemDependencies:
    te: Any
    tex: Any
    cudnn_wgrad: Any
    nvshmem: Any
    cuda_device_cls: Any


@dataclass
class _PaddedRoutes:
    activation: torch.Tensor
    router_weights: torch.Tensor
    top_experts: torch.Tensor
    local_counts: torch.Tensor
    padded_counts: torch.Tensor
    original_tokens: int


@dataclass
class FkForwardContext:
    """Per-autograd-call state that cannot live in the shared FK workspace."""

    original_tokens: int
    router_weights: torch.Tensor
    top_experts: torch.Tensor
    local_counts: torch.Tensor
    preactivation: torch.Tensor
    route_index: torch.Tensor
    fc1_x_data: torch.Tensor
    fc1_x_scale: torch.Tensor
    fc1_x_metadata: torch.Tensor | None


_SYSTEM_DEPS: _SystemDependencies | None = None
_FK_PACKAGES_SELECTED = False
_NVSHMEM_STATE: tuple[tuple[int, ...], int] | None = None
_RUNTIME_CACHE: dict[tuple[Any, ...], "FkRuntime"] = {}


def _mxfp8_scale_dtype() -> torch.dtype:
    """Resolve FK's public MXFP8 scale dtype for the selected data kind."""
    from common.host_utils import kind_scale_dtype

    return kind_scale_dtype("mxfp8_e4m3")


def _temporary_site_import(site_packages: str, module_name: str):
    sys.path.insert(0, site_packages)
    try:
        return __import__(module_name, fromlist=["*"])
    finally:
        try:
            sys.path.remove(site_packages)
        except ValueError:
            pass


def _prepare_system_dependencies() -> _SystemDependencies:
    """Resolve image TE/cuDNN before selecting FK's CuTe DSL package."""
    global _SYSTEM_DEPS
    if _SYSTEM_DEPS is not None:
        return _SYSTEM_DEPS
    site_packages = os.environ.get("FK_VENV_SITE_PACKAGES")
    if not site_packages or not os.path.isdir(site_packages):
        raise RuntimeError(
            "FK_VENV_SITE_PACKAGES must point at the prepared FK virtualenv site-packages"
        )
    import cudnn
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex

    nvshmem = _temporary_site_import(site_packages, "nvshmem.core")
    try:
        cuda_core = _temporary_site_import(site_packages, "cuda.core")
        cuda_device_cls = cuda_core.Device
    except (ImportError, AttributeError):
        cuda_core = _temporary_site_import(site_packages, "cuda.core.experimental")
        cuda_device_cls = cuda_core.Device
    _SYSTEM_DEPS = _SystemDependencies(
        te=te,
        tex=tex,
        cudnn_wgrad=cudnn.grouped_gemm_wgrad_wrapper_sm100,
        nvshmem=nvshmem,
        cuda_device_cls=cuda_device_cls,
    )
    return _SYSTEM_DEPS


def _select_fk_packages() -> None:
    """Select FK's CuTe DSL only after system cuDNN has been resolved."""
    global _FK_PACKAGES_SELECTED
    if _FK_PACKAGES_SELECTED:
        return
    site_packages = os.environ["FK_VENV_SITE_PACKAGES"]
    fk_root = os.environ.get("FK_ROOT")
    if not fk_root or not os.path.isdir(fk_root):
        raise RuntimeError("FK_ROOT must point at the frozen FK source tree")
    for name in tuple(sys.modules):
        if name == "nvidia_cutlass_dsl" or name.startswith("nvidia_cutlass_dsl."):
            del sys.modules[name]
        elif name == "cutlass" or name.startswith("cutlass."):
            del sys.modules[name]
    for path in (site_packages, fk_root):
        try:
            sys.path.remove(path)
        except ValueError:
            pass
        sys.path.insert(0, path)
    import nvidia_cutlass_dsl

    dsl_packages = os.path.join(nvidia_cutlass_dsl.__path__[0], "dsl_packages")
    try:
        sys.path.remove(dsl_packages)
    except ValueError:
        pass
    sys.path.insert(0, dsl_packages)
    _FK_PACKAGES_SELECTED = True


def _group_ranks(group: ProcessGroup) -> tuple[int, ...]:
    try:
        return tuple(dist.get_process_group_ranks(group))
    except AttributeError:
        return tuple(
            dist.get_global_rank(group, rank)
            for rank in range(dist.get_world_size(group))
        )


def _ensure_nvshmem(ep_group: ProcessGroup) -> tuple[int, int]:
    """Initialize exactly one NVSHMEM world from this rank's MCore EP group."""
    global _NVSHMEM_STATE
    if not dist.is_initialized():
        raise RuntimeError("FK requires torch.distributed to be initialized")
    deps = _prepare_system_dependencies()
    ranks = _group_ranks(ep_group)
    ep_rank = dist.get_rank(ep_group)
    ep_size = dist.get_world_size(ep_group)
    device_index = torch.cuda.current_device()
    if _NVSHMEM_STATE is not None:
        if _NVSHMEM_STATE != (ranks, device_index):
            raise RuntimeError(
                "One process cannot initialize FK NVSHMEM for multiple EP groups/devices: "
                f"existing={_NVSHMEM_STATE}, requested={(ranks, device_index)}"
            )
        return ep_rank, ep_size

    device = deps.cuda_device_cls(device_index)
    device.set_current()
    uid = deps.nvshmem.get_unique_id(empty=(ep_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).to(device="cuda")
    dist.broadcast(uid_tensor, src=ranks[0], group=ep_group)
    dist.barrier(group=ep_group)
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    deps.nvshmem.init(
        device=device,
        uid=uid,
        rank=ep_rank,
        nranks=ep_size,
        initializer_method="uid",
    )
    _NVSHMEM_STATE = (ranks, device_index)
    return ep_rank, ep_size


def _make_quantizer(*, rowwise: bool, columnwise: bool):
    deps = _prepare_system_dependencies()
    quantizer = deps.te.MXFP8Quantizer(
        deps.tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    quantizer.optimize_for_gemm = True
    return quantizer


def _group_quantize(
    tensor: torch.Tensor,
    *,
    groups: int,
    split_sizes: torch.Tensor | None = None,
    rowwise: bool = True,
    columnwise: bool = False,
):
    deps = _prepare_system_dependencies()
    return deps.tex.group_quantize(
        tensor,
        _make_quantizer(rowwise=rowwise, columnwise=columnwise),
        groups,
        split_sizes,
    )


def _unswizzle_row_scales(
    scale_inv: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    blocked = scale_inv.view(torch.uint8).reshape(rows // 128, columns // 128, 32, 4, 4)
    return (
        blocked.permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(rows, columns // 32)
        .view(torch.float8_e8m0fnu)
    )


def _to_cute(tensor: torch.Tensor, *, assumed_align: int = 16, static: bool = False):
    import cutlass.torch as cutlass_torch

    result = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    if static:
        return result
    return result.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))


def _raw_cudnn_operands(
    data: torch.Tensor,
    scales: torch.Tensor,
    total_tokens: int,
    features: int,
    *,
    transpose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = data[:total_tokens].reshape(total_tokens, features)
    if transpose:
        matrix = matrix.T
    scale_count = math.ceil(features / 128) * 128 * (total_tokens // 32)
    # Forward exposes a flat SF workspace while the backward runner exposes
    # the same byte layout through a 2D auxiliary tensor. Flatten before
    # truncating so both representations produce the cuDNN (M, K / 32) view.
    scale = scales.view(torch.uint8).reshape(-1)[:scale_count].view(
        math.ceil(features / 128) * 128, -1
    )
    return matrix, scale.view(torch.float8_e8m0fnu)


class FkRuntime:
    """Shared FK runners/workspaces for one EP group, shape, and token capacity."""

    def __init__(
        self,
        config: FkRuntimeConfig,
        ep_group: ProcessGroup,
        *,
        num_local_tokens: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.ep_group = ep_group
        self.device = device
        self.ep_rank, self.ep_size = _ensure_nvshmem(ep_group)
        if config.num_experts != self.ep_size * config.num_local_experts:
            raise RuntimeError(
                "FK assumes contiguous expert ownership: "
                f"experts={config.num_experts}, EP={self.ep_size}, "
                f"local_experts={config.num_local_experts}"
            )
        if torch.cuda.get_device_capability(device) != (10, 3):
            raise RuntimeError("FK MVP requires GB300/SM103")
        self.num_local_tokens = num_local_tokens
        self.local_capacity = calculate_local_route_capacity(
            num_local_tokens=num_local_tokens,
            topk=config.topk,
            num_local_experts=config.num_local_experts,
            capacity_factor=config.capacity_factor,
        )
        self.padded_num_local_tokens = self.local_capacity // config.topk
        self.forward_runner = None
        self.backward_runner = None
        self._col_requant_compiled = None
        self._col_requant_kwargs = None
        self._col_requant_sizes = None
        self._fc2_dy_data = None
        self._fc2_dy_scale = None
        self._forward_calls = 0
        self._backward_calls = 0
        self._debug_enabled = os.environ.get("FK_MCORE_DEBUG", "0") == "1"

        deps = _prepare_system_dependencies()
        self._precompile_wgrads(deps.cudnn_wgrad)
        _select_fk_packages()
        if self.ep_rank == 0:
            print(
                "FK_MCORE_RUNTIME "
                f"ep={self.ep_size} tokens={num_local_tokens} "
                f"padded_tokens={self.padded_num_local_tokens} "
                f"local_route_capacity={self.local_capacity} "
                f"capacity_factor={config.capacity_factor}",
                flush=True,
            )

    def _debug(self, event: str, *, synchronize: bool = False) -> None:
        """Emit opt-in runtime markers without affecting the default FK path."""
        if not self._debug_enabled:
            return
        if synchronize:
            torch.cuda.synchronize()
        print(
            "FK_MCORE_DEBUG "
            f"rank={self.ep_rank} forward_call={self._forward_calls} "
            f"backward_call={self._backward_calls} event={event}",
            flush=True,
        )

    def _debug_fc1_wgrad_alignment(self, context: FkForwardContext) -> None:
        """Report whether forward FC1 inputs match backward dGLU pool rows."""
        if not self._debug_enabled or context.fc1_x_metadata is None:
            return
        backward_metadata = self._workspace_view(
            self.backward_runner, "token_src_metadata", torch.int64
        )[: self.local_capacity]
        positional_matches = backward_metadata == context.fc1_x_metadata
        print(
            "FK_MCORE_ALIGNMENT "
            f"rank={self.ep_rank} forward_call={self._forward_calls} "
            f"backward_call={self._backward_calls} "
            f"matched_rows={int(positional_matches.sum().item())} "
            f"total_rows={self.local_capacity} "
            f"match_fraction={float(positional_matches.float().mean().item()):.6f}",
            flush=True,
        )

    def _ep_barrier(self) -> None:
        torch.cuda.synchronize()
        dist.barrier(group=self.ep_group)
        _prepare_system_dependencies().nvshmem.barrier_all(torch.cuda.current_stream())
        torch.cuda.synchronize()

    def _launch_distributed_kernel(self, compiled_kernel, kwargs) -> None:
        """Launch a reused FK kernel with the vendor runner's rendezvous protocol."""
        self._ep_barrier()
        compiled_kernel(**kwargs)
        self._ep_barrier()

    def _precompile_wgrads(self, cudnn_wgrad) -> None:
        counts = torch.full(
            (self.config.num_local_experts,),
            self.local_capacity // self.config.num_local_experts,
            dtype=torch.int32,
            device=self.device,
        )
        offsets = torch.cumsum(counts, dim=0).to(torch.int32)

        def compile_one(out_features: int, in_features: int) -> None:
            total = self.local_capacity
            dy = torch.zeros(
                (total, out_features), dtype=torch.float8_e4m3fn, device=self.device
            )
            x = torch.zeros(
                (total, in_features), dtype=torch.float8_e4m3fn, device=self.device
            )
            dy_sf = torch.zeros(
                (math.ceil(out_features / 128) * 128, total // 32),
                dtype=torch.uint8,
                device=self.device,
            ).view(torch.float8_e8m0fnu)
            x_sf = torch.zeros(
                (math.ceil(in_features / 128) * 128, total // 32),
                dtype=torch.uint8,
                device=self.device,
            ).view(torch.float8_e8m0fnu)
            output = torch.zeros(
                (self.config.num_local_experts, out_features, in_features),
                dtype=torch.bfloat16,
                device=self.device,
            )
            cudnn_wgrad(
                a_tensor=dy.T,
                b_tensor=x,
                sfa_tensor=dy_sf,
                sfb_tensor=x_sf,
                offsets_tensor=offsets,
                output_mode="dense",
                wgrad_tensor=output,
                acc_dtype=torch.float32,
                wgrad_dtype=torch.bfloat16,
                sf_vec_size=32,
                # Compile the exact mode used by training before FK replaces
                # the image-owned CuTeDSL modules in sys.modules.
                accumulate_on_output=True,
                current_stream=torch.cuda.current_stream().cuda_stream,
            )
            torch.cuda.synchronize()

        compile_one(2 * self.config.intermediate_size, self.config.hidden_size)
        compile_one(self.config.hidden_size, self.config.intermediate_size)

    def _pad_routes(
        self,
        activation: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
    ) -> _PaddedRoutes:
        if activation.shape != (self.num_local_tokens, self.config.hidden_size):
            raise RuntimeError(
                f"FK activation shape changed: {tuple(activation.shape)}"
            )
        expected_route_shape = (self.num_local_tokens, self.config.topk)
        if (
            router_weights.shape != expected_route_shape
            or top_experts.shape != expected_route_shape
        ):
            raise RuntimeError(
                "FK compact route shape mismatch: "
                f"weights={tuple(router_weights.shape)}, indices={tuple(top_experts.shape)}, "
                f"expected={expected_route_shape}"
            )
        if top_experts.dtype != torch.int64:
            raise TypeError("FK compact expert indices must be int64")
        invalid = (top_experts < 0) | (top_experts >= self.config.num_experts)
        torch._assert_async(
            torch.logical_not(invalid.any()),
            "FK MVP requires exactly topk valid expert indices per token",
        )
        self._debug("pad_routes_before_counts_all_reduce")
        counts = torch.bincount(
            top_experts.flatten(), minlength=self.config.num_experts
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.ep_group)
        self._debug("pad_routes_after_counts_all_reduce")
        padded_counts, dummy_experts_by_source_rank = build_route_padding_tensors(
            counts,
            ep_size=self.ep_size,
            num_local_tokens=self.num_local_tokens,
            topk=self.config.topk,
            local_capacity=self.local_capacity,
        )
        dummy_tokens = self.padded_num_local_tokens - self.num_local_tokens
        dummy_experts = dummy_experts_by_source_rank[self.ep_rank]
        padded_activation = torch.cat(
            (
                activation,
                torch.zeros(
                    (dummy_tokens, self.config.hidden_size),
                    dtype=activation.dtype,
                    device=self.device,
                ),
            ),
            dim=0,
        )
        padded_weights = torch.cat(
            (
                router_weights,
                torch.zeros(
                    (dummy_tokens, self.config.topk),
                    dtype=router_weights.dtype,
                    device=self.device,
                ),
            ),
            dim=0,
        )
        padded_experts = torch.cat((top_experts, dummy_experts), dim=0)
        local_counts = padded_counts.reshape(
            self.ep_size, self.config.num_local_experts
        )[self.ep_rank].to(torch.int32)
        return _PaddedRoutes(
            activation=padded_activation,
            router_weights=padded_weights,
            top_experts=padded_experts,
            local_counts=local_counts,
            padded_counts=padded_counts,
            original_tokens=self.num_local_tokens,
        )

    def _stage_row_quant(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        destination_scale: torch.Tensor,
    ):
        quantized = _group_quantize(source, groups=1, rowwise=True, columnwise=False)
        data = quantized.rowwise_data.view(torch.float8_e4m3fn).reshape(
            self.padded_num_local_tokens, self.config.hidden_size
        )
        scales = _unswizzle_row_scales(
            quantized.scale_inv, self.padded_num_local_tokens, self.config.hidden_size
        )
        destination.view(torch.uint8).copy_(data.view(torch.uint8))
        destination_scale.view(torch.uint8)[:, : self.config.hidden_size // 32].copy_(
            scales.view(torch.uint8)
        )
        return quantized

    def _set_forward_weights(
        self, runner, fc1: FkWeightView, fc2: FkWeightView
    ) -> None:
        runner.my_fc1_weight = fc1.forward_data
        runner.my_fc1_weight_sf = fc1.forward_scale
        runner.my_fc2_weight = fc2.forward_data
        runner.my_fc2_weight_sf = fc2.forward_scale

    def _build_forward_runner(
        self, routes: _PaddedRoutes, fc1: FkWeightView, fc2: FkWeightView
    ) -> None:
        from moe_mxfp8_glu.mega_runner import (
            MegaMoEMxfp8Tester,
            _sym_zeros,
            _sym_zeros_byte_view_1b,
        )
        from moe_mxfp8_glu.runner_common import TrainingForwardImplDesc
        from moe_nvfp4_swapab.mega_runner import MiscDesc, TokenCommProblemDesc
        from src.token_comm import CombineFormat

        combine = CombineFormat.parse("32e4m3xe8m0")
        problem = TokenCommProblemDesc(
            world_size=self.ep_size,
            num_tokens_per_rank=self.padded_num_local_tokens,
            num_topk=self.config.topk,
            num_total_experts=self.config.num_experts,
            hidden=self.config.hidden_size,
            intermediate=2 * self.config.intermediate_size,
            fc2_output_dtype=torch.bfloat16,
            combine_format=combine,
            route_distribution="balanced",
            power_law_exponent=1.0,
            gate_up_clamp=self.config.swiglu_limit,
        )
        impl = TrainingForwardImplDesc(
            mma_tiler_mnk=(256, 256, 128),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            enable_static_expert_shape=False,
            force_static_sched=True,
            clc_bundle_size=None,
            num_sched_stages=None,
            load_balance_mode="static",
            group_hint=self.config.fwd_group_hint,
            non_ubulk_fc2_store=True,
            in_kernel_fc2_reduce=False,
            token_back_mode="standalone_warps",
            epi_flag_batch=(1, 4),
            flag_batch=1,
            generate_c=True,
            col_quant_num_ctas=self.config.fwd_col_quant_num_ctas,
            use_stg_fc1=False,
            act_func="swiglu",
        )
        misc = MiscDesc(
            perf_run=False,
            skip_ref_check=True,
            run_target_kernel_only=False,
            enable_debug_checks=False,
            ref_compute_graph="transformers",
            enable_iket=False,
            seed=1234,
        )
        runner = MegaMoEMxfp8Tester(
            problem,
            impl,
            misc,
            rank=self.ep_rank,
            kind="mxfp8_e4m3",
            combine_format=combine,
        )
        t, h, k = (
            self.padded_num_local_tokens,
            self.config.hidden_size,
            self.config.topk,
        )
        runner.my_activation = _sym_zeros_byte_view_1b((t, h), torch.float8_e4m3fn)
        runner.my_activation_sf = _sym_zeros_byte_view_1b(
            (t, math.ceil(h / 32 / 4) * 4), _mxfp8_scale_dtype()
        )
        runner.my_topk_idx = _sym_zeros((t, k), torch.int64)
        runner.my_topk_weights = _sym_zeros((t, k), torch.float32)
        runner.output_activation = torch.zeros(
            (t, h), dtype=torch.bfloat16, device=self.device
        )
        self._set_forward_weights(runner, fc1, fc2)
        runner.my_topk_idx.copy_(routes.top_experts)
        runner.my_topk_weights.copy_(routes.router_weights)
        quantized = self._stage_row_quant(
            routes.activation, runner.my_activation, runner.my_activation_sf
        )
        counts = routes.padded_counts.to(torch.int64)
        runner._global_topk_idx = torch.repeat_interleave(
            torch.arange(self.config.num_experts, device=self.device),
            counts,
            output_size=self.ep_size * t * k,
        ).reshape(self.ep_size, t, k)
        self._ep_barrier()
        runner.run_kernel()
        self._ep_barrier()
        runner._global_topk_idx = None
        runner._mcore_activation_quant = quantized
        self.forward_runner = runner

    def _update_forward_runtime(
        self,
        routes: _PaddedRoutes,
        fc1: FkWeightView,
        fc2: FkWeightView,
        *,
        output: torch.Tensor,
        preactivation: torch.Tensor,
        route_index: torch.Tensor,
    ) -> None:
        import cuda.bindings.driver as cuda

        runner = self.forward_runner
        self._set_forward_weights(runner, fc1, fc2)
        runner.my_topk_idx.copy_(routes.top_experts)
        runner.my_topk_weights.copy_(routes.router_weights)
        quantized = self._stage_row_quant(
            routes.activation, runner.my_activation, runner.my_activation_sf
        )
        kwargs = runner._runtime_kwargs
        kwargs["fc1_weight"] = _to_cute(fc1.forward_data)
        kwargs["fc1_weight_sf"] = _to_cute(fc1.forward_scale)
        kwargs["fc2_weight"] = _to_cute(fc2.forward_data)
        kwargs["fc2_weight_sf"] = _to_cute(fc2.forward_scale)
        kwargs["output_activation"] = _to_cute(output)
        kwargs["fc1_c"] = _to_cute(preactivation)
        kwargs["fc1_c_route_index"] = _to_cute(route_index, assumed_align=4)
        kwargs["stream"] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._debug("update_forward_before_kernel")
        self._launch_distributed_kernel(runner._compiled_kernel, kwargs)
        self._debug("update_forward_after_kernel")
        runner._mcore_activation_quant = quantized

    def forward(
        self,
        activation: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        fc1: FkWeightView,
        fc2: FkWeightView,
    ) -> tuple[torch.Tensor, FkForwardContext]:
        self._forward_calls += 1
        self._debug("forward_enter")
        routes = self._pad_routes(activation, router_weights, top_experts)
        self._debug("forward_after_pad_routes")
        if self.forward_runner is None:
            self._debug("forward_before_build_runner")
            self._build_forward_runner(routes, fc1, fc2)
            self._debug("forward_after_build_runner", synchronize=True)
            runner = self.forward_runner
            output = runner.output_activation
            preactivation = runner._c_output
            route_index = runner._c_route_index
        else:
            output = torch.zeros(
                (self.padded_num_local_tokens, self.config.hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            preactivation = torch.empty(
                (self.local_capacity, 2 * self.config.intermediate_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            route_index = torch.empty(
                (self.ep_size * self.padded_num_local_tokens * self.config.topk,),
                dtype=torch.int32,
                device=self.device,
            )
            self._update_forward_runtime(
                routes,
                fc1,
                fc2,
                output=output,
                preactivation=preactivation,
                route_index=route_index,
            )
        self._debug("forward_before_context_copy")
        if preactivation.shape[0] != self.local_capacity:
            raise RuntimeError(
                "FK forward preactivation capacity mismatch: "
                f"got={preactivation.shape[0]}, expected={self.local_capacity}"
            )
        sf_count = (
            math.ceil(self.config.hidden_size / 128) * 128 * (self.local_capacity // 32)
        )
        fc1_x_data = self.forward_runner.col_quant_data[: self.local_capacity].clone()
        fc1_x_scale = self.forward_runner.col_quant_sf.view(torch.uint8)[
            :sf_count
        ].clone()
        fc1_x_metadata = None
        if self._debug_enabled:
            fc1_x_metadata = self._workspace_view(
                self.forward_runner, "token_src_metadata", torch.int64
            )[: self.local_capacity].clone()
        context = FkForwardContext(
            original_tokens=routes.original_tokens,
            router_weights=routes.router_weights,
            top_experts=routes.top_experts,
            local_counts=routes.local_counts,
            preactivation=preactivation,
            route_index=route_index,
            fc1_x_data=fc1_x_data,
            fc1_x_scale=fc1_x_scale,
            fc1_x_metadata=fc1_x_metadata,
        )
        self._debug("forward_exit", synchronize=True)
        return output[: routes.original_tokens], context

    def _set_backward_weights(
        self, runner, fc1: FkWeightView, fc2: FkWeightView
    ) -> None:
        # FK backward runner calls the down-projection dgrad weight "fc1" and
        # the gate/up dgrad weight "fc2".
        runner.fc1_weight = fc2.backward_data
        runner.fc1_weight_sf = fc2.backward_scale
        runner.fc2_weight = fc1.backward_data
        runner.fc2_weight_sf = fc1.backward_scale

    def _build_backward_runner(
        self,
        context: FkForwardContext,
        padded_grad_output: torch.Tensor,
        fc1: FkWeightView,
        fc2: FkWeightView,
    ) -> None:
        from moe_mxfp8_dglu.mega_runner import MegaDswigluMxfp8Tester, _sym_zeros
        from moe_mxfp8_glu.runner_common import TrainingBackwardImplDesc
        from moe_nvfp4_swapab.runner_fc12_common import MiscDesc, ProblemDesc
        from src.token_comm import CombineFormat

        combine = CombineFormat.parse("32e4m3xe8m0")
        problem = ProblemDesc(
            tokens_after_topk=self.padded_num_local_tokens * self.config.topk,
            experts=self.config.num_local_experts,
            hidden=self.config.hidden_size,
            intermediate=2 * self.config.intermediate_size,
            kind="mxfp8_e4m3",
        )
        impl = TrainingBackwardImplDesc(
            mma_tiler_mnk=(256, 256, 128),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            enable_static_expert_shape=False,
            force_static_sched=True,
            clc_bundle_size=None,
            num_sched_stages=None,
            load_balance_mode="static",
            group_hint=None,
            non_ubulk_fc2_store=True,
            epi_flag_batch=self.config.bwd_epi_flag_batch,
            token_back_mode=self.config.bwd_token_back_mode,
            in_kernel_fc2_reduce=False,
            dfc2_recompute=True,
            dfc2_col_output=True,
            act_func="swiglu",
        )
        misc = MiscDesc(
            perf_run=False,
            skip_ref_check=True,
            run_target_kernel_only=False,
            enable_debug_checks=False,
            ref_compute_graph="transformers",
            enable_iket=False,
            seed=1234,
        )
        runner = MegaDswigluMxfp8Tester(
            problem,
            impl,
            misc,
            rank=self.ep_rank,
            world_size=self.ep_size,
            num_topk=self.config.topk,
            max_tokens_per_rank=self.padded_num_local_tokens,
            route_distribution="balanced",
            combine_format=combine,
        )
        t, h, k = (
            self.padded_num_local_tokens,
            self.config.hidden_size,
            self.config.topk,
        )
        runner.my_grad_out = _sym_zeros((t, h), torch.uint8).view(torch.float8_e4m3fn)
        runner.my_grad_out_sf = _sym_zeros(
            (t, math.ceil(h / 32 / 4) * 4), torch.uint8
        ).view(_mxfp8_scale_dtype())
        runner.my_topk_idx = _sym_zeros((t, k), torch.int64)
        runner.my_topk_weights = _sym_zeros((t, k), torch.float32)
        runner.grad_activation = torch.zeros(
            (t, h), dtype=torch.bfloat16, device=self.device
        )
        runner.dprob = _sym_zeros((t, k), torch.float32)
        runner.beta = torch.ones(
            (self.config.num_local_experts,), dtype=torch.float32, device=self.device
        )
        self._set_backward_weights(runner, fc1, fc2)
        runner.saved_fc1_preact = context.preactivation
        runner.saved_preact_route_index = context.route_index
        runner.my_topk_idx.copy_(context.top_experts)
        runner.my_topk_weights.copy_(context.router_weights)
        quantized = self._stage_row_quant(
            padded_grad_output, runner.my_grad_out, runner.my_grad_out_sf
        )
        runner._dist_barrier = self._ep_barrier
        runner.run_kernel()
        runner._mcore_grad_quant = quantized
        self.backward_runner = runner
        self._compile_col_requant(context.local_counts)

    @staticmethod
    def _workspace_view(runner, name: str, dtype: torch.dtype) -> torch.Tensor:
        spec = runner._kernel._local_region_by_name[name]
        offset = runner._kernel._local_offsets[name]
        return runner.local_workspace[offset : offset + spec.nbytes].view(dtype)

    def _compile_col_requant(self, local_counts: torch.Tensor) -> None:
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from moe_mxfp8_glu.mxfp8_col_requant import Mxfp8ColRequant

        runner = self.backward_runner
        pool_capacity = runner._kernel.pool_token_capacity
        source_data = self._workspace_view(
            runner, "l1_token_buffer", torch.float8_e4m3fn
        ).reshape(pool_capacity, self.config.hidden_size)
        source_scale = self._workspace_view(runner, "l1_sf_buffer", torch.uint8)
        destination = torch.zeros(
            (pool_capacity, self.config.hidden_size),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        destination_scale = torch.zeros(
            (pool_capacity * self.config.hidden_size // 32,),
            dtype=torch.uint8,
            device=self.device,
        )
        self._col_requant_sizes = local_counts.to(torch.int32).contiguous().clone()
        launcher = Mxfp8ColRequant(
            hidden=self.config.hidden_size,
            num_experts=self.config.num_local_experts,
            max_total_tokens=self.ep_size
            * self.padded_num_local_tokens
            * self.config.topk,
            quant_type="mxfp8_e4m3",
            num_persistent_ctas=self.config.fwd_col_quant_num_ctas,
            token_padding_block=128,
            sf_padding_block=128,
        )
        kwargs = dict(
            src_data=_to_cute(source_data),
            src_sf_u8=_to_cute(source_scale),
            expert_token_sizes=_to_cute(self._col_requant_sizes),
            dst_data=_to_cute(destination),
            dst_sf_u8=_to_cute(destination_scale),
            cuda_stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        compiled = cute.compile(launcher, **kwargs)
        compiled(**kwargs)
        self._col_requant_compiled = compiled
        self._col_requant_kwargs = kwargs
        self._fc2_dy_data = destination
        self._fc2_dy_scale = destination_scale

    def _launch_col_requant(self, local_counts: torch.Tensor) -> None:
        import cuda.bindings.driver as cuda

        self._col_requant_sizes.copy_(local_counts)
        self._col_requant_kwargs["cuda_stream"] = cuda.CUstream(
            torch.cuda.current_stream().cuda_stream
        )
        self._col_requant_compiled(**self._col_requant_kwargs)

    def _update_backward_runtime(
        self,
        context: FkForwardContext,
        padded_grad_output: torch.Tensor,
        fc1: FkWeightView,
        fc2: FkWeightView,
    ) -> None:
        import cuda.bindings.driver as cuda

        runner = self.backward_runner
        self._set_backward_weights(runner, fc1, fc2)
        runner.my_topk_idx.copy_(context.top_experts)
        runner.my_topk_weights.copy_(context.router_weights)
        quantized = self._stage_row_quant(
            padded_grad_output, runner.my_grad_out, runner.my_grad_out_sf
        )
        # The kernel tail resets accumulating counters, but data/SF padding in
        # reused workspaces is not guaranteed to be overwritten by every route.
        runner._reset_local_counters()
        runner.grad_activation.zero_()
        runner.dprob.zero_()
        runner.saved_fc1_preact = context.preactivation
        runner.saved_preact_route_index = context.route_index
        kwargs = runner._runtime_kwargs
        kwargs["fc1_weight"] = _to_cute(fc2.backward_data)
        kwargs["fc1_weight_sf"] = _to_cute(fc2.backward_scale)
        kwargs["fc2_weight"] = _to_cute(fc1.backward_data)
        kwargs["fc2_weight_sf"] = _to_cute(fc1.backward_scale)
        kwargs["fc1_preact"] = _to_cute(context.preactivation, assumed_align=128)
        kwargs["saved_preact_route_index"] = _to_cute(context.route_index)
        kwargs["stream"] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._debug("update_backward_before_kernel")
        # Match MegaDswigluMxfp8Tester's repeated-launch protocol: local
        # counters are reset above, then every distributed kernel launch is
        # bracketed by torch.distributed + NVSHMEM barriers.  Reusing the
        # communication workspace without this rendezvous can leave peer
        # ranks observing counters from different launches and deadlock.
        self._launch_distributed_kernel(runner._compiled_kernel, kwargs)
        self._debug("update_backward_after_kernel")
        runner._mcore_grad_quant = quantized
        self._debug("update_backward_before_col_requant")
        self._launch_col_requant(context.local_counts)
        self._debug("update_backward_after_col_requant", synchronize=True)

    def _launch_wgrad(
        self,
        output: torch.Tensor,
        dy_data: torch.Tensor,
        dy_scale: torch.Tensor,
        x_data: torch.Tensor,
        x_scale: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        if output.dtype != torch.bfloat16:
            raise RuntimeError(f"FK MVP requires BF16 main_grad, got {output.dtype}")
        _prepare_system_dependencies().cudnn_wgrad(
            a_tensor=dy_data,
            b_tensor=x_data,
            sfa_tensor=dy_scale,
            sfb_tensor=x_scale,
            offsets_tensor=offsets,
            output_mode="dense",
            wgrad_tensor=output,
            acc_dtype=torch.float32,
            wgrad_dtype=torch.bfloat16,
            sf_vec_size=32,
            accumulate_on_output=True,
            current_stream=torch.cuda.current_stream().cuda_stream,
        )

    def backward(
        self,
        context: FkForwardContext,
        grad_output: torch.Tensor,
        fc1: FkWeightView,
        fc2: FkWeightView,
        fc1_main_grad: torch.Tensor,
        fc2_main_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._backward_calls += 1
        self._debug("backward_enter")
        dummy_tokens = self.padded_num_local_tokens - context.original_tokens
        padded_grad_output = torch.cat(
            (
                grad_output.contiguous(),
                torch.zeros(
                    (dummy_tokens, self.config.hidden_size),
                    dtype=grad_output.dtype,
                    device=self.device,
                ),
            ),
            dim=0,
        )
        if self.backward_runner is None:
            self._debug("backward_before_build_runner")
            self._build_backward_runner(context, padded_grad_output, fc1, fc2)
            self._debug("backward_after_build_runner", synchronize=True)
        else:
            self._update_backward_runtime(context, padded_grad_output, fc1, fc2)
        runner = self.backward_runner
        self._debug_fc1_wgrad_alignment(context)
        total = self.local_capacity
        offsets = torch.cumsum(context.local_counts, dim=0).to(torch.int32)
        fc1_dy, fc1_dy_scale = _raw_cudnn_operands(
            runner.fc1_col_output,
            runner.fc1_col_output_sf,
            total,
            2 * self.config.intermediate_size,
            transpose=True,
        )
        fc1_x, fc1_x_scale = _raw_cudnn_operands(
            context.fc1_x_data,
            context.fc1_x_scale,
            total,
            self.config.hidden_size,
            transpose=False,
        )
        fc2_dy, fc2_dy_scale = _raw_cudnn_operands(
            self._fc2_dy_data,
            self._fc2_dy_scale,
            total,
            self.config.hidden_size,
            transpose=True,
        )
        fc2_x, fc2_x_scale = _raw_cudnn_operands(
            runner.fc1_recompute,
            runner.fc1_recompute_sf,
            total,
            self.config.intermediate_size,
            transpose=False,
        )
        self._debug("backward_before_fc1_wgrad")
        self._launch_wgrad(
            fc1_main_grad, fc1_dy, fc1_dy_scale, fc1_x, fc1_x_scale, offsets
        )
        self._debug("backward_after_fc1_wgrad", synchronize=True)
        self._debug("backward_before_fc2_wgrad")
        self._launch_wgrad(
            fc2_main_grad, fc2_dy, fc2_dy_scale, fc2_x, fc2_x_scale, offsets
        )
        self._debug("backward_after_fc2_wgrad", synchronize=True)
        self._debug("backward_exit")
        return (
            runner.grad_activation[: context.original_tokens],
            runner.dprob[: context.original_tokens],
        )

    def release(self) -> None:
        """Best-effort release of the shared symmetric allocations at process exit."""
        deps = _SYSTEM_DEPS
        if deps is None:
            return
        for runner, names in (
            (
                self.forward_runner,
                (
                    "my_activation",
                    "my_activation_sf",
                    "my_topk_idx",
                    "my_topk_weights",
                    "output_activation",
                    "shared_workspace",
                ),
            ),
            (
                self.backward_runner,
                (
                    "my_grad_out",
                    "my_grad_out_sf",
                    "my_topk_idx",
                    "my_topk_weights",
                    "grad_activation",
                    "dprob",
                    "shared_workspace",
                ),
            ),
        ):
            if runner is None:
                continue
            runner._compiled_kernel = None
            runner._kernel = None
            for name in names:
                tensor = getattr(runner, name, None)
                if tensor is not None:
                    try:
                        deps.nvshmem.free_tensor(tensor)
                    except Exception:  # noqa: BLE001
                        pass
                setattr(runner, name, None)
        gc.collect()


def get_fk_runtime(
    config: FkRuntimeConfig,
    ep_group: ProcessGroup,
    *,
    num_local_tokens: int,
    device: torch.device,
) -> FkRuntime:
    """Return the process-local shared runtime for this EP group and shape."""
    ranks = _group_ranks(ep_group)
    key = (config, ranks, num_local_tokens, device.index)
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = FkRuntime(
            config,
            ep_group,
            num_local_tokens=num_local_tokens,
            device=device,
        )
        _RUNTIME_CACHE[key] = runtime
    return runtime


@atexit.register
def _finalize_fk_runtime() -> None:
    global _NVSHMEM_STATE
    for runtime in tuple(_RUNTIME_CACHE.values()):
        runtime.release()
    _RUNTIME_CACHE.clear()
    if _SYSTEM_DEPS is not None and _NVSHMEM_STATE is not None:
        try:
            _SYSTEM_DEPS.nvshmem.finalize()
        except Exception:  # noqa: BLE001
            pass
    _NVSHMEM_STATE = None
