# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MCore-owned adapter for the frozen FK MegaMoE training kernels.

MCore remains authoritative for routing, parameters, DDP, optimizer state, and
checkpoints. FK replaces only the routed expert dispatch/compute/combine path;
the shared expert remains the ordinary native MCore module.
"""

from __future__ import annotations

from contextlib import nullcontext
import weakref
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed import ProcessGroup

from megatron.core.transformer.moe.megakernel.backend import MegakernelBackend
from megatron.core.transformer.moe.megakernel.fk.runtime import (
    FkForwardContext,
    FkRuntimeConfig,
    get_fk_runtime,
)
from megatron.core.transformer.moe.megakernel.fk.weights import (
    native_single_grouped_weight_view,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    finish_weight_gradient,
    main_grad_buffer,
)
from megatron.core.transformer.moe.megakernel.route_adapter import (
    routing_map_to_compact_inputs,
)
from megatron.core.transformer.moe.paged_stash import (
    get_paged_stash_context,
    paged_stash_group_commit,
    paged_stash_group_start,
)
from megatron.core.typed_torch import apply_module

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


class _FkAutograd(torch.autograd.Function):
    """Bridge FK's functional routed path to MCore-owned parameters/main grads."""

    @staticmethod
    def forward(
        ctx,
        module: "FkMegakernel",
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        if len(parameters) != 2:
            raise RuntimeError(
                f"FK autograd expected routed FC1/FC2 parameters, got {len(parameters)}"
            )
        fc1_view, fc2_view = module.quantized_routed_weights()
        runtime = get_fk_runtime(
            module.runtime_config,
            module.ep_group,
            num_local_tokens=x.shape[0],
            device=x.device,
        )
        output, forward_context = runtime.forward(
            x, router_weights, top_experts, fc1_view, fc2_view
        )

        # Paged stash only considers tensors carrying this attribute. Keep the
        # route metadata resident, but let MCore page the three large, dense
        # activation payloads that FK needs in backward. The scale workspace is
        # columnwise and therefore has one logical token row per 32 data rows.
        forward_context.preactivation.grouped_tensor_scale_inv = False
        forward_context.fc1_x_data.grouped_tensor_scale_inv = False
        forward_context.fc1_x_scale.grouped_tensor_scale_inv = True
        # The first paged-stash iteration profiles the PP schedule before the
        # reusable CUDA page buffers exist.  Temporarily stage these large FK
        # contexts on host during that one profiling iteration so Full Model
        # can bootstrap; steady-state capture/replay still uses CUDA page stash.
        forward_context.preactivation.paged_stash_capture_to_host = True
        forward_context.fc1_x_data.paged_stash_capture_to_host = True
        forward_context.fc1_x_scale.paged_stash_capture_to_host = True

        ctx.module = module
        ctx.runtime = runtime
        ctx.original_tokens = forward_context.original_tokens
        ctx.has_fc1_x_metadata = forward_context.fc1_x_metadata is not None
        ctx.routed_weight_views = (fc1_view, fc2_view)
        saved_tensors = [
            *parameters,
            forward_context.router_weights,
            forward_context.top_experts,
            forward_context.local_counts,
            forward_context.preactivation,
            forward_context.route_index,
            forward_context.fc1_x_data,
            forward_context.fc1_x_scale,
        ]
        if forward_context.fc1_x_metadata is not None:
            saved_tensors.append(forward_context.fc1_x_metadata)
        # Saving the parameter aliases keeps their version counters and
        # autograd/DDP hook relationship explicit even though FK reads the
        # physical TE payloads. Saving the context tensors makes them visible
        # to MCore's saved_tensors_hooks and hence paged stash.
        ctx.save_for_backward(*saved_tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        module = ctx.module
        fc1_view, fc2_view = ctx.routed_weight_views
        saved_tensors = ctx.saved_tensors
        (
            router_weights,
            top_experts,
            local_counts,
            preactivation,
            route_index,
            fc1_x_data,
            fc1_x_scale,
        ) = saved_tensors[2:9]
        fc1_x_metadata = saved_tensors[9] if ctx.has_fc1_x_metadata else None
        forward_context = FkForwardContext(
            original_tokens=ctx.original_tokens,
            router_weights=router_weights,
            top_experts=top_experts,
            local_counts=local_counts,
            preactivation=preactivation,
            route_index=route_index,
            fc1_x_data=fc1_x_data,
            fc1_x_scale=fc1_x_scale,
            fc1_x_metadata=fc1_x_metadata,
        )
        d_x, d_router_weights = ctx.runtime.backward(
            forward_context,
            grad_output.contiguous(),
            fc1_view,
            fc2_view,
            main_grad_buffer(module.routed_fc1_weight),
            main_grad_buffer(module.routed_fc2_weight),
        )
        routed_parameter_grads = (
            finish_weight_gradient(module.routed_fc1_weight),
            finish_weight_gradient(module.routed_fc2_weight),
        )

        ctx.module = None
        ctx.runtime = None
        ctx.original_tokens = None
        ctx.has_fc1_x_metadata = None
        ctx.routed_weight_views = None
        return None, d_x, d_router_weights, None, *routed_parameter_grads


class FkMegakernel(MegakernelBackend):
    """Execute the routed MoE path with FK and the shared path with MCore."""

    def __init__(
        self,
        config: TransformerConfig,
        ep_group: ProcessGroup,
        routed_experts: nn.Module,
        shared_experts: nn.Module,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        if not config.gradient_accumulation_fusion:
            raise ValueError("FK requires gradient_accumulation_fusion=True")
        if not config.moe_single_grouped_weight:
            raise ValueError("FK MVP requires moe_single_grouped_weight=True")
        if shared_experts is None:
            raise ValueError("FK MVP requires native MCore shared experts")

        fc1 = routed_experts.linear_fc1
        fc2 = routed_experts.linear_fc2
        if not getattr(fc1, "single_grouped_weight", False) or not getattr(
            fc2, "single_grouped_weight", False
        ):
            raise ValueError("FK routed FC1/FC2 must use native single-grouped weights")

        self.config = config
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size
        self.topk = config.moe_router_topk
        self.runtime_config = FkRuntimeConfig.from_transformer_config(
            config, num_local_experts=num_local_experts
        )

        # Aliases let MCore DDP parameter-gather hooks see the parameters used by
        # this module. The native expert module stays their canonical owner.
        self.register_parameter("routed_fc1_weight", fc1.weight)
        self.register_parameter("routed_fc2_weight", fc2.weight)

        # Do not register the shared module a second time. Its ordinary MCore
        # forward below owns autograd, DDP hooks, optimizer state, and checkpoints.
        self._shared_experts_ref = weakref.ref(shared_experts)
        self.is_first_microbatch = True
        self._routed_weight_view_cache = None

    @property
    def shared_experts(self) -> nn.Module:
        module = self._shared_experts_ref()
        if module is None:
            raise RuntimeError(
                "The MCore shared expert module was released unexpectedly"
            )
        return module

    @torch.no_grad()
    def quantized_routed_weights(self):
        """Build or refresh stable FK views over native grouped MXFP8 storage."""
        if self._routed_weight_view_cache is None or self.is_first_microbatch:
            if self._routed_weight_view_cache is None:
                cached_fc1_view = cached_fc1_native = None
                cached_fc2_view = cached_fc2_native = None
            else:
                cached_fc1_view, cached_fc1_native = self._routed_weight_view_cache[:2]
                cached_fc2_view, cached_fc2_native = self._routed_weight_view_cache[2:]
            fc1_view, fc1_native = native_single_grouped_weight_view(
                self.routed_fc1_weight,
                num_experts=self.num_local_experts,
                rows=2 * self.intermediate_size,
                columns=self.hidden_size,
                cached_view=cached_fc1_view,
                cached_native_view=cached_fc1_native,
            )
            fc2_view, fc2_native = native_single_grouped_weight_view(
                self.routed_fc2_weight,
                num_experts=self.num_local_experts,
                rows=self.hidden_size,
                columns=self.intermediate_size,
                cached_view=cached_fc2_view,
                cached_native_view=cached_fc2_native,
            )
            self._routed_weight_view_cache = (
                fc1_view,
                fc1_native,
                fc2_view,
                fc2_native,
            )
        self.is_first_microbatch = False
        return self._routed_weight_view_cache[0], self._routed_weight_view_cache[2]

    # Native expert/shared modules emit the canonical checkpoint entries.
    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        del prefix, sharded_offsets, metadata
        return {}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        del destination, prefix, keep_vars

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        del state_dict, prefix, local_metadata, strict
        del missing_keys, unexpected_keys, error_msgs
        self._routed_weight_view_cache = None
        self.is_first_microbatch = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        probs = probs.reshape(x.shape[0], -1).contiguous()
        routing_map = routing_map.reshape(x.shape[0], -1).contiguous()
        router_weights, top_experts = routing_map_to_compact_inputs(
            probs, routing_map, self.topk, index_dtype=torch.int64
        )
        if self.config.moe_paged_stash:
            runtime = get_fk_runtime(
                self.runtime_config,
                self.ep_group,
                num_local_tokens=x.shape[0],
                device=x.device,
            )
            x = paged_stash_group_start(x)
            # FK backward consumes its fixed padded route capacity. Keep the
            # exact saved extent for correctness; the unpadded route count is
            # supplied separately as the allocation-sizing heuristic.
            num_tokens_tensor = torch.full(
                (), runtime.local_capacity, dtype=torch.int64, device=x.device
            )
            stash_context = get_paged_stash_context(
                name="fk_megamoe",
                max_num_tokens=runtime.local_capacity,
                num_tokens_tensor=num_tokens_tensor,
                avg_num_tokens=x.shape[0] * self.topk,
            )
        else:
            stash_context = nullcontext()

        with stash_context:
            routed_output = _FkAutograd.apply(
                self,
                x,
                router_weights,
                top_experts,
                self.routed_fc1_weight,
                self.routed_fc2_weight,
            )
        if self.config.moe_paged_stash:
            routed_output = paged_stash_group_commit(
                routed_output, name="fk_megamoe"
            )

        # The shared branch deliberately stays outside the custom autograd
        # function: native MCore/TE owns MXFP8 execution and weight gradients.
        shared_output = apply_module(self.shared_experts)(hidden_states)
        return (routed_output.view(original_shape) + shared_output).view(original_shape)
