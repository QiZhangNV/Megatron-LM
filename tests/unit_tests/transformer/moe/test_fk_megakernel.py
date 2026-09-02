# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.megakernel import factory
from megatron.core.transformer.moe.megakernel.fk import backend as fk_backend
from megatron.core.transformer.moe.megakernel.fk.route_padding import (
    FK_ROUTE_ALIGNMENT,
    build_route_padding_plan,
    calculate_local_route_capacity,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _fk_transformer_config(**overrides):
    values = {
        "num_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_ffn_hidden_size": 256,
        "moe_shared_expert_intermediate_size": 256,
        "expert_model_parallel_size": 8,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "gradient_accumulation_fusion": True,
        "moe_grouped_gemm": True,
        "moe_token_dispatcher_type": "flex",
        "moe_single_grouped_weight": True,
        "moe_mlp_glu_interleave_size": 32,
        "use_transformer_engine_op_fuser": False,
        "fp8": "hybrid",
        "fp8_recipe": "mxfp8",
        "fp8_param": True,
        "moe_megakernel_backend": "fk",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_fk_backend_accepts_mvp_configuration():
    config = _fk_transformer_config()

    assert config.moe_megakernel_backend == "fk"
    assert config.moe_single_grouped_weight
    assert config.moe_mlp_glu_interleave_size == 32


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"fp8": None, "fp8_param": False}, "MXFP8"),
        ({"moe_single_grouped_weight": False}, "single_grouped_weight"),
        ({"moe_mlp_glu_interleave_size": None}, "interleave_size=32"),
        ({"expert_model_parallel_size": 4}, "EP in"),
        ({"moe_shared_expert_overlap": True}, "shared-expert gate/overlap"),
        ({"cuda_graph_impl": "local"}, "does not support CUDA Graph"),
    ],
)
def test_fk_backend_rejects_unsupported_configuration(override, message):
    with pytest.raises(ValueError, match=message):
        _fk_transformer_config(**override)


def test_fk_shared_expert_keeps_native_mxfp8_config():
    config = _fk_transformer_config()

    assert factory.prepare_megakernel_shared_expert_config(config) is config
    with factory.megakernel_shared_expert_init_context(config):
        pass


def test_fk_route_padding_is_aligned_fixed_capacity_and_distinct():
    ep_size = 16
    num_experts = 96
    num_local_tokens = 4096
    topk = 6
    num_local_experts = num_experts // ep_size
    counts = [4096] * num_experts
    capacity = calculate_local_route_capacity(
        num_local_tokens=num_local_tokens,
        topk=topk,
        num_local_experts=num_local_experts,
        capacity_factor=1.0625,
    )

    plan = build_route_padding_plan(
        counts,
        ep_size=ep_size,
        num_local_tokens=num_local_tokens,
        topk=topk,
        local_capacity=capacity,
    )

    assert plan.padded_num_local_tokens > num_local_tokens
    assert all(count % FK_ROUTE_ALIGNMENT == 0 for count in plan.padded_counts)
    for ep_rank in range(ep_size):
        assert sum(plan.local_counts(ep_rank)) == capacity
    for rank_routes in plan.dummy_experts_by_source_rank:
        assert (
            len(rank_routes) == (plan.padded_num_local_tokens - num_local_tokens) * topk
        )
        for offset in range(0, len(rank_routes), topk):
            row = rank_routes[offset : offset + topk]
            assert len(set(row)) == topk

    dummy_counts = Counter(
        expert
        for rank_routes in plan.dummy_experts_by_source_rank
        for expert in rank_routes
    )
    assert (
        tuple(original + dummy_counts[expert] for expert, original in enumerate(counts))
        == plan.padded_counts
    )


def test_fk_route_padding_reports_destination_rank_overflow():
    ep_size = 8
    num_local_tokens = 128
    topk = 2
    counts = [32] * 64
    counts[0] += 896
    for expert in range(8, 36):
        counts[expert] -= 32
    capacity = calculate_local_route_capacity(
        num_local_tokens=num_local_tokens,
        topk=topk,
        num_local_experts=8,
        capacity_factor=1.0625,
    )

    with pytest.raises(RuntimeError, match="capacity overflow"):
        build_route_padding_plan(
            counts,
            ep_size=ep_size,
            num_local_tokens=num_local_tokens,
            topk=topk,
            local_capacity=capacity,
        )


def test_fk_autograd_accumulates_bf16_main_grads_and_returns_input_grads(monkeypatch):
    class FakeRuntime:
        def forward(self, x, router_weights, top_experts, fc1_view, fc2_view):
            del top_experts, fc1_view, fc2_view
            return torch.zeros_like(x), (x.shape, router_weights.shape)

        def backward(
            self,
            context,
            grad_output,
            fc1_view,
            fc2_view,
            fc1_main_grad,
            fc2_main_grad,
        ):
            del context, fc1_view, fc2_view
            fc1_main_grad.add_(2)
            fc2_main_grad.add_(3)
            return torch.full_like(grad_output, 4), torch.full((2, 2), 5.0)

    module = fk_backend.FkMegakernel.__new__(fk_backend.FkMegakernel)
    torch.nn.Module.__init__(module)
    module.runtime_config = object()
    module.ep_group = object()
    module.register_parameter(
        "routed_fc1_weight",
        torch.nn.Parameter(torch.zeros((1, 4, 4), dtype=torch.bfloat16)),
    )
    module.register_parameter(
        "routed_fc2_weight",
        torch.nn.Parameter(torch.zeros((1, 4, 2), dtype=torch.bfloat16)),
    )
    for parameter in (module.routed_fc1_weight, module.routed_fc2_weight):
        parameter.main_grad = torch.zeros_like(parameter)
        parameter.grad_added_to_main_grad = False
    module.quantized_routed_weights = lambda: (object(), object())
    monkeypatch.setattr(
        fk_backend, "get_fk_runtime", lambda *args, **kwargs: FakeRuntime()
    )

    x = torch.zeros((2, 4), requires_grad=True)
    router_weights = torch.zeros((2, 2), requires_grad=True)
    top_experts = torch.zeros((2, 2), dtype=torch.int64)
    output = fk_backend._FkAutograd.apply(
        module,
        x,
        router_weights,
        top_experts,
        module.routed_fc1_weight,
        module.routed_fc2_weight,
    )
    output.sum().backward()

    torch.testing.assert_close(x.grad, torch.full_like(x, 4))
    torch.testing.assert_close(router_weights.grad, torch.full_like(router_weights, 5))
    torch.testing.assert_close(
        module.routed_fc1_weight.main_grad,
        torch.full_like(module.routed_fc1_weight.main_grad, 2),
    )
    torch.testing.assert_close(
        module.routed_fc2_weight.main_grad,
        torch.full_like(module.routed_fc2_weight.main_grad, 3),
    )
    assert module.routed_fc1_weight.grad_added_to_main_grad
    assert module.routed_fc2_weight.grad_added_to_main_grad
