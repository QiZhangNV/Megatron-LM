# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import MoECudaGraphTensorStore
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)


def _config(backend="mok"):
    return SimpleNamespace(
        moe_megakernel_backend=backend,
        moe_shared_expert_intermediate_size=4,
        moe_shared_expert_overlap=False,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.moe_router],
        overlap_moe_expert_parallel_comm=False,
        delay_offload_until_cuda_graph=False,
        sequence_parallel=False,
    )


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 2, bias=False)
        self.calls = 0

    def forward(self, x, padding_mask, input_ids, packed_seq_params):
        self.calls += 1
        probs = self.proj(x.reshape(-1, 4)).softmax(dim=-1)
        return probs, torch.ones_like(probs, dtype=torch.bool)


class _Megakernel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.shared_scale = torch.nn.Parameter(torch.tensor(2.0))
        self.routed_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, probs, routing_map):
        self.calls += 1
        # A differentiable stand-in for the MOK boundary, not a kernel parity test.
        return self.shared_scale * x + self.routed_scale * probs[:, :1].reshape(*x.shape[:-1], 1)


def _moe():
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = _config()
    layer.attn_tp_group = SimpleNamespace(size=lambda: 1)
    layer.moe_layer_recompute = False
    layer.cudagraph_tensor_store = MoECudaGraphTensorStore()
    layer.router = _Router()
    layer.megakernel_experts = _Megakernel()
    return layer


def test_capture_stops_before_mok_and_replay_preserves_router_gradient(monkeypatch):
    layer = _moe()
    x = torch.randn(3, 1, 4, requires_grad=True)
    monkeypatch.setattr(moe_utils, "is_graph_capturing", lambda: True)

    captured = layer(x)
    assert len(captured) == 3
    assert captured[0] is x
    assert captured[2].dtype == torch.bool
    assert layer.router.calls == 1
    assert layer.megakernel_experts.calls == 0

    monkeypatch.setattr(moe_utils, "is_graph_capturing", lambda: False)
    layer.cudagraph_tensor_store.set(
        hidden_states=captured[0], probs=captured[1], routing_map=captured[2]
    )
    output, bias = layer(x)
    assert bias is None
    assert layer.router.calls == 1
    assert layer.megakernel_experts.calls == 1
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.router.proj.weight.grad is not None
    assert layer.router.proj.weight.grad.abs().sum() > 0


def test_mok_still_rejects_local_intermediate_tensor_protocol():
    with pytest.raises(RuntimeError, match="partial MoE"):
        _moe()(torch.randn(3, 1, 4), intermediate_tensors=())


def test_te_router_graph_replays_new_inputs_and_optimizer_updates(monkeypatch):
    """Exercise real TE graphs across the prefix/eager boundary with a mock MOK tail."""
    from transformer_engine.pytorch import make_graphed_callables

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    layer = _moe().cuda()
    reference = copy.deepcopy(layer)
    sample = torch.randn(3, 1, 4, device="cuda", requires_grad=True)
    monkeypatch.setattr(moe_utils, "is_graph_capturing", lambda: True)
    graphed_prefix = make_graphed_callables(
        layer, (sample,), num_warmup_iters=3, allow_unused_input=True
    )
    captured_router_calls = layer.router.calls
    assert layer.megakernel_experts.calls == 0
    monkeypatch.setattr(moe_utils, "is_graph_capturing", lambda: False)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)

    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        x = (torch.randn_like(sample) + step).requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        hidden, probs, routing_map = graphed_prefix(x)
        output = layer.megakernel_experts(hidden, probs, routing_map)
        expected, _ = reference(reference_x)
        torch.testing.assert_close(output, expected)
        output.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
        for param, ref_param in zip(layer.parameters(), reference.parameters()):
            torch.testing.assert_close(param.grad, ref_param.grad)
        assert layer.router.calls == captured_router_calls
        optimizer.step()
        reference_optimizer.step()
        for param, ref_param in zip(layer.parameters(), reference.parameters()):
            torch.testing.assert_close(param, ref_param)


class _EagerTail(torch.nn.Module):
    def __init__(self, backend, fail):
        super().__init__()
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()
        self.backend = backend
        self.fail = fail
        self.router = torch.nn.Identity()
        self.shared_experts = torch.nn.Identity()

    def forward(self, x):
        store = self.cudagraph_tensor_store
        assert x is store.hidden_states
        assert store.probs is not None and store.routing_map is not None
        if self.backend == "mok":
            assert store.shared_expert_output is None
        else:
            assert store.shared_expert_output is not None
        if self.fail:
            raise RuntimeError("eager tail failure")
        shared = 7 if self.backend == "mok" else store.shared_expert_output
        return x + store.probs + shared, None


@pytest.mark.parametrize("kind", ["ordinary", "mhc", "hybrid_helper"])
@pytest.mark.parametrize("backend", [None, "mok"])
@pytest.mark.parametrize("fail", [False, True])
def test_router_replay_shared_output_contract_and_cleanup(kind, backend, fail, monkeypatch):
    cls = HyperConnectionTransformerLayer if kind == "mhc" else TransformerLayer
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.config = _config(backend)
    layer.is_moe_layer = True
    layer.recompute_pre_mlp_layernorm = False
    layer.mlp = _EagerTail(backend, fail)
    x, probs, routing_map = torch.tensor(2.0), torch.tensor(3.0), torch.tensor(True)
    outputs = [x, probs, routing_map]
    if backend is None:
        outputs.append(torch.tensor(7.0))
    residual = torch.tensor(11.0)
    if kind == "mhc":
        h_post, h_res = torch.tensor(13.0), torch.tensor(17.0)
        outputs.extend([h_post, h_res])
        monkeypatch.setattr(layer, "_uses_mhc_recompute_attn_cuda_graph_split", lambda: False)
        monkeypatch.setattr(
            layer,
            "_forward_post_mlp_with_fused_hyper_connection",
            lambda raw, hr, res, hp: raw[0] + 2 * hp + 3 * hr + 5 * res,
        )
    else:
        monkeypatch.setattr(layer, "_forward_post_mlp", lambda raw, res: raw[0] + res)
    outputs.append(residual)
    monkeypatch.setattr(
        GraphableMegatronModule, "_te_cuda_graph_replay", lambda *args, **kwargs: tuple(outputs)
    )

    def replay():
        if kind == "hybrid_helper":
            return layer.resume_moe_experts_after_partial_cudagraph(list(outputs))
        return layer._te_cuda_graph_replay_impl((), {}, None)

    if fail:
        with pytest.raises(RuntimeError, match="eager tail failure"):
            replay()
    else:
        actual, _ = replay()
        expected = x + probs + 7
        if kind == "mhc":
            expected = expected + 2 * h_post + 3 * h_res + 5 * residual
        elif kind == "ordinary":
            expected = expected + residual
        torch.testing.assert_close(actual, expected)
    assert layer.mlp.cudagraph_tensor_store.is_empty()


@pytest.mark.parametrize("backend", [None, "mok"])
def test_router_graph_manual_hook_scope_excludes_mok_shared_experts(backend):
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = _config(backend)
    layer.is_moe_layer = True
    layer.pre_mlp_layernorm = torch.nn.Identity()
    layer.mlp = _EagerTail(backend, False)
    modules = layer._get_submodules_under_cudagraphs()
    assert layer.pre_mlp_layernorm in modules
    assert layer.mlp.router in modules
    assert (layer.mlp.shared_experts in modules) == (backend is None)
