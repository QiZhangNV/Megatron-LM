# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import inspect
import sys
import types
from argparse import ArgumentParser
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.megakernel import factory
from megatron.core.transformer.moe.megakernel.fk import backend as fk_backend
from megatron.core.transformer.moe.megakernel.fk import runtime as fk_runtime
from megatron.core.transformer.moe.megakernel.fk import weights as fk_weights
from megatron.core.transformer.moe.megakernel.fk.route_padding import (
    FK_ROUTE_ALIGNMENT,
    build_route_padding_plan,
    build_route_padding_tensors,
    calculate_local_route_capacity,
)
from megatron.core.transformer.moe.paged_stash import PagedStashManager
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory


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


def test_fk_mxfp8_scale_dtype_uses_host_utils_contract(monkeypatch):
    calls = []
    common = types.ModuleType("common")
    common.__path__ = []
    host_utils = types.ModuleType("common.host_utils")

    def kind_scale_dtype(kind):
        calls.append(kind)
        return torch.float8_e8m0fnu

    host_utils.kind_scale_dtype = kind_scale_dtype
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "common.host_utils", host_utils)

    assert fk_runtime._mxfp8_scale_dtype() is torch.float8_e8m0fnu
    assert calls == ["mxfp8_e4m3"]


def test_fk_cute_compile_uses_node_scoped_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("SLURM_JOB_ID", "645738")
    monkeypatch.setenv("SLURMD_NODENAME", "nvl72d117-T01")
    monkeypatch.setenv("FK_MCORE_COMPILE_LOCK_DIR", str(tmp_path))
    fk_runtime._COMPILE_ORDINALS.clear()
    calls = []

    def compile_fn(value, *, scale):
        calls.append((value, scale))
        return value * scale

    assert fk_runtime._compile_with_node_lock(
        "forward", compile_fn, 7, scale=6
    ) == 42
    assert calls == [(7, 6)]
    assert list(tmp_path.glob("645738/*.lock")) == [
        tmp_path / "645738" / "nvl72d117-T01.lock"
    ]

    assert list(tmp_path.glob("645738/*.o")) == []


def test_fk_cute_aot_sharing_preserves_hardware_info_compile_metadata():
    assert not fk_runtime._share_cute_aot_object("forward", 0)
    assert not fk_runtime._share_cute_aot_object("backward", 0)
    assert fk_runtime._share_cute_aot_object("forward", 1)
    assert fk_runtime._share_cute_aot_object("backward", 1)
    assert fk_runtime._share_cute_aot_object("col_requant", 0)


def test_fk_cute_compile_materializes_launcher_before_releasing_ir(monkeypatch):
    events = []
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")

    class Compiled:
        def __init__(self):
            self.ir_module = object()
            self.jit_module = None

        def to(self, device):
            events.append(("to", device, self.ir_module is not None))
            self.jit_module = object()
            return object()

    compiled = Compiled()
    monkeypatch.setattr(
        fk_runtime, "_trim_process_heap", lambda: events.append("trim")
    )

    result = fk_runtime._compile_with_node_lock("forward", lambda: compiled)

    assert result is compiled
    assert compiled.jit_module is not None
    assert compiled.ir_module is None
    assert events == [("to", None, True), "trim"]


def test_fk_cute_compile_exports_once_and_loads_node_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("SLURM_JOB_ID", "645739")
    monkeypatch.setenv("SLURMD_NODENAME", "nvl72d117-T01")
    monkeypatch.setenv("FK_MCORE_COMPILE_LOCK_DIR", str(tmp_path))
    events = []
    loaded = object()

    class SourceCompiled:
        def dump_to_object(self, prefix):
            events.append(("dump", prefix))
            return b"fake-cute-object"

    def load_cute_object(path, prefix):
        events.append(("load", path, prefix))
        return loaded

    monkeypatch.setattr(fk_runtime, "_load_cute_object", load_cute_object)
    monkeypatch.setattr(fk_runtime, "_trim_process_heap", lambda: True)
    fk_runtime._COMPILE_ORDINALS.clear()

    first = fk_runtime._compile_with_node_lock("col_requant", lambda: SourceCompiled())
    fk_runtime._COMPILE_ORDINALS.clear()
    second = fk_runtime._compile_with_node_lock(
        "col_requant",
        lambda: pytest.fail("a cached node artifact must not be recompiled"),
    )

    artifact = tmp_path / "645739" / "nvl72d117-T01.col_requant.0.o"
    prefix = "fk_mcore_col_requant_0"
    assert first is loaded
    assert second is loaded
    assert artifact.read_bytes() == b"fake-cute-object"
    assert events == [
        ("dump", prefix),
        ("load", str(artifact), prefix),
        ("load", str(artifact), prefix),
    ]


def test_fk_vendor_compile_wrapper_restores_cute_compile(monkeypatch):
    cutlass = types.ModuleType("cutlass")
    cutlass.__path__ = []
    cute = types.ModuleType("cutlass.cute")

    def original_compile(value):
        return ("compiled", value)

    cute.compile = original_compile
    cutlass.cute = cute
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    calls = []

    def compile_with_lock(label, compile_fn, *args, **kwargs):
        calls.append((label, compile_fn, args, kwargs))
        return compile_fn(*args, **kwargs)

    monkeypatch.setattr(fk_runtime, "_compile_with_node_lock", compile_with_lock)

    result = fk_runtime._run_with_node_serialized_cute_compiles(
        "backward", lambda: cute.compile(11)
    )

    assert result == ("compiled", 11)
    assert calls == [("backward", original_compile, (11,), {})]
    assert cute.compile is original_compile



def test_fk_resets_cudnn_dsa_children_and_lazy_symbol_cache(monkeypatch):
    root = types.ModuleType(fk_runtime._CUDNN_DSA_ROOT)
    child = types.ModuleType(f"{fk_runtime._CUDNN_DSA_ROOT}.utils.compiler")
    root._SYMBOLS = {"sparse_attention_backward_wrapper": ("unused", "unused")}
    root.sparse_attention_backward_wrapper = object()
    root.compiler = child
    root.unrelated = object()
    monkeypatch.setitem(sys.modules, root.__name__, root)
    monkeypatch.setitem(sys.modules, child.__name__, child)

    fk_runtime._reset_cudnn_dsa_modules_for_fk_cutlass()

    assert sys.modules[root.__name__] is root
    assert child.__name__ not in sys.modules
    assert not hasattr(root, "sparse_attention_backward_wrapper")
    assert not hasattr(root, "compiler")
    assert hasattr(root, "unrelated")


def test_fk_cudnn_operands_flatten_2d_runner_scale_workspace():
    total_tokens = 64
    features = 128
    data = torch.zeros((total_tokens, features), dtype=torch.uint8)
    scale_workspace = torch.arange(10 * features, dtype=torch.int32).to(
        torch.uint8
    ).reshape(10, features)

    matrix, scale = fk_runtime._raw_cudnn_operands(
        data,
        scale_workspace,
        total_tokens,
        features,
        transpose=True,
    )

    assert matrix.shape == (features, total_tokens)
    assert scale.shape == (features, total_tokens // 32)
    torch.testing.assert_close(
        scale.view(torch.uint8).reshape(-1),
        scale_workspace.reshape(-1)[: features * (total_tokens // 32)],
    )


def test_fk_reused_kernel_is_bracketed_by_ep_barriers():
    runtime = fk_runtime.FkRuntime.__new__(fk_runtime.FkRuntime)
    events = []
    runtime._ep_barrier = lambda: events.append("barrier")

    def compiled_kernel(**kwargs):
        events.append(("kernel", kwargs))

    runtime._launch_distributed_kernel(compiled_kernel, {"value": 7})

    assert events == ["barrier", ("kernel", {"value": 7}), "barrier"]


def test_fk_ep_barrier_uses_only_stream_ordered_nvshmem_during_capture(monkeypatch):
    runtime = fk_runtime.FkRuntime.__new__(fk_runtime.FkRuntime)
    runtime.ep_group = object()
    stream = object()
    events = []
    nvshmem = types.SimpleNamespace(
        barrier_all=lambda actual_stream: events.append(("nvshmem", actual_stream))
    )

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: events.append("cuda_synchronize")
    )
    monkeypatch.setattr(
        fk_runtime.dist, "barrier", lambda **kwargs: events.append(("dist", kwargs))
    )
    monkeypatch.setattr(
        fk_runtime,
        "_prepare_system_dependencies",
        lambda: types.SimpleNamespace(nvshmem=nvshmem),
    )

    runtime._ep_barrier()

    assert events == [("nvshmem", stream)]


def test_fk_route_counts_match_bincount_without_host_state():
    top_experts = torch.tensor([[3, 1, 5], [0, 3, 1], [4, 5, 3]])

    counts = fk_runtime._count_routes(top_experts, num_experts=8)

    assert torch.equal(counts, torch.bincount(top_experts.flatten(), minlength=8))


def test_fk_columnwise_payload_transpose_reuses_cached_storage(monkeypatch):
    experts, rows, columns = 2, 4, 6
    row_data = torch.arange(experts * rows * columns, dtype=torch.uint8).reshape(
        experts, rows, columns
    )
    column_data = (row_data + 37).remainder(251)
    row_scale = torch.zeros((experts, 2), dtype=torch.uint8)
    column_scale = torch.ones((experts, 2), dtype=torch.uint8)
    native_view = (row_data, row_scale, column_data, column_scale, True)

    monkeypatch.setattr(
        fk_weights,
        "_native_single_grouped_weight_view",
        lambda *args, **kwargs: native_view,
    )
    weight = torch.nn.Parameter(torch.zeros((experts, rows, columns)))
    first, cached_native = fk_weights.native_single_grouped_weight_view(
        weight,
        num_experts=experts,
        rows=rows,
        columns=columns,
    )

    expected_storage = column_data.transpose(1, 2).contiguous()
    torch.testing.assert_close(first.backward_data.transpose(1, 2), expected_storage)
    assert first.backward_data.shape == (experts, rows, columns)
    assert first.backward_data.stride() == (rows * columns, 1, rows)
    data_ptr = first.backward_data.data_ptr()

    column_data.add_(11)
    refreshed, _ = fk_weights.native_single_grouped_weight_view(
        weight,
        num_experts=experts,
        rows=rows,
        columns=columns,
        cached_view=first,
        cached_native_view=cached_native,
    )
    assert refreshed.backward_data.data_ptr() == data_ptr
    torch.testing.assert_close(
        refreshed.backward_data.transpose(1, 2),
        column_data.transpose(1, 2).contiguous(),
    )


def test_fk_native_mxfp8_columnwise_view_matches_transpose_quantization(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native MXFP8 parameter parity requires a Blackwell GPU")

    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling

    from megatron.core.fp8_utils import copy_tensor_to_quantized_param

    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    experts, rows, columns = 1, 7168, 3072
    torch.manual_seed(1234)
    source = torch.randn(
        (experts, rows, columns), device="cuda", dtype=torch.bfloat16
    )
    recipe = MXFP8BlockScaling(fp8_format=Format.HYBRID)
    constructor_kwargs = {
        "sequence_parallel": False,
        "fuse_wgrad_accumulation": True,
        "tp_group": None,
        "tp_size": 1,
        "get_rng_state_tracker": None,
        "init_method": torch.nn.init.normal_,
        "bias": False,
        "return_bias": False,
        "parallel_mode": None,
        "single_grouped_weight": True,
        "params_dtype": torch.bfloat16,
        "device": "cuda",
    }
    supported = inspect.signature(te.GroupedLinear.__init__).parameters
    if "single_grouped_weight" not in supported:
        pytest.skip("installed Transformer Engine lacks single grouped weight support")
    constructor_kwargs = {
        key: value for key, value in constructor_kwargs.items() if key in supported
    }
    with te.quantized_model_init(
        enabled=True,
        recipe=recipe,
        preserve_high_precision_init_val=True,
    ):
        layer = te.GroupedLinear(
            num_gemms=experts,
            in_features=columns,
            out_features=rows,
            **constructor_kwargs,
        )
    copy_tensor_to_quantized_param(layer.weight, source)

    actual, _ = fk_weights.native_single_grouped_weight_view(
        layer.weight,
        num_experts=experts,
        rows=rows,
        columns=columns,
    )
    quantizer = te.MXFP8Quantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    quantizer.optimize_for_gemm = True
    reference = tex.group_quantize(
        source.transpose(1, 2).contiguous().reshape(experts * columns, rows),
        quantizer,
        experts,
        None,
    )
    expected_data = (
        reference.rowwise_data.view(torch.float8_e4m3fn)
        .reshape(experts, columns, rows)
        .transpose(1, 2)
    )

    torch.testing.assert_close(
        actual.backward_data.view(torch.uint8),
        expected_data.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.backward_scale.view(torch.uint8).reshape(-1),
        reference.scale_inv.view(torch.uint8).reshape(-1),
        rtol=0,
        atol=0,
    )


def test_fk_bwd_epi_flag_batch_cli_contract():
    field_name = "fk_bwd_epi_flag_batch"
    exclude = [
        attribute.name
        for attribute in dataclasses.fields(TransformerConfig)
        if attribute.name != field_name
    ]
    parser = ArgumentParser()
    ArgumentGroupFactory(TransformerConfig, exclude=exclude).build_group(parser)

    assert parser.parse_args([]).fk_bwd_epi_flag_batch == (1, 1)
    assert parser.parse_args(
        ["--fk-bwd-epi-flag-batch", "1", "4"]
    ).fk_bwd_epi_flag_batch == [1, 4]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"fp8": None, "fp8_param": False}, "MXFP8"),
        ({"moe_single_grouped_weight": False}, "single_grouped_weight"),
        ({"moe_mlp_glu_interleave_size": None}, "interleave_size=32"),
        ({"expert_model_parallel_size": 4}, "EP in"),
        ({"moe_shared_expert_overlap": True}, "shared-expert gate/overlap"),
        ({"cuda_graph_impl": "local"}, "supports only cuda_graph_impl"),
    ],
)
def test_fk_backend_rejects_unsupported_configuration(override, message):
    with pytest.raises(ValueError, match=message):
        _fk_transformer_config(**override)


def test_fk_backend_accepts_full_iteration_cuda_graph():
    config = _fk_transformer_config(cuda_graph_impl="full_iteration")

    assert config.cuda_graph_impl == "full_iteration"


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


def test_fk_tensor_route_padding_matches_host_targets_and_preserves_deficits():
    ep_size = 16
    num_experts = 96
    num_local_tokens = 4096
    topk = 6
    num_local_experts = num_experts // ep_size
    counts = [4096] * num_experts
    for ep_rank in range(ep_size):
        begin = ep_rank * num_local_experts
        counts[begin] += 63
        counts[begin + 1] -= 63
        counts[begin + 2] += 17
        counts[begin + 3] -= 17
    capacity = calculate_local_route_capacity(
        num_local_tokens=num_local_tokens,
        topk=topk,
        num_local_experts=num_local_experts,
        capacity_factor=1.0625,
    )
    reference = build_route_padding_plan(
        counts,
        ep_size=ep_size,
        num_local_tokens=num_local_tokens,
        topk=topk,
        local_capacity=capacity,
    )

    padded_counts, dummy_experts = build_route_padding_tensors(
        torch.tensor(counts, dtype=torch.int64),
        ep_size=ep_size,
        num_local_tokens=num_local_tokens,
        topk=topk,
        local_capacity=capacity,
    )

    assert padded_counts.tolist() == list(reference.padded_counts)
    assert dummy_experts.shape == (
        ep_size,
        capacity // topk - num_local_tokens,
        topk,
    )
    sorted_rows = dummy_experts.sort(dim=-1).values
    assert torch.all(sorted_rows.diff(dim=-1) != 0)
    dummy_counts = torch.bincount(dummy_experts.reshape(-1), minlength=num_experts)
    assert torch.equal(torch.tensor(counts) + dummy_counts, padded_counts)


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
    paged_stash_tags = []

    class FakeRuntime:
        def forward(self, x, router_weights, top_experts, fc1_view, fc2_view):
            del fc1_view, fc2_view
            context = fk_runtime.FkForwardContext(
                original_tokens=x.shape[0],
                router_weights=router_weights,
                top_experts=top_experts,
                local_counts=torch.tensor([x.shape[0]], dtype=torch.int32),
                preactivation=torch.zeros((x.shape[0], 8)),
                route_index=torch.arange(x.shape[0], dtype=torch.int32),
                fc1_x_data=torch.zeros_like(x, dtype=torch.uint8),
                fc1_x_scale=torch.zeros((x.shape[0] // 2, x.shape[1]), dtype=torch.uint8),
                fc1_x_metadata=None,
            )
            return torch.zeros_like(x), context

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
    def pack(tensor):
        if hasattr(tensor, "grouped_tensor_scale_inv"):
            paged_stash_tags.append(
                (
                    tensor.grouped_tensor_scale_inv,
                    tensor.paged_stash_capture_to_host,
                )
            )
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
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
    assert Counter(paged_stash_tags) == Counter(
        {(False, True): 2, (True, True): 1}
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paged_stash_capture_host_round_trip_preserves_cuda_tensor():
    manager = PagedStashManager()
    manager.status = "capture"
    manager.max_num_tokens = 64
    manager.num_tokens_tensor = torch.tensor(48, dtype=torch.int64, device="cuda")
    manager.avg_num_tokens = 48
    manager.current_vp_stage = 0

    tensor = torch.arange(64 * 8, dtype=torch.float32, device="cuda").view(64, 8)
    tensor.grouped_tensor_scale_inv = False
    tensor.paged_stash_capture_to_host = True

    packed = manager.on_save_for_backward(tensor)
    assert packed.capture_to_host
    assert packed._tensor.device.type == "cpu"
    assert packed.device.type == "cuda"

    restored = manager.on_get_saved_tensor(packed)
    assert restored.device.type == "cuda"
    assert restored.shape == tensor.shape
    torch.testing.assert_close(restored[:48], tensor[:48])
    torch.testing.assert_close(restored[48:], torch.zeros_like(restored[48:]))
