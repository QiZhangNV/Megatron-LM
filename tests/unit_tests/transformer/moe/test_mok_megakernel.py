# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe import mok_megakernel


def _parameter_with_main_grad(shape=(4, 8)):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
    param.main_grad = torch.zeros(shape, dtype=torch.float32)
    param.grad_added_to_main_grad = False
    return param


def test_dummy_weight_gradient_reuses_parameter_storage():
    param = _parameter_with_main_grad()

    dummy = mok_megakernel._dummy_weight_gradient(param)

    assert dummy.shape == param.shape
    assert dummy.dtype == param.dtype
    assert dummy.data_ptr() == param.data_ptr()
    assert not dummy.requires_grad


def test_accumulate_weight_gradient_adds_to_main_grad_and_returns_dummy(monkeypatch):
    param = _parameter_with_main_grad()
    grad = torch.full_like(param, 0.5)
    dummy = torch.empty_like(param)
    monkeypatch.setattr(mok_megakernel, "_dummy_weight_gradient", lambda _: dummy)

    actual = mok_megakernel._accumulate_weight_gradient(param, grad)
    mok_megakernel._accumulate_weight_gradient(param, grad)

    assert actual is dummy
    torch.testing.assert_close(param.main_grad, torch.ones_like(param.main_grad))
    assert param.grad_added_to_main_grad


def test_accumulate_weight_gradient_requires_main_grad():
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="param.main_grad"):
        mok_megakernel._accumulate_weight_gradient(param, torch.zeros_like(param))


def test_accumulate_weight_gradient_rejects_shape_mismatch():
    param = _parameter_with_main_grad()

    with pytest.raises(RuntimeError, match="shape mismatch"):
        mok_megakernel._accumulate_weight_gradient(
            param, torch.zeros((2, 16), dtype=torch.bfloat16)
        )


def test_finish_weight_gradient_marks_ready_without_accumulating(monkeypatch):
    param = _parameter_with_main_grad()
    param.main_grad.fill_(0.25)
    dummy = torch.empty_like(param)
    monkeypatch.setattr(mok_megakernel, "_dummy_weight_gradient", lambda _: dummy)

    actual = mok_megakernel._finish_weight_gradient(param)

    assert actual is dummy
    torch.testing.assert_close(param.main_grad, torch.full_like(param.main_grad, 0.25))
    assert param.grad_added_to_main_grad


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_main_grad_buffer_requires_contiguous_fp32(dtype):
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))
    param.main_grad = torch.zeros((4, 8), dtype=dtype)

    with pytest.raises(RuntimeError, match="contiguous FP32"):
        mok_megakernel._main_grad_buffer(param)


def test_swizzle_mxfp8_scale_matches_tcgen05_lane_layout():
    rows = columns = 128
    logical = torch.arange(rows * (columns // 32), dtype=torch.int32).to(torch.uint8)
    logical = logical.reshape(1, rows, columns // 32)

    actual = mok_megakernel._swizzle_mxfp8_scale(
        logical, rows=rows, columns=columns
    )

    assert actual.shape == (1, 1, 32, 16)
    expected = torch.empty_like(actual)
    for lane in range(32):
        for row_group in range(4):
            for column_scale in range(4):
                expected[0, 0, lane, row_group * 4 + column_scale] = logical[
                    0, row_group * 32 + lane, column_scale
                ]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)



def test_mxfp8_backward_views_keep_native_columnwise_payload_zero_copy():
    num_experts, rows, columns = 1, 256, 128
    row_data = torch.empty((num_experts, rows, columns))
    row_scale = torch.zeros((num_experts, rows, columns // 32), dtype=torch.uint8)
    column_data = torch.empty_like(row_data)
    column_scale = torch.zeros(
        (num_experts, columns, rows // 32), dtype=torch.uint8
    )
    native = (row_data, row_scale, column_data, column_scale, True)

    actual = mok_megakernel._mok_mxfp8_backward_weight_views(
        native,
        rows=rows,
        columns=columns,
    )

    assert actual[0].data_ptr() == row_data.data_ptr()
    assert actual[2].data_ptr() == column_data.data_ptr()
    assert actual[4] is True


def test_native_single_grouped_bf16_views_alias_authoritative_parameters():
    num_experts, intermediate_size, hidden_size = 2, 4, 8
    fc1 = torch.nn.Parameter(
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
        )
    )
    fc2 = torch.nn.Parameter(
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16
        )
    )

    gate, up, down = mok_megakernel._native_single_grouped_weight_views(
        fc1,
        fc2,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        use_mxfp8=False,
    )

    assert gate is fc1
    assert up is fc1
    assert down is fc2


def _parameter_with_preserved_init(init_val):
    param = torch.nn.Parameter(torch.zeros(init_val.shape, dtype=torch.bfloat16))
    cleared = []
    param.get_high_precision_init_val = lambda: init_val
    param.clear_high_precision_init_val = lambda: cleared.append(True)
    return param, cleared


def test_import_weights_preserves_reordered_init_for_optimizer(monkeypatch):
    class Stub:
        pass

    monkeypatch.setattr(mok_megakernel, "_debug_tag", lambda *_: None)

    hidden_size = 3
    routed_intermediate = 2
    shared_intermediate = 1
    num_experts = 2

    routed = Stub()
    routed.linear_fc1 = Stub()
    routed.linear_fc2 = Stub()
    routed.linear_fc1.single_grouped_weight = False
    routed.linear_fc2.single_grouped_weight = False

    routed_fc1_init = []
    routed_fc2_init = []
    cleared = []
    for expert_idx in range(num_experts):
        fc1_init = (
            torch.arange(2 * routed_intermediate * hidden_size, dtype=torch.float32)
            .reshape(2 * routed_intermediate, hidden_size)
            .add_(100 * expert_idx + 0.125)
        )
        fc2_init = (
            torch.arange(hidden_size * routed_intermediate, dtype=torch.float32)
            .reshape(hidden_size, routed_intermediate)
            .add_(100 * expert_idx + 0.375)
        )
        fc1_param, fc1_cleared = _parameter_with_preserved_init(fc1_init)
        fc2_param, fc2_cleared = _parameter_with_preserved_init(fc2_init)
        setattr(routed.linear_fc1, f"weight{expert_idx}", fc1_param)
        setattr(routed.linear_fc2, f"weight{expert_idx}", fc2_param)
        routed_fc1_init.append(fc1_init)
        routed_fc2_init.append(fc2_init)
        cleared.extend((fc1_cleared, fc2_cleared))

    shared = Stub()
    shared.linear_fc1 = Stub()
    shared.linear_fc2 = Stub()
    shared_fc1_init = torch.arange(
        2 * shared_intermediate * hidden_size, dtype=torch.float32
    ).reshape(2 * shared_intermediate, hidden_size)
    shared_fc2_init = torch.arange(hidden_size * shared_intermediate, dtype=torch.float32).reshape(
        hidden_size, shared_intermediate
    )
    shared.linear_fc1.weight, shared_fc1_cleared = _parameter_with_preserved_init(shared_fc1_init)
    shared.linear_fc2.weight, shared_fc2_cleared = _parameter_with_preserved_init(shared_fc2_init)
    cleared.extend((shared_fc1_cleared, shared_fc2_cleared))

    module = mok_megakernel.MoKMegakernel.__new__(mok_megakernel.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.hidden_size = hidden_size
    module.intermediate_size = routed_intermediate
    module.shared_intermediate_size = shared_intermediate
    module.num_local_experts = num_experts
    module._debug_module_index = 0

    module._import_routed_weights(routed)
    module._import_shared_weights(shared)

    expected_routed_gate = torch.stack([value[:routed_intermediate] for value in routed_fc1_init])
    expected_routed_up = torch.stack([value[routed_intermediate:] for value in routed_fc1_init])
    expected_routed_down = torch.stack(routed_fc2_init)

    expected_shared_gate = torch.zeros((routed_intermediate, hidden_size))
    expected_shared_up = torch.zeros_like(expected_shared_gate)
    expected_shared_down = torch.zeros((hidden_size, routed_intermediate))
    expected_shared_gate[:shared_intermediate].copy_(shared_fc1_init[:shared_intermediate])
    expected_shared_up[:shared_intermediate].copy_(shared_fc1_init[shared_intermediate:])
    expected_shared_down[:, :shared_intermediate].copy_(shared_fc2_init)

    expected_by_param = {
        module.routed_gate_weight: expected_routed_gate,
        module.routed_up_weight: expected_routed_up,
        module.routed_down_weight: expected_routed_down,
        module.shared_gate_weight: expected_shared_gate,
        module.shared_up_weight: expected_shared_up,
        module.shared_down_weight: expected_shared_down,
    }
    from megatron.core.optimizer.optimizer import _pop_high_precision_init_val

    for param, expected in expected_by_param.items():
        torch.testing.assert_close(
            param.float(), expected.to(torch.bfloat16).float(), rtol=0, atol=0
        )
        preserved = _pop_high_precision_init_val(param)
        torch.testing.assert_close(preserved, expected, rtol=0, atol=0)
        assert _pop_high_precision_init_val(param) is None

    assert all(item == [True] for item in cleared)
