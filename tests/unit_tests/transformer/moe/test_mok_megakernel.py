# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe import mok_megakernel


def _parameter_with_main_grad(shape=(4, 8)):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
    param.main_grad = torch.zeros(shape, dtype=torch.float32)
    param.grad_added_to_main_grad = False
    return param


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
