# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer import clip_grads
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def test_split_grads_for_l2_norm_bounds_each_launch(monkeypatch):
    """Large gradient lists are split by both tensor count and indexing range."""
    monkeypatch.setattr(clip_grads, '_MAX_MULTI_TENSOR_L2_NORM_NUMEL', 8)
    monkeypatch.setattr(clip_grads, '_MAX_MULTI_TENSOR_L2_NORM_TENSORS', 2)

    grads = [torch.arange(size) for size in (3, 4, 2, 9)]
    batches = clip_grads._split_grads_for_l2_norm(grads)

    assert [[tensor.numel() for tensor in batch] for batch in batches] == [
        [3],
        [4],
        [2],
        [8],
        [1],
    ]
    assert all(len(batch) == 1 for batch in batches)
    assert all(sum(tensor.numel() for tensor in batch) <= 8 for batch in batches)


def test_split_grads_for_l2_norm_preserves_ordinary_launch():
    """Ordinary gradient lists retain the existing one-launch behavior."""
    grads = [torch.arange(size) for size in (3, 4)]

    batches = clip_grads._split_grads_for_l2_norm(grads)

    assert len(batches) == 1
    assert batches[0] is grads


def test_grad_norm_skip_threshold_config():
    """Test that grad_norm_skip_threshold config has correct default."""
    config = OptimizerConfig()
    assert config.grad_norm_skip_threshold == float('inf')


def test_default_grad_norm_skip_threshold_does_not_compare_grad_norm():
    """The disabled skip threshold must not inspect a device-backed gradient norm."""

    class UncomparableGradNorm:
        def __gt__(self, _other):
            raise AssertionError(
                "The default infinite threshold should short-circuit the comparison"
            )

    class MockOptimizer:
        def __init__(self):
            self.config = OptimizerConfig(clip_grad=0.0)
            self.param = torch.nn.Parameter(torch.ones(1))
            self.is_stub_optimizer = False
            self.step_called = False

        def prepare_grads(self):
            return False

        def get_grad_norm(self):
            return UncomparableGradNorm()

        def get_parameters(self):
            return [self.param]

        def step_with_ready_grads(self):
            self.step_called = True
            return True

    optimizer = MockOptimizer()

    update_successful, _, _ = ChainedOptimizer([optimizer]).step()

    assert update_successful
    assert optimizer.step_called
