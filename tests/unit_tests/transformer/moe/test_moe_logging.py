# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

from megatron.core.transformer.moe.moe_logging import (
    warmup_moe_metrics_pipeline_communicator,
)


def test_warmup_moe_metrics_pipeline_communicator_uses_explicit_group(monkeypatch):
    group = object()
    tensor = object()
    all_reduce = mock.Mock()
    synchronize = mock.Mock()

    monkeypatch.setattr(
        "megatron.core.transformer.moe.moe_logging.torch.distributed.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.moe_logging.torch.distributed.get_world_size",
        lambda group: 4,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.moe_logging.torch.distributed.all_reduce", all_reduce
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.moe_logging.torch.cuda.current_device", lambda: 2
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.moe_logging.torch.cuda.synchronize", synchronize
    )
    zeros = mock.Mock(return_value=tensor)
    monkeypatch.setattr("megatron.core.transformer.moe.moe_logging.torch.zeros", zeros)

    warmup_moe_metrics_pipeline_communicator(SimpleNamespace(pp=group))

    zeros.assert_called_once_with(1, device=2)
    all_reduce.assert_called_once_with(tensor, group=group)
    synchronize.assert_called_once_with()


def test_warmup_moe_metrics_pipeline_communicator_skips_single_rank_group(monkeypatch):
    group = object()
    all_reduce = mock.Mock()

    monkeypatch.setattr(
        "megatron.core.transformer.moe.moe_logging.torch.distributed.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.moe_logging.torch.distributed.get_world_size",
        lambda group: 1,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.moe_logging.torch.distributed.all_reduce", all_reduce
    )

    warmup_moe_metrics_pipeline_communicator(SimpleNamespace(pp=group))

    all_reduce.assert_not_called()
