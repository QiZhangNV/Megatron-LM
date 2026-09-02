# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Expose MCore single-grouped MXFP8 parameters in FK's tensor layouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from megatron.core.transformer.moe.megakernel.mok.weights import (
    _native_single_grouped_weight_view,
)


@dataclass(frozen=True)
class FkWeightView:
    """Forward row-quantized and backward column-quantized FK views."""

    forward_data: torch.Tensor
    forward_scale: torch.Tensor
    backward_data: torch.Tensor
    backward_scale: torch.Tensor


def native_single_grouped_weight_view(
    weight: nn.Parameter,
    *,
    num_experts: int,
    rows: int,
    columns: int,
    cached_native_view=None,
) -> tuple[FkWeightView, tuple]:
    """Build zero-copy FK payload views and stable converted scale buffers.

    TE's native columnwise payload is a contiguous quantization of the logical
    transpose. Re-view it as ``[E, K, M]`` and transpose the dimensions without
    copying, matching the K-major tensors used by the validated FK runner.
    """
    native_view = _native_single_grouped_weight_view(
        weight,
        num_experts=num_experts,
        rows=rows,
        columns=columns,
        use_mxfp8=True,
        cached_view=cached_native_view,
    )
    row_data, row_scale, column_storage, column_scale, _native_columnwise = native_view
    forward_data = row_data.transpose(-2, -1)
    backward_data = column_storage.reshape(num_experts, columns, rows).transpose(-2, -1)
    view = FkWeightView(
        forward_data=forward_data,
        forward_scale=row_scale.reshape(num_experts, -1),
        backward_data=backward_data,
        backward_scale=column_scale.reshape(num_experts, -1),
    )
    return view, native_view
