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
    cached_view: FkWeightView | None = None,
    cached_native_view=None,
) -> tuple[FkWeightView, tuple]:
    """Build FK payload views and stable converted data/scale buffers.

    TE keeps columnwise-quantized payload bytes in the original ``[E, M, K]``
    order, while FK expects the rowwise quantization of the logical transpose
    in physical ``[E, K, M]`` order. Refresh one cached transpose buffer and
    expose it through FK's K-major ``[E, M, K]`` tensor view. The scale
    conversion performed by the MOK adapter already matches FK's blocked scale
    layout exactly.
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
    column_data = column_storage.reshape(num_experts, rows, columns)
    transposed_column_data = column_data.transpose(-2, -1)
    if cached_view is None:
        backward_storage = transposed_column_data.contiguous()
    else:
        backward_storage = cached_view.backward_data.transpose(-2, -1)
        expected_shape = (num_experts, columns, rows)
        if (
            tuple(backward_storage.shape) != expected_shape
            or backward_storage.dtype != column_data.dtype
            or backward_storage.device != column_data.device
            or not backward_storage.is_contiguous()
        ):
            raise RuntimeError(
                "FK cached columnwise payload mismatch: "
                f"got shape={tuple(backward_storage.shape)}, "
                f"dtype={backward_storage.dtype}, device={backward_storage.device}, "
                f"contiguous={backward_storage.is_contiguous()}; expected "
                f"shape={expected_shape}, dtype={column_data.dtype}, "
                f"device={column_data.device}, contiguous=True"
            )
        backward_storage.copy_(transposed_column_data)
    backward_data = backward_storage.transpose(-2, -1)
    view = FkWeightView(
        forward_data=forward_data,
        forward_scale=row_scale.reshape(num_experts, -1),
        backward_data=backward_data,
        backward_scale=column_scale.reshape(num_experts, -1),
    )
    return view, native_view
