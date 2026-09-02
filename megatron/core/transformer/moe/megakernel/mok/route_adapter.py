# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MOK compatibility wrapper for the common compact-route adapter."""

from __future__ import annotations

import torch

from megatron.core.transformer.moe.megakernel.route_adapter import (
    RoutingMapToIndices,
    routing_map_to_compact_inputs,
)


def routing_map_to_mok_inputs(
    probs: torch.Tensor, routing_map: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MOK's FP32 router weights and int32 compact expert indices."""
    return routing_map_to_compact_inputs(
        probs, routing_map, topk, index_dtype=torch.int32
    )


__all__ = ["RoutingMapToIndices", "routing_map_to_mok_inputs"]
