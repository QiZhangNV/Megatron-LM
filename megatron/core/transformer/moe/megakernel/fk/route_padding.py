# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Plan zero-valued routes for FK's 128-row training-output contract."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

import torch


FK_ROUTE_ALIGNMENT = 128


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def calculate_local_route_capacity(
    *,
    num_local_tokens: int,
    topk: int,
    num_local_experts: int,
    capacity_factor: float,
) -> int:
    """Return a fixed per-destination-rank route capacity.

    The capacity is simultaneously divisible by ``topk`` (so every source
    rank appends a whole number of tokens) and by ``128 * local_experts`` (so
    FK's balanced setup route passes its col-output/recompute validation).
    Runtime routes still receive an explicit overflow check.
    """
    if num_local_tokens <= 0 or topk <= 0 or num_local_experts <= 0:
        raise ValueError("FK route-capacity dimensions must be positive")
    if capacity_factor < 1.0:
        raise ValueError("FK route capacity factor must be at least 1.0")
    alignment = math.lcm(FK_ROUTE_ALIGNMENT * num_local_experts, topk)
    requested = math.ceil(num_local_tokens * topk * capacity_factor)
    capacity = _round_up(requested, alignment)
    return capacity


@dataclass(frozen=True)
class RoutePaddingPlan:
    """Host-side plan shared identically by every rank in one EP group."""

    padded_counts: tuple[int, ...]
    dummy_experts_by_source_rank: tuple[tuple[int, ...], ...]
    local_capacity: int
    padded_num_local_tokens: int
    topk: int

    def local_counts(self, ep_rank: int) -> tuple[int, ...]:
        """Return padded counts for the experts owned by ``ep_rank``."""
        ep_size = len(self.dummy_experts_by_source_rank)
        if ep_rank < 0 or ep_rank >= ep_size:
            raise ValueError(f"EP rank {ep_rank} is outside [0, {ep_size})")
        num_local_experts = len(self.padded_counts) // ep_size
        begin = ep_rank * num_local_experts
        return self.padded_counts[begin : begin + num_local_experts]


def _schedule_distinct_dummy_routes(
    deficits: Sequence[int], topk: int
) -> tuple[int, ...]:
    """Pack a deficit multiset into token rows without duplicate experts."""
    total = sum(deficits)
    if total % topk:
        raise ValueError(
            f"FK dummy route count {total} is not divisible by topk={topk}"
        )
    heap = [(-count, expert) for expert, count in enumerate(deficits) if count]
    heapq.heapify(heap)
    schedule: list[int] = []
    for _ in range(total // topk):
        if len(heap) < topk:
            remaining = sorted((-count, expert) for count, expert in heap)
            raise ValueError(
                "FK route padding cannot form a token with distinct experts; "
                f"remaining={remaining}, topk={topk}"
            )
        selected = [heapq.heappop(heap) for _ in range(topk)]
        schedule.extend(expert for _neg_count, expert in selected)
        for neg_count, expert in selected:
            remaining = -neg_count - 1
            if remaining:
                heapq.heappush(heap, (-remaining, expert))
    if heap:
        raise AssertionError("FK dummy route scheduler left unconsumed deficits")
    return tuple(schedule)


def build_route_padding_plan(
    global_counts: Sequence[int],
    *,
    ep_size: int,
    num_local_tokens: int,
    topk: int,
    local_capacity: int,
) -> RoutePaddingPlan:
    """Pad every expert to 128 rows and every destination rank to one capacity.

    Dummy routes carry a zero hidden row and zero router probability at runtime,
    so they affect neither the real-token output nor any gradient. They only
    make FK's saved-preactivation and column-quantized wgrad operands legal.
    """
    counts = tuple(int(value) for value in global_counts)
    if ep_size <= 0 or len(counts) % ep_size:
        raise ValueError("FK global expert count must be divisible by EP size")
    if any(value < 0 for value in counts):
        raise ValueError("FK route counts must be non-negative")
    expected_routes = ep_size * num_local_tokens * topk
    if sum(counts) != expected_routes:
        raise ValueError(
            "FK requires exactly topk valid routes per token: "
            f"observed={sum(counts)}, expected={expected_routes}"
        )
    if local_capacity % FK_ROUTE_ALIGNMENT or local_capacity % topk:
        raise ValueError(
            "FK local route capacity must be divisible by both 128 and topk"
        )
    original_local_routes = num_local_tokens * topk
    if local_capacity < original_local_routes:
        raise ValueError(
            f"FK local route capacity {local_capacity} must cover {original_local_routes}"
        )

    num_local_experts = len(counts) // ep_size
    padded_counts = [0] * len(counts)
    for ep_rank in range(ep_size):
        begin = ep_rank * num_local_experts
        rank_counts = counts[begin : begin + num_local_experts]
        targets = [
            _round_up(value, FK_ROUTE_ALIGNMENT) if value else 0
            for value in rank_counts
        ]
        physical = sum(targets)
        if physical > local_capacity:
            raise RuntimeError(
                "FK expert-rank route capacity overflow: "
                f"ep_rank={ep_rank}, physical_routes={physical}, "
                f"capacity={local_capacity}, counts={rank_counts}"
            )
        remaining = local_capacity - physical
        if remaining % FK_ROUTE_ALIGNMENT:
            raise AssertionError(
                "FK aligned rank capacity left a partial 128-row block"
            )
        while remaining:
            expert = min(range(num_local_experts), key=lambda idx: (targets[idx], idx))
            targets[expert] += FK_ROUTE_ALIGNMENT
            remaining -= FK_ROUTE_ALIGNMENT
        padded_counts[begin : begin + num_local_experts] = targets

    deficits = [
        target - count for target, count in zip(padded_counts, counts, strict=True)
    ]
    schedule = _schedule_distinct_dummy_routes(deficits, topk)
    padded_num_local_tokens = local_capacity // topk
    dummy_tokens_per_rank = padded_num_local_tokens - num_local_tokens
    routes_per_source_rank = dummy_tokens_per_rank * topk
    expected_dummy_routes = ep_size * routes_per_source_rank
    if len(schedule) != expected_dummy_routes:
        raise AssertionError(
            "FK padding plan size mismatch: "
            f"scheduled={len(schedule)}, expected={expected_dummy_routes}"
        )
    per_rank = tuple(
        schedule[rank * routes_per_source_rank : (rank + 1) * routes_per_source_rank]
        for rank in range(ep_size)
    )
    return RoutePaddingPlan(
        padded_counts=tuple(padded_counts),
        dummy_experts_by_source_rank=per_rank,
        local_capacity=local_capacity,
        padded_num_local_tokens=padded_num_local_tokens,
        topk=topk,
    )


def build_route_padding_tensors(
    global_counts: torch.Tensor,
    *,
    ep_size: int,
    num_local_tokens: int,
    topk: int,
    local_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the equivalent padding plan without synchronizing to the host.

    The returned tensors are ``padded_counts[num_experts]`` and
    ``dummy_experts[ep_size, dummy_tokens_per_rank, topk]``.  All tensor
    dimensions are fixed by the model configuration, so this path can be used
    from a CUDA Graph after the surrounding distributed rendezvous is made
    capture-safe.

    Dummy routes are scheduled by flattening each expert's deficit into one
    contiguous occurrence range and assigning occurrence ``p`` to token
    ``p % num_dummy_tokens``.  Since a valid expert deficit cannot exceed the
    number of dummy tokens, one expert never appears twice in a token row.
    The flattened range has exactly ``num_dummy_tokens * topk`` entries, so
    every token receives exactly ``topk`` distinct experts.
    """
    if global_counts.ndim != 1:
        raise ValueError(
            f"FK global route counts must be one-dimensional, got {global_counts.shape}"
        )
    num_experts = global_counts.numel()
    if ep_size <= 0 or num_experts % ep_size:
        raise ValueError("FK global expert count must be divisible by EP size")
    if local_capacity % FK_ROUTE_ALIGNMENT or local_capacity % topk:
        raise ValueError(
            "FK local route capacity must be divisible by both 128 and topk"
        )

    original_local_routes = num_local_tokens * topk
    if local_capacity < original_local_routes:
        raise ValueError(
            f"FK local route capacity {local_capacity} must cover {original_local_routes}"
        )

    counts = global_counts.to(torch.int64)
    expected_routes = ep_size * original_local_routes
    torch._assert_async(
        torch.all(counts >= 0), "FK route counts must be non-negative"
    )
    torch._assert_async(
        counts.sum() == expected_routes,
        "FK requires exactly topk valid routes per token",
    )

    num_local_experts = num_experts // ep_size
    capacity_blocks = local_capacity // FK_ROUTE_ALIGNMENT
    base_blocks = torch.div(
        counts + FK_ROUTE_ALIGNMENT - 1,
        FK_ROUTE_ALIGNMENT,
        rounding_mode="floor",
    ).reshape(ep_size, num_local_experts)
    torch._assert_async(
        torch.all(base_blocks.sum(dim=1) <= capacity_blocks),
        "FK expert-rank route capacity overflow",
    )

    # Vectorized water filling.  This is equivalent to repeatedly adding one
    # 128-row block to the currently smallest expert target, but it launches a
    # fixed set of tensor operations instead of branching on device values.
    levels = torch.arange(
        capacity_blocks + 1, dtype=base_blocks.dtype, device=counts.device
    )
    totals_by_level = torch.maximum(
        base_blocks.unsqueeze(-1), levels.reshape(1, 1, -1)
    ).sum(dim=1)
    feasible_levels = torch.where(
        totals_by_level <= capacity_blocks,
        levels.reshape(1, -1),
        torch.full_like(totals_by_level, -1),
    )
    water_level = feasible_levels.amax(dim=1).clamp_min(0)
    padded_blocks = torch.maximum(base_blocks, water_level.unsqueeze(1))
    remaining_blocks = capacity_blocks - padded_blocks.sum(dim=1)

    expert_indices = torch.arange(
        num_local_experts, dtype=base_blocks.dtype, device=counts.device
    ).reshape(1, -1)
    order = torch.argsort(
        padded_blocks * (num_local_experts + 1) + expert_indices, dim=1
    )
    increments_in_order = (
        torch.arange(num_local_experts, device=counts.device).reshape(1, -1)
        < remaining_blocks.unsqueeze(1)
    ).to(base_blocks.dtype)
    increments = torch.zeros_like(padded_blocks).scatter(
        1, order, increments_in_order
    )
    padded_counts = ((padded_blocks + increments) * FK_ROUTE_ALIGNMENT).reshape(-1)

    deficits = padded_counts - counts
    dummy_tokens_per_rank = local_capacity // topk - num_local_tokens
    total_dummy_tokens = ep_size * dummy_tokens_per_rank
    total_dummy_routes = total_dummy_tokens * topk
    torch._assert_async(
        torch.all(deficits >= 0), "FK padded expert counts must cover real routes"
    )
    torch._assert_async(
        torch.all(deficits <= total_dummy_tokens),
        "FK route padding cannot form distinct dummy expert rows",
    )
    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=counts.device)
    expanded = torch.repeat_interleave(
        expert_ids, deficits, output_size=total_dummy_routes
    )
    dummy_experts = (
        expanded.reshape(topk, total_dummy_tokens)
        .transpose(0, 1)
        .reshape(ep_size, dummy_tokens_per_rank, topk)
        .contiguous()
    )
    return padded_counts, dummy_experts
