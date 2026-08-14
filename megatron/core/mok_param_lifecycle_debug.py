# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in diagnostics for the experimental MOK parameter lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import torch

_COUNTS: dict[tuple[str, str], int] = {}
_LOCK = threading.Lock()


def _output_path() -> str | None:
    return os.environ.get("MOK_DEBUG_PARAM_LIFECYCLE_PATH")


def enabled() -> bool:
    """Return whether lifecycle tracing is enabled for this process."""
    return bool(_output_path())


def tag_parameter(param: torch.nn.Parameter, name: str) -> None:
    """Attach the stable diagnostic name used across optimizer and DDP stages."""
    param._mok_lifecycle_name = name


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _selected(param: torch.Tensor) -> bool:
    name = getattr(param, "_mok_lifecycle_name", "")
    requested = os.environ.get(
        "MOK_DEBUG_PARAM_LIFECYCLE_NAME", "module0.routed_gate_weight"
    )
    return _rank() == 0 and name == requested


def _tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach()
    flat = tensor.reshape(-1)
    sample_count = min(
        int(os.environ.get("MOK_DEBUG_PARAM_LIFECYCLE_SAMPLES", "4096")), flat.numel()
    )
    if sample_count == 0:
        sampled = torch.empty(0, dtype=torch.float32)
    elif sample_count == flat.numel():
        sampled = flat
    else:
        indices = torch.linspace(
            0, flat.numel() - 1, sample_count, device=flat.device, dtype=torch.float64
        ).to(torch.long)
        sampled = flat.index_select(0, indices)

    sampled_cpu = sampled.to(device="cpu").contiguous()
    raw_bytes = sampled_cpu.view(torch.uint8).numpy().tobytes()
    if sampled_cpu.dtype == torch.uint8:
        canonical_bf16_sha256 = None
        numeric = sampled_cpu.float()
    else:
        canonical = sampled_cpu.to(torch.bfloat16).contiguous()
        canonical_bf16_sha256 = hashlib.sha256(
            canonical.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        numeric = sampled_cpu.float()

    # TE GroupedTensor is a logical wrapper around member row/column backing
    # tensors and intentionally has no single valid Python storage pointer.
    try:
        storage_ptr = tensor.untyped_storage().data_ptr()
        storage_offset_bytes = tensor.data_ptr() - storage_ptr
        data_ptr = tensor.data_ptr()
    except RuntimeError:
        storage_ptr = None
        storage_offset_bytes = None
        data_ptr = None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "data_ptr": data_ptr,
        "storage_ptr": storage_ptr,
        "storage_offset_bytes": storage_offset_bytes,
        "raw_sample_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "canonical_bf16_sample_sha256": canonical_bf16_sha256,
        "sample_count": sample_count,
        "sample_first8": numeric[:8].tolist(),
        "sample_min": numeric.min().item() if numeric.numel() else None,
        "sample_max": numeric.max().item() if numeric.numel() else None,
        "sample_mean": numeric.mean().item() if numeric.numel() else None,
    }


def record(
    stage: str,
    param: torch.Tensor,
    *,
    tensors: dict[str, torch.Tensor | None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one selected parameter snapshot as a JSON line."""
    path = _output_path()
    if path is None or not _selected(param):
        return

    param_name = getattr(param, "_mok_lifecycle_name")
    key = (param_name, stage)
    max_events = int(os.environ.get("MOK_DEBUG_PARAM_LIFECYCLE_MAX_EVENTS", "8"))
    with _LOCK:
        count = _COUNTS.get(key, 0)
        if count >= max_events:
            return
        _COUNTS[key] = count + 1

    payload: dict[str, Any] = {
        "rank": _rank(),
        "stage": stage,
        "stage_occurrence": count,
        "parameter_name": param_name,
        "parameter": _tensor_signature(param),
    }
    for name, tensor in (tensors or {}).items():
        payload[name] = None if tensor is None else _tensor_signature(tensor)
    if metadata:
        payload["metadata"] = metadata

    resolved = Path(path.format(rank=_rank()))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, resolved.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
