# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for fixed-shape vision metadata used by full-iteration CUDA graphs."""

from types import SimpleNamespace

import torch

import examples.multimodal_dev.forward_step as forward_step
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import (
    Qwen35VLVisionEncoder,
    build_vision_grid_metadata,
)


class _FakeVisionRotaryEmbedding:
    def __init__(self, axis_dim):
        self.axis_dim = axis_dim

    def __call__(self, seqlen, device=None):
        positions = torch.arange(seqlen, device=device, dtype=torch.float32)[:, None]
        dims = torch.arange(self.axis_dim, device=device, dtype=torch.float32)[None, :]
        return positions * 0.125 + dims * 0.01


def _fake_encoder():
    torch.manual_seed(123)
    return SimpleNamespace(
        spatial_merge_size=2,
        num_grid_per_side=8,
        pos_embed=torch.nn.Embedding(64, 6),
        rot_pos_emb=_FakeVisionRotaryEmbedding(axis_dim=4),
        config=SimpleNamespace(mrope_section=None),
    )


def test_precomputed_vision_metadata_matches_dynamic_implementation():
    grid_thw = torch.tensor([[1, 4, 4], [2, 2, 2]], dtype=torch.long)
    metadata = build_vision_grid_metadata(
        grid_thw,
        spatial_merge_size=2,
        num_grid_per_side=8,
    )
    encoder = _fake_encoder()

    dynamic_pos = Qwen35VLVisionEncoder._fast_pos_embed_interpolate(encoder, grid_thw)
    static_pos = Qwen35VLVisionEncoder._fast_pos_embed_interpolate(
        encoder, grid_thw, metadata
    )
    torch.testing.assert_close(static_pos, dynamic_pos)

    dynamic_rotary = Qwen35VLVisionEncoder._compute_rotary_pos_emb(encoder, grid_thw)
    static_rotary = Qwen35VLVisionEncoder._compute_rotary_pos_emb(
        encoder, grid_thw, metadata
    )
    torch.testing.assert_close(static_rotary, dynamic_rotary)

    dynamic_packed = Qwen35VLVisionEncoder._build_packed_seq_params(grid_thw)
    static_packed = Qwen35VLVisionEncoder._build_packed_seq_params(grid_thw, metadata)
    torch.testing.assert_close(
        static_packed.cu_seqlens_q,
        dynamic_packed.cu_seqlens_q,
    )
    assert static_packed.max_seqlen_q == dynamic_packed.max_seqlen_q


def _sample(base, seq_length=4):
    grid_thw = torch.tensor([[2, 4, 4]], dtype=torch.long)
    metadata = build_vision_grid_metadata(
        grid_thw,
        spatial_merge_size=2,
        num_grid_per_side=8,
    )
    return {
        "input_ids": torch.arange(base, base + seq_length),
        "labels": torch.arange(base + 1, base + seq_length + 1),
        "loss_mask": torch.ones(seq_length),
        "position_ids": torch.arange(seq_length).repeat(3, 1) + base,
        "pixel_values": torch.full((32, 8), float(base)),
        "image_grid_thw": grid_thw,
        "vision_pos_embed_indices": metadata["pos_embed_indices"],
        "vision_pos_embed_weights": metadata["pos_embed_weights"],
        "vision_rotary_pos_ids": metadata["rotary_pos_ids"],
        "vision_cu_seqlens": metadata["cu_seqlens"],
        "vision_max_seqlen": metadata["max_seqlen"],
    }


def test_bshd_batch_preserves_positions_and_merges_vision_metadata(monkeypatch):
    monkeypatch.setattr(
        forward_step.mpu,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        forward_step.mpu,
        "get_context_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        forward_step.mpu,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        forward_step,
        "get_args",
        lambda: SimpleNamespace(
            sequence_parallel=False,
            cuda_graph_impl="none",
        ),
    )

    batch = forward_step.pack_or_pad_batch(
        [_sample(0), _sample(10)],
        use_packed_sequence=False,
        seq_length=4,
        device="cpu",
    )

    assert batch["position_ids"].shape == (2, 3, 4)
    torch.testing.assert_close(
        batch["position_ids"][1],
        torch.arange(4).repeat(3, 1) + 10,
    )
    assert batch["vision_pos_embed_indices"].shape == (4, 64)
    assert batch["vision_pos_embed_weights"].shape == (4, 64)
    assert batch["vision_rotary_pos_ids"].shape == (64, 2)
    torch.testing.assert_close(
        batch["vision_cu_seqlens"],
        torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32),
    )
    assert batch["vision_max_seqlen"] == 16
