# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA Graph forward/backward coverage for the Qwen3.5-VL patch merger.

Run with::

    torchrun --nproc_per_node=1 \\
        examples/multimodal_dev/tests/test_vision_patch_merger_cuda_graph.py
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../.."),
)
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLPatchMerger


HIDDEN_SIZE = 1152
OUT_HIDDEN_SIZE = 3584
SPATIAL_MERGE_SIZE = 2
NUM_PATCHES = 64


class TorchPatchMerger(nn.Module):
    """PyTorch-native control with the same patch-merger topology."""

    def __init__(self) -> None:
        super().__init__()
        merge_dim = HIDDEN_SIZE * (SPATIAL_MERGE_SIZE**2)
        self.merge_dim = merge_dim
        self.norm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-6)
        self.linear_fc1 = nn.Linear(merge_dim, merge_dim)
        self.linear_fc2 = nn.Linear(merge_dim, OUT_HIDDEN_SIZE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.merge_dim)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states, approximate="none")
        return self.linear_fc2(hidden_states)


class PatchMergerWithScatter(nn.Module):
    """Append the same image-token masked scatter used by the VL model."""

    def __init__(self, merger: nn.Module, scatter_impl: str) -> None:
        super().__init__()
        self.merger = merger
        self.scatter_impl = scatter_impl
        num_visual_tokens = NUM_PATCHES // (SPATIAL_MERGE_SIZE**2)
        image_mask = torch.zeros(1, NUM_PATCHES, dtype=torch.bool)
        image_mask[:, :num_visual_tokens] = True
        self.register_buffer("image_mask", image_mask, persistent=False)
        self.register_buffer(
            "image_token_indices", torch.arange(num_visual_tokens), persistent=False
        )
        self.text_embeddings = nn.Parameter(
            torch.randn(NUM_PATCHES, 1, OUT_HIDDEN_SIZE, dtype=torch.bfloat16)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        vision_embeddings = self.merger(hidden_states)
        combined = self.text_embeddings.transpose(0, 1).contiguous()
        if self.scatter_impl == "masked_scatter":
            mask_expanded = self.image_mask.unsqueeze(-1).expand_as(combined)
            combined = combined.masked_scatter(mask_expanded, vision_embeddings)
        else:
            combined_flat = combined.view(-1, combined.shape[-1])
            combined_flat = combined_flat.index_copy(
                0, self.image_token_indices, vision_embeddings
            )
            combined = combined_flat.view_as(combined)
        return combined.transpose(0, 1).contiguous()


def _init_distributed() -> torch.device:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(42)
    return torch.device("cuda", local_rank)


def _build_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=HIDDEN_SIZE,
        num_attention_heads=8,
        kv_channels=HIDDEN_SIZE // 8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        add_bias_linear=True,
        gated_linear_unit=False,
        normalization="LayerNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


def _capture_forward_backward(name: str, module: nn.Module, device: torch.device) -> None:
    module.train()
    static_input = torch.randn(
        NUM_PATCHES,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            module.zero_grad(set_to_none=True)
            static_input.grad = None
            output = module(static_input)
            output.float().square().mean().backward()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    module.zero_grad(set_to_none=True)
    static_input.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = module(static_input)
        static_loss = static_output.float().square().mean()
        static_loss.backward()

    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(static_loss).all(), f"{name}: non-finite loss"
    assert static_input.grad is not None and torch.isfinite(static_input.grad).all(), (
        f"{name}: invalid input gradient"
    )
    for parameter_name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"{name}: missing gradient for {parameter_name}"
        assert torch.isfinite(parameter.grad).all(), (
            f"{name}: non-finite gradient for {parameter_name}"
        )
    print(f"PASS: {name} forward/backward capture and replay", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "torch",
            "mcore",
            "torch_masked_scatter",
            "mcore_masked_scatter",
            "torch_index_copy",
            "mcore_index_copy",
        ),
        required=True,
    )
    args = parser.parse_args()

    device = _init_distributed()
    torch.manual_seed(42)

    if args.case.startswith("torch"):
        merger = TorchPatchMerger()
    else:
        merger = Qwen35VLPatchMerger(
            config=_build_config(),
            hidden_size=HIDDEN_SIZE,
            out_hidden_size=OUT_HIDDEN_SIZE,
            spatial_merge_size=SPATIAL_MERGE_SIZE,
        )
    if args.case.endswith("masked_scatter"):
        module = PatchMergerWithScatter(merger, "masked_scatter")
    elif args.case.endswith("index_copy"):
        module = PatchMergerWithScatter(merger, "index_copy")
    else:
        module = merger
    module = module.to(device=device, dtype=torch.bfloat16)
    _capture_forward_backward(args.case, module, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
