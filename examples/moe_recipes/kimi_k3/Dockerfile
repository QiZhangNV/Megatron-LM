# syntax=docker/dockerfile:1.7
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Kimi-K3 training dependencies for GB200/GB300 (ARM64).
FROM nvcr.io/nvidia/pytorch:26.04-py3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Keep runtime and headers on the same NCCL release; preserve NGC CUDA/PyTorch.
ARG NCCL_DEB_VERSION=2.32.3-1+cuda13.4
ENV NCCL_VERSION=2.32.3

ARG TE_COMMIT=0a2ebe942b9f6a66aa0292502966e73114a7b912
ARG CUDNN_FRONTEND_COMMIT=5ba9432e652e0b4df3aaca85cf3f95fe6bb7375a
ARG EMERGING_OPTIMIZERS_COMMIT=a44d1f83a4950b445f2a77ca71177c5f033d45d5
# causal-conv1d 1.6.2.post1. gdn_pre_gated_delta_rule_fusion uses its C++
# depthwise-convolution backward; the KDA layer raises ImportError without it.
ARG CAUSAL_CONV1D_COMMIT=4f6ae4e26ae5fe8af9372f8d312ab25cc4595223
# HybridEP from the DeepEP hybrid-ep branch. Revisions before 17cfb81 (#625)
# read a per-call pinned host token count from the permute/unpermute kernels
# in blocking mode and fail intermittently with an illegal memory access on
# long packed sequences (DeepEP issue 756); 10d4dd7 (2026-09-11) no longer
# does and ran the 512K packed-THD recipe cleanly. Place each EP64 group
# within one NVLink domain; this is not a cross-domain RDMA build.
ARG HYBRID_EP_COMMIT=10d4dd7377d5bce900fbb4b80cce863764892b95
ARG NVTE_CUDA_ARCHS="100a;103a"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates gettext libfabric-dev python3-venv \
        "libnccl2=${NCCL_DEB_VERSION}" "libnccl-dev=${NCCL_DEB_VERSION}" && \
    ldconfig && \
    sed -i "s/^NCCL_VERSION=.*/NCCL_VERSION=${NCCL_VERSION}/" /etc/environment && \
    rm -rf /var/lib/apt/lists/* /opt/megatron-lm

# Fail the build if the linker still resolves an older NCCL runtime.
RUN python -c 'import ctypes; lib = ctypes.CDLL("libnccl.so.2"); version = ctypes.c_int(); assert lib.ncclGetVersion(ctypes.byref(version)) == 0; assert version.value == 23203, version.value'

# Inherit NGC PyTorch/CUDA while keeping pip additions separate from
# OS-managed packages such as python3-yaml.
RUN python -m venv --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
        packaging ninja==1.11.1.1 cmake 'pybind11[global]>=2.13,<3' \
        sentencepiece tiktoken einops wandb tqdm datasets tensorboard \
        pydantic pyyaml omegaconf nvtx psutil flask-restful wrapt \
        tensorstore multi-storage-client opentelemetry-api \
        transformers==5.16.1 flash-linear-attention==0.5.2 \
        nvidia-resiliency-ext==0.6.0 nvidia-mathdx==25.1.1 \
        'nvidia-cutlass-dsl[cu13]==4.5.2' apache-tvm-ffi==0.1.11

RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
        --no-build-isolation \
        "nvidia-cudnn-frontend[cutedsl] @ git+https://github.com/NVIDIA/cudnn-frontend.git@${CUDNN_FRONTEND_COMMIT}"

# These recipes use HybridEP. Keep the unused TE NCCL EP extension disabled
# when upgrading the NCCL runtime.
RUN unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" NVTE_BUILD_THREADS_PER_JOB=8 \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 \
    python -m pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/NVIDIA/TransformerEngine.git@${TE_COMMIT}"

# Muon with per-head QKV splitting. Pin the source revision rather than
# the older release used by some published container tags.
RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir \
        --no-build-isolation \
        "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@${EMERGING_OPTIMIZERS_COMMIT}"

# No prebuilt causal-conv1d wheel matches this base, so build from source.
# nvcc 13 adds the sm_100 and sm_103 cubins to the default architecture list.
RUN unset PIP_CONSTRAINT && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    python -m pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/Dao-AILab/causal-conv1d.git@${CAUSAL_CONV1D_COMMIT}"

# The hybrid-ep build links libibverbs / libmlx5 from the NGC base image's source build of
# rdma-core (60, /workspace/rdma-core/build/lib) and records that directory as the extension's
# RUNPATH. When deep_ep is imported before NCCL creates its first communicator (Megatron imports
# it at module level in megatron/core/transformer/moe/fused_a2a.py), that rdma-core is loaded
# first, NCCL's network plugin then loads the system rdma-core (59.1) as well, its devx HCA query
# fails with "No space left on device" and NCCL silently falls back to NET/Socket: every transfer
# that leaves the NVLink domain (pipeline parallelism across racks, cross-rack all-reduces) runs
# over TCP at about 3 GB/s. Dropping the RUNPATH makes the extension resolve the system libraries
# like NCCL does; the HYBRID_EP_MULTINODE=0 build uses no RDMA path of its own.
RUN git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP && \
    cd /opt/DeepEP && git checkout "${HYBRID_EP_COMMIT}" && \
    unset PIP_CONSTRAINT && \
    HYBRID_EP_MULTINODE=0 TORCH_CUDA_ARCH_LIST="10.0" \
    python -m pip install --no-build-isolation --no-cache-dir . && \
    patchelf --remove-rpath /usr/local/lib/python3.12/dist-packages/hybrid_ep_cpp*.so && \
    ! readelf -d /usr/local/lib/python3.12/dist-packages/hybrid_ep_cpp*.so | grep -q rdma-core

# NGC's triton-kernels declares pytest as a dependency.
# Preserve dependency consistency as a build requirement.
RUN unset PIP_CONSTRAINT && python -m pip install --no-cache-dir pytest && \
    python -m pip check && rm -rf /root/.cache/pip
WORKDIR /workspace/Megatron-LM
