# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the Qwen3.5-VL decoder-only proxy switch."""

from types import SimpleNamespace

from examples.multimodal_dev.forward_step import _use_text_only_proxy_bypass


def test_text_only_proxy_bypass_is_independent_of_graph_mode(monkeypatch):
    monkeypatch.setenv("MCORE_QWEN35_VL_TEXT_ONLY_PROXY", "1")

    assert _use_text_only_proxy_bypass(SimpleNamespace(cuda_graph_impl="none"))
    assert _use_text_only_proxy_bypass(SimpleNamespace(cuda_graph_impl="full_iteration"))


def test_legacy_text_only_bypass_remains_full_cg_only(monkeypatch):
    monkeypatch.delenv("MCORE_QWEN35_VL_TEXT_ONLY_PROXY", raising=False)
    monkeypatch.setenv("MCORE_QWEN35_VL_FULLCG_TEXT_ONLY", "1")

    assert not _use_text_only_proxy_bypass(SimpleNamespace(cuda_graph_impl="none"))
    assert _use_text_only_proxy_bypass(SimpleNamespace(cuda_graph_impl="full_iteration"))


def test_text_only_proxy_bypass_defaults_off(monkeypatch):
    monkeypatch.delenv("MCORE_QWEN35_VL_TEXT_ONLY_PROXY", raising=False)
    monkeypatch.delenv("MCORE_QWEN35_VL_FULLCG_TEXT_ONLY", raising=False)

    assert not _use_text_only_proxy_bypass(SimpleNamespace(cuda_graph_impl="none"))
