"""Numerical-stability tests for HGT under BF16 autocast.

These tests probe whether the post-aggregation block in HGTLayer
(out projection + LayerNorm + FFN) can safely run in BF16 instead of
the current FP32 fallback. The fallback exists because hidden_dim=384
matmuls in FP16 can overflow, but BF16 has FP32's dynamic range and
should be safe — these tests prove or disprove it.
"""
from __future__ import annotations

import pytest
import torch

from graphslm_ids.models.hgt import HGTLayer


def _make_layer(hidden_dim: int = 384, num_heads: int = 12) -> HGTLayer:
    node_types = ["flow", "packet", "technique", "tactic"]
    edge_types = [
        ("flow", "contains", "packet"),
        ("packet", "next_packet", "packet"),
        ("flow", "matches_technique", "technique"),
    ]
    return HGTLayer(
        node_types=node_types,
        edge_types=edge_types,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.0,
        ffn_multiplier=4,
    )


def _make_inputs(
    hidden_dim: int = 384,
    n_flow: int = 32,
    n_packet: int = 128,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    x_dict = {
        "flow": torch.randn(n_flow, hidden_dim, dtype=dtype, device=device),
        "packet": torch.randn(n_packet, hidden_dim, dtype=dtype, device=device),
        "technique": torch.randn(8, hidden_dim, dtype=dtype, device=device),
        "tactic": torch.randn(4, hidden_dim, dtype=dtype, device=device),
    }
    edge_index_dict = {
        ("flow", "contains", "packet"): torch.stack(
            [
                torch.randint(0, n_flow, (200,)),
                torch.randint(0, n_packet, (200,)),
            ]
        ).to(device),
        ("packet", "next_packet", "packet"): torch.stack(
            [
                torch.randint(0, n_packet, (300,)),
                torch.randint(0, n_packet, (300,)),
            ]
        ).to(device),
        ("flow", "matches_technique", "technique"): torch.stack(
            [
                torch.randint(0, n_flow, (50,)),
                torch.randint(0, 8, (50,)),
            ]
        ).to(device),
    }
    return x_dict, edge_index_dict


def test_hgt_layer_forward_bf16_on_cpu_no_nan():
    """Forward pass with BF16 inputs must produce no NaN/Inf on CPU."""
    torch.manual_seed(42)
    layer = _make_layer().to(dtype=torch.float32)
    x_dict, edge_index_dict = _make_inputs(dtype=torch.bfloat16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        out = layer(x_dict, edge_index_dict)
    for node_type, tensor in out.items():
        assert torch.isfinite(tensor).all(), f"NaN/Inf in {node_type} output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hgt_layer_backward_bf16_on_cuda_no_nan_grad():
    """Backward pass under CUDA BF16 autocast must produce no NaN gradients."""
    torch.manual_seed(42)
    layer = _make_layer().cuda()
    x_dict, edge_index_dict = _make_inputs(device="cuda", dtype=torch.bfloat16)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        out = layer(x_dict, edge_index_dict)
        loss = sum(t.float().mean() for t in out.values())
    loss.backward()
    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hgt_layer_bf16_full_repro_stable_across_3_runs():
    """3 forward+backward passes with same seed must all stay finite."""
    for trial in range(3):
        torch.manual_seed(42 + trial)
        layer = _make_layer().cuda()
        x_dict, edge_index_dict = _make_inputs(device="cuda", dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            out = layer(x_dict, edge_index_dict)
            loss = sum(t.float().mean() for t in out.values())
        loss.backward()
        for name, p in layer.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"trial {trial}: NaN in {name}"
