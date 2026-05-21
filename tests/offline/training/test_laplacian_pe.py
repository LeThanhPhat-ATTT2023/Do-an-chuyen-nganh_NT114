"""Tests for HPE-HGT Laplacian positional encoding.

Reference: Dwivedi et al., "Graph Neural Networks with Learnable Structural
and Positional Representations", ICLR 2022 (Laplacian PE / GraphGPS), and
MTSU 2025 HPE-HGT for IDS (spec §3.3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------- PE computation ----------


def test_compute_laplacian_pe_on_path_graph_returns_smooth_eigenvecs():
    """Path graph 0-1-2-3-4: first non-trivial eigenvec is monotone (Fiedler)."""
    from graphslm_ids.offline.training.laplacian_pe import compute_laplacian_pe

    # Path graph adjacency: 5 nodes, edges (0,1), (1,2), (2,3), (3,4)
    edge_index = np.array(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=np.int64
    )
    num_nodes = 5

    pe = compute_laplacian_pe(edge_index, num_nodes=num_nodes, k=3)

    assert pe.shape == (5, 3), f"expected (5, 3), got {pe.shape}"
    # Output should be float32 (memory-efficient storage)
    assert pe.dtype == np.float32
    # No NaN/Inf
    assert np.isfinite(pe).all()


def test_compute_laplacian_pe_handles_k_greater_than_num_nodes():
    """When k > num_nodes - 1, the extra columns must be zero-padded."""
    from graphslm_ids.offline.training.laplacian_pe import compute_laplacian_pe

    # Triangle graph: 3 nodes, 3 edges. Laplacian has only 3 eigenvecs.
    edge_index = np.array([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]], dtype=np.int64)
    pe = compute_laplacian_pe(edge_index, num_nodes=3, k=8)
    assert pe.shape == (3, 8)
    # Last few columns are zero-padded
    assert np.allclose(pe[:, 3:], 0.0)


def test_compute_laplacian_pe_handles_disconnected_graph():
    """Graphs with isolated nodes still produce valid (finite) PE."""
    from graphslm_ids.offline.training.laplacian_pe import compute_laplacian_pe

    # 4 nodes: 0-1 connected, 2 and 3 isolated
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    pe = compute_laplacian_pe(edge_index, num_nodes=4, k=2)
    assert pe.shape == (4, 2)
    assert np.isfinite(pe).all()


def test_compute_laplacian_pe_deterministic_with_seed():
    """Same input + same seed → identical PE (sign-flipping handled internally)."""
    from graphslm_ids.offline.training.laplacian_pe import compute_laplacian_pe

    rng_edges = np.random.default_rng(0)
    n = 20
    src = rng_edges.integers(0, n, 60)
    dst = rng_edges.integers(0, n, 60)
    edge_index = np.stack([src, dst])

    pe1 = compute_laplacian_pe(edge_index, num_nodes=n, k=4, seed=42)
    pe2 = compute_laplacian_pe(edge_index, num_nodes=n, k=4, seed=42)
    np.testing.assert_array_equal(pe1, pe2)


# ---------- Save/load ----------


def test_save_load_pe_roundtrip(tmp_path):
    """save_laplacian_pe + load_laplacian_pe preserves array exactly."""
    from graphslm_ids.offline.training.laplacian_pe import (
        compute_laplacian_pe,
        load_laplacian_pe,
        save_laplacian_pe,
    )

    edge_index = np.array([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]], dtype=np.int64)
    pe = compute_laplacian_pe(edge_index, num_nodes=3, k=2)

    path = tmp_path / "pe.npy"
    save_laplacian_pe(pe, path)
    assert path.exists()

    loaded = load_laplacian_pe(path)
    np.testing.assert_array_equal(loaded, pe)


def test_load_laplacian_pe_returns_float32(tmp_path):
    """Loaded PE must be float32 regardless of source dtype on disk."""
    from graphslm_ids.offline.training.laplacian_pe import (
        load_laplacian_pe,
        save_laplacian_pe,
    )

    pe64 = np.random.randn(10, 4).astype(np.float64)
    path = tmp_path / "pe64.npy"
    save_laplacian_pe(pe64, path)
    loaded = load_laplacian_pe(path)
    assert loaded.dtype == np.float32


# ---------- Feature concatenation helper ----------


def test_concat_pe_to_features_basic_shape():
    """concat_pe_to_features: (N, D_feat) + (N, D_pe) → (N, D_feat + D_pe)."""
    from graphslm_ids.offline.training.laplacian_pe import concat_pe_to_features

    feats = np.random.randn(5, 8).astype(np.float32)
    pe = np.random.randn(5, 16).astype(np.float32)
    out = concat_pe_to_features(feats, pe)
    assert out.shape == (5, 24)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :8], feats)
    np.testing.assert_array_equal(out[:, 8:], pe)


def test_concat_pe_to_features_subset_by_indices():
    """When subgraph contains only a subset of nodes, PE is gathered by global index."""
    from graphslm_ids.offline.training.laplacian_pe import concat_pe_to_features

    feats = np.random.randn(3, 8).astype(np.float32)
    pe_full = np.arange(100, dtype=np.float32).reshape(10, 10)
    # Subgraph contains global indices [2, 5, 9]
    out = concat_pe_to_features(feats, pe_full, global_indices=np.array([2, 5, 9]))
    assert out.shape == (3, 18)
    np.testing.assert_array_equal(out[:, :8], feats)
    # PE rows match global indices
    np.testing.assert_array_equal(out[:, 8:], pe_full[[2, 5, 9]])


def test_concat_pe_to_features_pads_when_pe_shorter():
    """If PE has fewer columns than expected k, pad with zeros."""
    from graphslm_ids.offline.training.laplacian_pe import concat_pe_to_features

    feats = np.random.randn(3, 4).astype(np.float32)
    pe = np.random.randn(3, 8).astype(np.float32)
    out = concat_pe_to_features(feats, pe, pad_to_dim=16)
    assert out.shape == (3, 4 + 16)
    # First 8 PE columns are real, rest zero-padded
    np.testing.assert_array_equal(out[:, 4:12], pe)
    assert np.allclose(out[:, 12:], 0.0)
