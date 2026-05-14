from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from graphslm_ids.offline_path.training.hetero_graph_artifact import HeteroGraphArtifact
from graphslm_ids.offline_path.training.neighbor_sampling import (
    HeteroNeighborSampler,
    InMemoryNeighborBackend,
)
from graphslm_ids.offline_path.training.on_disk_graph_store import (
    edge_index_to_csr,
    gather_csr_neighbors,
)


def _random_csr(num_src: int, avg_degree: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    degrees = rng.integers(0, 2 * avg_degree + 1, size=num_src)
    indptr = np.empty((num_src + 1,), dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degrees, out=indptr[1:])
    num_edges = int(indptr[-1])
    indices = rng.integers(0, num_src, size=num_edges).astype(np.int64)
    attr = rng.random(num_edges).astype(np.float32)
    return indptr, indices, attr


def test_gather_csr_returns_full_neighbors_when_fanout_none() -> None:
    indptr, indices, attr = _random_csr(num_src=20, avg_degree=4, seed=0)
    src = np.array([3, 7, 11, 19, 0], dtype=np.int64)
    local_indptr, nbrs, weights = gather_csr_neighbors(indptr, indices, attr, src, fanout=None)

    # local_indptr lengths match per-row CSR degree
    expected = (indptr[src + 1] - indptr[src]).astype(np.int64)
    assert np.array_equal(np.diff(local_indptr), expected)
    # Concatenated payload equals the original CSR slices
    expected_nbrs = np.concatenate([indices[indptr[i]:indptr[i + 1]] for i in src.tolist()])
    expected_w = np.concatenate([attr[indptr[i]:indptr[i + 1]] for i in src.tolist()])
    assert np.array_equal(nbrs, expected_nbrs)
    assert np.allclose(weights, expected_w)


def test_gather_csr_fanout_caps_each_row_and_keeps_csr_order() -> None:
    indptr, indices, attr = _random_csr(num_src=50, avg_degree=6, seed=1)
    src = np.arange(50, dtype=np.int64)
    rng = np.random.default_rng(123)
    fanout = 3
    local_indptr, nbrs, weights = gather_csr_neighbors(
        indptr, indices, attr, src, fanout=fanout, rng=rng
    )

    degrees = (indptr[src + 1] - indptr[src]).astype(np.int64)
    expected_counts = np.minimum(degrees, fanout)
    assert np.array_equal(np.diff(local_indptr), expected_counts)
    assert nbrs.shape[0] == int(expected_counts.sum())

    # Sampled neighbors must be a subset of the original neighbours for that row.
    for row, node_id in enumerate(src.tolist()):
        row_nbrs = nbrs[local_indptr[row] : local_indptr[row + 1]]
        row_attr = weights[local_indptr[row] : local_indptr[row + 1]]
        full_nbrs = indices[indptr[node_id] : indptr[node_id + 1]]
        full_attr = attr[indptr[node_id] : indptr[node_id + 1]]
        # Every sampled neighbour and its weight live in the original row.
        for nbr, wt in zip(row_nbrs.tolist(), row_attr.tolist()):
            matches = (full_nbrs == nbr) & np.isclose(full_attr, wt)
            assert bool(matches.any()), f"row {node_id} sample {nbr} not in source CSR"


def test_gather_csr_fanout_zero_returns_empty() -> None:
    indptr, indices, attr = _random_csr(num_src=10, avg_degree=4, seed=2)
    src = np.arange(10, dtype=np.int64)
    local_indptr, nbrs, weights = gather_csr_neighbors(indptr, indices, attr, src, fanout=0)
    assert nbrs.size == 0
    assert weights.size == 0
    assert local_indptr.shape == (11,)
    assert int(local_indptr[-1]) == 0


def test_gather_csr_handles_zero_degree_rows() -> None:
    # Source with several empty rows interleaved.
    indptr = np.array([0, 2, 2, 3, 3, 5], dtype=np.int64)
    indices = np.array([10, 11, 20, 30, 31], dtype=np.int64)
    attr = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    src = np.arange(5, dtype=np.int64)
    local_indptr, nbrs, weights = gather_csr_neighbors(indptr, indices, attr, src, fanout=None)
    assert np.array_equal(np.diff(local_indptr), [2, 0, 1, 0, 2])
    assert nbrs.tolist() == indices.tolist()
    assert np.allclose(weights, attr)


def _three_tier_toy() -> HeteroGraphArtifact:
    """A graph slightly bigger than the existing tiny fixture, with multi-hop paths."""
    edge_index = {
        ("flow", "contains", "packet"): np.array(
            [[0, 0, 0, 1, 1, 2, 2, 3, 4],
             [0, 1, 2, 3, 4, 5, 6, 7, 8]],
            dtype=np.int64,
        ),
        ("packet", "rev_contains", "flow"): np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8],
             [0, 0, 0, 1, 1, 2, 2, 3, 4]],
            dtype=np.int64,
        ),
        ("packet", "matches_technique", "technique"): np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8],
             [0, 0, 1, 1, 2, 2, 0, 1, 2]],
            dtype=np.int64,
        ),
        ("technique", "belongs_to_tactic", "tactic"): np.array(
            [[0, 1, 2], [0, 0, 1]],
            dtype=np.int64,
        ),
    }
    edge_attr = {
        key: np.ones((value.shape[1], 1), dtype=np.float32) for key, value in edge_index.items()
    }
    flow_x = np.tile(np.arange(6, dtype=np.float32), (5, 1))
    return HeteroGraphArtifact(
        node_features={
            "flow": flow_x,
            "packet": np.eye(9, 4, dtype=np.float32),
            "technique": np.eye(3, 4, dtype=np.float32),
            "tactic": np.arange(2, dtype=np.int64)[:, None],
        },
        edge_index=edge_index,
        edge_attr=edge_attr,
        flow_y=np.array([0, 1, 0, 1, 0], dtype=np.int64),
        metadata={"label_mapping": {"benign": 0, "malicious": 1}},
    )


def _make_sampler(seed: int) -> HeteroNeighborSampler:
    backend = InMemoryNeighborBackend(_three_tier_toy())
    return HeteroNeighborSampler(
        backend,
        hops=2,
        fanouts={
            "flow__contains__packet": 2,
            "packet__matches_technique__technique": 2,
            "technique__belongs_to_tactic__tactic": 1,
        },
        reverse_fanouts={"rev_contains": 0},
        always_include_all_tactics=True,
        always_include_all_techniques=True,
        standardize_flow_features=False,
        seed=seed,
    )


def test_sampler_includes_static_knowledge_and_seed_flow() -> None:
    sampler = _make_sampler(seed=7)
    batch = sampler.sample([0, 2])

    # Seeds preserved in order, with correct labels.
    assert batch.seed_flow_ids.tolist() == [0, 2]
    assert batch.seed_labels.tolist() == [0, 0]
    assert batch.seed_mask.shape == (batch.node_features["flow"].shape[0],)
    assert batch.seed_mask.sum() == 2

    # Static knowledge always present (always_include_all_*).
    assert batch.local_to_global["technique"].tolist() == [0, 1, 2]
    assert batch.local_to_global["tactic"].tolist() == [0, 1]

    # technique→tactic edges always emitted (3 techniques, one tactic each).
    tt_edge = batch.edge_index[("technique", "belongs_to_tactic", "tactic")]
    assert tt_edge.shape[1] == 3


def test_sampler_caps_fanout_per_relation() -> None:
    sampler = _make_sampler(seed=11)
    batch = sampler.sample([0])
    flow_packet_edges = batch.edge_index[("flow", "contains", "packet")]
    # flow 0 has 3 packets but fanout is 2 → at most 2 outgoing per seed flow.
    assert flow_packet_edges.shape[1] <= 2


def test_sampler_is_deterministic_given_seed() -> None:
    sampler_a = _make_sampler(seed=42)
    sampler_b = _make_sampler(seed=42)
    batch_a = sampler_a.sample([0, 1, 2])
    batch_b = sampler_b.sample([0, 1, 2])
    for edge_type, ei_a in batch_a.edge_index.items():
        ei_b = batch_b.edge_index[edge_type]
        assert ei_a.shape == ei_b.shape
        if ei_a.size:
            assert np.array_equal(ei_a, ei_b)
