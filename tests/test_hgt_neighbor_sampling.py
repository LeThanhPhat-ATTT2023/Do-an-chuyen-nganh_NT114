from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from graphslm_ids.offline.training.hetero_graph_artifact import HeteroGraphArtifact
from graphslm_ids.offline.training.neighbor_sampling import (
    HeteroNeighborSampler,
    InMemoryNeighborBackend,
)


def _tiny_artifact() -> HeteroGraphArtifact:
    edge_index = {
        ("flow", "contains", "packet"): np.asarray([[0, 0, 1], [0, 1, 2]], dtype=np.int64),
        ("packet", "rev_contains", "flow"): np.asarray([[0, 1, 2], [0, 0, 1]], dtype=np.int64),
        ("packet", "matches_technique", "technique"): np.asarray(
            [[0, 1, 2], [0, 1, 1]],
            dtype=np.int64,
        ),
        ("technique", "belongs_to_tactic", "tactic"): np.asarray(
            [[0, 1], [0, 1]],
            dtype=np.int64,
        ),
    }
    edge_attr = {
        key: np.ones((value.shape[1], 1), dtype=np.float32) for key, value in edge_index.items()
    }
    edge_attr[("packet", "matches_technique", "technique")] = np.asarray(
        [[0.9], [0.8], [0.7]],
        dtype=np.float32,
    )
    return HeteroGraphArtifact(
        node_features={
            "flow": np.asarray([[1, 10, 0, 1, 2, 1], [1, 20, 0, 1, 3, 1]], dtype=np.float32),
            "packet": np.eye(3, 4, dtype=np.float32),
            "technique": np.eye(2, 4, dtype=np.float32),
            "tactic": np.arange(2, dtype=np.int64)[:, None],
        },
        edge_index=edge_index,
        edge_attr=edge_attr,
        flow_y=np.asarray([1, 0], dtype=np.int64),
        metadata={"label_mapping": {"benign": 0, "malicious": 1}},
    )


def test_neighbor_sampler_keeps_seed_labels_and_global_tactics() -> None:
    backend = InMemoryNeighborBackend(_tiny_artifact())
    sampler = HeteroNeighborSampler(
        backend,
        hops=2,
        fanouts={
            "flow__contains__packet": 10,
            "flow__matches_technique__technique": 10,
            "packet__matches_technique__technique": 10,
            "technique__belongs_to_tactic__tactic": 1,
        },
        reverse_fanouts={"rev_contains": 0},
        always_include_all_tactics=True,
        always_include_all_techniques=True,
        standardize_flow_features=False,
    )

    batch = sampler.sample([0])

    assert batch.seed_flow_ids.tolist() == [0]
    assert batch.seed_labels.tolist() == [1]
    assert batch.seed_mask.tolist() == [True]
    assert batch.local_to_global["packet"].tolist() == [0, 1]
    assert batch.local_to_global["technique"].tolist() == [0, 1]
    assert batch.local_to_global["tactic"].tolist() == [0, 1]
    assert batch.node_features["tactic"].reshape(-1).tolist() == [0, 1]
    assert ("flow", "contains", "packet") in batch.edge_index
    assert batch.edge_index[("flow", "contains", "packet")].shape[1] == 2
    assert batch.edge_index[("technique", "belongs_to_tactic", "tactic")].shape[1] == 2


def test_node_stats_count_deferred_packets() -> None:
    """When packet features are deferred to the feature store, node_features['packet']
    is an empty (0, dim) array. The reported node-count stat must still reflect the
    real number of sampled packets (from local_to_global), not the empty feature array.
    Regression test for avg_packet_nodes=0.0 in training logs.
    """
    backend = InMemoryNeighborBackend(_tiny_artifact())
    sampler = HeteroNeighborSampler(
        backend,
        hops=2,
        fanouts={
            "flow__contains__packet": 10,
            "packet__matches_technique__technique": 10,
            "technique__belongs_to_tactic__tactic": 1,
        },
        reverse_fanouts={"rev_contains": 0},
        always_include_all_tactics=True,
        always_include_all_techniques=True,
        standardize_flow_features=False,
        defer_packet_features=True,
    )

    batch = sampler.sample([0])

    # Features are deferred, so the feature array is empty...
    assert batch.node_features["packet"].shape[0] == 0
    # ...but two packets (0, 1) were genuinely sampled.
    assert batch.local_to_global["packet"].tolist() == [0, 1]
    # The stat must report the real sampled count, not the deferred-empty array.
    assert batch.stats["nodes"]["packet"] == 2
