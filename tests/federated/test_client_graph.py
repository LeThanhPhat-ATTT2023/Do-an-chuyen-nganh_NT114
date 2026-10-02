import numpy as np

from graphslm_ids.offline.federated.client_graph import (
    FROM_HOST,
    TO_HOST,
    build_client_artifact,
    packet_owner,
    recompute_host_features,
)
from graphslm_ids.offline.training.hetero_graph_artifact import HeteroGraphArtifact


def test_client_graph_keeps_only_owned_flow_and_packet_edges(tiny_artifact):
    art = tiny_artifact
    client = np.arange(0, art.flow_y.shape[0], 2)
    sub = build_client_artifact(art, client)
    owner = packet_owner(art)
    flows = set(client.tolist())
    assert sub.edge_index.keys() == art.edge_index.keys()
    for key, edges in sub.edge_index.items():
        for pos, ntype in ((0, key[0]), (1, key[2])):
            ids = edges[pos]
            if ntype == "flow":
                assert set(ids.tolist()) <= flows, key
            if ntype == "packet":
                assert set(owner[ids].tolist()) <= flows, key
        assert sub.edge_attr[key].shape[0] == edges.shape[1]
    burst = sub.edge_index[("flow", "burst_neighbor", "flow")]
    assert burst.shape[1] > 0 and np.all(burst % 2 == 0)


def test_client_graph_keeps_mitre_knowledge_edges(tiny_artifact):
    sub = build_client_artifact(tiny_artifact, np.array([0]))
    for key in (
        ("technique", "has_subtechnique", "technique"),
        ("technique", "belongs_to_tactic", "tactic"),
        ("tactic", "rev_belongs_to_tactic", "technique"),
    ):
        assert sub.edge_index[key].shape[1] == tiny_artifact.edge_index[key].shape[1]


def test_client_graph_shares_large_arrays(tiny_artifact):
    sub = build_client_artifact(tiny_artifact, np.arange(5))
    assert sub.node_features["packet"] is tiny_artifact.node_features["packet"]
    assert sub.node_features["flow"] is tiny_artifact.node_features["flow"]
    assert sub.flow_y is tiny_artifact.flow_y


def test_recompute_host_features_matches_hand_count():
    flow_x = np.array([[6, 1000, 80], [6, 1001, 80], [6, 1002, 22]], dtype=np.float32)
    art = HeteroGraphArtifact(
        node_features={
            "flow": flow_x,
            "packet": np.zeros((0, 2), dtype=np.float32),
            "host": np.full((3, 4), 99.0, dtype=np.float32),
            "technique": np.zeros((1, 2), dtype=np.float32),
            "tactic": np.arange(1)[:, None],
        },
        edge_index={
            FROM_HOST: np.array([[0, 1, 2], [0, 0, 1]]),
            TO_HOST: np.array([[0, 1, 2], [1, 2, 2]]),
        },
        edge_attr={},
        flow_y=np.zeros(3, dtype=np.int64),
        metadata={"flow_feature_names": ["proto", "src_port", "dst_port"]},
    )
    hx = recompute_host_features(art, np.array([0, 1]))
    assert hx.tolist() == [[2, 0, 1, 2], [0, 1, 0, 0], [0, 1, 0, 0]]
