import json

import numpy as np

from graphslm_ids.offline.training.hetero_graph_artifact import load_three_tier_graph_artifact


def test_load_three_tier_graph_artifact_adds_reverse_edges(tmp_path) -> None:
    graph_npz = tmp_path / "graph_artifact_3tier.npz"
    graph_meta = tmp_path / "graph_artifact_3tier.meta.json"

    np.savez_compressed(
        graph_npz,
        flow_x=np.ones((2, 6), dtype=np.float32),
        flow_y=np.array([0, 1], dtype=np.int64),
        packet_x=np.ones((3, 256), dtype=np.uint8),
        packet_semantic_x=np.ones((3, 8), dtype=np.float32),
        technique_x=np.ones((2, 8), dtype=np.float32),
        contain_edge_index=np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int64),
        link_edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        link_edge_attr=np.ones((2, 1), dtype=np.float32),
        packet_technique_edge_index=np.array([[0, 2], [0, 1]], dtype=np.int64),
        packet_technique_edge_attr=np.array([[0.9], [0.8]], dtype=np.float32),
        flow_technique_edge_index=np.array([[0], [1]], dtype=np.int64),
        flow_technique_edge_attr=np.array([[0.95]], dtype=np.float32),
        technique_tactic_edge_index=np.array([[0, 1], [0, 1]], dtype=np.int64),
        technique_tactic_edge_attr=np.ones((2, 1), dtype=np.float32),
    )
    graph_meta.write_text(json.dumps({"num_tactics": 2}), encoding="utf-8")

    artifact = load_three_tier_graph_artifact(graph_npz, graph_meta, add_reverse_edges=True)

    assert artifact.node_features["flow"].shape == (2, 6)
    assert artifact.node_features["packet"].shape == (3, 8)
    assert artifact.node_features["technique"].shape == (2, 8)
    assert artifact.node_features["tactic"].shape == (2, 1)
    assert ("packet", "rev_contains", "flow") in artifact.edge_index
    assert artifact.edge_index[("packet", "rev_contains", "flow")].shape == (2, 3)


def test_load_v3_artifact_preserves_packet_dtype(tmp_path):
    import json
    import numpy as np
    from graphslm_ids.offline.training.hetero_graph_artifact import load_v3_artifact

    npz = tmp_path / "g.npz"
    meta = tmp_path / "g.meta.json"
    np.savez(
        str(npz),
        flow_x=np.zeros((2, 4), dtype=np.float32),
        flow_y=np.array([0, 1], dtype=np.int64),
        packet_x=np.ones((3, 2323), dtype=np.float16),
        host_x=np.zeros((1, 4), dtype=np.float32),
        technique_x=np.zeros((5, 8), dtype=np.float32),
    )
    meta.write_text(json.dumps({"artifact_version": "v3", "num_tactics": 2}))

    art32 = load_v3_artifact(npz, meta)
    assert art32.node_features["packet"].dtype == np.float32

    art16 = load_v3_artifact(npz, meta, packet_dtype="preserve")
    assert art16.node_features["packet"].dtype == np.float16
