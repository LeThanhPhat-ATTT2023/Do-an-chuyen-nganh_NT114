"""End-to-end loader test: build a tiny v2 artifact and load it through the
hetero_graph_artifact API the trainer uses."""
from __future__ import annotations

from pathlib import Path

import pytest

from graphslm_ids.offline.preprocessing.v2.extractor import extract_packets
from graphslm_ids.offline.preprocessing.v2.graph_builder import (
    build_v2_graph_artifact,
    save_v2_artifact,
)
from graphslm_ids.offline.training.hetero_graph_artifact import load_v2_artifact
from tests.v2._fixtures.build_tiny_pcap import build_demo_pcap

MITRE_TECH_CSV = Path("data/mitre/mitre_techniques.csv")
MITRE_TECH_NPY = Path("data/mitre/mitre_techniques_embeddings.npy")
MITRE_TT_CSV = Path("data/mitre/mitre_technique_tactic_edges.csv")


pytestmark = pytest.mark.skipif(
    not (MITRE_TECH_CSV.exists() and MITRE_TECH_NPY.exists() and MITRE_TT_CSV.exists()),
    reason="MITRE assets not present locally",
)


def test_v2_loader_round_trip(tmp_path: Path) -> None:
    pcap = tmp_path / "demo.pcap"
    build_demo_pcap(pcap)
    df = extract_packets(pcap, label="Demo")
    arts = build_v2_graph_artifact(
        df,
        mitre_techniques_csv=MITRE_TECH_CSV,
        mitre_technique_embeddings_npy=MITRE_TECH_NPY,
        mitre_technique_tactic_csv=MITRE_TT_CSV,
    )
    npz = tmp_path / "g.npz"
    meta = tmp_path / "g.meta.json"
    save_v2_artifact(arts, npz, meta)

    loaded = load_v2_artifact(graph_npz=npz, graph_meta_json=meta, add_reverse_edges=True)
    # Node types all present.
    for k in ("flow", "packet", "technique", "tactic"):
        assert k in loaded.node_features
    # All five canonical edge types present (+ five reverse types).
    expected_edge_types = {
        ("flow", "contains", "packet"),
        ("packet", "next_packet", "packet"),
        ("packet", "matches_technique", "technique"),
        ("flow", "matches_technique", "technique"),
        ("technique", "belongs_to_tactic", "tactic"),
    }
    for et in expected_edge_types:
        assert et in loaded.edge_index, f"missing edge type {et}"
    # Reverse edges added.
    assert ("packet", "rev_contains", "flow") in loaded.edge_index
    assert loaded.metadata["artifact_version"] == "v2"
    # flow_y is per-flow integer label.
    assert loaded.flow_y.ndim == 1 and loaded.flow_y.shape[0] == loaded.node_features["flow"].shape[0]


def test_v2_loader_rejects_v1_artifact(tmp_path: Path) -> None:
    """If someone points the v2 loader at a v1 artifact, it must fail loudly."""
    import json

    import numpy as np

    npz = tmp_path / "v1.npz"
    meta = tmp_path / "v1.meta.json"
    np.savez(
        str(npz),
        flow_x=np.zeros((1, 6), dtype=np.float32),
        flow_y=np.zeros(1, dtype=np.int64),
        packet_x=np.zeros((1, 256), dtype=np.float32),
        technique_x=np.zeros((1, 768), dtype=np.float32),
        contain_edge_index=np.zeros((2, 1), dtype=np.int64),
        link_edge_index=np.zeros((2, 0), dtype=np.int64),
        link_edge_attr=np.zeros((0, 1), dtype=np.float32),
        packet_technique_edge_index=np.zeros((2, 0), dtype=np.int64),
        flow_technique_edge_index=np.zeros((2, 0), dtype=np.int64),
        technique_tactic_edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    meta.write_text(json.dumps({"artifact_version": "v1"}))

    with pytest.raises(ValueError, match="artifact_version"):
        load_v2_artifact(graph_npz=npz, graph_meta_json=meta)
