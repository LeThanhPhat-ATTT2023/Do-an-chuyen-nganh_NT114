"""Integration test for v2 graph artifact assembly."""
from __future__ import annotations

from pathlib import Path

import pytest

from graphslm_ids.offline.preprocessing.v2.extractor import extract_packets
from graphslm_ids.offline.preprocessing.v2.graph_builder import (
    build_v2_graph_artifact,
    save_v2_artifact,
)
from graphslm_ids.offline.preprocessing.v2.payload_features import FEATURE_DIM
from tests.v2._fixtures.build_tiny_pcap import build_demo_pcap

MITRE_TECH_CSV = Path("data/mitre/mitre_techniques.csv")
MITRE_TECH_NPY = Path("data/mitre/mitre_techniques_embeddings.npy")
MITRE_TT_CSV = Path("data/mitre/mitre_technique_tactic_edges.csv")


pytestmark = pytest.mark.skipif(
    not (MITRE_TECH_CSV.exists() and MITRE_TECH_NPY.exists() and MITRE_TT_CSV.exists()),
    reason="MITRE assets not present locally",
)


def test_v2_artifact_shapes_and_metadata(tmp_path: Path) -> None:
    pcap = tmp_path / "demo.pcap"
    build_demo_pcap(pcap)
    df = extract_packets(pcap, label="Demo")
    arts = build_v2_graph_artifact(
        df,
        mitre_techniques_csv=MITRE_TECH_CSV,
        mitre_technique_embeddings_npy=MITRE_TECH_NPY,
        mitre_technique_tactic_csv=MITRE_TT_CSV,
    )
    assert arts["metadata"]["artifact_version"] == "v2"
    # 5 packets -> 3 flows (1 TCP bidirectional + 1 UDP + 1 ICMP)
    assert arts["flow_x"].shape[0] == 3
    assert arts["packet_x"].shape == (5, FEATURE_DIM)
    # technique embeddings come straight from the MITRE NPY (~691 rows)
    assert arts["technique_x"].shape[0] > 100
    # contain edges: one per packet (every packet belongs to exactly one flow)
    assert arts["contain_edge_index"].shape == (2, 5)
    # link edges: only consecutive within the same flow -> 2 inside the TCP flow
    assert arts["link_edge_index"].shape[0] == 2
    # technique_tactic edges: deterministic from the MITRE CSV
    assert arts["technique_tactic_edge_index"].shape[0] == 2
    assert arts["technique_tactic_edge_index"].shape[1] > 100


def test_v2_artifact_round_trip(tmp_path: Path) -> None:
    pcap = tmp_path / "demo.pcap"
    build_demo_pcap(pcap)
    df = extract_packets(pcap, label="Demo")
    arts = build_v2_graph_artifact(
        df,
        mitre_techniques_csv=MITRE_TECH_CSV,
        mitre_technique_embeddings_npy=MITRE_TECH_NPY,
        mitre_technique_tactic_csv=MITRE_TT_CSV,
    )
    out_npz = tmp_path / "g.npz"
    out_meta = tmp_path / "g.meta.json"
    save_v2_artifact(arts, out_npz, out_meta)
    assert out_npz.exists() and out_meta.exists()
    import json

    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    assert meta["artifact_version"] == "v2"
    assert "flow_feature_names" in meta and len(meta["flow_feature_names"]) >= 40
