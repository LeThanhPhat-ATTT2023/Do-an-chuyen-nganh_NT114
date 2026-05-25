"""Smoke / integration tests for the v3 graph artifact assembly.

Real production builds take ~90 minutes on the 14 GB dataset; these tests
exercise the assembly path with a minimal fixture so the public contract
(node / edge counts, dtype, schema keys, metadata roundtrip) is locked in.

What we test:

* ``build_v3_graph_artifact`` returns the documented array set + metadata
  with ``artifact_version='v3'``.
* Control packets (``payload_len == 0``) are dropped from the packet tier.
* ``has_subtechnique`` edges are emitted when the techniques CSV contains
  sub-techniques.
* ``save_v3_artifact`` writes an NPZ + JSON sidecar that round-trips.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphslm_ids.offline.preprocessing.v2.payload_features import (
    FEATURE_DIM as PAYLOAD_FEATURE_DIM,
)
from graphslm_ids.offline.preprocessing.v3.graph_builder import (
    _build_has_subtechnique_edges,
    build_v3_graph_artifact,
    save_v3_artifact,
)


@pytest.fixture
def gb_fixture(tmp_path: Path, mitre_dir_fixture: Path) -> dict:
    """Materialise every input the v3 graph builder consumes.

    The fixture conftest's tactic CSV uses ``tactic_id`` but the v2-inherited
    ``_load_mitre_tactics`` helper requires ``tactic_shortname``; we rewrite a
    fresh tactic CSV here under ``tmp_path`` to satisfy the contract without
    touching shared fixtures.
    """
    # Flows + packets — 4 flows, 8 packets (2 per flow), 2 attack classes.
    flows = pd.DataFrame(
        [
            {
                "flow_id": "f0",
                "label": "SqlInjection",
                "flow_start_ts": 1000.0,
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_port": 1024,
                "dst_port": 80,
                "proto": 0,
                "fwd_bytes": 200.0,
                "bwd_bytes": 400.0,
            },
            {
                "flow_id": "f1",
                "label": "SqlInjection",
                "flow_start_ts": 1001.0,
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.3",
                "src_port": 1025,
                "dst_port": 80,
                "proto": 0,
                "fwd_bytes": 100.0,
                "bwd_bytes": 200.0,
            },
            {
                "flow_id": "f2",
                "label": "CommandInjection",
                "flow_start_ts": 2000.0,
                "src_ip": "10.0.0.4",
                "dst_ip": "10.0.0.5",
                "src_port": 2048,
                "dst_port": 22,
                "proto": 0,
                "fwd_bytes": 50.0,
                "bwd_bytes": 75.0,
            },
            {
                "flow_id": "f3",
                "label": "CommandInjection",
                "flow_start_ts": 2001.0,
                "src_ip": "10.0.0.6",
                "dst_ip": "10.0.0.5",
                "src_port": 2049,
                "dst_port": 22,
                "proto": 0,
                "fwd_bytes": 30.0,
                "bwd_bytes": 50.0,
            },
        ]
    ).set_index("flow_id")

    # Packets: 2 per flow, one with non-empty payload (kept) + one control (dropped).
    packet_rows = []
    sql_payload = b"GET /?id=1 UNION SELECT * FROM information_schema.users HTTP/1.1\r\n\r\n"
    cmd_payload = b"POST /run HTTP/1.1\r\n\r\ncmd.exe /c whoami"
    for fid, base_ts, payload, label in (
        ("f0", 1000.0, sql_payload, "SqlInjection"),
        ("f1", 1001.0, sql_payload, "SqlInjection"),
        ("f2", 2000.0, cmd_payload, "CommandInjection"),
        ("f3", 2001.0, cmd_payload, "CommandInjection"),
    ):
        # Real packet (kept).
        packet_rows.append(
            {
                "ts": base_ts,
                "flow_id": fid,
                "payload": payload,
                "payload_len": len(payload),
                "label": label,
            }
        )
        # Control packet (dropped during build).
        packet_rows.append(
            {
                "ts": base_ts + 0.001,
                "flow_id": fid,
                "payload": b"",
                "payload_len": 0,
                "label": label,
            }
        )
    packets = pd.DataFrame(packet_rows)

    # Embeddings npy — 5 rows aligned with mitre_techniques.csv (768-d to match
    # the production embedding shape, though smaller would also work).
    techs = pd.read_csv(mitre_dir_fixture / "mitre_techniques.csv")
    emb = np.random.default_rng(7).standard_normal((len(techs), 768)).astype(np.float32)
    emb_path = mitre_dir_fixture / "mitre_techniques_embeddings.npy"
    np.save(emb_path, emb)

    # Tactic CSV — rewrite with the v2-required ``tactic_shortname`` column.
    tactic_csv = tmp_path / "mitre_technique_tactic_edges.csv"
    pd.DataFrame(
        [
            {"technique_id": "T1190", "tactic_shortname": "initial-access"},
            {"technique_id": "T1059", "tactic_shortname": "execution"},
            {"technique_id": "T1046", "tactic_shortname": "discovery"},
        ]
    ).to_csv(tactic_csv, index=False)

    # Class -> technique map covering the two fixture classes.
    class_map_csv = tmp_path / "class_technique_map.csv"
    pd.DataFrame(
        [
            {"class": "SqlInjection", "technique": "T1190", "weight": 1.0},
            {"class": "CommandInjection", "technique": "T1059", "weight": 1.0},
        ]
    ).to_csv(class_map_csv, index=False)

    # Technique -> family routing.
    tech_family_csv = tmp_path / "technique_family.csv"
    pd.DataFrame(
        [
            {"technique": "T1190", "family": "injection"},
            {"technique": "T1059", "family": "command_exec"},
            {"technique": "T1046", "family": "recon"},
        ]
    ).to_csv(tech_family_csv, index=False)

    # PMI table — minimal entries so the ensemble has something to look up.
    # Tokens chosen to match the synthetic payloads above.
    pmi_table_path = tmp_path / "pmi_table.parquet"
    pd.DataFrame(
        [
            {"token": "t:union", "technique": "T1190", "family": "injection", "weight": 0.9},
            {"token": "t:select", "technique": "T1190", "family": "injection", "weight": 0.8},
            {"token": "t:cmd.exe", "technique": "T1059", "family": "command_exec", "weight": 0.9},
            {"token": "t:whoami", "technique": "T1059", "family": "command_exec", "weight": 0.7},
        ]
    ).to_parquet(pmi_table_path, index=False)

    # Splits dict matching the make_splits shape.
    splits = {
        "random": {
            "train": np.array(["f0", "f2"], dtype=object),
            "val": np.array(["f1"], dtype=object),
            "test": np.array(["f3"], dtype=object),
        },
        "temporal": {
            "train": np.array(["f0", "f2"], dtype=object),
            "val": np.array(["f1"], dtype=object),
            "test": np.array(["f3"], dtype=object),
        },
    }

    return {
        "packets": packets,
        "feats": flows,
        "splits": splits,
        "mitre_techniques_csv": mitre_dir_fixture / "mitre_techniques.csv",
        "mitre_techniques_npy": emb_path,
        "mitre_tactic_csv": tactic_csv,
        "mitre_stix": mitre_dir_fixture / "enterprise-attack.json",
        "class_map_csv": class_map_csv,
        "tech_family_csv": tech_family_csv,
        "pmi_table": pmi_table_path,
    }


def test_build_has_subtechnique_edges_emits_parent_to_child() -> None:
    techs = pd.DataFrame(
        [
            {"technique_id": "T1190", "is_subtechnique": False},
            {"technique_id": "T1059", "is_subtechnique": False},
            {"technique_id": "T1059.004", "is_subtechnique": True},
            {"technique_id": "T1059.007", "is_subtechnique": True},
        ]
    )
    tid_to_idx = {tid: i for i, tid in enumerate(techs["technique_id"].astype(str).tolist())}
    edges = _build_has_subtechnique_edges(techs, tid_to_idx)
    # Two sub-techniques -> two parent->child edges.
    assert edges.shape == (2, 2)
    # Every edge maps a parent (T1059) src to a sub-technique dst.
    parent_idx = tid_to_idx["T1059"]
    assert np.all(edges[0] == parent_idx)


def test_build_v3_artifact_smoke(gb_fixture: dict) -> None:
    arts = build_v3_graph_artifact(
        gb_fixture["packets"],
        gb_fixture["feats"],
        gb_fixture["splits"],
        mitre_techniques_csv=gb_fixture["mitre_techniques_csv"],
        mitre_technique_embeddings_npy=gb_fixture["mitre_techniques_npy"],
        mitre_technique_tactic_csv=gb_fixture["mitre_tactic_csv"],
        mitre_stix_json=gb_fixture["mitre_stix"],
        class_technique_map_csv=gb_fixture["class_map_csv"],
        technique_family_csv=gb_fixture["tech_family_csv"],
        pmi_table_parquet=gb_fixture["pmi_table"],
        payload_length=256,
        tau_edge=0.4,
    )

    md = arts["metadata"]
    assert md["artifact_version"] == "v3"
    assert md["num_flows"] == 4
    # 8 raw packets; 4 control packets dropped -> 4 kept.
    assert md["num_packets"] == 4
    assert md["num_packets_raw"] == 8

    # Node features have the documented dtypes and shapes.
    assert arts["flow_x"].dtype == np.float32
    assert arts["packet_x"].shape == (4, PAYLOAD_FEATURE_DIM)
    assert arts["packet_x"].dtype == np.float32

    # Contain edges: one per kept packet.
    assert arts["contain_edge_index"].shape[1] == 4

    # The fixture has sub-techniques (T1059.004 / T1059.007) -> hierarchy edges fire.
    assert arts["has_subtechnique_edge_index"].shape[1] == 2

    # Splits round-trip through metadata as JSON-safe lists of strings.
    assert set(md["splits"]) == {"random", "temporal"}
    assert md["splits"]["random"]["train"] == ["f0", "f2"]

    # The 5 typed-evidence edge types must all appear in the array set, even
    # if their edge count is 0 (audit gate relies on the keys existing).
    for fam in ("injection", "command_exec", "file_upload", "recon", "c2_beacon"):
        assert f"evidence_{fam}_edge_index" in arts
        assert f"evidence_{fam}_edge_attr" in arts


def test_save_v3_artifact_roundtrips(gb_fixture: dict, tmp_path: Path) -> None:
    arts = build_v3_graph_artifact(
        gb_fixture["packets"],
        gb_fixture["feats"],
        gb_fixture["splits"],
        mitre_techniques_csv=gb_fixture["mitre_techniques_csv"],
        mitre_technique_embeddings_npy=gb_fixture["mitre_techniques_npy"],
        mitre_technique_tactic_csv=gb_fixture["mitre_tactic_csv"],
        mitre_stix_json=gb_fixture["mitre_stix"],
        class_technique_map_csv=gb_fixture["class_map_csv"],
        technique_family_csv=gb_fixture["tech_family_csv"],
        pmi_table_parquet=gb_fixture["pmi_table"],
        payload_length=256,
        tau_edge=0.4,
    )
    out_npz = tmp_path / "graph.npz"
    out_meta = tmp_path / "graph.meta.json"
    save_v3_artifact(arts, out_npz, out_meta)

    assert out_npz.exists() and out_meta.exists()

    # Meta JSON loads cleanly and preserves the artifact version + counts.
    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    assert meta["artifact_version"] == "v3"
    assert meta["num_flows"] == 4
    assert meta["num_packets"] == 4

    # NPZ contains the documented array set.
    with np.load(out_npz, allow_pickle=False) as npz:
        assert "flow_x" in npz.files
        assert "packet_x" in npz.files
        assert "contain_edge_index" in npz.files


def test_build_v3_artifact_empty_packets_raises(gb_fixture: dict) -> None:
    with pytest.raises(ValueError, match="empty packets_df"):
        build_v3_graph_artifact(
            pd.DataFrame(columns=gb_fixture["packets"].columns),
            gb_fixture["feats"],
            gb_fixture["splits"],
            mitre_techniques_csv=gb_fixture["mitre_techniques_csv"],
            mitre_technique_embeddings_npy=gb_fixture["mitre_techniques_npy"],
            mitre_technique_tactic_csv=gb_fixture["mitre_tactic_csv"],
            mitre_stix_json=gb_fixture["mitre_stix"],
            class_technique_map_csv=gb_fixture["class_map_csv"],
            technique_family_csv=gb_fixture["tech_family_csv"],
            pmi_table_parquet=gb_fixture["pmi_table"],
        )
