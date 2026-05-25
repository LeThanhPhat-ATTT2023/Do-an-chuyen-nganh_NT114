"""Tests for the v2 evidence-edge builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.v2.evidence_edges import (
    build_evidence_edges,
)


def test_scan_flow_emits_t1046_edge_and_sqli_packet_emits_t1190() -> None:
    flow_features = pd.DataFrame(
        [
            # f0: classic port scan (should hit T1046 via flow:syn_fanout)
            {
                "proto": 0,
                "fan_dst_port": 1200,
                "fan_dst_ip": 1,
                "r_syn": 0.95,
                "plen_pay_mean": 0,
                "pkts_per_s": 5,
                "r_rst": 0,
                "r_psh": 0,
                "tot_pkts": 1200,
                "dur": 60.0,
                "fwd_bytes": 60_000,
                "bwd_bytes": 0,
            },
            # f1: ordinary HTTP-ish flow, no flow signature fires
            {
                "proto": 0,
                "fan_dst_port": 2,
                "fan_dst_ip": 1,
                "r_syn": 0.1,
                "plen_pay_mean": 100,
                "pkts_per_s": 5,
                "r_rst": 0,
                "r_psh": 0.4,
                "tot_pkts": 20,
                "dur": 4.0,
                "fwd_bytes": 2_000,
                "bwd_bytes": 2_000,
            },
        ],
        index=pd.Index(["f0", "f1"], name="flow_id"),
    )

    # one packet (row 0) belongs to f1 and carries a SQLi payload
    sqli = b"GET /?id=1' OR 1=1-- HTTP/1.1"
    payload_matrix = np.zeros((1, 64), dtype=np.uint8)
    payload_matrix[0, : len(sqli)] = list(sqli)
    packet_payload_idx = {
        "f0": np.array([], dtype=np.int64),
        "f1": np.array([0], dtype=np.int64),
    }
    tech_idx = {"T1046": 100, "T1190": 200, "T1018": 300}

    out = build_evidence_edges(flow_features, payload_matrix, packet_payload_idx, tech_idx)

    # f0 (scan) -> T1046 (mapped to integer 100)
    ft = out["flow_technique_edge_index"]
    ft_attr = out["flow_technique_edge_attr"]
    assert ft.shape[0] == 2, "edge_index must be (2, E)"
    assert (ft[1] == 100).any(), "expected a T1046 edge from the scan flow"
    # ...and the source must be the scan flow (row 0).
    scan_edge_mask = ft[1] == 100
    assert (ft[0][scan_edge_mask] == 0).any()
    assert (ft_attr > 0).all() and (ft_attr <= 1).all()

    # SQLi packet (row 0) -> T1190 (mapped to integer 200)
    pt = out["packet_technique_edge_index"]
    assert pt.shape[0] == 2
    assert (pt[1] == 200).any(), "expected a T1190 edge from the SQLi packet"


def test_unknown_technique_dropped_silently() -> None:
    flow_features = pd.DataFrame(
        [
            {
                "proto": 0,
                "fan_dst_port": 1200,
                "fan_dst_ip": 1,
                "r_syn": 0.95,
                "plen_pay_mean": 0,
                "pkts_per_s": 5,
                "r_rst": 0,
                "r_psh": 0,
                "tot_pkts": 1200,
                "dur": 60.0,
                "fwd_bytes": 60_000,
                "bwd_bytes": 0,
            }
        ],
        index=pd.Index(["f0"], name="flow_id"),
    )
    # T1046 deliberately omitted -> evidence-edge builder must skip it.
    tech_idx = {"T1190": 200}
    out = build_evidence_edges(
        flow_features,
        np.zeros((0, 16), dtype=np.uint8),
        {"f0": np.array([], dtype=np.int64)},
        tech_idx,
    )
    assert out["flow_technique_edge_index"].shape[1] == 0
    assert out["packet_technique_edge_index"].shape[1] == 0


def test_weights_dedupe_by_summing_then_clip() -> None:
    flow_features = pd.DataFrame(
        [
            # This row triggers BOTH flow:syn_fanout (T1046) and flow:host_fanout (T1018)
            {
                "proto": 0,
                "fan_dst_port": 1200,
                "fan_dst_ip": 500,
                "r_syn": 0.95,
                "plen_pay_mean": 0,
                "pkts_per_s": 5,
                "r_rst": 0,
                "r_psh": 0,
                "tot_pkts": 1200,
                "dur": 60.0,
                "fwd_bytes": 60_000,
                "bwd_bytes": 0,
            }
        ],
        index=pd.Index(["f0"], name="flow_id"),
    )
    tech_idx = {"T1046": 0, "T1018": 1}
    out = build_evidence_edges(
        flow_features,
        np.zeros((0, 16), dtype=np.uint8),
        {"f0": np.array([], dtype=np.int64)},
        tech_idx,
    )
    ft_attr = out["flow_technique_edge_attr"]
    assert (ft_attr <= 1.0).all(), "weights must be clipped to 1.0"
    # two distinct (src, dst) pairs were added (one per technique)
    assert out["flow_technique_edge_index"].shape[1] == 2
