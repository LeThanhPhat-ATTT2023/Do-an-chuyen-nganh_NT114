"""Unit tests for v2 flow assembly + features."""
from __future__ import annotations

import pandas as pd

from graphslm_ids.offline.preprocessing.flows import (
    assign_flows,
    build_flow_features,
)

_COLS = [
    "ts",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "proto",
    "flags",
    "ip_len",
    "payload_len",
    "label",
]


def _toy_packets() -> pd.DataFrame:
    """5 packets: 3 TCP (one bidirectional flow), 1 UDP, 1 ICMP -> 3 flows."""
    rows = [
        (0.000, "10.0.0.1", "10.0.0.2", 5000, 80, 0, 0x02, 60, 0, "X"),  # SYN fwd
        (0.001, "10.0.0.2", "10.0.0.1", 80, 5000, 0, 0x12, 60, 0, "X"),  # SYN+ACK bwd
        (0.002, "10.0.0.1", "10.0.0.2", 5000, 80, 0, 0x18, 200, 140, "X"),  # PSH+ACK fwd
        (0.003, "10.0.0.1", "8.8.8.8", 33333, 53, 1, 0, 50, 30, "X"),  # UDP
        (0.004, "10.0.0.1", "10.0.0.3", -1, -1, 2, 0, 40, 4, "X"),  # ICMP
    ]
    return pd.DataFrame(rows, columns=_COLS)


def test_bidirectional_grouping_three_flows() -> None:
    tagged = assign_flows(_toy_packets())
    # 3 TCP rows share one flow_id; UDP and ICMP each own one.
    assert tagged["flow_id"].nunique() == 3
    # In the TCP flow: first packet is fwd; reply is bwd; third (same direction
    # as the first) is fwd again.
    tcp = (
        tagged[tagged["proto"] == 0]
        .sort_values("ts")
        .reset_index(drop=True)
    )
    assert tcp.loc[0, "is_fwd"] and not tcp.loc[1, "is_fwd"] and tcp.loc[2, "is_fwd"]


def test_flow_features_carry_flag_counts_and_fanout() -> None:
    tagged = assign_flows(_toy_packets())
    feats, meta = build_flow_features(tagged)
    assert feats.shape[0] == 3
    # Required columns from the spec must exist.
    must_have = {
        "tot_pkts",
        "fwd_pkts",
        "bwd_pkts",
        "tot_bytes",
        "fwd_bytes",
        "bwd_bytes",
        "dur",
        "bytes_per_s",
        "pkts_per_s",
        "down_up_ratio",
        "mean_pkt_size",
        "n_syn",
        "r_syn",
        "n_ack",
        "n_psh",
        "fan_dst_ip",
        "fan_dst_port",
        "fan_syn",
        "fan_pkts",
        "iat_mean",
        "plen_mean",
        "plen_pay_mean",
        "mean_ip_len_fwd_minus_bwd",
    }
    missing = must_have.difference(feats.columns)
    assert not missing, f"missing feature columns: {missing}"

    # TCP flow: 1 plain SYN + 1 SYN+ACK -> 2 SYN, 2 ACK, 1 PSH.
    tcp_flow = feats[feats["proto"] == 0].iloc[0]
    assert tcp_flow["n_syn"] == 2
    assert tcp_flow["n_ack"] == 2
    assert tcp_flow["n_psh"] == 1
    # UDP and ICMP rows must have zero flag counts (they don't carry TCP flags).
    udp_flow = feats[feats["proto"] == 1].iloc[0]
    assert udp_flow["n_syn"] == 0 and udp_flow["n_ack"] == 0

    # Fan-out: 10.0.0.1 reaches 3 distinct dst-ips in this batch
    # (10.0.0.2, 8.8.8.8, 10.0.0.3).
    src_a_flows = feats[
        feats.index.str.contains("10.0.0.1", regex=False)
    ]
    assert (src_a_flows["fan_dst_ip"] == 3).any()

    assert meta["n_flows"] == 3
    assert "feature_names" in meta and len(meta["feature_names"]) >= 60


def test_empty_input_returns_empty_meta() -> None:
    empty = pd.DataFrame(columns=_COLS)
    feats, meta = build_flow_features(assign_flows(empty))
    assert feats.empty
    assert meta["n_flows"] == 0
