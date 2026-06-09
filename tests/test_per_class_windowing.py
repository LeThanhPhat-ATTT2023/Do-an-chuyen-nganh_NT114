"""Unit tests for per-class sub-flow windowing in ``assign_flows``.

Phase 2 of the EG-HGT v5 design (``docs/superpowers/specs/
2026-06-07-v5-deterministic-representation-design.md``): volumetric flood
classes are split with a SMALLER window so a flood of many packets yields many
homogeneous sub-flows (recovering sample count), while other classes keep the
default window. The split must stay deterministic, leakage-free, and
byte-for-byte backward compatible when ``per_class_max_packets is None``.
"""
from __future__ import annotations

import math

import pandas as pd

from graphslm_ids.offline.preprocessing.flows import assign_flows

# Same packet-row schema ``assign_flows`` consumes (see tests/preprocessing/test_flows.py).
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


def _flood_segment(
    n: int,
    label: str = "DDoS-ICMP_Flood",
    *,
    src_ip: str = "10.0.0.9",
    dst_ip: str = "10.0.0.1",
    proto: int = 2,
    dt: float = 0.001,
    t0: float = 0.0,
) -> pd.DataFrame:
    """A single timeout-contiguous segment of ``n`` packets for one flow key.

    All packets share the same 5-tuple/label and are spaced ``dt`` apart (well
    under the 30 s idle timeout), so without windowing they form exactly one
    flow; with windowing they split purely on packet order.
    """
    rows = []
    for i in range(n):
        rows.append(
            (t0 + i * dt, src_ip, dst_ip, -1, -1, proto, 0, 40, 4, label)
        )
    return pd.DataFrame(rows, columns=_COLS)


def test_volumetric_class_splits_into_ceil_n_over_window() -> None:
    """A DDoS-ICMP_Flood segment of N packets with window 32 -> ceil(N/32)."""
    n = 100
    window = 32
    df = _flood_segment(n, label="DDoS-ICMP_Flood")
    tagged = assign_flows(df, per_class_max_packets={"DDoS-ICMP_Flood": window})
    assert tagged["flow_id"].nunique() == math.ceil(n / window)  # 100/32 -> 4
    # Every sub-flow except possibly the last holds exactly ``window`` packets.
    sizes = tagged.groupby("flow_id").size().tolist()
    full = [s for s in sizes if s == window]
    assert len(full) == n // window  # 3 full chunks of 32
    assert sum(sizes) == n  # no packet dropped or duplicated


def test_non_volumetric_short_flow_stays_single() -> None:
    """A short (5-packet) non-volumetric flow remains exactly one flow.

    The class is NOT in ``per_class_max_packets`` so it falls back to the
    default ``max_packets_per_flow`` (256) and 5 < 256 => 1 flow.
    """
    df = _flood_segment(5, label="XSS", proto=0, src_ip="10.0.0.2", dst_ip="10.0.0.3")
    tagged = assign_flows(df, per_class_max_packets={"DDoS-ICMP_Flood": 32})
    assert tagged["flow_id"].nunique() == 1


def test_default_class_unaffected_by_per_class_override() -> None:
    """A class absent from the dict uses the default window, not the small one."""
    # 50-packet Benign segment, override only targets DDoS-ICMP_Flood.
    df = _flood_segment(50, label="Benign", proto=0)
    tagged = assign_flows(df, per_class_max_packets={"DDoS-ICMP_Flood": 8})
    # 50 < default 256 -> single flow (the size-8 override must not apply here).
    assert tagged["flow_id"].nunique() == 1


def test_mixed_input_applies_window_per_class() -> None:
    """A volumetric class and a normal class in the SAME batch window separately."""
    flood = _flood_segment(70, label="Mirai-udpplain", proto=1, src_ip="10.0.0.5")
    web = _flood_segment(
        6, label="XSS", proto=0, src_ip="10.0.0.2", dst_ip="10.0.0.7", t0=500.0
    )
    df = pd.concat([flood, web], ignore_index=True)
    tagged = assign_flows(df, per_class_max_packets={"Mirai-udpplain": 16})

    mirai = tagged[tagged["label"] == "Mirai-udpplain"]
    xss = tagged[tagged["label"] == "XSS"]
    assert mirai["flow_id"].nunique() == math.ceil(70 / 16)  # 70/16 -> 5
    assert xss["flow_id"].nunique() == 1  # 6 < default 256


def test_determinism_identical_calls_identical_assignment() -> None:
    """Two identical calls produce identical flow_id assignments (row-aligned)."""
    flood = _flood_segment(80, label="DDoS-ICMP_Fragmentation", src_ip="10.0.0.8")
    web = _flood_segment(
        5, label="SqlInjection", proto=0, src_ip="10.0.0.4", dst_ip="10.0.0.6", t0=900.0
    )
    df = pd.concat([flood, web], ignore_index=True)
    pcm = {"DDoS-ICMP_Fragmentation": 24}

    a = assign_flows(df.copy(), per_class_max_packets=pcm)
    b = assign_flows(df.copy(), per_class_max_packets=pcm)
    # Sort by a stable key so we compare the same logical rows.
    sort_cols = ["label", "ts", "src_ip", "dst_ip"]
    a_sorted = a.sort_values(sort_cols).reset_index(drop=True)
    b_sorted = b.sort_values(sort_cols).reset_index(drop=True)
    assert a_sorted["flow_id"].tolist() == b_sorted["flow_id"].tolist()
    assert a_sorted["is_fwd"].tolist() == b_sorted["is_fwd"].tolist()


def test_backward_compat_none_matches_default_behavior() -> None:
    """``per_class_max_packets=None`` is byte-for-byte identical to the default.

    Build a mixed batch (a long flood segment that DOES window under the
    default 256 plus short flows that do not) and assert both the flow_id and
    is_fwd columns are identical with and without the (None) per-class arg.
    """
    flood = _flood_segment(600, label="DDoS-ICMP_Flood", src_ip="10.0.0.9")
    udp = _flood_segment(
        3, label="Benign", proto=1, src_ip="10.0.0.1", dst_ip="8.8.8.8", t0=700.0
    )
    web = _flood_segment(
        7, label="XSS", proto=0, src_ip="10.0.0.2", dst_ip="10.0.0.3", t0=800.0
    )
    df = pd.concat([flood, udp, web], ignore_index=True)

    default = assign_flows(df.copy())  # current default behavior
    explicit_none = assign_flows(df.copy(), per_class_max_packets=None)

    # Same row order (assign_flows sorts deterministically), so compare directly.
    assert default["flow_id"].tolist() == explicit_none["flow_id"].tolist()
    assert default["is_fwd"].tolist() == explicit_none["is_fwd"].tolist()
    # The 600-packet flood must have windowed under the default 256.
    n_flood = default[default["label"] == "DDoS-ICMP_Flood"]["flow_id"].nunique()
    assert n_flood == math.ceil(600 / 256)  # -> 3


def test_window_larger_than_segment_is_noop() -> None:
    """A per-class window >= segment length leaves the segment as one flow."""
    df = _flood_segment(10, label="DDoS-ICMP_Flood")
    tagged = assign_flows(df, per_class_max_packets={"DDoS-ICMP_Flood": 64})
    assert tagged["flow_id"].nunique() == 1
