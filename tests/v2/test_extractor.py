"""Unit tests for the v2 PCAP extractor.

The whole point of v2 is that we DO NOT drop control packets or ICMP — these
tests pin that contract so a regression can't silently sneak the v1 bug back in.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from graphslm_ids.offline.preprocessing.v2.extractor import (
    COLUMNS,
    extract_packets,
    extract_packets_dir,
)
from tests.v2._fixtures.build_tiny_pcap import build_demo_pcap


def test_extractor_keeps_every_packet_kind(tmp_path: Path) -> None:
    pcap = tmp_path / "demo.pcap"
    build_demo_pcap(pcap)
    df = extract_packets(pcap, label="Demo")
    # 5 packets in the demo pcap: 3 TCP + 1 UDP + 1 ICMP.
    assert len(df) == 5, f"expected 5 packets, got {len(df)}"
    assert set(COLUMNS) <= set(df.columns)
    # Every required protocol family survived (regression guard against v1).
    protos = set(df["proto"].tolist())
    assert {0, 1, 2} <= protos, f"missing TCP/UDP/ICMP in {protos}"
    # The TCP SYN packet must (a) have zero L4 payload AND (b) the SYN flag set.
    syn_rows = df[(df["proto"] == 0) & (df["payload_len"] == 0)]
    assert len(syn_rows) >= 1, "TCP control packet was dropped"
    assert (syn_rows["flags"].astype(int) & 0x02).any(), "SYN bit lost"
    # The TCP PSH+ACK packet must still carry payload bytes.
    assert (df["payload_len"] > 0).any(), "no TCP packet with payload kept"
    # ip_len > 0 always (IP header is at least 20 bytes).
    assert (df["ip_len"] >= 20).all()
    # Label propagates to every row.
    assert (df["label"] == "Demo").all()


def test_extract_packets_dir_concats_class_folders(tmp_path: Path) -> None:
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    build_demo_pcap(tmp_path / "A" / "a.pcap")
    build_demo_pcap(tmp_path / "B" / "b.pcap")
    df = extract_packets_dir(tmp_path)
    assert set(df["label"]) == {"A", "B"}
    assert len(df) == 10  # 5 packets per pcap, 2 pcaps


def test_extractor_handles_missing_or_unreadable_pcap(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pcap"
    bad.write_bytes(b"not a pcap at all")
    df = extract_packets(bad, label="X")
    # Must return an empty frame with the right schema, not raise.
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == COLUMNS
    assert len(df) == 0
