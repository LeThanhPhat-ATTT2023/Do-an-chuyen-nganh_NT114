"""Unit tests for NIDSDataset graph structure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "baselines" / "gnn4id"))

import ast
import pandas as pd
import torch
import tempfile
import os
from utils.functions import NIDSDataset, _safe_parse_list


def _make_synthetic_csv(path: str, label: str, n_rows: int = 3) -> None:
    """Write a minimal nfstream-style CSV for testing."""
    rows = []
    for i in range(n_rows):
        n_pkts = 5
        rows.append({
            "bidirectional_first_seen_ms": i * 1000,
            "src_ip": "192.168.1.1",
            "dst_ip": "10.0.0.1",
            "src_port": 12345,
            "dst_port": 80,
            "protocol": 6,
            "bidirectional_packets": n_pkts,
            "bidirectional_duration_ms": 100,
            "bidirectional_ack_packets": 3,
            "bidirectional_syn_packets": 1,
            "bidirectional_fin_packets": 1,
            "bidirectional_rst_packets": 0,
            "bidirectional_psh_packets": 2,
            "src2dst_packets": 3,
            "src2dst_min_ps": 40,
            "src2dst_max_ps": 1500,
            "dst2src_min_ps": 40,
            "dst2src_max_ps": 1500,
            "expiration_id": 0,
            "pkt_hex": str(["deadbeef" * 10] * n_pkts),
            "pkt_delta": str([0, 10, 5, 8, 3]),
            "pkt_dir": str([0, 1, 0, 1, 0]),
            "pkt_ip_size": str([60, 80, 60, 80, 60]),
            "pkt_transport_size": str([20, 40, 20, 40, 20]),
            "pkt_payload_size": str([0, 20, 0, 20, 0]),
            "label": label,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_dataset_builds_graphs():
    label_mapping = {"BruteForce": 0, "DDoS": 1}
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = os.path.join(tmpdir, "bruteforce.csv")
        csv2 = os.path.join(tmpdir, "ddos.csv")
        _make_synthetic_csv(csv1, "BruteForce", n_rows=3)
        _make_synthetic_csv(csv2, "DDoS", n_rows=2)
        dataset = NIDSDataset(
            csv_files=[(csv1, "BruteForce"), (csv2, "DDoS")],
            label_mapping=label_mapping,
        )
    assert len(dataset) == 5


def test_graph_node_types():
    label_mapping = {"BruteForce": 0}
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = os.path.join(tmpdir, "test.csv")
        _make_synthetic_csv(csv1, "BruteForce", n_rows=1)
        dataset = NIDSDataset([(csv1, "BruteForce")], label_mapping)
    g = dataset[0]
    assert "flow" in g.node_types
    assert "packet" in g.node_types
    assert g["flow"].x.shape[0] == 1
    assert g["flow"].y.item() == 0


def test_graph_edge_types():
    label_mapping = {"BruteForce": 0}
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = os.path.join(tmpdir, "test.csv")
        _make_synthetic_csv(csv1, "BruteForce", n_rows=1)
        dataset = NIDSDataset([(csv1, "BruteForce")], label_mapping)
    g = dataset[0]
    assert ("flow", "contains", "packet") in g.edge_types
    assert ("packet", "next_packet", "packet") in g.edge_types
    # contains: 5 packet nodes → 5 edges
    assert g["flow", "contains", "packet"].edge_index.shape[1] == 5
    # link: 5 packets → 4 link edges
    assert g["packet", "next_packet", "packet"].edge_index.shape[1] == 4


def test_safe_parse_list():
    assert _safe_parse_list("[1, 2, 3]") == [1, 2, 3]
    assert _safe_parse_list("5") == [5]
    assert _safe_parse_list("bad") is None
