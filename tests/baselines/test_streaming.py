"""Tests for the OOM-hardening of GNN4ID preprocessing:

  * ``_can_admit`` byte-budget admission rule for the extraction scheduler,
  * ``stream_graphs_from_csv`` parity with the legacy ``NIDSDataset``,
  * ``write_graph_shards`` / ``load_graphs`` round-trip (sharded + legacy .pt).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "baselines" / "gnn4id"))

import pandas as pd
import torch

import preprocess
from preprocess import _can_admit
from utils.functions import (
    NIDSDataset,
    ShardWriter,
    stream_graphs_from_csv,
    write_graph_shards,
    load_graphs,
    MANIFEST_FORMAT,
)


def _make_synthetic_csv(path: str, label: str, n_rows: int = 3) -> None:
    n_pkts = 5
    rows = []
    for i in range(n_rows):
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


# ── _can_admit truth table ───────────────────────────────────────────────────

def test_can_admit_respects_worker_cap():
    # At the worker cap, nothing more is admitted even with budget to spare.
    assert _can_admit(inflight_bytes=0, inflight_count=2, task_bytes=1,
                      budget_bytes=10**9, max_workers=2) is False


def test_can_admit_always_allows_first_task():
    # A single task larger than the whole budget must still be admittable.
    assert _can_admit(inflight_bytes=0, inflight_count=0, task_bytes=10**12,
                      budget_bytes=2000, max_workers=4) is True


def test_can_admit_within_budget():
    assert _can_admit(inflight_bytes=500, inflight_count=1, task_bytes=400,
                      budget_bytes=1000, max_workers=4) is True


def test_can_admit_rejects_over_budget():
    assert _can_admit(inflight_bytes=900, inflight_count=1, task_bytes=400,
                      budget_bytes=1000, max_workers=4) is False


def test_scheduler_invariant_peak_bounded():
    """Simulate the admission loop with _can_admit as oracle; peak concurrent
    bytes must never exceed max(budget, largest single task), count <= workers."""
    sizes = [1953, 1953, 979, 308, 217, 191, 33, 27, 12, 10, 9, 7, 3]  # MB-ish
    budget, workers = 2000, 4
    pending = sorted(sizes, reverse=True)
    running: list[int] = []
    inflight = 0
    peak_bytes = 0
    peak_count = 0
    # Deterministic schedule: admit what fits, then "complete" the smallest task.
    while pending or running:
        progressed = True
        while progressed and pending:
            progressed = False
            for idx in range(len(pending)):
                if _can_admit(inflight, len(running), pending[idx], budget, workers):
                    running.append(pending.pop(idx))
                    inflight += running[-1]
                    progressed = True
                    break
        peak_bytes = max(peak_bytes, inflight)
        peak_count = max(peak_count, len(running))
        if running:
            done = running.pop(0)
            inflight -= done
    assert peak_count <= workers
    assert peak_bytes <= max(budget, max(sizes))


# ── streaming parity + shard round-trip ──────────────────────────────────────

def test_stream_matches_nidsdataset():
    lm = {"DDoS": 0}
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "ddos.csv")
        _make_synthetic_csv(csv, "DDoS", n_rows=4)
        streamed = list(stream_graphs_from_csv(csv, 0))
        legacy = NIDSDataset([(csv, "DDoS")], lm)
    assert len(streamed) == len(legacy) == 4
    g = streamed[0]
    assert g["flow"].y.item() == 0
    assert g["flow", "contains", "packet"].edge_index.shape[1] == 5


def test_write_and_load_shards_roundtrip():
    lm = {"BruteForce": 0, "DDoS": 1}
    with tempfile.TemporaryDirectory() as d:
        c1 = os.path.join(d, "bf.csv")
        c2 = os.path.join(d, "ddos.csv")
        _make_synthetic_csv(c1, "BruteForce", n_rows=5)
        _make_synthetic_csv(c2, "DDoS", n_rows=3)
        manifest_path = os.path.join(d, "graphs.manifest.json")
        manifest = write_graph_shards(
            csv_files=[(c1, "BruteForce"), (c2, "DDoS")],
            label_mapping=lm,
            manifest_path=manifest_path,
            max_graphs_per_shard=2,   # force multiple shards per CSV
        )
        assert manifest["format"] == MANIFEST_FORMAT
        assert manifest["num_graphs"] == 8
        # 5 rows -> 3 shards, 3 rows -> 2 shards
        assert len(manifest["shards"]) == 5
        assert Path(manifest_path).exists()

        graphs, lm_loaded = load_graphs(manifest_path)
    assert lm_loaded == lm
    assert len(graphs) == 8
    labels = sorted(g["flow"].y.item() for g in graphs)
    assert labels == [0] * 5 + [1] * 3


def test_load_graphs_legacy_pt():
    lm = {"DDoS": 0}
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "ddos.csv")
        _make_synthetic_csv(csv, "DDoS", n_rows=2)
        graphs = list(stream_graphs_from_csv(csv, 0))
        legacy_pt = os.path.join(d, "graphs.pt")
        torch.save({"graphs": graphs, "label_mapping": lm}, legacy_pt)
        loaded, lm_loaded = load_graphs(legacy_pt)
    assert lm_loaded == lm
    assert len(loaded) == 2


# ── uint8 packet features + model compatibility ──────────────────────────────

def test_packet_x_is_uint8():
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "ddos.csv")
        _make_synthetic_csv(csv, "DDoS", n_rows=2)
        g = next(stream_graphs_from_csv(csv, 0))
    assert g["packet"].x.dtype == torch.uint8
    assert g["packet"].x.shape[1] == 1500


def test_model_forward_accepts_uint8():
    """End-to-end: uint8 packet.x must flow through the model (cast → float)."""
    from torch_geometric.loader import DataLoader
    from model import HeteroGNN_Edge

    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "ddos.csv")
        _make_synthetic_csv(csv, "DDoS", n_rows=4)
        graphs = list(stream_graphs_from_csv(csv, 0))
    assert graphs[0]["packet"].x.dtype == torch.uint8
    batch = next(iter(DataLoader(graphs, batch_size=2)))
    model = HeteroGNN_Edge(graphs[0].metadata(), hidden_channels=16, num_classes=2)
    model.eval()
    out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.batch_dict)
    assert out.shape == (2, 2)
    assert not torch.isnan(out).any()


# ── subsampling + ShardWriter resume / per-class cap ─────────────────────────

def test_subsample_caps_and_is_deterministic():
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "ddos.csv")
        _make_synthetic_csv(csv, "DDoS", n_rows=50)
        full = list(stream_graphs_from_csv(csv, 0))
        sub_a = list(stream_graphs_from_csv(csv, 0, max_flows=10, seed=42))
        sub_b = list(stream_graphs_from_csv(csv, 0, max_flows=10, seed=42))
    assert len(full) == 50
    assert len(sub_a) == 10
    # same seed → same selection (deterministic)
    assert len(sub_b) == 10
    # n_total <= max_flows → keep all
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "small.csv")
        _make_synthetic_csv(csv, "DDoS", n_rows=3)
        assert len(list(stream_graphs_from_csv(csv, 0, max_flows=10))) == 3


def test_shardwriter_resume_skips_built():
    lm = {"DDoS": 0}
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "ddos.csv")
        _make_synthetic_csv(csv, "DDoS", n_rows=4)
        manifest = os.path.join(d, "g.manifest.json")
        w1 = ShardWriter(manifest, lm)
        added1 = w1.add_csv(csv, "DDoS")
        w1.finalize()
        # fresh writer reloads the manifest and must skip the already-built csv
        w2 = ShardWriter(manifest, lm)
        assert w2.already_built(csv, "DDoS")
        added2 = w2.add_csv(csv, "DDoS")
    assert added1 == 4
    assert added2 == 0


def test_shardwriter_per_class_cap_across_csvs():
    lm = {"DDoS": 0}
    with tempfile.TemporaryDirectory() as d:
        c1 = os.path.join(d, "ddos1.csv")
        c2 = os.path.join(d, "ddos2.csv")
        _make_synthetic_csv(c1, "DDoS", n_rows=8)
        _make_synthetic_csv(c2, "DDoS", n_rows=8)
        manifest = os.path.join(d, "g.manifest.json")
        w = ShardWriter(manifest, lm, max_flows_per_class=10, seed=42)
        a1 = w.add_csv(c1, "DDoS")
        a2 = w.add_csv(c2, "DDoS")
        m = w.finalize()
    assert a1 == 8          # first csv fits under the cap of 10
    assert a2 == 2          # second csv contributes only the remaining 2
    assert m["per_class"]["DDoS"] == 10
    assert m["num_graphs"] == 10
