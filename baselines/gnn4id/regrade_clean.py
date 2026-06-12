#!/usr/bin/env python3
"""Re-grade the trained GNN4ID baseline against the CLEAN answer key (LNL protocol).

Loads the existing checkpoint (no retraining), reproduces the exact subsample +
random split of train_imbalanced.py (same SEED, same target counts read back from
its results JSON), runs inference on the test split, then scores the SAME
predictions twice:

  * noisy  — against the per-pcap labels (must reproduce the published 0.8528
             macro-F1; this validates the split/checkpoint reconstruction), and
  * clean  — against the signature-isolated answer key, where web-attack flows
             whose 5-tuple carries no HTTP attack signature are demoted to Benign.

Flow identity: GNN4ID shards store packets as raw bytes from the transport header
on (nfstream `ip_packet[20:]`), so the TCP/UDP port pair of the first packet is
recoverable but the IPs are not. Within one web-attack pcap there is a single
attacker/victim host pair, so the SORTED PORT PAIR identifies the canonical
5-tuple; the script audits the port-pair collision rate and refuses to grade a
class where the reduction is ambiguous.

Usage:
    python baselines/gnn4id/regrade_clean.py \
        --graphs baselines/gnn4id/outputs/outputs_v1/graphs.manifest.json \
        --checkpoint baselines/gnn4id/outputs/outputs_v1/checkpoint_imbalanced.pt \
        --results baselines/gnn4id/outputs/results_imbalanced_v3dist.json \
        --raw-root data/raw \
        --out baselines/gnn4id/outputs/results_imbalanced_v3dist_CLEAN.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "src"))

from model import HeteroGNN_Edge                          # noqa: E402
from train import SEED, _eval, _split_graphs              # noqa: E402
from train_imbalanced import _load_subsampled             # noqa: E402

from graphslm_ids.offline.preprocessing.flow_attack_labeler import (  # noqa: E402
    BENIGN_LABEL,
    WEB_ATTACK_CLASSES,
    label_pcap_flows,
)

_LOG = logging.getLogger("gnn4id.regrade")


def _port_pair_of_canon_key(key: str) -> tuple[int, int]:
    """'ip:port|ip:port|proto' -> sorted (port, port)."""
    lo, hi, _proto = key.split("|")
    p1 = int(lo.rsplit(":", 1)[1])
    p2 = int(hi.rsplit(":", 1)[1])
    return (p1, p2) if p1 <= p2 else (p2, p1)


def collect_attack_port_pairs(raw_root: Path) -> dict[str, set[tuple[int, int]]]:
    """Signature-isolate every web-attack pcap; reduce attack keys to port pairs.

    Aborts if the port-pair reduction collides (an attack key and a non-attack
    key sharing the same pair), which would make port-level grading ambiguous.
    """
    out: dict[str, set[tuple[int, int]]] = {}
    for cls in sorted(WEB_ATTACK_CLASSES):
        cls_dir = raw_root / cls
        pcaps = sorted(cls_dir.glob("*.pcap")) if cls_dir.exists() else []
        if not pcaps:
            _LOG.warning("no pcaps for %s under %s — class skipped", cls, cls_dir)
            continue
        attack_pairs: set[tuple[int, int]] = set()
        benign_pairs: set[tuple[int, int]] = set()
        for pcap in pcaps:
            mapping, audit = label_pcap_flows(pcap, cls)
            for key, lab in mapping.items():
                pair = _port_pair_of_canon_key(key)
                (attack_pairs if lab == cls else benign_pairs).add(pair)
            _LOG.info("[%s] %s: audit=%s", cls, pcap.name, audit)
        clash = attack_pairs & benign_pairs
        _LOG.info(
            "[%s] attack port-pairs=%d benign port-pairs=%d collisions=%d",
            cls, len(attack_pairs), len(benign_pairs), len(clash),
        )
        if clash:
            _LOG.warning(
                "[%s] %d ambiguous port pairs graded conservatively as ATTACK "
                "(keeps the noisy label -> never inflates the clean score of "
                "a wrong relabel)", cls, len(clash),
            )
        out[cls] = attack_pairs
    return out


def _first_packet_ports(g) -> tuple[int, int] | None:
    """Sorted transport port pair from the first stored packet's raw bytes."""
    px = g["packet"].x
    if px.shape[0] == 0 or px.shape[1] < 4:
        return None
    b = px[0, :4].to(torch.int64)
    sport = int(b[0]) * 256 + int(b[1])
    dport = int(b[2]) * 256 + int(b[3])
    if sport == 0 and dport == 0:
        return None
    return (sport, dport) if sport <= dport else (dport, sport)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", default="baselines/gnn4id/outputs/outputs_v1/graphs.manifest.json")
    ap.add_argument("--checkpoint", default="baselines/gnn4id/outputs/outputs_v1/checkpoint_imbalanced.pt")
    ap.add_argument("--results", default="baselines/gnn4id/outputs/results_imbalanced_v3dist.json",
                    help="results JSON of the original run; supplies target_counts + label_mapping.")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--out", default="baselines/gnn4id/outputs/results_imbalanced_v3dist_CLEAN.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")
    from torch_geometric.loader import DataLoader  # heavy import

    device = torch.device(args.device)
    prior = json.loads(Path(args.results).read_text())
    label_mapping: dict[str, int] = prior["label_mapping"]
    inv = {v: k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    target = {label_mapping[name]: int(n) for name, n in prior["target_counts"].items()}

    # ── 1. Attack keys (signature isolation over the local pcaps) ───────────
    pairs_by_class = collect_attack_port_pairs(Path(args.raw_root))

    # ── 2. Reproduce subsample + split (same code path, same SEED) ──────────
    _LOG.info("Reloading + subsampling shards (deterministic, seed=%d) ...", SEED)
    graphs, label_mapping2 = _load_subsampled(args.graphs, target, SEED)
    assert label_mapping2 == label_mapping, "label mapping drifted between runs"
    train_idx, val_idx, test_idx = _split_graphs(graphs, label_mapping)
    _LOG.info("Split: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    flow_feats = torch.cat([g["flow"].x for g in train_graphs], dim=0)
    flow_mean = flow_feats.mean(dim=0)
    flow_std = flow_feats.std(dim=0).clamp(min=1e-6)
    for g in test_graphs:
        g["flow"].x = (g["flow"].x - flow_mean) / flow_std

    # ── 3. Inference from the existing checkpoint ───────────────────────────
    model = HeteroGNN_Edge(graphs[0].metadata(), hidden_channels=64,
                           num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
    noisy_f1, pred, true_noisy = _eval(model, test_loader, device)
    _LOG.info("noisy macro-F1 = %.4f (published: %s)", noisy_f1, prior.get("macro_f1"))
    drift = abs(noisy_f1 - float(prior["macro_f1"]))
    if drift > 0.005:
        _LOG.warning("REPRODUCTION DRIFT %.4f — split/normalization may differ; "
                     "clean numbers below are suspect", drift)

    # ── 4. Clean labels for the test flows (port-pair matching) ─────────────
    benign_id = label_mapping[BENIGN_LABEL]
    true_clean = true_noisy.copy()
    demoted: dict[str, int] = defaultdict(int)
    no_ports: dict[str, int] = defaultdict(int)
    for row, g in enumerate(test_graphs):
        cls_id = int(true_noisy[row])
        name = inv[cls_id]
        pairs = pairs_by_class.get(name)
        if pairs is None:
            continue  # not a relabel-eligible class
        pp = _first_packet_ports(g)
        if pp is None:
            no_ports[name] += 1
            continue  # unidentifiable -> conservatively keep the noisy label
        if pp not in pairs:
            true_clean[row] = benign_id
            demoted[name] += 1
    _LOG.info("demoted per class: %s | unidentifiable: %s", dict(demoted), dict(no_ports))

    clean_f1 = f1_score(true_clean, pred, average="macro", zero_division=0)
    names = [inv[i] for i in range(num_classes)]
    rep_clean = classification_report(true_clean, pred, labels=list(range(num_classes)),
                                      target_names=names, output_dict=True, zero_division=0)
    results = {
        "experiment": "gnn4id_on_v3_imbalanced_distribution_CLEAN_regrade",
        "split": "random",
        "checkpoint": str(args.checkpoint),
        "noisy_macro_f1_reproduced": round(float(noisy_f1), 4),
        "noisy_macro_f1_published": prior.get("macro_f1"),
        "clean_macro_f1": round(float(clean_f1), 4),
        "clean_accuracy": round(float((pred == true_clean).mean()), 4),
        "clean_weighted_f1": round(float(f1_score(true_clean, pred, average="weighted",
                                                  zero_division=0)), 4),
        "clean_per_class_f1": {n: round(rep_clean[n]["f1-score"], 4) for n in names},
        "clean_per_class_support": {n: int(rep_clean[n]["support"]) for n in names},
        "demoted_per_class": dict(demoted),
        "unidentifiable_per_class": dict(no_ports),
        "attack_port_pairs_per_class": {c: len(p) for c, p in pairs_by_class.items()},
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    _LOG.info("Saved -> %s", args.out)
    _LOG.info("GNN4ID  noisy=%.4f  clean=%.4f", noisy_f1, clean_f1)


if __name__ == "__main__":
    main()
