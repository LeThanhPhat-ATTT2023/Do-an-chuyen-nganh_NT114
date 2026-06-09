#!/usr/bin/env python3
"""Controlled experiment: train GNN4ID on the EG-HGT v3 imbalanced flow distribution.

Why
---
The headline GNN4ID vs EG-HGT v3 comparison is unfair: GNN4ID's preprocessor caps
each majority class at ~16k flows AND its nfstream extraction produces 100-350x MORE
flows for volumetric attacks (DDoS-ICMP_Flood: GNN4ID 16,225 vs v3 46; Mirai: 16,225
vs 133). So GNN4ID wins macro-F1 mostly because it *has the samples* v3 lacks.

This script removes that advantage: it subsamples GNN4ID's existing graph shards to
match v3's EXACT per-class flow counts (read from outputs/v3/graph.npz), then runs the
*identical* GNN4ID training recipe. If GNN4ID's macro-F1 collapses toward v3's level
(esp. ICMP_Flood/Mirai/ICMP_Fragmentation F1 cratering), the gap is data scarcity from
the flow-extraction unit, NOT model quality.

No re-extraction needed — reuses the shards already built under --graphs.

Usage (server, GPU recommended; CPU works but slow):
    python baselines/gnn4id/train_imbalanced.py \
        --graphs baselines/gnn4id/outputs/outputs/graphs.manifest.json \
        --v3-graph outputs/v3/graph.npz \
        --out-dir baselines/gnn4id/outputs/outputs \
        --epochs 50

Smoke test (fast, caps every class small to verify the pipeline runs):
    python baselines/gnn4id/train_imbalanced.py --smoke
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model import HeteroGNN_Edge                       # noqa: E402
from train import _split_graphs, _class_weights, _eval, SEED  # noqa: E402

_LOG = logging.getLogger("gnn4id.train_imbalanced")


def _v3_target_counts(v3_graph: str, label_mapping: dict[str, int]) -> dict[int, int]:
    """Per-class flow counts of the v3 artifact, keyed by GNN4ID label index.

    v3 stores integer flow labels in graph.npz['flow_y']; the class *name* order
    comes from the sibling graph.meta.json['label_mapping']. We translate v3 names
    → GNN4ID indices so the two label spaces line up even if index order differs.
    """
    z = np.load(v3_graph, allow_pickle=True)
    y = z["flow_y"]
    meta = json.loads((Path(v3_graph).with_suffix("").parent / "graph.meta.json").read_text())
    v3_inv = {v: k for k, v in meta["label_mapping"].items()}  # v3 idx -> name
    name_counts = Counter(v3_inv[int(i)] for i in y.tolist())
    out: dict[int, int] = {}
    for name, cnt in name_counts.items():
        if name not in label_mapping:
            _LOG.warning("v3 class %r absent from GNN4ID label_mapping — skipped", name)
            continue
        out[label_mapping[name]] = int(cnt)
    return out


def _load_subsampled(manifest_path: str, target: dict[int, int], seed: int) -> tuple[list, dict]:
    """Load shards one at a time, keeping at most target[class] graphs per class.

    Bounds peak RAM to one shard (+ the kept subset) instead of the whole dataset.
    """
    p = Path(manifest_path)
    if p.is_dir():
        p = sorted(p.glob("*.manifest.json"))[0]
    manifest = json.loads(p.read_text())
    label_mapping = manifest["label_mapping"]
    rng = np.random.default_rng(seed)

    kept_by_class: dict[int, list] = defaultdict(list)
    for rel in manifest["shards"]:
        shard = torch.load(str(p.parent / rel), map_location="cpu", weights_only=False)
        # group this shard's graphs by class (shards are per-class, but be safe)
        by_cls: dict[int, list] = defaultdict(list)
        for g in shard:
            by_cls[int(g["flow"].y.item())].append(g)
        for cls, graphs in by_cls.items():
            cap = target.get(cls)
            already = len(kept_by_class[cls])
            room = (cap - already) if cap is not None else len(graphs)
            if room <= 0:
                continue
            if len(graphs) <= room:
                kept_by_class[cls].extend(graphs)
            else:
                idx = rng.choice(len(graphs), size=room, replace=False)
                kept_by_class[cls].extend(graphs[i] for i in idx)
        del shard

    graphs = [g for cls in kept_by_class for g in kept_by_class[cls]]
    rng.shuffle(graphs)
    return graphs, label_mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", default="baselines/gnn4id/outputs/outputs/graphs.manifest.json")
    ap.add_argument("--v3-graph", default="outputs/v3/graph.npz",
                    help="v3 artifact whose per-class flow distribution GNN4ID must match.")
    ap.add_argument("--out-dir", default="baselines/gnn4id/outputs/outputs")
    ap.add_argument("--results-name", default="results_imbalanced_v3dist.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="Fast sanity run: cap every class at --smoke-cap, 2 epochs.")
    ap.add_argument("--smoke-cap", type=int, default=300)
    ap.add_argument("--cap", type=int, default=0,
                    help="Upper-cap each class at min(v3_count, cap). 0 = no cap (exact v3 "
                         "distribution). Use e.g. 2000 to shrink majority classes for a fast "
                         "foreground run while KEEPING scarce classes at their v3 counts.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")
    from torch_geometric.loader import DataLoader  # local import (heavy)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Target distribution = min(GNN4ID available, v3 count) per class ──────
    manifest = json.loads(Path(args.graphs).read_text())
    label_mapping = manifest["label_mapping"]
    if args.smoke:
        target = {i: args.smoke_cap for i in label_mapping.values()}
        epochs = 2
    else:
        target = _v3_target_counts(args.v3_graph, label_mapping)
        if args.cap > 0:
            target = {c: min(n, args.cap) for c, n in target.items()}
        epochs = args.epochs

    inv = {v: k for k, v in label_mapping.items()}
    _LOG.info("Target per-class counts (v3 distribution):")
    for cls in sorted(target):
        _LOG.info("  %-28s %d", inv[cls], target[cls])

    # ── Load shards subsampled to that distribution ─────────────────────────
    _LOG.info("Loading + subsampling shards from %s ...", args.graphs)
    graphs, label_mapping = _load_subsampled(args.graphs, target, SEED)
    num_classes = len(label_mapping)
    realized = Counter(int(g["flow"].y.item()) for g in graphs)
    _LOG.info("Loaded %d graphs across %d classes (realized counts):", len(graphs), num_classes)
    for cls in sorted(realized):
        _LOG.info("  %-28s %d", inv[cls], realized[cls])

    # ── Split / normalize / train (identical recipe to train.py) ────────────
    train_idx, val_idx, test_idx = _split_graphs(graphs, label_mapping)
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    _LOG.info("Split: train=%d val=%d test=%d", len(train_graphs), len(val_graphs), len(test_graphs))

    flow_feats = torch.cat([g["flow"].x for g in train_graphs], dim=0)
    flow_mean = flow_feats.mean(dim=0)
    flow_std = flow_feats.std(dim=0).clamp(min=1e-6)

    def _norm(gs):
        for g in gs:
            g["flow"].x = (g["flow"].x - flow_mean) / flow_std
    _norm(train_graphs); _norm(val_graphs); _norm(test_graphs)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    model = HeteroGNN_Edge(graphs[0].metadata(), hidden_channels=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, threshold=0.01, min_lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(train_graphs, num_classes).to(device))

    best_val_f1, patience_count = 0.0, 0
    best_ckpt = str(out_dir / "checkpoint_imbalanced.pt")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.batch_dict)
            loss = criterion(out, batch["flow"].y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_f1, _, _ = _eval(model, val_loader, device)
        scheduler.step(val_f1)
        _LOG.info("Epoch %d | loss=%.4f | val_macro_f1=%.4f | lr=%.2e", epoch,
                  total_loss / max(1, len(train_loader)), val_f1, optimizer.param_groups[0]["lr"])
        if val_f1 > best_val_f1:
            best_val_f1, patience_count = val_f1, 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                _LOG.info("Early stopping at epoch %d", epoch)
                break

    # ── Test ────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    test_macro_f1, test_pred, test_true = _eval(model, test_loader, device)
    report = classification_report(test_true, test_pred,
                                   target_names=[inv[i] for i in range(num_classes)],
                                   output_dict=True, zero_division=0)
    results = {
        "split": "random",
        "experiment": "gnn4id_on_v3_imbalanced_distribution",
        "macro_f1": round(float(test_macro_f1), 4),
        "weighted_f1": round(float(f1_score(test_true, test_pred, average="weighted", zero_division=0)), 4),
        "accuracy": round(float((test_pred == test_true).mean()), 4),
        "per_class_f1": {inv[i]: round(report[inv[i]]["f1-score"], 4) for i in range(num_classes)},
        "per_class_support": {inv[i]: int(report[inv[i]]["support"]) for i in range(num_classes)},
        "target_counts": {inv[c]: target.get(c, 0) for c in sorted(target)},
        "label_mapping": label_mapping,
    }
    (out_dir / args.results_name).write_text(json.dumps(results, indent=2))
    _LOG.info("Saved → %s | test macro-F1 = %.4f", out_dir / args.results_name, test_macro_f1)


if __name__ == "__main__":
    main()
