#!/usr/bin/env python3
"""GNN4ID training: graphs.pt → checkpoint.pt + results.json.

Usage:
    python baselines/gnn4id/train.py \
        --graphs baselines/gnn4id/outputs/graphs.pt \
        --out-dir baselines/gnn4id/outputs
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model import HeteroGNN_Edge

_LOG = logging.getLogger("gnn4id.train")

CLASS_WEIGHT_CAP = 10.0
SEED = 42


def _split_graphs(graphs, label_mapping):
    labels = [g["flow"].y.item() for g in graphs]
    idx = list(range(len(graphs)))
    train_idx, tmp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=SEED
    )
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, stratify=tmp_labels, random_state=SEED
    )
    return train_idx, val_idx, test_idx


def _class_weights(train_graphs, num_classes: int) -> torch.Tensor:
    ys = np.array([g["flow"].y.item() for g in train_graphs])
    classes = np.arange(num_classes)
    w = compute_class_weight("balanced", classes=classes, y=ys)
    w = np.clip(w, None, CLASS_WEIGHT_CAP)
    return torch.tensor(w, dtype=torch.float)


def _eval(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x_dict, batch.edge_index_dict,
                batch.edge_attr_dict, batch.batch_dict,
            )
            pred = out.argmax(dim=1).cpu().numpy()
            true = batch["flow"].y.cpu().numpy()
            all_pred.append(pred)
            all_true.append(true)
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return macro_f1, all_pred, all_true


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", default="baselines/gnn4id/outputs/graphs.pt")
    ap.add_argument("--out-dir", default="baselines/gnn4id/outputs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load ───────────────────────────────────────────────────────────────
    _LOG.info("Loading graphs from %s ...", args.graphs)
    saved = torch.load(args.graphs, map_location="cpu")
    graphs: list = saved["graphs"]
    label_mapping: dict[str, int] = saved["label_mapping"]
    num_classes = len(label_mapping)
    _LOG.info("  %d graphs, %d classes", len(graphs), num_classes)

    # ── Split ──────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = _split_graphs(graphs, label_mapping)
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    _LOG.info("Split: train=%d val=%d test=%d", len(train_graphs), len(val_graphs), len(test_graphs))

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size)

    # ── Model ──────────────────────────────────────────────────────────────
    metadata = graphs[0].metadata()
    model = HeteroGNN_Edge(metadata, hidden_channels=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_w = _class_weights(train_graphs, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_w)

    # ── Train ──────────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    patience_count = 0
    best_ckpt = str(out_dir / "checkpoint.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(
                batch.x_dict, batch.edge_index_dict,
                batch.edge_attr_dict, batch.batch_dict,
            )
            loss = criterion(out, batch["flow"].y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_f1, _, _ = _eval(model, val_loader, device)
        _LOG.info("Epoch %d | loss=%.4f | val_macro_f1=%.4f", epoch,
                  total_loss / len(train_loader), val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                _LOG.info("Early stopping at epoch %d", epoch)
                break

    # ── Evaluate on test set ───────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_macro_f1, test_pred, test_true = _eval(model, test_loader, device)
    idx_to_label = {v: k for k, v in label_mapping.items()}
    report = classification_report(
        test_true, test_pred,
        target_names=[idx_to_label[i] for i in range(num_classes)],
        output_dict=True, zero_division=0,
    )

    weighted_f1 = f1_score(test_true, test_pred, average="weighted", zero_division=0)
    accuracy    = float((test_pred == test_true).mean())

    per_class_f1 = {
        idx_to_label[i]: round(report[idx_to_label[i]]["f1-score"], 4)
        for i in range(num_classes)
    }

    results = {
        "split": "random",
        "macro_f1":    round(float(test_macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "accuracy":    round(float(accuracy), 4),
        "per_class_f1": per_class_f1,
        "label_mapping": label_mapping,
    }

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    _LOG.info("Results saved to %s", results_path)
    _LOG.info("Test macro F1 = %.4f", test_macro_f1)


if __name__ == "__main__":
    main()
