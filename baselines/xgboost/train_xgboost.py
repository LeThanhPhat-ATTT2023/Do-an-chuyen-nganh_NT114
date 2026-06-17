#!/usr/bin/env python3
"""XGBoost flow-classifier baseline on the v3_ob graph (tabular control).

A non-graph, non-deep baseline: XGBoost over the 79 CICFlowMeter flow features
(``flow_x``) only — no graph structure, no packet bytes. This answers the
reviewer's standing question "is the heterogeneous graph actually necessary, or
do tabular flow stats suffice?" and, graded on the clean answer key, tests
whether even a classic gradient-boosted-tree baseline memorizes the per-pcap
label noise the way the graph models do.

Controlled comparison (apples-to-apples with HGT/GNN4ID/de-inflated):
  * same artifact   — flow_x + flow_y from outputs/v3_ob/graph.npz
  * same split      — splits.json["random"], seed 42, mapped via flow_id_order
  * same grading    — noisy (per-pcap) AND clean (signature-isolated answer key),
                      plus the pooled binary web-attack detection metric.

Model config follows the published CICIoT2023 XGBoost baseline
(Anis 2024, github.com/FarihaAnis/...-XGBoost-using-CICIoT2023-Dataset):
multi:softprob, reg_alpha=0.5, reg_lambda=0, random_state=42, per-class balanced
sample weights. Tree models are scale-invariant so the upstream RobustScaler is
omitted (it cannot change tree splits).

Usage (metis):
    PYTHONPATH=src ~/venv/bin/python baselines/xgboost/train_xgboost.py \
        --graph-npz outputs/v3_ob/graph.npz \
        --graph-meta outputs/v3_ob/graph.meta.json \
        --splits-json outputs/v3_ob/splits.json \
        --clean-labels outputs/v3_ob/clean_eval_labels.npy \
        --out baselines/xgboost/results_v3_ob.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

WEB_CLASSES = ("CommandInjection", "XSS", "SqlInjection", "Uploading_Attack")


def _idx_from_splits(splits_json: Path, flow_id_order: list[str], protocol: str
                     ) -> dict[str, np.ndarray]:
    sp = json.loads(splits_json.read_text(encoding="utf-8"))[protocol]
    pos = {fid: i for i, fid in enumerate(flow_id_order)}
    out = {}
    for part in ("train", "val", "test"):
        out[part] = np.array([pos[f] for f in sp[part]], dtype=np.int64)
    return out


def _per_class_table(y_true, y_pred, num_classes, inv):
    out = {}
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[inv[c]] = {"precision": prec, "recall": rec, "f1": f1,
                       "support": int((y_true == c).sum())}
    return out


def _macro_present(y_true, y_pred, num_classes):
    """Macro-F1 over classes with support>0 (trainer's definition)."""
    f1s, sup = [], []
    for c in range(num_classes):
        s = int((y_true == c).sum())
        sup.append(s)
        if s == 0:
            f1s.append(0.0)
            continue
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    f1s, sup = np.array(f1s), np.array(sup)
    return float(f1s[sup > 0].mean()) if (sup > 0).any() else 0.0


def _web_binary(y_true, y_pred, web_ids):
    tb = np.isin(y_true, web_ids)
    pb = np.isin(y_pred, web_ids)
    tp = int((tb & pb).sum()); fp = int((~tb & pb).sum()); fn = int((tb & ~pb).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
            "support": int(tb.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-npz", type=Path, default=Path("outputs/v3_ob/graph.npz"))
    ap.add_argument("--graph-meta", type=Path, default=Path("outputs/v3_ob/graph.meta.json"))
    ap.add_argument("--splits-json", type=Path, default=Path("outputs/v3_ob/splits.json"))
    ap.add_argument("--clean-labels", type=Path, default=Path("outputs/v3_ob/clean_eval_labels.npy"))
    ap.add_argument("--protocol", default="random", choices=["random", "temporal"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import xgboost as xgb

    z = np.load(args.graph_npz)
    X = np.asarray(z["flow_x"], dtype=np.float32)
    y = np.asarray(z["flow_y"], dtype=np.int64)
    meta = json.loads(args.graph_meta.read_text(encoding="utf-8"))
    label_mapping = meta["label_mapping"]
    inv = {v: k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    flow_id_order = meta["flow_id_order"]
    idx = _idx_from_splits(args.splits_json, flow_id_order, args.protocol)
    web_ids = np.array(sorted(label_mapping[c] for c in WEB_CLASSES if c in label_mapping))

    Xtr, ytr = X[idx["train"]], y[idx["train"]]
    Xte, yte = X[idx["test"]], y[idx["test"]]
    print(f"[xgb] train={len(ytr)} test={len(yte)} feats={X.shape[1]} classes={num_classes}",
          flush=True)

    # Per-class balanced sample weights (Anis 2024 multiclass recipe).
    sw = compute_sample_weight("balanced", ytr)
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        reg_alpha=0.5, reg_lambda=0,          # Anis 2024 regularization
        tree_method="hist",
        device=args.device,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    clf.fit(Xtr, ytr, sample_weight=sw)
    pred = clf.predict(Xte).astype(np.int64)

    # ── Noisy grading (per-pcap labels) ───────────────────────────────────────
    noisy_macro = float(f1_score(yte, pred, average="macro", labels=list(range(num_classes)),
                                 zero_division=0))
    noisy_acc = float((pred == yte).mean())

    # ── Clean grading (signature-isolated answer key) ─────────────────────────
    clean_all = np.asarray(np.load(args.clean_labels), dtype=np.int64)
    yte_clean = clean_all[idx["test"]]
    clean_macro = _macro_present(yte_clean, pred, num_classes)
    clean_acc = float((pred == yte_clean).mean())

    results = {
        "experiment": "xgboost_flow_features_v3_ob (Anis2024 config, our pipeline)",
        "split": args.protocol,
        "model": "XGBClassifier multi:softprob reg_alpha=0.5 reg_lambda=0",
        "n_features": int(X.shape[1]),
        "noisy_macro_f1": round(noisy_macro, 4),
        "noisy_accuracy": round(noisy_acc, 4),
        "clean_macro_f1": round(clean_macro, 4),
        "clean_accuracy": round(clean_acc, 4),
        "noisy_minus_clean": round(noisy_macro - clean_macro, 4),
        "web_binary_clean": _web_binary(yte_clean, pred, web_ids),
        "per_class_noisy": _per_class_table(yte, pred, num_classes, inv),
        "per_class_clean": _per_class_table(yte_clean, pred, num_classes, inv),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"[xgb] noisy macro-F1={noisy_macro:.4f}  clean macro-F1={clean_macro:.4f}  "
          f"web-bin recall={results['web_binary_clean']['recall']:.4f} "
          f"prec={results['web_binary_clean']['precision']:.4f}", flush=True)
    print(f"[xgb] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
