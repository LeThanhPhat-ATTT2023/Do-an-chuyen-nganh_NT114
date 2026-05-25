"""Full-scale tabular ceiling test on v2 features.

Validates that the v2 pipeline produces feature vectors a strong tabular model
can separate into the 13 NT114 attack classes. Compares against the v1 cosine
GNN baseline (~0.12 macro-F1) and the flag-aware tree baseline (~0.955).

Pipeline:
  1. Parse all pcaps under data/raw/14gb (v2 extractor keeps every packet)
  2. Bidirectional flow assembly + ~80-dim CICFlowMeter features
  3. Stratified subsample (cap per class) to keep runtime under control
  4. Train HistGradientBoostingClassifier; report macro-F1 + per-class F1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.v2.extractor import extract_packets_dir
from graphslm_ids.offline.preprocessing.v2.flows import (
    assign_flows,
    build_flow_features,
)


def _stratified_subsample(
    labels: np.ndarray, max_per_class: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if max_per_class > 0 and idx.shape[0] > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument(
        "--out-json", default="outputs/v2/v2_ceiling_results.json"
    )
    ap.add_argument(
        "--max-per-class-packets",
        type=int,
        default=500_000,
        help="Cap packets parsed per class (0 = unlimited)",
    )
    ap.add_argument(
        "--max-per-class-flows",
        type=int,
        default=30_000,
        help="Cap flows used to train HistGBM (0 = unlimited)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(
        f"[v2-ceiling] parsing pcaps under {args.raw_root} "
        f"(cap={args.max_per_class_packets:,} pkts/class) ...",
        flush=True,
    )
    cap = None if args.max_per_class_packets == 0 else args.max_per_class_packets
    df = extract_packets_dir(Path(args.raw_root), max_per_class=cap)
    print(
        f"[v2-ceiling] parsed packets = {len(df):,}  "
        f"proto mix = {df['proto'].value_counts().to_dict()}",
        flush=True,
    )

    print("[v2-ceiling] assembling bidirectional flows ...", flush=True)
    tagged = assign_flows(df)
    feats, meta = build_flow_features(tagged)
    print(f"[v2-ceiling] flows = {len(feats):,}  features = {len(feats.columns) - 1}",
          flush=True)

    y_all = feats["label"].astype("category")
    label_names = list(y_all.cat.categories)
    y = y_all.cat.codes.to_numpy(dtype=np.int64)
    X = feats.drop(columns=["label"]).to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Subsample for speed and to mirror the existing tabular_baseline harness.
    sel = _stratified_subsample(y, args.max_per_class_flows, args.seed)
    X_sub, y_sub = X[sel], y[sel]
    print(
        f"[v2-ceiling] training on {sel.shape[0]:,} flows "
        f"(<= {args.max_per_class_flows:,}/class)",
        flush=True,
    )

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    Xtr, Xte, ytr, yte = train_test_split(
        X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=args.seed
    )
    sw = compute_sample_weight("balanced", ytr)
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=args.seed,
    )
    fit_t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sw)
    fit_s = time.time() - fit_t0
    pred = clf.predict(Xte)
    macro = float(f1_score(yte, pred, average="macro"))
    acc = float(accuracy_score(yte, pred))
    rep = classification_report(
        yte,
        pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: round(rep[name]["f1-score"], 4) for name in label_names
    }
    print(f"\n[v2-ceiling] fit={fit_s:.1f}s  acc={acc:.4f}  macro_f1={macro:.4f}")
    print("[v2-ceiling] per-class f1 (sorted):")
    for cls, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:28s} {v:.4f}")
    print(
        f"\n[v2-ceiling] reference ceilings:\n"
        f"  GNN v8.5            macro_f1 = 0.12\n"
        f"  v1 base6 tree       macro_f1 = 0.34\n"
        f"  v1 +payload (hist)  macro_f1 = 0.47\n"
        f"  v1 flags+fanout     macro_f1 = 0.955  <-- target"
    )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "n_packets_parsed": int(len(df)),
                "n_flows_total": int(len(feats)),
                "n_flows_trained": int(sel.shape[0]),
                "n_features": int(X.shape[1]),
                "accuracy": acc,
                "macro_f1": macro,
                "per_class_f1": per_class,
                "wall_s": time.time() - t0,
                "proto_counts": {int(k): int(v) for k, v in meta.get("proto_counts", {}).items()},
            },
            indent=2,
        )
    )
    print(f"\n[v2-ceiling] wrote {out}  wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
