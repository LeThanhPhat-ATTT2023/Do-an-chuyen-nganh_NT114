"""End-to-end verification of the v2 ceiling result.

The 0.9768 macro-F1 from `v2_ceiling_test.py` is *suspiciously* clean, so this
script runs three independent sanity checks before we trust it:

  1. Label-permutation control. If we shuffle the labels uniformly at random
     and the model can still fit them, there's a hidden leak in the features.
     A clean pipeline should collapse to chance accuracy (~1/13 = 0.077).

  2. Label-free fan-out re-build. The v2 flow features group fan-out by
     ``(label, src_ip)``, which uses the ground-truth label as a partitioning
     key. We re-compute the same features but with the groupby keyed on
     ``src_ip`` alone, then re-train and compare macro-F1. If the label-keyed
     and label-free numbers are close, the label partition is just a
     convenience that approximates the deployment-time per-window grouping,
     not a leak.

  3. Feature importance ranking. We print the HistGBM's top-15 feature
     importances. If any single feature looks like a label proxy (e.g., a
     value that is trivially unique per class), it shows up here.

This is read-only on the dataset. It does NOT alter any artefact on disk; it
just prints + writes a JSON summary to ``outputs/v2/v2_verify_results.json``.
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


def _train_and_score(
    X: np.ndarray, y: np.ndarray, seed: int, label_names: list[str]
) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    sw = compute_sample_weight("balanced", ytr)
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=seed,
    )
    t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sw)
    fit_s = time.time() - t0
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
    per_class = {name: round(rep[name]["f1-score"], 4) for name in label_names}
    cm = confusion_matrix(yte, pred, labels=list(range(len(label_names))))
    return {
        "macro_f1": macro,
        "accuracy": acc,
        "per_class_f1": per_class,
        "confusion": cm.tolist(),
        "fit_s": fit_s,
        "clf": clf,
    }


def _build_features_label_free_fanout(df_packets: pd.DataFrame) -> pd.DataFrame:
    """Recompute v2 features with fan-out grouped on src_ip only (no label).

    Mirrors :func:`build_flow_features` but partitions fan-out so the feature
    value does not depend on the ground-truth label.
    """
    tagged = assign_flows(df_packets)
    feats, _ = build_flow_features(tagged)

    # Recompute fan-out without using label as a grouping key.
    sub = tagged.copy()
    # Pre-compute syn flag inside this function to mirror the main builder.
    _SYN = 0x02
    sub["syn"] = ((sub["flags"].astype(int) & _SYN) > 0) & (sub["proto"] == 0)
    fan = (
        sub.groupby("src_ip", sort=False)
        .agg(
            fan_dst_ip=("dst_ip", "nunique"),
            fan_dst_port=("dst_port", "nunique"),
            fan_syn=("syn", "sum"),
            fan_pkts=("ts", "count"),
        )
        .reset_index()
    )
    flow_to_src = tagged.groupby("flow_id", sort=False)["src_ip"].first().rename(
        "_first_src_ip"
    )
    feats = feats.join(flow_to_src)
    feats = feats.drop(columns=["fan_dst_ip", "fan_dst_port", "fan_syn", "fan_pkts"])
    # Preserve the flow_id index across the merge: pandas .merge loses index,
    # so we reset, merge, then set_index back. Without this the row order
    # diverges from feats_labeled and the verify script crashes its alignment
    # assertion.
    feats = (
        feats.reset_index()
        .merge(fan, left_on="_first_src_ip", right_on="src_ip", how="left")
        .drop(columns=["_first_src_ip", "src_ip"])
        .set_index("flow_id")
    )
    for col in ("fan_dst_ip", "fan_dst_port", "fan_syn", "fan_pkts"):
        feats[col] = feats[col].fillna(0.0).astype(np.float64)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument("--out-json", default="outputs/v2/v2_verify_results.json")
    ap.add_argument("--max-per-class-packets", type=int, default=500_000)
    ap.add_argument("--max-per-class-flows", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[verify] parsing pcaps under {args.raw_root} ...", flush=True)
    cap = None if args.max_per_class_packets == 0 else args.max_per_class_packets
    packets_df = extract_packets_dir(Path(args.raw_root), max_per_class=cap)
    print(
        f"[verify] parsed {len(packets_df):,} packets across "
        f"{packets_df['label'].nunique()} classes",
        flush=True,
    )

    print("[verify] building label-keyed features (v2 default) ...", flush=True)
    tagged = assign_flows(packets_df)
    feats_labeled, _ = build_flow_features(tagged)

    print(
        "[verify] building label-FREE features (fan-out by src_ip only) ...",
        flush=True,
    )
    feats_labelfree = _build_features_label_free_fanout(packets_df)
    # Re-align to the label-keyed row order so the subsample indexes match.
    feats_labelfree = feats_labelfree.reindex(feats_labeled.index)
    assert list(feats_labeled.index) == list(feats_labelfree.index)

    # Encode labels once.
    y_cat = feats_labeled["label"].astype("category")
    label_names = list(y_cat.cat.categories)
    y_all = y_cat.cat.codes.to_numpy(dtype=np.int64)
    print(
        f"[verify] flows = {len(feats_labeled):,}  classes = {len(label_names)}",
        flush=True,
    )

    # Stratified subsample for speed -- same protocol as v2_ceiling_test.
    sel = _stratified_subsample(y_all, args.max_per_class_flows, args.seed)
    y_sub = y_all[sel]
    Xa = feats_labeled.drop(columns=["label"]).to_numpy(np.float32)[sel]
    Xb = feats_labelfree.drop(columns=["label"]).to_numpy(np.float32)[sel]
    feat_names_a = [c for c in feats_labeled.columns if c != "label"]
    print(
        f"[verify] training on {sel.shape[0]:,} flows  "
        f"(<= {args.max_per_class_flows:,}/class)  features={Xa.shape[1]}",
        flush=True,
    )

    # === A. Real labels, label-keyed features (replicates v2 ceiling) ========
    print("\n[A] training with REAL labels (label-keyed fan-out) ...", flush=True)
    A = _train_and_score(Xa, y_sub, args.seed, label_names)
    print(f"  acc={A['accuracy']:.4f}  macro_f1={A['macro_f1']:.4f}")

    # === B. Real labels, label-FREE fan-out features =========================
    print(
        "\n[B] training with REAL labels (label-FREE fan-out) ...",
        flush=True,
    )
    B = _train_and_score(Xb, y_sub, args.seed, label_names)
    print(f"  acc={B['accuracy']:.4f}  macro_f1={B['macro_f1']:.4f}")

    # === C. SHUFFLED labels (permutation control) ============================
    # If the features hide a leak, even a shuffled-label model fits well. A
    # clean pipeline should land near 1/num_classes accuracy.
    rng = np.random.default_rng(args.seed)
    y_shuffled = y_sub.copy()
    rng.shuffle(y_shuffled)
    print(
        "\n[C] training with SHUFFLED labels (permutation control) ...",
        flush=True,
    )
    C = _train_and_score(Xa, y_shuffled, args.seed, label_names)
    chance = 1.0 / len(label_names)
    print(
        f"  acc={C['accuracy']:.4f}  macro_f1={C['macro_f1']:.4f}   "
        f"chance ~ {chance:.4f}",
    )

    # === Feature importances (from A's classifier) ===========================
    print("\n[D] HGBM feature importances (top 20, label-keyed features):")
    clf_A = A["clf"]
    imps = getattr(clf_A, "feature_importances_", None)
    if imps is None:
        # HistGradientBoostingClassifier doesn't expose feature_importances_,
        # so we fall back to permutation_importance on the test split.
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        _, Xte, _, yte = train_test_split(
            Xa, y_sub, test_size=0.2, stratify=y_sub, random_state=args.seed
        )
        pi = permutation_importance(
            clf_A, Xte, yte, n_repeats=3, random_state=args.seed, n_jobs=1
        )
        imps = pi.importances_mean
    top_idx = np.argsort(-imps)[:20]
    top_imports = [(feat_names_a[i], float(imps[i])) for i in top_idx]
    for name, val in top_imports:
        print(f"  {name:30s} {val:.4f}")

    # === Summary =============================================================
    leak_ok = C["macro_f1"] < 0.25  # very loose threshold; chance ~0.077
    free_vs_keyed_delta = A["macro_f1"] - B["macro_f1"]
    print("\n[summary]")
    print(f"  A label-keyed:   macro_f1 = {A['macro_f1']:.4f}")
    print(f"  B label-free:    macro_f1 = {B['macro_f1']:.4f}   (delta = {free_vs_keyed_delta:+.4f})")
    print(f"  C shuffled ctrl: macro_f1 = {C['macro_f1']:.4f}   (chance = {chance:.4f})")
    if leak_ok:
        print("  -> permutation control LOOKS CLEAN (model cannot fit shuffled labels)")
    else:
        print("  -> WARNING: model fit shuffled labels above chance -- investigate!")
    if abs(free_vs_keyed_delta) < 0.03:
        print("  -> label-free vs label-keyed fan-out: within 3% -- label partition is NOT the source of the lift")
    else:
        print(f"  -> label-free vs label-keyed differ by {free_vs_keyed_delta:.4f}; non-trivial")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "n_flows_total": int(len(feats_labeled)),
                "n_flows_trained": int(sel.shape[0]),
                "n_features": int(Xa.shape[1]),
                "label_names": label_names,
                "A_label_keyed": {
                    "macro_f1": A["macro_f1"],
                    "accuracy": A["accuracy"],
                    "per_class_f1": A["per_class_f1"],
                    "confusion": A["confusion"],
                },
                "B_label_free": {
                    "macro_f1": B["macro_f1"],
                    "accuracy": B["accuracy"],
                    "per_class_f1": B["per_class_f1"],
                    "confusion": B["confusion"],
                },
                "C_shuffled_control": {
                    "macro_f1": C["macro_f1"],
                    "accuracy": C["accuracy"],
                    "per_class_f1": C["per_class_f1"],
                },
                "top20_feature_importances": [
                    {"feature": n, "importance": v} for n, v in top_imports
                ],
                "leak_control_passed": bool(leak_ok),
                "free_vs_keyed_delta": float(free_vs_keyed_delta),
                "wall_s": time.time() - t0,
            },
            indent=2,
        )
    )
    print(f"\n[verify] wrote {out}  wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
