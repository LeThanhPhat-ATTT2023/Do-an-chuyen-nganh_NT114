"""Unit A — v4 confusion-matrix + per-class error-flow analysis (EG-HGT v5 Phase 0).

Why
---
v4 reached test macro-F1 0.865 but five classes drag it: Benign (recall 0.63),
DDoS-ICMP_Fragmentation (0.45), Backdoor_Malware (0.57, regressed from v3's 0.87),
Recon-PingSweep (0.70), SqlInjection (0.80). The training_summary has per-class P/R/F1
but NOT the confusion matrix, so we cannot see WHERE each class's errors flow. This script
dumps the full confusion matrix + the top error destinations for the draggers, so v5
Phases 1-3 target real failure modes instead of guesses.

How
---
Reuses the EXACT production inference path (`evaluate_neighbor_sampling`) so there is zero
metric drift. That function builds `pred_np`/`label_np` internally and passes them to
`metrics_from_predictions`; we monkeypatch that one function to capture the raw arrays,
then rebuild the confusion matrix from them. As a self-check, the per-class F1 recomputed
from the confusion matrix is diffed against training_summary.json — they must match.

Usage (server, venv python):
    /home/ubuntu/venv/bin/python scripts/diagnostics/v4_confusion_analysis.py \
        --config configs/eg_hgt_v4.yaml \
        --checkpoint outputs/v3/hgt_v4/hgt_flow_best.pt \
        --training-summary outputs/v3/hgt_v4/training_summary.json \
        --out outputs/v3/hgt_v4/v4_confusion.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import graphslm_ids.offline.training.train_hgt_flow_classifier as T
from graphslm_ids.offline.training.hetero_graph_artifact import load_v3_artifact
from graphslm_ids.offline.training.neighbor_sampling import (
    HeteroNeighborSampler,
    InMemoryNeighborBackend,
)
from graphslm_ids.models.hgt import HeteroGraphTransformer

DRAGGERS = [
    "Benign",
    "DDoS-ICMP_Fragmentation",
    "Backdoor_Malware",
    "Recon-PingSweep",
    "SqlInjection",
]


def _build_model_from_checkpoint(state, backend, device):
    """Rebuild the model from the fields the trainer SAVED in the checkpoint
    (node_input_dims/edge_types/num_classes/num_tactics) — NOT from the backend —
    so the architecture matches the weights exactly, then strict-load with a report."""
    cfg = state.get("config") or {}
    model_cfg = (cfg.get("model") if isinstance(cfg, dict) else {}) or {}
    node_input_dims = {nt: int(d) for nt, d in state["node_input_dims"].items()}
    edge_types = [tuple(e) for e in state["edge_types"]]
    num_classes = int(state["num_classes"])
    num_tactics = int(state["num_tactics"])
    derived_node_types = list(node_input_dims.keys()) + ["tactic"]
    model = HeteroGraphTransformer(
        node_input_dims=node_input_dims,
        edge_types=edge_types,
        num_classes=num_classes,
        num_tactics=num_tactics,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        ffn_multiplier=int(model_cfg.get("ffn_multiplier", 2)),
        activation_checkpointing=False,
        node_types=derived_node_types,
    ).to(device)
    sd = state["model_state_dict"]
    stripped = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        stripped[nk] = v
    incompat = model.load_state_dict(stripped, strict=False)
    miss, unexp = list(incompat.missing_keys), list(incompat.unexpected_keys)
    print(f"[A] load_state_dict: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    if miss:
        print(f"[A]   sample missing: {miss[:5]}", flush=True)
    if unexp:
        print(f"[A]   sample unexpected: {unexp[:5]}", flush=True)
    if miss or unexp:
        raise SystemExit("[A] ABORT: checkpoint keys do not match the rebuilt model — "
                         "weights would be random. Fix architecture before trusting metrics.")
    model.eval()
    return model, num_classes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--training-summary", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-seed-flows", type=int, default=384)
    ap.add_argument("--la-taus", default="",
                    help="comma list of logit-adjustment taus to sweep at inference; "
                         "tau<0 PENALIZES rare classes (fixes over-prediction precision sinks), "
                         "tau>0 is standard Menon (favors rare). e.g. '-1.5,-1.0,-0.5,-0.25,0.5'")
    args = ap.parse_args()

    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]

    artifact = load_v3_artifact(
        graph_npz=Path(data_cfg["graph_npz"]),
        graph_meta_json=Path(data_cfg["graph_meta_json"]),
        add_reverse_edges=bool(data_cfg.get("add_reverse_edges", True)),
    )
    backend = InMemoryNeighborBackend(artifact)
    labels_np = artifact.flow_y
    num_classes = int(labels_np.max()) + 1
    label_names = T.label_name_mapping(artifact.metadata, labels_np)

    # Same split the trainer used (reads splits.json via config, protocol=random).
    train_idx, val_idx, test_idx = T.backend_splits(backend, labels_np, config, seed=42)
    print(f"[A] num_classes={num_classes} test={test_idx.shape[0]}", flush=True)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, num_classes = _build_model_from_checkpoint(state, backend, device)

    sampler_cfg = config.get("sampler") or {}
    # Prefer the checkpoint's saved flow_feature_stats so standardization matches training.
    ckpt_stats = state.get("flow_feature_stats")
    manifest_stats = ckpt_stats if isinstance(ckpt_stats, dict) else (backend.manifest or {}).get("flow_feature_stats")
    flow_feature_stats = None
    if isinstance(manifest_stats, dict) and "mean" in manifest_stats and "std" in manifest_stats:
        flow_feature_stats = {"mean": list(manifest_stats["mean"]), "std": list(manifest_stats["std"])}

    sampler = HeteroNeighborSampler(
        backend,
        hops=int(sampler_cfg.get("hops", 4)),
        fanouts=dict(sampler_cfg.get("fanouts") or {}),
        reverse_fanouts=dict(sampler_cfg.get("reverse_fanouts") or {}),
        always_include_all_tactics=bool(sampler_cfg.get("always_include_all_tactics", True)),
        always_include_all_techniques=bool(sampler_cfg.get("always_include_all_techniques", True)),
        flow_feature_stats=flow_feature_stats,
        standardize_flow_features=bool(data_cfg.get("standardize_flow_features", True)),
        seed=42,
    )
    eval_cfg = {
        "train": {"batch_seed_flows": int(args.batch_seed_flows), "seed": 42},
        "dataloader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
    }
    test_loader, _ = T.make_neighbor_loader(
        test_idx, sampler, eval_cfg, shuffle=False, world_size=1, rank=0, seed=42
    )

    # ── Capture raw predictions by wrapping metrics_from_predictions ──────────
    captured: dict[str, np.ndarray] = {}
    _orig = T.metrics_from_predictions

    def _capture(pred_np, label_np, *a, **k):
        captured["pred"] = np.asarray(pred_np)
        captured["label"] = np.asarray(label_np)
        return _orig(pred_np, label_np, *a, **k)

    T.metrics_from_predictions = _capture
    try:
        T.evaluate_neighbor_sampling(
            model=model,
            loader=test_loader,
            edge_types=list(backend.edge_types),
            device=device,
            use_amp=False,
            use_semantic_edge_weights=bool(data_cfg.get("use_semantic_edge_weights", True)),
            num_classes=num_classes,
            label_names=label_names,
            epoch=0,
            split_name="test_random",
            is_ddp=False,
        )
    finally:
        T.metrics_from_predictions = _orig

    pred = captured["pred"]
    true = captured["label"]
    assert pred.shape == true.shape and pred.size > 0, "no predictions captured"

    # ── Confusion matrix (rows=true, cols=pred) ──────────────────────────────
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true.tolist(), pred.tolist()):
        cm[t, p] += 1

    # ── Self-check: per-class F1 from CM vs training_summary ─────────────────
    per_class = {}
    for c in range(num_classes):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[label_names[c]] = {"precision": prec, "recall": rec, "f1": f1,
                                      "support": int(cm[c, :].sum())}
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    acc = float(np.trace(cm) / cm.sum())

    check = None
    if args.training_summary and args.training_summary.exists():
        ts = json.loads(args.training_summary.read_text())["best_test_metrics"]["per_class"]
        diffs = {n: abs(per_class[n]["f1"] - ts[n]["f1"]) for n in ts if n in per_class}
        check = {"max_abs_f1_diff": max(diffs.values()), "macro_f1_here": macro_f1,
                 "macro_f1_summary": json.loads(args.training_summary.read_text())["best_test_metrics"]["macro_f1"]}

    # ── Error-flow for the draggers: where do their misclassifications go? ────
    error_flow = {}
    name_to_idx = {n: i for i, n in label_names.items()}
    for name in DRAGGERS:
        if name not in name_to_idx:
            continue
        c = name_to_idx[name]
        row = cm[c].copy()
        total = int(row.sum())
        row[c] = 0  # zero out correct → only errors
        order = np.argsort(row)[::-1]
        dests = [{"to": label_names[int(j)], "count": int(row[j]),
                  "pct_of_class": round(100 * row[j] / total, 1)}
                 for j in order[:4] if row[j] > 0]
        error_flow[name] = {"support": total, "correct": int(cm[c, c]),
                            "recall": per_class[name]["recall"],
                            "precision": per_class[name]["precision"],
                            "top_error_destinations": dests}

    out = {
        "num_classes": num_classes, "n_test": int(true.size),
        "accuracy": acc, "macro_f1": macro_f1,
        "self_check_vs_training_summary": check,
        "per_class": per_class,
        "dragger_error_flow": error_flow,
        "label_names": {str(k): v for k, v in label_names.items()},
        "confusion_matrix": cm.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # ── Human-readable summary ───────────────────────────────────────────────
    print(f"\n[A] accuracy={acc:.4f} macro_f1={macro_f1:.4f}")
    if check:
        print(f"[A] self-check vs training_summary: max|ΔF1|={check['max_abs_f1_diff']:.4f} "
              f"(summary macro={check['macro_f1_summary']:.4f}) "
              f"{'OK' if check['max_abs_f1_diff'] < 0.02 else 'MISMATCH!'}")
    print("\n[A] DRAGGER ERROR FLOW (where misclassifications go):")
    for name, info in error_flow.items():
        print(f"  {name:26s} sup={info['support']:5d} R={info['recall']:.2f} P={info['precision']:.2f}")
        for d in info["top_error_destinations"]:
            print(f"      → {d['to']:26s} {d['count']:5d}  ({d['pct_of_class']}%)")
    print(f"\n[A] wrote {args.out}", flush=True)

    # ── Optional: logit-adjustment τ sweep, VAL-tuned then reported on TEST ────
    if args.la_taus.strip():
        taus = [0.0] + [float(t) for t in args.la_taus.split(",") if t.strip() and float(t) != 0.0]
        train_labels = labels_np[train_idx]
        prior = np.maximum(
            np.bincount(train_labels, minlength=num_classes).astype(np.float64)
            / max(len(train_labels), 1), 1e-12)
        log_prior = np.log(prior)
        watch = [w for w in ["Benign", "Backdoor_Malware", "DDoS-ICMP_Fragmentation",
                             "Recon-PingSweep", "SqlInjection", "CommandInjection", "XSS",
                             "Uploading_Attack"] if w in set(label_names.values())]

        val_loader, _ = T.make_neighbor_loader(
            val_idx, sampler, eval_cfg, shuffle=False, world_size=1, rank=0, seed=42)

        def _eval_tau(loader, tau, split):
            la = None if tau == 0.0 else torch.from_numpy(
                (tau * log_prior).astype(np.float32)).to(device)
            m = T.evaluate_neighbor_sampling(
                model=model, loader=loader, edge_types=list(backend.edge_types),
                device=device, use_amp=False,
                use_semantic_edge_weights=bool(data_cfg.get("use_semantic_edge_weights", True)),
                num_classes=num_classes, label_names=label_names, epoch=0,
                split_name=split, is_ddp=False, logit_adjustment=la)
            src = m if tau == 0.0 else (m.get("logit_adjusted") or {})
            pcf = {n: float(e["f1"]) for n, e in (src.get("per_class") or {}).items()}
            return float(src.get("macro_f1") or 0.0), pcf

        def _row(tau, macro, pcf):
            return f"  {tau:+5.2f}  {macro:7.4f}   " + " ".join(f"{pcf.get(w, 0.0):5.2f}" for w in watch)

        val_res, test_res = {}, {}
        print("\n[A] τ SWEEP — VAL (selection) then TEST (report):")
        print("   tau   val_mF1  test_mF1  | TEST per-class: " + " ".join(w[:5] for w in watch))
        for tau in taus:
            vm, _ = _eval_tau(val_loader, tau, f"val{tau:+.2f}")
            tm, tpcf = _eval_tau(test_loader, tau, f"test{tau:+.2f}")
            val_res[tau], test_res[tau] = vm, (tm, tpcf)
            print(f"  {tau:+5.2f}  {vm:7.4f}  {tm:7.4f}   " + " ".join(f"{tpcf.get(w,0.0):5.2f}" for w in watch))

        best_tau = max(val_res, key=lambda t: val_res[t])
        test_at_best = test_res[best_tau][0]
        sweep_out = {"val_macro_f1": val_res,
                     "test_macro_f1": {t: v[0] for t, v in test_res.items()},
                     "best_tau_on_val": best_tau,
                     "test_macro_f1_at_best_tau": test_at_best,
                     "v4_raw_test_macro_f1": macro_f1}
        Path(str(args.out).replace(".json", "_la_sweep.json")).write_text(json.dumps(sweep_out, indent=2))
        print(f"\n[A] τ TUNED ON VAL = {best_tau:+.2f}  →  TEST macro-F1 = {test_at_best:.4f}  "
              f"(v4 raw {macro_f1:.4f}, Δ {test_at_best - macro_f1:+.4f}) — honest, no test peeking",
              flush=True)


if __name__ == "__main__":
    main()
