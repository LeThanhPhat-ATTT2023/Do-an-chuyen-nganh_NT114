"""Post-hoc per-class threshold (additive-bias) calibration for the HGT classifier.

Why
---
The FINAL HGT run reaches test macro-F1 ~0.857. The shortfall is concentrated in a
single failure mode visible in the confusion matrix: a few minority classes are
**over-predicted** (high recall, low precision) and eat their neighbours — the
classic precision-sink that aggressive class re-weighting (cb_beta + focal +
weight cap) produces. The web-content cluster (CmdInj/XSS/Upload) is a genuine
information ceiling (encrypted payload + label noise — see
docs/reports/2026-06-06-web-attack-encryption-ceiling.md) and is NOT the target
here; the recoverable points are the calibration sinks (XSS, SqlInjection,
VulnerabilityScan, DDoS-ICMP_Fragmentation).

What
----
This is a *decision-rule* calibration, NOT a relabel and NOT a retrain: it learns
a per-class additive logit bias `b` (length C) that maximises macro-F1, then
predicts `argmax(logits + b)`. It is the multiclass generalisation of per-class
threshold tuning to maximise F1 (Lipton et al. 2014) and a learned-per-class
cousin of logit adjustment (Menon et al. ICLR 2021).

Honest-evaluation contract (no test peeking)
--------------------------------------------
The bias is tuned on the VALIDATION split only and then applied unchanged to the
TEST split. Validation and test never mix. The script reports BOTH the raw and the
calibrated macro-F1 on test so the lift is auditable.

Fair comparison
---------------
The same calibration protocol can be applied to any model that can emit per-flow
logits (e.g. the GNN4ID baseline), so the comparison stays apples-to-apples — the
calibration tier is part of *our method*, applied identically to both sides when a
2x2 (raw/calibrated x HGT/GNN4ID) is desired.

Usage (server, venv python):
    /home/ubuntu/venv/bin/python scripts/eval/calibrate_thresholds.py \
        --config configs/eg_hgt_v6_ob_focal.yaml \
        --checkpoint outputs/v3_ob_focal/hgt_flow_best.pt \
        --training-summary outputs/v3_ob_focal/training_summary.json \
        --out outputs/v3_ob_focal/confusion_calibrated.json
"""
from __future__ import annotations

import argparse
import json
import sys
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

# Reuse the model-rebuild helper from the confusion-analysis tool (same checkpoint
# format, same strict-load safety check) and the pure calibration core.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_confusion_analysis import _build_model_from_checkpoint  # noqa: E402
import eval_reporting as er  # noqa: E402


def _dump_logits(
    *,
    model,
    loader,
    edge_types,
    device,
    use_semantic_edge_weights: bool,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over ``loader`` and return ``(logits, labels)``.

    Mirrors the single-device branch of ``evaluate_neighbor_sampling`` exactly:
    builds each batch with ``to_torch_batch``, forwards through the model, and
    selects the seed rows — so the RAW argmax of these logits reproduces the
    trainer's reported per-class F1 (verified by the self-check in main()).
    """
    model.eval()
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            node_features, edge_index, edge_weight, seed_mask, seed_labels = T.to_torch_batch(
                batch, edge_types, device, use_semantic_edge_weights,
            )
            logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
            seed_logits = logits[seed_mask].detach().float().cpu().numpy()
            logits_chunks.append(seed_logits)
            label_chunks.append(seed_labels.detach().cpu().numpy())
    return (
        np.concatenate(logits_chunks, axis=0),
        np.concatenate(label_chunks, axis=0).astype(np.int64),
    )


def _per_class_table(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, label_names: dict[int, str]
) -> dict[str, dict[str, float]]:
    cm = er.confusion_matrix(y_true, y_pred, num_classes)
    out: dict[str, dict[str, float]] = {}
    for c in range(num_classes):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[label_names[c]] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": int(cm[c, :].sum()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--training-summary", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-seed-flows", type=int, default=384)
    ap.add_argument("--cal-seed", type=int, default=42,
                    help="seed for the coordinate-ascent class order (deterministic)")
    ap.add_argument("--cal-rounds", type=int, default=12)
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

    train_idx, val_idx, test_idx = T.backend_splits(backend, labels_np, config, seed=42)
    print(f"[cal] num_classes={num_classes} val={val_idx.shape[0]} test={test_idx.shape[0]}",
          flush=True)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, num_classes = _build_model_from_checkpoint(state, backend, device)

    sampler_cfg = config.get("sampler") or {}
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
    use_sew = bool(data_cfg.get("use_semantic_edge_weights", True))
    edge_types = list(backend.edge_types)

    val_loader, _ = T.make_neighbor_loader(
        val_idx, sampler, eval_cfg, shuffle=False, world_size=1, rank=0, seed=42
    )
    test_loader, _ = T.make_neighbor_loader(
        test_idx, sampler, eval_cfg, shuffle=False, world_size=1, rank=0, seed=42
    )

    print("[cal] dumping VAL logits ...", flush=True)
    val_logits, val_labels = _dump_logits(
        model=model, loader=val_loader, edge_types=edge_types, device=device,
        use_semantic_edge_weights=use_sew, num_classes=num_classes,
    )
    print("[cal] dumping TEST logits ...", flush=True)
    test_logits, test_labels = _dump_logits(
        model=model, loader=test_loader, edge_types=edge_types, device=device,
        use_semantic_edge_weights=use_sew, num_classes=num_classes,
    )

    # ── Self-check: RAW test argmax reproduces the trainer's headline macro-F1 ──
    raw_test_pred = er.apply_bias(test_logits, np.zeros(num_classes))
    raw_f1, raw_sup = er.per_class_f1_from_labels(test_labels, raw_test_pred, num_classes)
    raw_test_macro = er._macro_f1(raw_f1, raw_sup)
    check = None
    if args.training_summary and args.training_summary.exists():
        summ_macro = json.loads(args.training_summary.read_text())["best_test_metrics"]["macro_f1"]
        check = {"raw_test_macro_here": raw_test_macro, "summary_macro": summ_macro,
                 "abs_diff": abs(raw_test_macro - summ_macro)}
        print(f"[cal] self-check raw test macro={raw_test_macro:.4f} "
              f"(summary {summ_macro:.4f}, delta {check['abs_diff']:.4f})", flush=True)

    # ── Tune the per-class bias on VALIDATION, apply to TEST (no peeking) ───────
    bias, cal_info = er.calibrate_bias_for_macro_f1(
        val_logits, val_labels, num_classes,
        n_rounds=int(args.cal_rounds), seed=int(args.cal_seed),
    )
    val_raw = cal_info["macro_f1_raw"]
    val_cal = cal_info["macro_f1_calibrated"]
    print(f"[cal] VAL macro-F1: raw={val_raw:.4f} -> calibrated={val_cal:.4f} "
          f"(delta {val_cal - val_raw:+.4f}, rounds={cal_info['rounds_run']})", flush=True)

    cal_test_pred = er.apply_bias(test_logits, bias)
    cal_f1, cal_sup = er.per_class_f1_from_labels(test_labels, cal_test_pred, num_classes)
    cal_test_macro = er._macro_f1(cal_f1, cal_sup)
    print(f"[cal] TEST macro-F1: raw={raw_test_macro:.4f} -> calibrated={cal_test_macro:.4f} "
          f"(delta {cal_test_macro - raw_test_macro:+.4f})  <-- HONEST (bias tuned on val)",
          flush=True)

    out = {
        "method": "per_class_additive_logit_bias (coordinate-ascent, macro-F1, tuned-on-val)",
        "num_classes": num_classes,
        "n_val": int(val_labels.shape[0]),
        "n_test": int(test_labels.shape[0]),
        "bias_per_class": {label_names[c]: float(bias[c]) for c in range(num_classes)},
        "macro_f1": {
            "val_raw": float(val_raw),
            "val_calibrated": float(val_cal),
            "test_raw": float(raw_test_macro),
            "test_calibrated": float(cal_test_macro),
            "test_lift": float(cal_test_macro - raw_test_macro),
        },
        "per_class_raw": _per_class_table(test_labels, raw_test_pred, num_classes, label_names),
        "per_class_calibrated": _per_class_table(test_labels, cal_test_pred, num_classes, label_names),
        "calibration_info": cal_info,
        "self_check_vs_summary": check,
        "label_names": {str(k): v for k, v in label_names.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[cal] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
