"""Tier-0 fair-compare calibration: per-class bias (val-tuned) + TTA, NO retrain.

Runs the EXISTING checkpoint through evaluate_neighbor_sampling(collect_logits=True)
K times with different sampler seeds, averages softmax (TTA), tunes a per-class bias
on VAL, and reports TEST macro-F1 for: raw / +bias / +TTA / +bias+TTA. Selection is
on VAL only (honest, no test-peek). Self-checks raw macro-F1 vs the known baseline.

Server usage:
    /home/ubuntu/venv/bin/python scripts/diagnostics/tier0_calibration.py \
        --config configs/eg_hgt_v5_origlabels.yaml \
        --checkpoint outputs/v3/hgt_v5_origlabels/hgt_flow_best.pt \
        --out outputs/v3/hgt_v5_origlabels/tier0_calibration.json \
        --tta 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_confusion_analysis import _build_model_from_checkpoint  # noqa: E402

import graphslm_ids.offline.training.train_hgt_flow_classifier as T  # noqa: E402
from graphslm_ids.offline.training.hetero_graph_artifact import load_v3_artifact  # noqa: E402
from graphslm_ids.offline.training.neighbor_sampling import (  # noqa: E402
    HeteroNeighborSampler,
    InMemoryNeighborBackend,
)
from graphslm_ids.offline.training.calibration import (  # noqa: E402
    apply_bias,
    combine_tta,
    macro_f1,
    tune_per_class_bias,
)


def _dump(model, idx, sampler_cfg, data_cfg, backend, label_names, num_classes,
          flow_feature_stats, device, seed, batch_seed_flows):
    sampler = HeteroNeighborSampler(
        backend,
        hops=int(sampler_cfg.get("hops", 4)),
        fanouts=dict(sampler_cfg.get("fanouts") or {}),
        reverse_fanouts=dict(sampler_cfg.get("reverse_fanouts") or {}),
        always_include_all_tactics=bool(sampler_cfg.get("always_include_all_tactics", True)),
        always_include_all_techniques=bool(sampler_cfg.get("always_include_all_techniques", True)),
        flow_feature_stats=flow_feature_stats,
        standardize_flow_features=bool(data_cfg.get("standardize_flow_features", True)),
        seed=seed,
    )
    eval_cfg = {
        "train": {"batch_seed_flows": int(batch_seed_flows), "seed": seed},
        "dataloader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
    }
    loader, _ = T.make_neighbor_loader(idx, sampler, eval_cfg, shuffle=False,
                                       world_size=1, rank=0, seed=seed)
    m = T.evaluate_neighbor_sampling(
        model=model, loader=loader, edge_types=list(backend.edge_types), device=device,
        use_amp=False, use_semantic_edge_weights=bool(data_cfg.get("use_semantic_edge_weights", True)),
        num_classes=num_classes, label_names=label_names, epoch=0, split_name="dump",
        is_ddp=False, collect_logits=True)
    return m["_logits"], m["_labels"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tta", type=int, default=5, help="number of stochastic sampling passes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-seed-flows", type=int, default=384)
    args = ap.parse_args()

    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    artifact = load_v3_artifact(
        graph_npz=Path(data_cfg["graph_npz"]),
        graph_meta_json=Path(data_cfg["graph_meta_json"]),
        add_reverse_edges=bool(data_cfg.get("add_reverse_edges", True)))
    backend = InMemoryNeighborBackend(artifact)
    labels_np = artifact.flow_y
    num_classes = int(labels_np.max()) + 1
    label_names = T.label_name_mapping(artifact.metadata, labels_np)
    train_idx, val_idx, test_idx = T.backend_splits(backend, labels_np, config, seed=42)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, num_classes = _build_model_from_checkpoint(state, backend, device)
    ckpt_stats = state.get("flow_feature_stats")
    flow_feature_stats = None
    if isinstance(ckpt_stats, dict) and "mean" in ckpt_stats and "std" in ckpt_stats:
        flow_feature_stats = {"mean": list(ckpt_stats["mean"]), "std": list(ckpt_stats["std"])}
    sampler_cfg = config.get("sampler") or {}

    # K stochastic passes for VAL and TEST (seed 42..42+K-1). Pass 0 = canonical (seed 42).
    val_logits_k, test_logits_k = [], []
    val_labels = test_labels = None
    for k in range(max(1, args.tta)):
        vl, vy = _dump(model, val_idx, sampler_cfg, data_cfg, backend, label_names,
                       num_classes, flow_feature_stats, device, 42 + k, args.batch_seed_flows)
        tl, ty = _dump(model, test_idx, sampler_cfg, data_cfg, backend, label_names,
                       num_classes, flow_feature_stats, device, 42 + k, args.batch_seed_flows)
        val_logits_k.append(vl); test_logits_k.append(tl)
        val_labels, test_labels = vy, ty
        print(f"[T0] dumped pass {k} (seed {42 + k})", flush=True)

    raw_val, raw_test = val_logits_k[0], test_logits_k[0]            # single-pass (seed 42)
    tta_val, tta_test = combine_tta(val_logits_k), combine_tta(test_logits_k)

    def report(name, val_log, test_log, bias):
        return {
            "name": name,
            "val_macro_f1": macro_f1(apply_bias(val_log, bias), val_labels, num_classes),
            "test_macro_f1": macro_f1(apply_bias(test_log, bias), test_labels, num_classes),
        }

    zero = np.zeros(num_classes)
    bias_raw = tune_per_class_bias(raw_val, val_labels, num_classes)
    bias_tta = tune_per_class_bias(tta_val, val_labels, num_classes)
    rows = [
        report("raw", raw_val, raw_test, zero),
        report("raw+bias", raw_val, raw_test, bias_raw),
        report("tta", tta_val, tta_test, zero),
        report("tta+bias", tta_val, tta_test, bias_tta),
    ]
    best = max(rows, key=lambda r: r["val_macro_f1"])  # SELECT ON VAL
    out = {
        "n_val": int(val_labels.size), "n_test": int(test_labels.size),
        "tta_passes": int(args.tta), "num_classes": num_classes,
        "rows": rows, "selected_on_val": best["name"],
        "selected_test_macro_f1": best["test_macro_f1"],
        "baseline_raw_test_macro_f1": rows[0]["test_macro_f1"],
        "bias_raw": bias_raw.tolist(), "bias_tta": bias_tta.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("\n[T0]  variant      VAL_mF1   TEST_mF1")
    for r in rows:
        print(f"[T0]  {r['name']:10s}  {r['val_macro_f1']:.4f}   {r['test_macro_f1']:.4f}")
    print(f"\n[T0] SELECTED-ON-VAL = {best['name']}  ->  TEST macro-F1 = {best['test_macro_f1']:.4f}  "
          f"(raw {rows[0]['test_macro_f1']:.4f}, delta {best['test_macro_f1'] - rows[0]['test_macro_f1']:+.4f})")
    # Self-check: raw must reproduce the known fair-compare baseline.
    assert abs(rows[0]["test_macro_f1"] - 0.8535) < 0.01, \
        f"raw macro-F1 {rows[0]['test_macro_f1']:.4f} != known 0.8535 - pipeline drift"
    print(f"[T0] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
