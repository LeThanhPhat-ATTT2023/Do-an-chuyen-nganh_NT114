"""Diagnostic: dump an 18x18 confusion matrix for a trained v3 HGT checkpoint.

Reuses the production eval setup (artifact load, backend, sampler, neighbor
loader, model build) from ``v3_eval_both_splits`` but, instead of only
returning aggregate metrics, captures raw argmax predictions per seed flow and
accumulates a confusion matrix. For each "broken" class it prints where the
TRUE samples actually land (the absorbing classes), which is the decisive
signal for the Tier-2 root-cause analysis.

Usage (on the server, graph + checkpoint resident):

    python scripts/diagnostics/v2_confusion.py \
        --checkpoint outputs/v3/hgt_v2/hgt_flow_best.pt \
        --graph      outputs/v3/graph.npz \
        --graph-meta outputs/v3/graph.meta.json \
        --splits     outputs/v3/splits.json \
        --protocol   random \
        --out        outputs/v3/hgt_v2/confusion_random.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from graphslm_ids.offline.training.hetero_graph_artifact import load_v3_artifact
from graphslm_ids.offline.training.neighbor_sampling import (
    HeteroNeighborSampler,
    InMemoryNeighborBackend,
)
from graphslm_ids.offline.training.train_hgt_flow_classifier import (
    label_name_mapping,
    make_neighbor_loader,
    to_torch_batch,
)

# Reuse the checkpoint->model builder from the eval script (single source of truth).
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_v3_eval_both_splits",
    str(Path(__file__).resolve().parents[1] / "eval" / "v3_eval_both_splits.py"),
)
_eval_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)  # type: ignore[union-attr]
_build_model_from_checkpoint = _eval_mod._build_model_from_checkpoint


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump confusion matrix for a v3 HGT checkpoint.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--graph-meta", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--protocol", default="random", choices=["random", "temporal"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-seed-flows", type=int, default=384)
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_test_ids(artifact, splits_payload, protocol: str) -> np.ndarray:
    flow_id_order = artifact.metadata.get("flow_id_order") or []
    raw = splits_payload[protocol]["test"]
    if flow_id_order and isinstance(raw[0], str):
        id_to_idx = {fid: i for i, fid in enumerate(flow_id_order)}
        return np.asarray(
            [id_to_idx[str(x)] for x in raw if str(x) in id_to_idx], dtype=np.int64
        )
    return np.asarray(raw, dtype=np.int64)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)

    artifact = load_v3_artifact(
        graph_npz=args.graph, graph_meta_json=args.graph_meta, add_reverse_edges=True
    )
    backend = InMemoryNeighborBackend(artifact)
    labels_np = artifact.flow_y
    num_classes = int(labels_np.max()) + 1
    label_names = label_name_mapping(artifact.metadata, labels_np)

    with args.splits.open("r", encoding="utf-8") as fh:
        splits_payload = json.load(fh)
    test_ids = _resolve_test_ids(artifact, splits_payload, args.protocol)
    print(f"[confusion] protocol={args.protocol} n_test={test_ids.shape[0]} "
          f"num_classes={num_classes}", flush=True)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(state, backend, num_classes, device)
    model.eval()

    cfg = state.get("config") or {}
    sampler_cfg = (cfg.get("sampler") if isinstance(cfg, dict) else {}) or {}
    data_cfg = (cfg.get("data") if isinstance(cfg, dict) else {}) or {}

    # Flow standardization stats: the trainer bakes the TRAIN-split mean/std into
    # the checkpoint under 'flow_feature_stats'. backend.manifest usually does NOT
    # carry them, so prefer the checkpoint stats (mismatch => unnormalized inputs
    # => garbage predictions). Fall back to manifest only if the ckpt lacks them.
    flow_feature_stats = None
    ckpt_stats = state.get("flow_feature_stats")
    manifest_stats = (backend.manifest or {}).get("flow_feature_stats")
    chosen = ckpt_stats if isinstance(ckpt_stats, dict) else manifest_stats
    if isinstance(chosen, dict) and "mean" in chosen and "std" in chosen:
        flow_feature_stats = {
            "mean": list(chosen["mean"]),
            "std": list(chosen["std"]),
        }
        print(f"[confusion] using flow_feature_stats from "
              f"{'checkpoint' if chosen is ckpt_stats else 'manifest'}", flush=True)
    else:
        print("[confusion] WARN: no flow_feature_stats found — flow features NOT "
              "standardized (predictions will be unreliable)", flush=True)

    sampler = HeteroNeighborSampler(
        backend,
        hops=int(sampler_cfg.get("hops") or 4),
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
    loader, _ = make_neighbor_loader(
        test_ids, sampler, eval_cfg, shuffle=False, world_size=1, rank=0, seed=42
    )

    use_sew = bool(data_cfg.get("use_semantic_edge_weights", True))
    edge_types = list(backend.edge_types)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)  # [true, pred]
    with torch.inference_mode():
        for i, batch in enumerate(loader, start=1):
            node_features, edge_index, edge_weight, seed_mask, seed_labels = to_torch_batch(
                batch, edge_types, device, use_sew, packet_store=None
            )
            logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
            seed_logits = logits[seed_mask]
            preds = seed_logits.float().argmax(dim=1).cpu().numpy()
            trues = seed_labels.cpu().numpy()
            np.add.at(confusion, (trues, preds), 1)
            if i % 10 == 0 or i == len(loader):
                print(f"[confusion] batch {i}/{len(loader)}", flush=True)

    # --- Report: for each class, where do its TRUE samples go? ---
    idx_to_name = {i: label_names.get(i, str(i)) for i in range(num_classes)}
    report: dict[str, object] = {
        "protocol": args.protocol,
        "n_test": int(test_ids.shape[0]),
        "label_names": idx_to_name,
        "confusion": confusion.tolist(),
        "per_class_breakdown": {},
    }
    print("\n===== Per-class breakdown (TRUE -> where predicted) =====", flush=True)
    for true_id in range(num_classes):
        support = int(confusion[true_id].sum())
        if support == 0:
            continue
        row = confusion[true_id]
        correct = int(row[true_id])
        recall = correct / support
        # Top-3 predicted destinations (excluding none).
        order = np.argsort(row)[::-1]
        dests = []
        for pred_id in order[:4]:
            cnt = int(row[pred_id])
            if cnt == 0:
                continue
            dests.append({
                "pred": idx_to_name[int(pred_id)],
                "count": cnt,
                "pct": round(100.0 * cnt / support, 1),
                "is_correct": bool(int(pred_id) == true_id),
            })
        report["per_class_breakdown"][idx_to_name[true_id]] = {
            "support": support,
            "recall": round(recall, 3),
            "destinations": dests,
        }
        dest_str = "  ".join(
            f"{d['pred']}={d['pct']}%{'*' if d['is_correct'] else ''}" for d in dests
        )
        print(f"  {idx_to_name[true_id]:<28} n={support:<5} recall={recall:.3f} | {dest_str}",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[confusion] written -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
