"""v3 Smart-BOTH Hybrid: evaluate the SAME model architecture on both the
random-stratified and temporal test splits, write a side-by-side results
table, and emit the random-minus-temporal gap (the contribution number).

Usage
-----
::

    python scripts/eval/v3_eval_both_splits.py \\
        --checkpoint-random   outputs/v3/checkpoint_random.pt \\
        --checkpoint-temporal outputs/v3/checkpoint_temporal.pt \\
        --graph               outputs/v3/graph.npz \\
        --graph-meta          outputs/v3/graph.meta.json \\
        --splits              outputs/v3/splits.json \\
        --out                 outputs/v3/results.json

Reuses the production trainer's ``evaluate_neighbor_sampling`` helper so the
metrics are computed *exactly* the same way as during training (no risk of a
silent definition drift between train-time validation and final evaluation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from graphslm_ids.offline.training.hetero_graph_artifact import load_v3_artifact
from graphslm_ids.offline.training.neighbor_sampling import (
    HeteroNeighborSampler,
    InMemoryNeighborBackend,
)
from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.offline.training.train_hgt_flow_classifier import (
    evaluate_neighbor_sampling,
    label_name_mapping,
    make_neighbor_loader,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate v3 HGT on random + temporal test splits and report the gap."
        )
    )
    p.add_argument("--checkpoint-random", required=True, type=Path)
    p.add_argument("--checkpoint-temporal", required=True, type=Path)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--graph-meta", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--device", default="auto",
        help="Device to run inference on (cpu | cuda | auto). Default: auto.",
    )
    p.add_argument(
        "--batch-seed-flows", type=int, default=384,
        help="Seed flows per eval batch (memory tunable). Default: 384.",
    )
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_model_from_checkpoint(
    state: dict[str, Any],
    backend: InMemoryNeighborBackend,
    num_classes: int,
    device: torch.device,
) -> HeteroGraphTransformer:
    # Hyperparameters expected to be saved alongside the checkpoint under
    # 'config' or 'hparams'. Fall back to common v3 defaults if absent.
    cfg = state.get("config") or state.get("hparams") or {}
    model_cfg = (cfg.get("model") if isinstance(cfg, dict) else {}) or {}
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 4))
    num_heads = int(model_cfg.get("num_heads", 8))
    dropout = float(model_cfg.get("dropout", 0.2))
    ffn_multiplier = int(model_cfg.get("ffn_multiplier", 2))

    # v3 has 5 node types (adds 'host' to v2's 4). Mirror the trainer:
    # derive node_input_dims from backend.feature_dims so 'host' is included
    # whenever the artifact contains it. The HGT model accepts a `node_types`
    # parameter — pass keys + 'tactic' (id-only embedding, no input dim).
    node_input_dims = {nt: int(d) for nt, d in backend.feature_dims.items()}
    derived_node_types = list(node_input_dims.keys()) + ["tactic"]
    model = HeteroGraphTransformer(
        node_input_dims=node_input_dims,
        edge_types=list(backend.edge_types),
        num_classes=num_classes,
        num_tactics=int(backend.num_tactics),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        ffn_multiplier=ffn_multiplier,
        activation_checkpointing=False,
        node_types=derived_node_types,
    ).to(device)

    state_dict = state.get("model_state_dict") or state.get("state_dict") or state
    if isinstance(state_dict, dict):
        # Strip optional DDP / torch.compile prefixes.
        stripped = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            if new_k.startswith("_orig_mod."):
                new_k = new_k[len("_orig_mod."):]
            stripped[new_k] = v
        model.load_state_dict(stripped, strict=False)
    return model


def _eval_one_protocol(
    *,
    protocol: str,
    checkpoint_path: Path,
    backend: InMemoryNeighborBackend,
    test_ids: np.ndarray,
    num_classes: int,
    label_names: dict[int, str],
    device: torch.device,
    batch_seed_flows: int,
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(state, backend, num_classes, device)

    cfg = state.get("config") or {}
    sampler_cfg = (cfg.get("sampler") if isinstance(cfg, dict) else {}) or {}
    fanouts = dict(sampler_cfg.get("fanouts") or {})
    reverse_fanouts = dict(sampler_cfg.get("reverse_fanouts") or {})
    sampler_hops = int(sampler_cfg.get("hops") or 4)

    flow_feature_stats = None
    manifest_stats = (backend.manifest or {}).get("flow_feature_stats")
    if isinstance(manifest_stats, dict) and "mean" in manifest_stats and "std" in manifest_stats:
        flow_feature_stats = {
            "mean": list(manifest_stats["mean"]),
            "std": list(manifest_stats["std"]),
        }

    sampler = HeteroNeighborSampler(
        backend,
        hops=sampler_hops,
        fanouts=fanouts,
        reverse_fanouts=reverse_fanouts,
        always_include_all_tactics=bool(sampler_cfg.get("always_include_all_tactics", True)),
        always_include_all_techniques=bool(sampler_cfg.get("always_include_all_techniques", True)),
        flow_feature_stats=flow_feature_stats,
        standardize_flow_features=bool(
            (cfg.get("data") or {}).get("standardize_flow_features", True)
        ),
        seed=42,
    )

    eval_cfg = {
        "train": {"batch_seed_flows": int(batch_seed_flows), "seed": 42},
        "dataloader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
    }
    test_loader, _ = make_neighbor_loader(
        test_ids, sampler, eval_cfg, shuffle=False,
        world_size=1, rank=0, seed=42,
    )

    metrics = evaluate_neighbor_sampling(
        model=model,
        loader=test_loader,
        edge_types=list(backend.edge_types),
        device=device,
        use_amp=False,
        use_semantic_edge_weights=bool(
            (cfg.get("data") or {}).get("use_semantic_edge_weights", True)
        ),
        num_classes=num_classes,
        label_names=label_names,
        epoch=0,
        split_name=f"test_{protocol}",
        is_ddp=False,
    )
    # Re-shape the metrics dict into a stable summary for cross-protocol diff.
    per_class_f1 = {
        name: float(entry["f1"])
        for name, entry in (metrics.get("per_class") or {}).items()
    }
    return {
        "protocol": protocol,
        "n_test": int(test_ids.shape[0]),
        "accuracy": float(metrics.get("accuracy") or 0.0),
        "macro_f1": float(metrics.get("macro_f1") or 0.0),
        "loss": metrics.get("loss"),
        "per_class_f1": per_class_f1,
    }


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)

    artifact = load_v3_artifact(
        graph_npz=args.graph,
        graph_meta_json=args.graph_meta,
        add_reverse_edges=True,
    )
    backend = InMemoryNeighborBackend(artifact)
    labels_np = artifact.flow_y
    num_classes = int(labels_np.max()) + 1
    label_names = label_name_mapping(artifact.metadata, labels_np)

    with args.splits.open("r", encoding="utf-8") as fh:
        splits_payload = json.load(fh)
    if "random" not in splits_payload or "temporal" not in splits_payload:
        raise SystemExit(
            f"--splits {args.splits} must contain both 'random' and 'temporal' keys; "
            f"got {sorted(splits_payload.keys())}."
        )

    # splits.json stores STRING flow IDs from v3/split.py. Convert to integer
    # node indices using the artifact's canonical flow_id_order.
    flow_id_order = artifact.metadata.get("flow_id_order") or []
    if flow_id_order and isinstance(splits_payload["random"]["test"][0], str):
        _id_to_idx = {fid: i for i, fid in enumerate(flow_id_order)}
        random_test = np.asarray(
            [_id_to_idx[str(x)] for x in splits_payload["random"]["test"] if str(x) in _id_to_idx],
            dtype=np.int64,
        )
        temporal_test = np.asarray(
            [_id_to_idx[str(x)] for x in splits_payload["temporal"]["test"] if str(x) in _id_to_idx],
            dtype=np.int64,
        )
    else:
        # Already integer-indexed (e.g. pre-converted splits file).
        random_test = np.asarray(splits_payload["random"]["test"], dtype=np.int64)
        temporal_test = np.asarray(splits_payload["temporal"]["test"], dtype=np.int64)

    print(
        f"[v3-eval] num_classes={num_classes} "
        f"random_test={random_test.shape[0]} temporal_test={temporal_test.shape[0]}",
        flush=True,
    )

    random_res = _eval_one_protocol(
        protocol="random",
        checkpoint_path=args.checkpoint_random,
        backend=backend,
        test_ids=random_test,
        num_classes=num_classes,
        label_names=label_names,
        device=device,
        batch_seed_flows=args.batch_seed_flows,
    )
    temporal_res = _eval_one_protocol(
        protocol="temporal",
        checkpoint_path=args.checkpoint_temporal,
        backend=backend,
        test_ids=temporal_test,
        num_classes=num_classes,
        label_names=label_names,
        device=device,
        batch_seed_flows=args.batch_seed_flows,
    )

    gap = {
        "macro_f1": float(random_res["macro_f1"] - temporal_res["macro_f1"]),
        "accuracy": float(random_res["accuracy"] - temporal_res["accuracy"]),
    }
    payload = {
        "random": random_res,
        "temporal": temporal_res,
        "gap_random_minus_temporal": gap,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human-readable summary table.
    print()
    print("=" * 72)
    print(f"{'metric':<14} {'random':>12} {'temporal':>12} {'gap':>12}")
    print("-" * 72)
    print(
        f"{'macro_f1':<14} "
        f"{random_res['macro_f1']:>12.4f} "
        f"{temporal_res['macro_f1']:>12.4f} "
        f"{gap['macro_f1']:>+12.4f}"
    )
    print(
        f"{'accuracy':<14} "
        f"{random_res['accuracy']:>12.4f} "
        f"{temporal_res['accuracy']:>12.4f} "
        f"{gap['accuracy']:>+12.4f}"
    )
    print("=" * 72)
    print("per-class F1 (random / temporal):")
    all_classes = sorted(
        set(random_res["per_class_f1"]) | set(temporal_res["per_class_f1"])
    )
    for cls in all_classes:
        r = random_res["per_class_f1"].get(cls, float("nan"))
        t = temporal_res["per_class_f1"].get(cls, float("nan"))
        d = r - t
        print(f"  {cls:<32} {r:>8.4f}  {t:>8.4f}  {d:>+8.4f}")
    print()
    print(f"[v3-eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
