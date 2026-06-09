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
        --out                 outputs/v3/results.json \\
        --tag                 v5-ordered-byte

Reuses the production trainer's ``evaluate_neighbor_sampling`` helper so the
metrics are computed *exactly* the same way as during training (no risk of a
silent definition drift between train-time validation and final evaluation).

Defensibility add-ons (v5)
--------------------------
Per split, the structured JSON also records, via ``eval_reporting``:

  * ``confusion_matrix`` (rows = true, cols = pred) + label order,
  * ``per_class_support`` (test-sample count per class),
  * ``bootstrap_f1_ci`` — seeded 95% CIs for macro-F1 and each per-class F1,
  * ``feature_flags`` — which ablation levers (attack-isolation / ordered-byte)
    were active, read from the checkpoint config, plus the ``--tag``.

Two runs with different ``--tag`` / flags can then be diffed as an ablation.
"""
from __future__ import annotations

import argparse
import json
import sys
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
    to_torch_batch,
)

# eval_reporting lives next to this script (scripts/eval/) and is not part of
# the installed package — import it by absolute path so the script works from
# any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_reporting as er  # noqa: E402


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
    p.add_argument(
        "--tag", default=None,
        help=(
            "Free-form run label recorded in the JSON (e.g. 'v5-ordered-byte' "
            "or 'v4-baseline'). Lets two runs be diffed as an ablation."
        ),
    )
    p.add_argument(
        "--bootstrap-n", type=int, default=1000,
        help="Bootstrap resamples for the per-class/macro-F1 CIs. Default: 1000.",
    )
    p.add_argument(
        "--bootstrap-seed", type=int, default=42,
        help="Seed for the (deterministic) bootstrap CIs. Default: 42.",
    )
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# Config keys we probe for ablation flags. Build-time decisions (attack
# isolation / evidence relabel) and the v5 ordered-byte feature may live in
# different config sections depending on which pipeline wrote them, so we scan
# a small set of likely locations and report whatever we find (None if the
# checkpoint config doesn't carry the flag).
_ISOLATION_FLAG_KEYS = (
    "attack_isolation",
    "evidence_relabel",
    "relabel",
    "isolate_attack_flows",
    "use_attack_isolation",
)
_ORDERED_BYTE_FLAG_KEYS = (
    "ordered_bytes",
    "ordered_byte",
    "use_ordered_bytes",
    "ordered_byte_block",
    "pmi_ngram",
    "payload_length",
)
_FLAG_SECTIONS = ("features", "feature", "data", "preprocessing", "experiment")


def _find_flag(cfg: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first matching flag value across known sections (or None).

    Checks the top level first, then each known sub-section. Returns the raw
    value (bool / int / str) so e.g. ``payload_length: 512`` is recorded as-is.
    """
    if not isinstance(cfg, dict):
        return None
    for key in keys:
        if key in cfg:
            return cfg[key]
    for section in _FLAG_SECTIONS:
        sub = cfg.get(section)
        if isinstance(sub, dict):
            for key in keys:
                if key in sub:
                    return sub[key]
    return None


def _extract_feature_flags(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Pull the ablation-relevant feature flags out of a checkpoint's config.

    Records the two v5 levers (evidence-relabel / attack-isolation, and the
    deterministic ordered-byte representation) so two result files can be
    compared as an ablation. Missing flags are reported as ``None`` rather than
    guessed, and ``artifact_version`` / ``graph_npz`` are echoed for provenance.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    return {
        "attack_isolation": _find_flag(cfg, _ISOLATION_FLAG_KEYS),
        "ordered_byte_feature": _find_flag(cfg, _ORDERED_BYTE_FLAG_KEYS),
        "artifact_version": data_cfg.get("artifact_version"),
        "graph_npz": data_cfg.get("graph_npz"),
        "config_present": bool(cfg),
    }


def _collect_predictions(
    *,
    model: HeteroGraphTransformer,
    loader: Any,
    edge_types: list[tuple[str, str, str]],
    device: torch.device,
    use_semantic_edge_weights: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return ``(y_true, y_pred)`` int64 arrays.

    Mirrors the single-device path of the trainer's
    ``evaluate_neighbor_sampling`` (raw-logits ``argmax`` over the seed mask),
    so the predictions used for the confusion matrix / bootstrap are identical
    to those behind the headline metrics. We collect the arrays here because
    ``evaluate_neighbor_sampling`` only returns the aggregated metric dict, not
    the raw predictions, and the trainer must not be modified.
    """
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            node_features, edge_index, edge_weight, seed_mask, seed_labels = to_torch_batch(
                batch, edge_types, device, use_semantic_edge_weights,
            )
            logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
            seed_logits = logits[seed_mask].detach().float()
            preds.append(seed_logits.argmax(dim=1).cpu().numpy())
            labels.append(seed_labels.detach().cpu().numpy())
    y_pred = (
        np.concatenate(preds).astype(np.int64)
        if preds else np.empty((0,), dtype=np.int64)
    )
    y_true = (
        np.concatenate(labels).astype(np.int64)
        if labels else np.empty((0,), dtype=np.int64)
    )
    return y_true, y_pred


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
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
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

    use_semantic_edge_weights = bool(
        (cfg.get("data") or {}).get("use_semantic_edge_weights", True)
    )
    metrics = evaluate_neighbor_sampling(
        model=model,
        loader=test_loader,
        edge_types=list(backend.edge_types),
        device=device,
        use_amp=False,
        use_semantic_edge_weights=use_semantic_edge_weights,
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

    # Second pass on the SAME loader to collect raw predictions for the
    # defensibility reports (confusion matrix + per-class support + bootstrap
    # CIs). evaluate_neighbor_sampling returns only aggregated metrics, and the
    # trainer must not be modified, so we re-run inference here with the
    # identical raw-logits-argmax rule. The loader is deterministic
    # (shuffle=False, seed=42), so this is an exact replay.
    y_true, y_pred = _collect_predictions(
        model=model,
        loader=test_loader,
        edge_types=list(backend.edge_types),
        device=device,
        use_semantic_edge_weights=use_semantic_edge_weights,
    )

    cm = er.confusion_matrix(y_true, y_pred, num_classes)
    support_by_id = er.per_class_support(y_true, num_classes)
    bootstrap = er.bootstrap_f1_ci(
        y_true, y_pred, num_classes,
        n_boot=int(bootstrap_n), seed=int(bootstrap_seed),
    )

    # Re-key the integer-id support / bootstrap maps by human label name so the
    # JSON is readable; keep the id alongside for unambiguous joins.
    support_named = {
        label_names.get(cid, str(cid)): cnt for cid, cnt in support_by_id.items()
    }
    bootstrap_named = {}
    for cid, entry in bootstrap.get("per_class", {}).items():
        item = dict(entry)
        item["class_id"] = int(cid)
        bootstrap_named[label_names.get(cid, str(cid))] = item

    return {
        "protocol": protocol,
        "n_test": int(test_ids.shape[0]),
        "n_pred": int(y_true.shape[0]),
        "accuracy": float(metrics.get("accuracy") or 0.0),
        "macro_f1": float(metrics.get("macro_f1") or 0.0),
        "loss": metrics.get("loss"),
        "per_class_f1": per_class_f1,
        "per_class_support": support_named,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [
            label_names.get(c, str(c)) for c in range(num_classes)
        ],
        "bootstrap_f1_ci": {
            "n_boot": bootstrap["n_boot"],
            "seed": bootstrap["seed"],
            "alpha": bootstrap["alpha"],
            "macro_f1": bootstrap["macro_f1"],
            "per_class": bootstrap_named,
        },
        "feature_flags": _extract_feature_flags(cfg),
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
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
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
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )

    gap = {
        "macro_f1": float(random_res["macro_f1"] - temporal_res["macro_f1"]),
        "accuracy": float(random_res["accuracy"] - temporal_res["accuracy"]),
    }
    # Surface the ablation flags at the top level too. Both checkpoints SHOULD
    # share the same feature build; if they disagree, record both so the
    # discrepancy is visible rather than silently picking one.
    feature_flags = random_res.get("feature_flags") or {}
    if random_res.get("feature_flags") != temporal_res.get("feature_flags"):
        feature_flags = {
            "random": random_res.get("feature_flags"),
            "temporal": temporal_res.get("feature_flags"),
            "_mismatch": True,
        }
    payload = {
        "tag": args.tag,
        "feature_flags": feature_flags,
        "num_classes": int(num_classes),
        "bootstrap": {"n_boot": int(args.bootstrap_n), "seed": int(args.bootstrap_seed)},
        "random": random_res,
        "temporal": temporal_res,
        "gap_random_minus_temporal": gap,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human-readable summary table.
    print()
    if args.tag:
        print(f"[v3-eval] tag={args.tag}")
    print(f"[v3-eval] feature_flags={json.dumps(feature_flags)}")
    _rci = random_res["bootstrap_f1_ci"]["macro_f1"]
    _tci = temporal_res["bootstrap_f1_ci"]["macro_f1"]
    print(
        f"[v3-eval] macro-F1 95% CI  "
        f"random=[{_rci['lo']:.4f}, {_rci['hi']:.4f}]  "
        f"temporal=[{_tci['lo']:.4f}, {_tci['hi']:.4f}]"
    )
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
