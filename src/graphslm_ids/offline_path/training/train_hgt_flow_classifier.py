from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.offline_path.training.hetero_graph_artifact import load_three_tier_graph_artifact
from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.utils.io import ensure_dir, write_json


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "graph_npz": "data/processed/graph_artifact_3tier_t082_k5.npz",
        "graph_meta_json": "data/processed/graph_artifact_3tier_t082_k5.meta.json",
        "packet_feature": "semantic",
        "add_reverse_edges": True,
        "standardize_flow_features": True,
        "use_semantic_edge_weights": True,
    },
    "model": {
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "ffn_multiplier": 2,
    },
    "train": {
        "output_dir": "outputs/hgt_flow_classifier_t082_k5_l3_d01",
        "epochs": 150,
        "batch_mode": "full",
        "lr": 1e-3,
        "weight_decay": 5e-5,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "patience": 30,
        "class_weight": "balanced",
        "seed": 42,
        "device": "cpu",
        "monitor": "val_macro_f1",
        "log_every": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact HGT-style flow classifier on the selected t082 three-tier graph artifact."
    )
    parser.add_argument("--config", default="configs/hgt.example.yaml")
    parser.add_argument("--graph-npz", default=None)
    parser.add_argument("--graph-meta-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("HGT config must be a YAML mapping.")
        config = deep_update(config, loaded)
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = deepcopy(config)
    if args.graph_npz is not None:
        config["data"]["graph_npz"] = args.graph_npz
    if args.graph_meta_json is not None:
        config["data"]["graph_meta_json"] = args.graph_meta_json
    if args.output_dir is not None:
        config["train"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    if args.num_heads is not None:
        config["model"]["num_heads"] = args.num_heads
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["train"]["weight_decay"] = args.weight_decay
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def stratified_split(
    labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label in sorted(np.unique(labels).tolist()):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        n = int(class_indices.shape[0])
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3 and test_ratio > 0:
            n_test = max(1, n_test)
        if n >= 3 and val_ratio > 0:
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            overflow = n_test + n_val - n + 1
            n_val = max(0, n_val - overflow)

        test_indices.extend(class_indices[:n_test].tolist())
        val_indices.extend(class_indices[n_test : n_test + n_val].tolist())
        train_indices.extend(class_indices[n_test + n_val :].tolist())

    train = np.asarray(train_indices, dtype=np.int64)
    val = np.asarray(val_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def standardize_flow_features(
    flow_x: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = flow_x[train_idx].mean(axis=0, keepdims=True)
    std = flow_x[train_idx].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    normalized = (flow_x - mean) / std
    stats = {
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
    }
    return normalized.astype(np.float32), stats


def to_torch_dict(
    node_features: dict[str, np.ndarray],
    edge_index: dict[tuple[str, str, str], np.ndarray],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
    node_tensors = {
        key: torch.from_numpy(np.asarray(value)).to(device)
        for key, value in node_features.items()
    }
    edge_tensors = {
        key: torch.from_numpy(np.asarray(value, dtype=np.int64)).to(device)
        for key, value in edge_index.items()
    }
    return node_tensors, edge_tensors


def to_torch_edge_weights(
    edge_attr: dict[tuple[str, str, str], np.ndarray],
    device: torch.device,
    use_semantic_edge_weights: bool,
) -> dict[tuple[str, str, str], torch.Tensor] | None:
    if not use_semantic_edge_weights:
        return None

    edge_weights: dict[tuple[str, str, str], torch.Tensor] = {}
    for edge_key, attr in edge_attr.items():
        relation = edge_key[1]
        if "matches_technique" not in relation:
            continue
        values = np.asarray(attr, dtype=np.float32)
        if values.ndim == 2:
            values = values[:, 0]
        edge_weights[edge_key] = torch.from_numpy(values).to(device)
    return edge_weights


def class_weights(labels: np.ndarray, train_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels[train_idx], minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (float(num_classes) * counts[nonzero])
    return torch.from_numpy(weights)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    label_names: dict[int, str],
) -> dict[str, Any]:
    if indices.numel() == 0:
        return {
            "count": 0,
            "loss": None,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {},
        }

    selected_logits = logits[indices]
    selected_labels = labels[indices]
    loss = F.cross_entropy(selected_logits, selected_labels).item()
    pred = selected_logits.argmax(dim=1)
    correct = (pred == selected_labels).sum().item()
    count = int(selected_labels.numel())

    pred_np = pred.detach().cpu().numpy()
    label_np = selected_labels.detach().cpu().numpy()
    num_classes = int(logits.shape[1])
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []

    for class_id in range(num_classes):
        tp = int(((pred_np == class_id) & (label_np == class_id)).sum())
        fp = int(((pred_np == class_id) & (label_np != class_id)).sum())
        fn = int(((pred_np != class_id) & (label_np == class_id)).sum())
        support = int((label_np == class_id).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if support > 0:
            f1_values.append(float(f1))
        per_class[label_names.get(class_id, str(class_id))] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return {
        "count": count,
        "loss": float(loss),
        "accuracy": float(correct / max(count, 1)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def label_name_mapping(metadata: dict[str, Any], labels: np.ndarray) -> dict[int, str]:
    mapping = metadata.get("label_mapping", {})
    if isinstance(mapping, dict) and mapping:
        return {int(idx): str(name) for name, idx in mapping.items()}
    return {int(label): str(label) for label in sorted(np.unique(labels).tolist())}


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(Path(args.config)), args)

    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["train"]["device"]))
    if str(config["train"]["batch_mode"]).lower() != "full":
        raise ValueError("Only full batch HGT training is supported in this script.")

    artifact = load_three_tier_graph_artifact(
        graph_npz=Path(config["data"]["graph_npz"]),
        graph_meta_json=Path(config["data"]["graph_meta_json"]),
        packet_feature=str(config["data"]["packet_feature"]),
        add_reverse_edges=bool(config["data"]["add_reverse_edges"]),
    )

    labels_np = artifact.flow_y.astype(np.int64)
    num_classes = int(labels_np.max()) + 1
    train_idx_np, val_idx_np, test_idx_np = stratified_split(
        labels=labels_np,
        val_ratio=float(config["train"]["val_ratio"]),
        test_ratio=float(config["train"]["test_ratio"]),
        seed=seed,
    )

    flow_feature_stats: dict[str, list[float]] | None = None
    if bool(config["data"]["standardize_flow_features"]):
        artifact.node_features["flow"], flow_feature_stats = standardize_flow_features(
            artifact.node_features["flow"],
            train_idx_np,
        )

    node_features, edge_index = to_torch_dict(artifact.node_features, artifact.edge_index, device)
    edge_weight = to_torch_edge_weights(
        artifact.edge_attr,
        device=device,
        use_semantic_edge_weights=bool(config["data"]["use_semantic_edge_weights"]),
    )
    labels = torch.from_numpy(labels_np).long().to(device)
    train_idx = torch.from_numpy(train_idx_np).long().to(device)
    val_idx = torch.from_numpy(val_idx_np).long().to(device)
    test_idx = torch.from_numpy(test_idx_np).long().to(device)

    node_input_dims = {
        "flow": int(node_features["flow"].shape[1]),
        "packet": int(node_features["packet"].shape[1]),
        "technique": int(node_features["technique"].shape[1]),
    }
    num_tactics = int(node_features["tactic"].shape[0])

    model = HeteroGraphTransformer(
        node_input_dims=node_input_dims,
        edge_types=list(edge_index.keys()),
        num_classes=num_classes,
        num_tactics=num_tactics,
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        num_heads=int(config["model"]["num_heads"]),
        dropout=float(config["model"]["dropout"]),
        ffn_multiplier=int(config["model"]["ffn_multiplier"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    weight = None
    if str(config["train"]["class_weight"]).lower() == "balanced":
        weight = class_weights(labels_np, train_idx_np, num_classes).to(device)

    output_dir = ensure_dir(Path(config["train"]["output_dir"]))
    best_checkpoint = output_dir / "hgt_flow_best.pt"
    label_names = label_name_mapping(artifact.metadata, labels_np)

    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    monitor = str(config["train"]["monitor"])
    log_every = max(1, int(config["train"]["log_every"]))

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(node_features, edge_index, edge_weight_dict=edge_weight)
            train_metrics = compute_metrics(logits_eval, labels, train_idx, label_names)
            val_metrics = compute_metrics(logits_eval, labels, val_idx, label_names)
            test_metrics = compute_metrics(logits_eval, labels, test_idx, label_names)

        entry = {
            "epoch": epoch,
            "train": {key: value for key, value in train_metrics.items() if key != "per_class"},
            "val": {key: value for key, value in val_metrics.items() if key != "per_class"},
            "test": {key: value for key, value in test_metrics.items() if key != "per_class"},
        }
        history.append(entry)

        monitor_score = float(val_metrics["macro_f1"] if monitor == "val_macro_f1" else -val_metrics["loss"])
        if monitor_score > best_score:
            best_score = monitor_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "node_input_dims": node_input_dims,
                    "edge_types": [list(edge_key) for edge_key in edge_index.keys()],
                    "num_classes": num_classes,
                    "num_tactics": num_tactics,
                    "label_names": label_names,
                    "flow_feature_stats": flow_feature_stats,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                best_checkpoint,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % log_every == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={float(loss.item()):.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"test_macro_f1={test_metrics['macro_f1']:.4f}"
            )

        if epochs_without_improvement >= int(config["train"]["patience"]):
            print(f"Early stopping at epoch {epoch}.")
            break

    best_payload = load_checkpoint(best_checkpoint, device)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "device": str(device),
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "num_flows": int(labels_np.shape[0]),
        "num_packets": int(artifact.node_features["packet"].shape[0]),
        "num_techniques": int(artifact.node_features["technique"].shape[0]),
        "num_tactics": int(num_tactics),
        "num_edge_types": int(len(edge_index)),
        "splits": {
            "train": int(train_idx_np.shape[0]),
            "val": int(val_idx_np.shape[0]),
            "test": int(test_idx_np.shape[0]),
        },
        "label_names": {str(key): value for key, value in label_names.items()},
        "best_val_metrics": best_payload["val_metrics"],
        "best_test_metrics": best_payload["test_metrics"],
        "history": history,
    }
    write_json(output_dir / "training_summary.json", summary)

    print(f"[OK] Best checkpoint: {best_checkpoint}")
    print(f"[OK] Training summary: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
