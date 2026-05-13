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
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.offline_path.training.hetero_graph_artifact import (
    load_graph_store_artifact,
    load_three_tier_graph_artifact,
)
from graphslm_ids.offline_path.training.neighbor_sampling import (
    FlowSeedDataset,
    HeteroNeighborSampler,
    InMemoryNeighborBackend,
    MiniBatchSubgraph,
    NeighborBackend,
    NeighborSamplingCollator,
)
from graphslm_ids.offline_path.training.on_disk_graph_store import OnDiskHeteroGraphStore
from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.utils.io import ensure_dir, write_json


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "source": "npz",
        "graph_npz": "data/processed/graph_artifact_3tier_t082_k5.npz",
        "graph_meta_json": "data/processed/graph_artifact_3tier_t082_k5.meta.json",
        "graph_store_root": "data/graph_store_v1",
        "read_sealed_only": True,
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
        "device": "auto",
        "multi_gpu": True,
        "monitor": "val_macro_f1",
        "log_every": 1,
        "amp": False,
        "activation_checkpointing": False,
        "batch_seed_flows": 256,
        "grad_accum_steps": 1,
    },
    "sampler": {
        "hops": None,
        "fanouts": {
            "flow__contains__packet": 20,
            "packet__next_packet__packet": 4,
            "packet__matches_technique__technique": 5,
            "flow__matches_technique__technique": 5,
            "technique__belongs_to_tactic__tactic": 1,
        },
        "reverse_fanouts": {
            "rev_contains": 1,
            "rev_next_packet": 1,
            "rev_matches_technique": 0,
            "rev_belongs_to_tactic": 0,
        },
        "always_include_all_tactics": True,
        "always_include_all_techniques": True,
    },
    "dataloader": {
        "num_workers": 0,
        "prefetch_factor": 2,
        "pin_memory": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact HGT-style flow classifier on the selected t082 three-tier graph artifact."
    )
    parser.add_argument("--config", default="configs/hgt.example.yaml")
    parser.add_argument("--graph-npz", default=None)
    parser.add_argument("--graph-meta-json", default=None)
    parser.add_argument("--graph-store-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision training/evaluation.")
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Trade extra compute for lower activation memory during HGT training.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--multi-gpu", dest="multi_gpu", action="store_true", default=None,
                        help="Use all available CUDA GPUs (neighbor_sampling mode only).")
    parser.add_argument("--no-multi-gpu", dest="multi_gpu", action="store_false",
                        help="Disable multi-GPU and force single GPU/CPU.")
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
    if args.graph_store_root is not None:
        config["data"]["source"] = "graph_store"
        config["data"]["graph_store_root"] = args.graph_store_root
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
    if args.amp:
        config["train"]["amp"] = True
    if args.activation_checkpointing:
        config["train"]["activation_checkpointing"] = True
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.multi_gpu is not None:
        config["train"]["multi_gpu"] = args.multi_gpu
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


def get_training_devices(primary: torch.device, multi_gpu: bool) -> list[torch.device]:
    """Return all available CUDA devices when multi_gpu=True, else just the primary device."""
    if primary.type != "cuda" or not multi_gpu:
        return [primary]
    n = torch.cuda.device_count()
    if n <= 1:
        return [primary]
    return [torch.device("cuda", i) for i in range(n)]


def _multi_gpu_train_step(
    model: HeteroGraphTransformer,
    batch_group: list[MiniBatchSubgraph],
    devices: list[torch.device],
    edge_types: list[tuple[str, str, str]],
    weight: torch.Tensor | None,
    scaler: torch.amp.GradScaler,
    use_semantic_edge_weights: bool,
    grad_accum_steps: int,
) -> tuple[float, int, list[np.ndarray], list[np.ndarray], dict[str, list[int]], dict[str, list[int]]]:
    """Forward-backward across multiple GPUs using parallel_apply, then sync gradients to primary model."""
    import torch.nn.parallel as P

    n = min(len(batch_group), len(devices))
    primary = devices[0]
    device_ids = [d.index for d in devices[:n]]

    replicas = P.replicate(model, device_ids, detach=False)

    inputs_args: list[tuple] = []
    inputs_kwargs: list[dict[str, Any]] = []
    masks: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch, device in zip(batch_group[:n], devices[:n]):
        nf, ei, ew, sm, sl = to_torch_batch(batch, edge_types, device, use_semantic_edge_weights)
        inputs_args.append((nf, ei))
        inputs_kwargs.append({"edge_weight_dict": ew})
        masks.append(sm)
        all_labels.append(sl)

    all_logits = P.parallel_apply(replicas, inputs_args, inputs_kwargs)

    preds: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    gpu_losses: list[torch.Tensor] = []
    loss_sum_f = 0.0
    examples = 0

    for logits, sm, sl, device in zip(all_logits, masks, all_labels, devices[:n]):
        seed_logits = logits[sm].float()
        w = weight.to(device) if weight is not None else None
        loss = F.cross_entropy(seed_logits, sl, weight=w)
        gpu_losses.append(loss)
        preds.append(seed_logits.detach().argmax(dim=1).cpu().numpy())
        labels_out.append(sl.detach().cpu().numpy())
        batch_n = int(sl.numel())
        loss_sum_f += float(loss.item()) * batch_n
        examples += batch_n

    # Gather losses to primary device, scale by 1/(n_gpus * grad_accum), backward
    avg_loss = sum(lv.to(primary) for lv in gpu_losses) / (n * grad_accum_steps)
    scaler.scale(avg_loss).backward()

    # Aggregate replica gradients to the original model's parameters
    replica_params = [list(r.parameters()) for r in replicas]
    for param_idx, p_model in enumerate(model.parameters()):
        grad_agg: torch.Tensor | None = None
        for rp_list in replica_params:
            p_r = rp_list[param_idx]
            if p_r.grad is not None:
                g = p_r.grad.to(primary)
                grad_agg = g if grad_agg is None else grad_agg.add_(g)
        if grad_agg is not None:
            if p_model.grad is None:
                p_model.grad = grad_agg
            else:
                p_model.grad.add_(grad_agg)

    node_stats: dict[str, list[int]] = {}
    edge_stats: dict[str, list[int]] = {}
    for batch in batch_group[:n]:
        for nt, cnt in batch.stats.get("nodes", {}).items():
            node_stats.setdefault(nt, []).append(int(cnt))
        for et, cnt in batch.stats.get("edges", {}).items():
            edge_stats.setdefault(et, []).append(int(cnt))

    return loss_sum_f, examples, preds, labels_out, node_stats, edge_stats


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


def load_neighbor_backend(config: dict[str, Any]) -> NeighborBackend:
    source = str(config["data"].get("source", "npz")).lower()
    if source in {"on_disk_graph_store", "graph_store_csr", "graph_store"}:
        graph_store_root = Path(config["data"]["graph_store_root"])
        try:
            return OnDiskHeteroGraphStore(graph_store_root)
        except (FileNotFoundError, ValueError):
            if source != "graph_store":
                raise
            artifact = load_graph_store_artifact(
                graph_store_root=graph_store_root,
                packet_feature=str(config["data"]["packet_feature"]),
                add_reverse_edges=bool(config["data"]["add_reverse_edges"]),
                sealed_only=bool(config["data"].get("read_sealed_only", True)),
            )
            return InMemoryNeighborBackend(artifact)

    artifact = load_three_tier_graph_artifact(
        graph_npz=Path(config["data"]["graph_npz"]),
        graph_meta_json=Path(config["data"]["graph_meta_json"]),
        packet_feature=str(config["data"]["packet_feature"]),
        add_reverse_edges=bool(config["data"]["add_reverse_edges"]),
    )
    return InMemoryNeighborBackend(artifact)


def backend_splits(
    backend: NeighborBackend,
    labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_ids = getattr(backend, "split_ids", None)
    if callable(split_ids):
        train = split_ids("train")
        val = split_ids("val")
        test = split_ids("test")
        if train is not None and val is not None and test is not None:
            return (
                np.asarray(train, dtype=np.int64),
                np.asarray(val, dtype=np.int64),
                np.asarray(test, dtype=np.int64),
            )
    return stratified_split(
        labels=labels,
        val_ratio=float(config["train"]["val_ratio"]),
        test_ratio=float(config["train"]["test_ratio"]),
        seed=seed,
    )


def compute_flow_feature_stats_backend(
    backend: NeighborBackend,
    train_ids: np.ndarray,
    chunk_size: int = 65_536,
) -> dict[str, list[float]]:
    train_ids = np.asarray(train_ids, dtype=np.int64)
    dim = int(backend.feature_dims["flow"])
    if train_ids.size == 0:
        return {"mean": [0.0] * dim, "std": [1.0] * dim}

    total = np.zeros((dim,), dtype=np.float64)
    total_sq = np.zeros((dim,), dtype=np.float64)
    count = 0
    for start in range(0, int(train_ids.shape[0]), chunk_size):
        ids = train_ids[start : start + chunk_size]
        rows = backend.get_flow_features(ids).astype(np.float64)
        total += rows.sum(axis=0)
        total_sq += np.square(rows).sum(axis=0)
        count += int(rows.shape[0])
    mean = total / max(count, 1)
    var = np.maximum(total_sq / max(count, 1) - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-6)
    return {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def class_weights_from_backend(
    backend: NeighborBackend,
    train_idx: np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    manifest_weights = (backend.manifest or {}).get("class_weights")
    if isinstance(manifest_weights, list) and len(manifest_weights) >= num_classes:
        return torch.tensor(manifest_weights[:num_classes], dtype=torch.float32)
    labels = backend.get_flow_labels(np.asarray(train_idx, dtype=np.int64))
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (float(num_classes) * counts[nonzero])
    return torch.from_numpy(weights)


def to_torch_batch(
    batch: MiniBatchSubgraph,
    edge_types: list[tuple[str, str, str]],
    device: torch.device,
    use_semantic_edge_weights: bool,
) -> tuple[
    dict[str, torch.Tensor],
    dict[tuple[str, str, str], torch.Tensor],
    dict[tuple[str, str, str], torch.Tensor] | None,
    torch.Tensor,
    torch.Tensor,
]:
    node_tensors = {
        key: torch.from_numpy(np.asarray(value)).to(device)
        for key, value in batch.node_features.items()
    }
    edge_tensors: dict[tuple[str, str, str], torch.Tensor] = {}
    for edge_type in edge_types:
        value = batch.edge_index.get(edge_type)
        if value is None:
            value = np.empty((2, 0), dtype=np.int64)
        edge_tensors[edge_type] = torch.from_numpy(np.asarray(value, dtype=np.int64)).to(device)

    edge_weights: dict[tuple[str, str, str], torch.Tensor] = {}
    if use_semantic_edge_weights:
        for edge_type in edge_types:
            if "matches_technique" not in edge_type[1]:
                continue
            values = np.asarray(batch.edge_attr.get(edge_type, np.empty((0,), dtype=np.float32)), dtype=np.float32)
            edge_weights[edge_type] = torch.from_numpy(values.reshape(-1)).to(device)

    seed_mask = torch.from_numpy(np.asarray(batch.seed_mask, dtype=bool)).to(device)
    seed_labels = torch.from_numpy(np.asarray(batch.seed_labels, dtype=np.int64)).to(device)
    return node_tensors, edge_tensors, edge_weights or None, seed_mask, seed_labels


def metrics_from_predictions(
    pred_np: np.ndarray,
    label_np: np.ndarray,
    num_classes: int,
    label_names: dict[int, str],
    loss_sum: float | None = None,
) -> dict[str, Any]:
    if label_np.size == 0:
        return {"count": 0, "loss": None, "accuracy": None, "macro_f1": None, "per_class": {}}

    correct = int((pred_np == label_np).sum())
    count = int(label_np.shape[0])
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
        "loss": float(loss_sum / max(count, 1)) if loss_sum is not None else None,
        "accuracy": float(correct / max(count, 1)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def make_neighbor_loader(
    flow_ids: np.ndarray,
    sampler: HeteroNeighborSampler,
    config: dict[str, Any],
    *,
    shuffle: bool,
) -> DataLoader:
    num_workers = int(config.get("dataloader", {}).get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(config["train"].get("batch_seed_flows", 256)),
        "shuffle": bool(shuffle),
        "collate_fn": NeighborSamplingCollator(sampler),
        "num_workers": num_workers,
        "pin_memory": bool(config.get("dataloader", {}).get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.get("dataloader", {}).get("prefetch_factor", 2))
    return DataLoader(FlowSeedDataset(flow_ids), **loader_kwargs)


def evaluate_neighbor_sampling(
    *,
    model: HeteroGraphTransformer,
    loader: DataLoader,
    edge_types: list[tuple[str, str, str]],
    device: torch.device,
    use_amp: bool,
    use_semantic_edge_weights: bool,
    num_classes: int,
    label_names: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            node_features, edge_index, edge_weight, seed_mask, seed_labels = to_torch_batch(
                batch,
                edge_types,
                device,
                use_semantic_edge_weights,
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
                seed_logits = logits[seed_mask]
                loss = F.cross_entropy(seed_logits.float(), seed_labels, reduction="sum")
            loss_sum += float(loss.item())
            preds.append(seed_logits.detach().float().argmax(dim=1).cpu().numpy())
            labels.append(seed_labels.detach().cpu().numpy())
    pred_np = np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)
    label_np = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)
    return metrics_from_predictions(pred_np, label_np, num_classes, label_names, loss_sum)


def train_neighbor_sampling(config: dict[str, Any], seed: int, device: torch.device) -> None:
    multi_gpu = bool(config["train"].get("multi_gpu", True))
    devices = get_training_devices(device, multi_gpu)
    n_gpus = len(devices)
    if n_gpus > 1:
        print(f"[Multi-GPU] Using {n_gpus} GPUs: {', '.join(str(d) for d in devices)}")

    backend = load_neighbor_backend(config)
    all_flow_ids = np.arange(int(backend.num_flows), dtype=np.int64)
    labels_np = backend.get_flow_labels(all_flow_ids)
    if labels_np.size == 0:
        raise ValueError("Cannot train HGT: graph has no flow labels.")
    num_classes = int(labels_np.max()) + 1
    train_idx_np, val_idx_np, test_idx_np = backend_splits(backend, labels_np, config, seed)

    flow_feature_stats: dict[str, list[float]] | None = None
    if bool(config["data"]["standardize_flow_features"]):
        manifest_stats = (backend.manifest or {}).get("flow_feature_stats")
        if isinstance(manifest_stats, dict) and "mean" in manifest_stats and "std" in manifest_stats:
            flow_feature_stats = {
                "mean": list(manifest_stats["mean"]),
                "std": list(manifest_stats["std"]),
            }
        else:
            flow_feature_stats = compute_flow_feature_stats_backend(backend, train_idx_np)

    sampler_cfg = dict(config.get("sampler") or {})
    sampler_hops = sampler_cfg.get("hops")
    if sampler_hops is None:
        sampler_hops = int(config["model"]["num_layers"])
    sampler = HeteroNeighborSampler(
        backend,
        hops=int(sampler_hops),
        fanouts=dict(sampler_cfg.get("fanouts") or {}),
        reverse_fanouts=dict(sampler_cfg.get("reverse_fanouts") or {}),
        always_include_all_tactics=bool(sampler_cfg.get("always_include_all_tactics", True)),
        always_include_all_techniques=bool(sampler_cfg.get("always_include_all_techniques", True)),
        flow_feature_stats=flow_feature_stats,
        standardize_flow_features=bool(config["data"]["standardize_flow_features"]),
        seed=seed,
    )
    train_loader = make_neighbor_loader(train_idx_np, sampler, config, shuffle=True)
    val_loader = make_neighbor_loader(val_idx_np, sampler, config, shuffle=False)
    test_loader = make_neighbor_loader(test_idx_np, sampler, config, shuffle=False)

    edge_types = list(backend.edge_types)
    node_input_dims = {
        "flow": int(backend.feature_dims["flow"]),
        "packet": int(backend.feature_dims["packet"]),
        "technique": int(backend.feature_dims["technique"]),
    }
    model = HeteroGraphTransformer(
        node_input_dims=node_input_dims,
        edge_types=edge_types,
        num_classes=num_classes,
        num_tactics=int(backend.num_tactics),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        num_heads=int(config["model"]["num_heads"]),
        dropout=float(config["model"]["dropout"]),
        ffn_multiplier=int(config["model"]["ffn_multiplier"]),
        activation_checkpointing=bool(config["train"].get("activation_checkpointing", False)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    weight = None
    if str(config["train"]["class_weight"]).lower() == "balanced":
        weight = class_weights_from_backend(backend, train_idx_np, num_classes).to(device)

    output_dir = ensure_dir(Path(config["train"]["output_dir"]))
    best_checkpoint = output_dir / "hgt_flow_best.pt"
    label_names = label_name_mapping(backend.manifest, labels_np)
    monitor = str(config["train"]["monitor"])
    log_every = max(1, int(config["train"]["log_every"]))
    use_amp = bool(config["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = max(1, int(config["train"].get("grad_accum_steps", 1)))
    use_semantic_edge_weights = bool(config["data"]["use_semantic_edge_weights"])

    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        preds: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        loss_sum = 0.0
        examples = 0
        pending_step = False
        sampled_nodes: dict[str, list[int]] = {"flow": [], "packet": [], "technique": [], "tactic": []}
        sampled_edges: dict[str, list[int]] = {}

        train_iter = iter(train_loader)
        step = 0
        while True:
            # Collect one batch per GPU to maximise parallelism
            batch_group: list[MiniBatchSubgraph] = []
            for _ in range(n_gpus):
                try:
                    batch_group.append(next(train_iter))
                except StopIteration:
                    break
            if not batch_group:
                break

            if n_gpus > 1 and len(batch_group) > 1:
                ls, ex, bp, bl, ns, es = _multi_gpu_train_step(
                    model, batch_group, devices, edge_types, weight,
                    scaler, use_semantic_edge_weights, grad_accum_steps,
                )
                loss_sum += ls
                examples += ex
                preds.extend(bp)
                labels.extend(bl)
                for nt, cnts in ns.items():
                    sampled_nodes.setdefault(nt, []).extend(cnts)
                for et, cnts in es.items():
                    sampled_edges.setdefault(et, []).extend(cnts)
            else:
                batch = batch_group[0]
                node_features, edge_index, edge_weight, seed_mask, seed_labels = to_torch_batch(
                    batch, edge_types, device, use_semantic_edge_weights,
                )
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
                    seed_logits = logits[seed_mask]
                    loss = F.cross_entropy(seed_logits, seed_labels, weight=weight)
                    scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
                batch_count = int(seed_labels.numel())
                loss_sum += float(loss.detach().item()) * batch_count
                examples += batch_count
                preds.append(seed_logits.detach().float().argmax(dim=1).cpu().numpy())
                labels.append(seed_labels.detach().cpu().numpy())
                for node_type, count in batch.stats.get("nodes", {}).items():
                    sampled_nodes.setdefault(node_type, []).append(int(count))
                for edge_name, count in batch.stats.get("edges", {}).items():
                    sampled_edges.setdefault(edge_name, []).append(int(count))

            pending_step = True
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_step = False
            step += 1

        if pending_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_pred = np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)
        train_label = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)
        train_metrics = metrics_from_predictions(
            train_pred,
            train_label,
            num_classes,
            label_names,
            loss_sum if examples else None,
        )
        val_metrics = evaluate_neighbor_sampling(
            model=model,
            loader=val_loader,
            edge_types=edge_types,
            device=device,
            use_amp=use_amp,
            use_semantic_edge_weights=use_semantic_edge_weights,
            num_classes=num_classes,
            label_names=label_names,
        )
        test_metrics = evaluate_neighbor_sampling(
            model=model,
            loader=test_loader,
            edge_types=edge_types,
            device=device,
            use_amp=use_amp,
            use_semantic_edge_weights=use_semantic_edge_weights,
            num_classes=num_classes,
            label_names=label_names,
        )

        avg_nodes = {
            node_type: float(np.mean(values)) if values else 0.0
            for node_type, values in sampled_nodes.items()
        }
        avg_edges = {
            edge_name: float(np.mean(values)) if values else 0.0
            for edge_name, values in sampled_edges.items()
        }
        entry = {
            "epoch": epoch,
            "train": {key: value for key, value in train_metrics.items() if key != "per_class"},
            "val": {key: value for key, value in val_metrics.items() if key != "per_class"},
            "test": {key: value for key, value in test_metrics.items() if key != "per_class"},
            "sampler": {
                "avg_subgraph_nodes": avg_nodes,
                "avg_subgraph_edges": avg_edges,
            },
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
                    "edge_types": [list(edge_key) for edge_key in edge_types],
                    "num_classes": num_classes,
                    "num_tactics": int(backend.num_tactics),
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
                f"loss={float(train_metrics['loss'] or 0.0):.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"test_macro_f1={test_metrics['macro_f1']:.4f} "
                f"avg_flow_nodes={avg_nodes.get('flow', 0.0):.1f} "
                f"avg_packet_nodes={avg_nodes.get('packet', 0.0):.1f}"
            )

        if epochs_without_improvement >= int(config["train"]["patience"]):
            print(f"Early stopping at epoch {epoch}.")
            break

    best_payload = load_checkpoint(best_checkpoint, device)
    device_str = str(device) + (f" x{n_gpus}" if n_gpus > 1 else "")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "device": device_str,
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "num_flows": int(backend.num_flows),
        "num_techniques": int(backend.num_techniques),
        "num_tactics": int(backend.num_tactics),
        "num_edge_types": int(len(edge_types)),
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
    batch_mode = str(config["train"]["batch_mode"]).lower()
    if batch_mode in {"neighbor_sampling", "neighbor", "mini_batch", "minibatch"}:
        train_neighbor_sampling(config, seed, device)
        return
    if batch_mode != "full":
        raise ValueError("train.batch_mode must be 'full' or 'neighbor_sampling'.")

    if str(config["data"].get("source", "npz")).lower() == "graph_store":
        artifact = load_graph_store_artifact(
            graph_store_root=Path(config["data"]["graph_store_root"]),
            packet_feature=str(config["data"]["packet_feature"]),
            add_reverse_edges=bool(config["data"]["add_reverse_edges"]),
            sealed_only=bool(config["data"].get("read_sealed_only", True)),
        )
    else:
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
        activation_checkpointing=bool(config["train"].get("activation_checkpointing", False)),
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
    use_amp = bool(config["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_eval = model(node_features, edge_index, edge_weight_dict=edge_weight)
            logits_eval = logits_eval.float()
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
