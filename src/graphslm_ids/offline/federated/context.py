"""Experiment data context: the v3 artifact, splits, labels and clean key, loaded once.

The context depends only on ``cfg["data"]`` (fingerprinted by :func:`data_key`), so a
matrix runner can reuse one context across many runs. ``client_cache`` / ``cache_tag``
belong to the simulation (it caches per-client model data there); nothing in this module
writes to them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from graphslm_ids.offline.federated.client_graph import packet_owner
from graphslm_ids.offline.federated.records import to_jsonable
from graphslm_ids.offline.training.hetero_graph_artifact import HeteroGraphArtifact, load_v3_artifact
from graphslm_ids.offline.training.neighbor_sampling import InMemoryNeighborBackend
from graphslm_ids.offline.training.train_hgt_flow_classifier import backend_splits, label_name_mapping


@dataclass
class ExperimentContext:
    data_key: str
    artifact: HeteroGraphArtifact
    backend: InMemoryNeighborBackend
    labels: np.ndarray
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    label_names: dict[int, str]
    num_classes: int
    clean_labels: np.ndarray | None
    packet_owner: np.ndarray
    flow_order: np.ndarray | None
    client_cache: dict[int, Any] = field(default_factory=dict)
    cache_tag: str | None = None


def data_key(cfg: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(cfg["data"]), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def require_file(value: Any, key: str) -> Path:
    """``value`` as an existing file path; a missing or unset path is a config error."""
    if not value:
        raise ValueError(f"{key} must be set")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{key} points to {str(path)!r}, which is not an existing file")
    return path


def flow_start_times(artifact: HeteroGraphArtifact) -> np.ndarray | None:
    """Per-flow start time (float64) used to order flows in the explicit partition, or None.

    v3 keeps it as the ``flow_start_ts`` column of ``flow_x`` (named in ``metadata
    ["flow_feature_names"]``); a ``metadata["flow_start_ts"]`` array is the fallback. ``flow_x`` is
    float32, so unix-second timestamps resolve to about two minutes; ties fall back to flow id.
    """
    metadata = artifact.metadata or {}
    names = list(metadata.get("flow_feature_names") or [])
    if "flow_start_ts" in names:
        return np.asarray(artifact.node_features["flow"][:, names.index("flow_start_ts")], dtype=np.float64)
    if "flow_start_ts" in metadata:
        return np.asarray(metadata["flow_start_ts"], dtype=np.float64)
    return None


def load_context(cfg: dict[str, Any]) -> ExperimentContext:
    data = cfg["data"]
    if str(data.get("artifact_version", "")).lower() != "v3":
        raise ValueError("the FL testbed requires data.artifact_version: v3")
    # A missing splits file would make backend_splits fall back to a seeded random split
    # that data_key() does not capture, so it is required here.
    require_file(data.get("splits_json"), "data.splits_json")
    artifact = load_v3_artifact(
        require_file(data.get("graph_npz"), "data.graph_npz"),
        require_file(data.get("graph_meta_json"), "data.graph_meta_json"),
        add_reverse_edges=bool(data.get("add_reverse_edges", True)),
        packet_dtype=str(data.get("packet_dtype", "float32")),
    )
    backend = InMemoryNeighborBackend(artifact)
    labels = np.asarray(artifact.flow_y, dtype=np.int64)
    split_cfg = {"data": data, "train": {"val_ratio": 0.1, "test_ratio": 0.1}}
    # Fixed seed: the context must be a function of cfg["data"] alone (see data_key).
    train_ids, val_ids, test_ids = backend_splits(backend, labels, split_cfg, 42)
    label_names = label_name_mapping(backend.manifest, labels)
    num_classes = max(len(label_names), int(labels.max()) + 1 if labels.size else 0)

    clean = None
    if data.get("clean_eval_labels"):
        arr = np.load(require_file(data["clean_eval_labels"], "data.clean_eval_labels"))
        if arr.shape != labels.shape:
            raise ValueError(
                f"clean_eval_labels has shape {arr.shape} but the artifact has {labels.shape[0]} flows"
            )
        clean = np.asarray(arr, dtype=np.int64)
        if clean.size and (clean.min() < 0 or clean.max() >= num_classes):
            raise ValueError(f"clean_eval_labels entries must lie in [0, {num_classes})")

    flow_order = flow_start_times(artifact)
    if flow_order is not None and flow_order.shape != labels.shape:
        raise ValueError(f"flow_start_ts has shape {flow_order.shape}, expected ({labels.shape[0]},)")

    return ExperimentContext(
        data_key=data_key(cfg),
        artifact=artifact,
        backend=backend,
        labels=labels,
        train_ids=np.asarray(train_ids, dtype=np.int64),
        val_ids=np.asarray(val_ids, dtype=np.int64),
        test_ids=np.asarray(test_ids, dtype=np.int64),
        label_names=label_names,
        num_classes=int(num_classes),
        clean_labels=clean,
        packet_owner=packet_owner(artifact),
        flow_order=flow_order,
    )


def federated_flow_stats(flow_x: np.ndarray, client_flow_ids: list[np.ndarray]) -> dict[str, list[float]]:
    """Round-0 protocol message: each client sends ``(n, sum x, sum x^2)`` of its own
    ``flow_x`` rows; the server pools them into a mean/std (float64, std floored at 1e-6)."""
    dim = int(flow_x.shape[1])
    n = 0
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)
    for ids in client_flow_ids:
        rows = np.asarray(flow_x[np.asarray(ids, dtype=np.int64)], dtype=np.float64)
        n += int(rows.shape[0])
        total += rows.sum(axis=0)
        total_sq += np.square(rows).sum(axis=0)
    if n == 0:
        # Same convention as the centralized trainer for an empty training set.
        return {"mean": [0.0] * dim, "std": [1.0] * dim}
    mean = total / n
    std = np.maximum(np.sqrt(np.maximum(total_sq / n - np.square(mean), 0.0)), 1e-6)
    return {"mean": mean.tolist(), "std": std.tolist()}
