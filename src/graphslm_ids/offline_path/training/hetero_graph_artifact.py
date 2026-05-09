from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


EdgeKey = tuple[str, str, str]


@dataclass
class HeteroGraphArtifact:
    node_features: dict[str, np.ndarray]
    edge_index: dict[EdgeKey, np.ndarray]
    edge_attr: dict[EdgeKey, np.ndarray]
    flow_y: np.ndarray
    metadata: dict[str, Any]


def _require_array(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"Graph artifact is missing required array: {key}")
    return arrays[key]


def _maybe_edge_attr(arrays: dict[str, np.ndarray], key: str, num_edges: int) -> np.ndarray:
    if key in arrays:
        attr = np.asarray(arrays[key], dtype=np.float32)
        if attr.ndim == 1:
            attr = attr[:, None]
        if attr.shape[0] != num_edges:
            raise ValueError(f"Edge attr {key} row count does not match edge count.")
        return attr
    return np.ones((num_edges, 1), dtype=np.float32)


def _reverse_edge_index(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    return edge_index[[1, 0], :]


def _empty_tactic_features(num_tactics: int) -> np.ndarray:
    return np.arange(num_tactics, dtype=np.int64)[:, None]


def load_three_tier_graph_artifact(
    graph_npz: Path,
    graph_meta_json: Path | None = None,
    packet_feature: str = "semantic",
    add_reverse_edges: bool = True,
) -> HeteroGraphArtifact:
    """Load the three-tier graph NPZ into node/edge dictionaries for HGT training."""
    graph_npz = Path(graph_npz)
    if graph_meta_json is None:
        graph_meta_json = graph_npz.with_suffix(".meta.json")
    graph_meta_json = Path(graph_meta_json)

    with np.load(graph_npz, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    metadata: dict[str, Any] = {}
    if graph_meta_json.exists():
        with graph_meta_json.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    flow_x = np.asarray(_require_array(arrays, "flow_x"), dtype=np.float32)
    flow_y = np.asarray(_require_array(arrays, "flow_y"), dtype=np.int64)

    if packet_feature == "semantic":
        packet_x = np.asarray(_require_array(arrays, "packet_semantic_x"), dtype=np.float32)
    elif packet_feature == "payload":
        packet_x = np.asarray(_require_array(arrays, "packet_x"), dtype=np.float32) / 255.0
    else:
        raise ValueError("packet_feature must be either 'semantic' or 'payload'.")

    technique_x = np.asarray(_require_array(arrays, "technique_x"), dtype=np.float32)

    technique_tactic_edge_index = np.asarray(
        _require_array(arrays, "technique_tactic_edge_index"),
        dtype=np.int64,
    )
    if "num_tactics" in metadata:
        num_tactics = int(metadata["num_tactics"])
    elif technique_tactic_edge_index.shape[1] > 0:
        num_tactics = int(technique_tactic_edge_index[1].max()) + 1
    else:
        num_tactics = 0

    node_features = {
        "flow": flow_x,
        "packet": packet_x,
        "technique": technique_x,
        "tactic": _empty_tactic_features(num_tactics),
    }

    edge_specs: list[tuple[EdgeKey, str, str | None]] = [
        (("flow", "contains", "packet"), "contain_edge_index", None),
        (("packet", "next_packet", "packet"), "link_edge_index", "link_edge_attr"),
        (
            ("packet", "matches_technique", "technique"),
            "packet_technique_edge_index",
            "packet_technique_edge_attr",
        ),
        (
            ("flow", "matches_technique", "technique"),
            "flow_technique_edge_index",
            "flow_technique_edge_attr",
        ),
        (
            ("technique", "belongs_to_tactic", "tactic"),
            "technique_tactic_edge_index",
            "technique_tactic_edge_attr",
        ),
    ]

    edge_index: dict[EdgeKey, np.ndarray] = {}
    edge_attr: dict[EdgeKey, np.ndarray] = {}
    for edge_key, edge_name, attr_name in edge_specs:
        edge = np.asarray(_require_array(arrays, edge_name), dtype=np.int64)
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f"{edge_name} must have shape (2, num_edges).")
        edge_index[edge_key] = edge
        edge_attr[edge_key] = _maybe_edge_attr(
            arrays,
            attr_name or "",
            int(edge.shape[1]),
        )

        if add_reverse_edges:
            src_type, relation, dst_type = edge_key
            reverse_key = (dst_type, f"rev_{relation}", src_type)
            edge_index[reverse_key] = _reverse_edge_index(edge)
            edge_attr[reverse_key] = edge_attr[edge_key].copy()

    return HeteroGraphArtifact(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        flow_y=flow_y,
        metadata=metadata,
    )
