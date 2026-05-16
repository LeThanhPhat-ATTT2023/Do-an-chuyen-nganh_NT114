from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from graphslm_ids.utils.io import read_json


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
    packet_semantic_npy: Path | None = None,
) -> HeteroGraphArtifact:
    """Load the three-tier graph NPZ into node/edge dictionaries for HGT training."""
    graph_npz = Path(graph_npz)
    if graph_meta_json is None:
        graph_meta_json = graph_npz.with_suffix(".meta.json")
    graph_meta_json = Path(graph_meta_json)

    # Load metadata first so packet_semantic_x_npy path recorded at build-time is available.
    metadata: dict[str, Any] = {}
    if graph_meta_json.exists():
        metadata = read_json(graph_meta_json)

    with np.load(graph_npz, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    # packet_semantic_x may be stored as a separate mmap .npy sidecar (not inside the
    # NPZ zip) to avoid the ~59 GB decompression cost.  Try several candidate paths:
    # 1. explicit caller-supplied path, 2. path recorded in meta.json at build time,
    # 3. conventional sidecar next to NPZ with stem suffix "_packet_semantic_x.npy".
    if "packet_semantic_x" not in arrays:
        sidecar_candidates: list[Path] = []
        if packet_semantic_npy is not None:
            sidecar_candidates.append(Path(packet_semantic_npy))
        meta_sidecar = metadata.get("packet_semantic_x_npy")
        if meta_sidecar:
            sidecar_candidates.append(Path(meta_sidecar))
        sidecar_candidates.append(graph_npz.with_name(graph_npz.stem + "_packet_semantic_x.npy"))
        for candidate in sidecar_candidates:
            if candidate.exists():
                arrays["packet_semantic_x"] = np.load(str(candidate), mmap_mode="r")
                break

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


def load_graph_store_artifact(
    graph_store_root: Path,
    packet_feature: str = "semantic",
    add_reverse_edges: bool = True,
    sealed_only: bool = True,
) -> HeteroGraphArtifact:
    """Load a graph store through the legacy full-batch artifact API."""
    manifest_path = Path(graph_store_root) / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("layout") == "numpy_memmap_csr":
            return _load_on_disk_graph_store_artifact(Path(graph_store_root), add_reverse_edges)

    from graphslm_ids.runtime.pipeline.graph_store import PersistentGraphStore

    store = PersistentGraphStore(graph_store_root)
    arrays, metadata = store.to_three_tier_arrays(sealed_only=sealed_only)

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
    num_tactics = int(metadata.get("num_tactics", 0))
    if num_tactics == 0 and technique_tactic_edge_index.shape[1] > 0:
        num_tactics = int(technique_tactic_edge_index[1].max()) + 1

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
        edge_attr[edge_key] = _maybe_edge_attr(arrays, attr_name or "", int(edge.shape[1]))
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


def _load_on_disk_graph_store_artifact(
    graph_store_root: Path,
    add_reverse_edges: bool,
) -> HeteroGraphArtifact:
    from graphslm_ids.offline.training.on_disk_graph_store import OnDiskHeteroGraphStore

    store = OnDiskHeteroGraphStore(graph_store_root)
    flow_ids = np.arange(store.num_flows, dtype=np.int64)
    packet_ids = np.arange(store.node_counts["packet"], dtype=np.int64)
    tactic_ids = np.arange(store.num_tactics, dtype=np.int64)

    node_features = {
        "flow": store.get_flow_features(flow_ids),
        "packet": store.get_packet_features(packet_ids),
        "technique": store.get_technique_features(),
        "tactic": tactic_ids[:, None],
    }
    edge_index = {edge_key: store.get_edge_index(edge_key) for edge_key in store.edge_types}
    edge_attr = {edge_key: store.get_edge_attr(edge_key) for edge_key in store.edge_types}

    if add_reverse_edges:
        for edge_key, edge in list(edge_index.items()):
            if edge_key[1].startswith("rev_"):
                continue
            src_type, relation, dst_type = edge_key
            reverse_key = (dst_type, f"rev_{relation}", src_type)
            if reverse_key in edge_index:
                continue
            edge_index[reverse_key] = _reverse_edge_index(edge)
            edge_attr[reverse_key] = edge_attr[edge_key].copy()

    return HeteroGraphArtifact(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        flow_y=store.get_flow_labels(flow_ids),
        metadata=store.manifest,
    )
