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


def load_v2_artifact(
    graph_npz: Path,
    graph_meta_json: Path | None = None,
    add_reverse_edges: bool = True,
) -> HeteroGraphArtifact:
    """Load a v2 evidence-grounded graph artifact (artifact_version='v2').

    Returns the same ``HeteroGraphArtifact`` dataclass as
    :func:`load_three_tier_graph_artifact` so the rest of the training pipeline
    (neighbor sampler, HGT, evaluator) needs no per-version code paths.

    Differences vs v1:
      * ``flow_x`` carries ~80 CICFlowMeter-style features (instead of 6)
      * ``packet_x`` carries 2323-dim deterministic payload features (instead
        of either 256B raw bytes or 768-d SecureBERT embedding)
      * ``matches_technique`` edges are evidence-weighted (from OWASP CRS +
        flow signatures), not cosine-thresholded

    The flow_y label encoding is preserved through metadata['label_mapping']
    exactly as v1 — so monitor metrics and label_name lookups continue to
    work without changes in the trainer.
    """
    graph_npz = Path(graph_npz)
    if graph_meta_json is None:
        graph_meta_json = graph_npz.with_suffix(".meta.json")
    graph_meta_json = Path(graph_meta_json)

    metadata: dict[str, Any] = {}
    if graph_meta_json.exists():
        metadata = read_json(graph_meta_json)

    with np.load(graph_npz, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    if metadata.get("artifact_version") != "v2":
        # Not a fatal mismatch -- the caller might point us at a v1 artifact by
        # accident. Surface a clear error rather than producing silently wrong
        # features.
        raise ValueError(
            "load_v2_artifact called but metadata.artifact_version is "
            f"{metadata.get('artifact_version')!r}; expected 'v2'."
        )

    flow_x = np.asarray(_require_array(arrays, "flow_x"), dtype=np.float32)
    flow_y = np.asarray(_require_array(arrays, "flow_y"), dtype=np.int64)
    packet_x = np.asarray(_require_array(arrays, "packet_x"), dtype=np.float32)
    technique_x = np.asarray(_require_array(arrays, "technique_x"), dtype=np.float32)

    technique_tactic_edge_index = np.asarray(
        _require_array(arrays, "technique_tactic_edge_index"), dtype=np.int64
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
        edge_attr[edge_key] = _maybe_edge_attr(
            arrays, attr_name or "", int(edge.shape[1])
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


def load_v3_artifact(
    graph_npz: Path,
    graph_meta_json: Path | None = None,
    add_reverse_edges: bool = True,
) -> HeteroGraphArtifact:
    """Load a v3 Smart-BOTH Hybrid graph artifact (``artifact_version='v3'``).

    Parallel to :func:`load_v2_artifact` but introduces a 5-node-type schema:
    flow, packet, host, technique, tactic. Edge types are typed by attack family
    (injection / command_exec / file_upload / recon / c2_beacon) replacing the
    single collapsed ``packet -> matches_technique -> technique`` of v2.

    v3 schema differences vs v2:
      * NEW ``host`` node type — 4-d aggregate features from flow src/dst IPs
      * NEW ``flow -> burst_neighbor -> flow`` homophily edges
      * NEW ``flow -> from_host -> host`` / ``flow -> to_host -> host`` edges
      * NEW ``technique -> has_subtechnique -> technique`` hierarchy edges
      * The single ``packet -> matches_technique -> technique`` of v2 is
        REPLACED by FIVE typed ``packet -> evidence_{family} -> technique``
        edges (one per attack family). HGT can now learn per-family attention.

    Missing optional edge keys in the NPZ are tolerated: we emit empty
    ``(2, 0)`` int64 arrays so tiny test fixtures load without crashing.
    Required: ``flow_x``, ``flow_y``, ``packet_x``, ``technique_x``, ``host_x``.

    Returns the same :class:`HeteroGraphArtifact` shape as ``load_v2_artifact``
    so the rest of the pipeline (sampler, HGT model) is unchanged.
    """
    graph_npz = Path(graph_npz)
    if graph_meta_json is None:
        graph_meta_json = graph_npz.with_suffix(".meta.json")
    graph_meta_json = Path(graph_meta_json)

    metadata: dict[str, Any] = {}
    if graph_meta_json.exists():
        metadata = read_json(graph_meta_json)

    with np.load(graph_npz, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    if metadata.get("artifact_version") != "v3":
        raise ValueError(
            "load_v3_artifact called but metadata.artifact_version is "
            f"{metadata.get('artifact_version')!r}; expected 'v3'."
        )

    flow_x = np.asarray(_require_array(arrays, "flow_x"), dtype=np.float32)
    flow_y = np.asarray(_require_array(arrays, "flow_y"), dtype=np.int64)
    packet_x = np.asarray(_require_array(arrays, "packet_x"), dtype=np.float32)
    host_x = np.asarray(_require_array(arrays, "host_x"), dtype=np.float32)
    technique_x = np.asarray(_require_array(arrays, "technique_x"), dtype=np.float32)

    # Tactic feature derivation matches v2: integer id column, sized from
    # metadata or from the max tactic index touched by belongs_to_tactic edges.
    if "technique_tactic_edge_index" in arrays:
        technique_tactic_edge_index = np.asarray(
            arrays["technique_tactic_edge_index"], dtype=np.int64
        )
    else:
        technique_tactic_edge_index = np.zeros((2, 0), dtype=np.int64)
    num_tactics = int(metadata.get("num_tactics", 0))
    if num_tactics == 0 and technique_tactic_edge_index.shape[1] > 0:
        num_tactics = int(technique_tactic_edge_index[1].max()) + 1

    node_features = {
        "flow": flow_x,
        "packet": packet_x,
        "host": host_x,
        "technique": technique_x,
        "tactic": _empty_tactic_features(num_tactics),
    }

    # (edge_key, edge_index_array_name, edge_attr_array_name_or_None)
    # Missing keys are tolerated (emit empty edges).
    edge_specs: list[tuple[EdgeKey, str, str | None]] = [
        # Structural
        (("flow", "contains", "packet"), "contain_edge_index", None),
        (("packet", "next_packet", "packet"), "link_edge_index", "link_edge_attr"),
        (("flow", "from_host", "host"), "from_host_edge_index", "from_host_edge_attr"),
        (("flow", "to_host", "host"), "to_host_edge_index", "to_host_edge_attr"),
        # Flow homophily
        (
            ("flow", "burst_neighbor", "flow"),
            "burst_neighbor_edge_index",
            "burst_neighbor_edge_attr",
        ),
        # Typed evidence (5 attack families) — PRIMARY v3 contribution
        (
            ("packet", "evidence_injection", "technique"),
            "evidence_injection_edge_index",
            "evidence_injection_edge_attr",
        ),
        (
            ("packet", "evidence_command_exec", "technique"),
            "evidence_command_exec_edge_index",
            "evidence_command_exec_edge_attr",
        ),
        (
            ("packet", "evidence_file_upload", "technique"),
            "evidence_file_upload_edge_index",
            "evidence_file_upload_edge_attr",
        ),
        (
            ("packet", "evidence_recon", "technique"),
            "evidence_recon_edge_index",
            "evidence_recon_edge_attr",
        ),
        (
            ("packet", "evidence_c2_beacon", "technique"),
            "evidence_c2_beacon_edge_index",
            "evidence_c2_beacon_edge_attr",
        ),
        # Knowledge
        (
            ("flow", "matches_technique", "technique"),
            "flow_technique_edge_index",
            "flow_technique_edge_attr",
        ),
        (
            ("technique", "has_subtechnique", "technique"),
            "has_subtechnique_edge_index",
            None,
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
        if edge_name in arrays:
            edge = np.asarray(arrays[edge_name], dtype=np.int64)
            if edge.ndim != 2 or edge.shape[0] != 2:
                raise ValueError(f"{edge_name} must have shape (2, num_edges).")
        else:
            # Tolerated absent edge — small fixtures / ablation runs.
            edge = np.zeros((2, 0), dtype=np.int64)
        edge_index[edge_key] = edge
        edge_attr[edge_key] = _maybe_edge_attr(
            arrays, attr_name or "", int(edge.shape[1])
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
