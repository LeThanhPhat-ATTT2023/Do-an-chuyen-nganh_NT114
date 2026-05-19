from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from graphslm_ids.offline.training.hetero_graph_artifact import (
    HeteroGraphArtifact,
    load_three_tier_graph_artifact,
)
from graphslm_ids.utils.io import read_json


EdgeKey = tuple[str, str, str]


def edge_key_to_name(edge_key: EdgeKey) -> str:
    return "__".join(edge_key)


def edge_name_to_key(edge_name: str) -> EdgeKey:
    parts = str(edge_name).split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid edge relation name: {edge_name!r}")
    return (parts[0], parts[1], parts[2])


class OnDiskHeteroGraphStore:
    """Memory-mapped CSR graph store for scalable HGT training.

    This is the training-side store described in the scalable HGT design: large
    flow/packet arrays and relation CSR arrays stay on disk, while small static
    knowledge tensors are cheap to read eagerly through the same API.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Graph store manifest not found: {manifest_path}")
        self.manifest: dict[str, Any] = read_json(manifest_path)

        layout = str(self.manifest.get("layout", ""))
        if layout != "numpy_memmap_csr":
            raise ValueError(
                "Unsupported graph store layout for training sampler: "
                f"{layout!r}. Expected 'numpy_memmap_csr'."
            )

        self.node_counts = {str(k): int(v) for k, v in self.manifest["node_counts"].items()}
        self.feature_dims = {str(k): int(v) for k, v in self.manifest["feature_dims"].items()}
        self.edge_types = [edge_name_to_key(name) for name in self.manifest["edge_files"]]

        self.flow_features = self._memmap_node_features(
            self.manifest["node_files"]["flow"]["features"],
            (self.node_counts["flow"], self.feature_dims["flow"]),
        )
        self.flow_labels = self._memmap(
            self.manifest["node_files"]["flow"]["labels"],
            np.int64,
            (self.node_counts["flow"],),
        )
        self.flow_shard_index = self._optional_memmap(
            self.manifest["node_files"]["flow"].get("shard_index"),
            np.int64,
            (self.node_counts["flow"],),
        )
        self.packet_features = self._memmap_node_features(
            self.manifest["node_files"]["packet"]["features"],
            (self.node_counts["packet"], self.feature_dims["packet"]),
        )
        self.packet_shard_index = self._optional_memmap(
            self.manifest["node_files"]["packet"].get("shard_index"),
            np.int64,
            (self.node_counts["packet"],),
        )
        self.technique_features = self._memmap_node_features(
            self.manifest["node_files"]["technique"]["features"],
            (self.node_counts["technique"], self.feature_dims["technique"]),
        )
        self.tactic_ids = self._memmap(
            self.manifest["node_files"]["tactic"]["ids"],
            np.int64,
            (self.node_counts["tactic"],),
        )

        self._edge_csr: dict[EdgeKey, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for edge_name, files in self.manifest["edge_files"].items():
            edge_key = edge_name_to_key(edge_name)
            num_src = self.node_counts[edge_key[0]]
            num_edges = int(self.manifest["edge_counts"].get(edge_name, 0))
            indptr = self._memmap(files["indptr"], np.int64, (num_src + 1,))
            indices = self._memmap(files["indices"], np.int64, (num_edges,))
            attr = self._memmap(files["attr"], np.float32, (num_edges,))
            self._edge_csr[edge_key] = (indptr, indices, attr)

    @property
    def num_flows(self) -> int:
        return self.node_counts["flow"]

    @property
    def num_tactics(self) -> int:
        return self.node_counts["tactic"]

    @property
    def num_techniques(self) -> int:
        return self.node_counts["technique"]

    def get_flow_features(self, flow_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.flow_features[np.asarray(flow_ids, dtype=np.int64)], dtype=np.float32)

    def get_packet_features(self, packet_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.packet_features[np.asarray(packet_ids, dtype=np.int64)], dtype=np.float32)

    def get_technique_features(self, technique_ids: np.ndarray | None = None) -> np.ndarray:
        if technique_ids is None:
            return np.asarray(self.technique_features, dtype=np.float32)
        return np.asarray(
            self.technique_features[np.asarray(technique_ids, dtype=np.int64)],
            dtype=np.float32,
        )

    def get_tactic_index(self, tactic_ids: np.ndarray | None = None) -> np.ndarray:
        if tactic_ids is None:
            return np.asarray(self.tactic_ids, dtype=np.int64)
        return np.asarray(self.tactic_ids[np.asarray(tactic_ids, dtype=np.int64)], dtype=np.int64)

    def get_flow_labels(self, flow_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.flow_labels[np.asarray(flow_ids, dtype=np.int64)], dtype=np.int64)

    def split_ids(self, split_name: str) -> np.ndarray | None:
        splits = self.manifest.get("splits", {})
        rel_path = splits.get(split_name)
        if not rel_path:
            return None
        path = self.root / rel_path
        if not path.exists():
            return None
        count = path.stat().st_size // np.dtype(np.int64).itemsize
        return np.asarray(np.memmap(path, dtype=np.int64, mode="r", shape=(count,)), dtype=np.int64)

    def out_neighbors(
        self,
        edge_type: EdgeKey,
        src_ids: np.ndarray,
        fanout: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edge_key = tuple(edge_type)
        if edge_key not in self._edge_csr:
            src_count = int(np.asarray(src_ids).shape[0])
            return (
                np.zeros((src_count + 1,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
        indptr, indices, attr = self._edge_csr[edge_key]
        return gather_csr_neighbors(indptr, indices, attr, src_ids, fanout=fanout, rng=rng)

    def get_edge_index(self, edge_type: EdgeKey) -> np.ndarray:
        edge_key = tuple(edge_type)
        if edge_key not in self._edge_csr:
            return np.empty((2, 0), dtype=np.int64)
        indptr, indices, _ = self._edge_csr[edge_key]
        counts = np.diff(np.asarray(indptr, dtype=np.int64))
        src = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
        return np.vstack([src, np.asarray(indices, dtype=np.int64)])

    def get_edge_attr(self, edge_type: EdgeKey) -> np.ndarray:
        edge_key = tuple(edge_type)
        if edge_key not in self._edge_csr:
            return np.empty((0, 1), dtype=np.float32)
        _, _, attr = self._edge_csr[edge_key]
        return np.asarray(attr, dtype=np.float32).reshape(-1, 1)

    def _memmap(self, rel_path: str, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        if int(np.prod(shape, dtype=np.int64)) == 0:
            return np.empty(shape, dtype=dtype)
        return np.memmap(self.root / rel_path, dtype=dtype, mode="r", shape=shape)

    def _memmap_node_features(self, rel_path: str, shape: tuple[int, ...]) -> np.ndarray:
        """Memory-map a float32 node-feature array, tolerant of two on-disk forms.

        * Raw float32 dump (written by ``_write_raw``)  -> plain ``np.memmap``.
        * A ``.npy`` file -> ``np.load(mmap_mode='r')``.

        The second case matters because ``--symlink-packet-features`` points
        ``nodes/packet/features.f32`` straight at the packet-semantic ``.npy``
        sidecar. A ``.npy`` carries a 64-128 byte header; reading it with a raw
        ``np.memmap`` would interpret that header as feature data and shift
        EVERY packet row, silently corrupting the whole semantic tier.
        ``np.load`` parses the header and returns a correctly-offset memmap.
        """
        path = self.root / rel_path
        if int(np.prod(shape, dtype=np.int64)) == 0:
            return np.empty(shape, dtype=np.float32)
        with open(path, "rb") as handle:
            is_npy = handle.read(6) == b"\x93NUMPY"
        if is_npy:
            arr = np.load(path, mmap_mode="r")
            if tuple(arr.shape) != tuple(shape):
                arr = arr.reshape(shape)
            return arr
        return np.memmap(path, dtype=np.float32, mode="r", shape=shape)

    def _optional_memmap(
        self,
        rel_path: str | None,
        dtype: Any,
        shape: tuple[int, ...],
    ) -> np.ndarray | None:
        if not rel_path:
            return None
        path = self.root / rel_path
        if not path.exists():
            return None
        if int(np.prod(shape, dtype=np.int64)) == 0:
            return np.empty(shape, dtype=dtype)
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def convert_npz_to_on_disk_graph_store(
    *,
    graph_npz: str | Path,
    output_root: str | Path,
    graph_meta_json: str | Path | None = None,
    packet_feature: str = "semantic",
    add_reverse_edges: bool = True,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    packet_semantic_npy: str | Path | None = None,
    symlink_packet_features: bool = False,
) -> dict[str, Any]:
    """Convert the legacy NPZ artifact into the mmap + CSR graph store layout."""

    artifact = load_three_tier_graph_artifact(
        graph_npz=Path(graph_npz),
        graph_meta_json=Path(graph_meta_json) if graph_meta_json is not None else None,
        packet_feature=packet_feature,
        add_reverse_edges=add_reverse_edges,
        packet_semantic_npy=Path(packet_semantic_npy) if packet_semantic_npy is not None else None,
    )
    root = Path(output_root)
    _ensure_store_dirs(root)

    flow_x = np.asarray(artifact.node_features["flow"], dtype=np.float32)
    packet_x = np.asarray(artifact.node_features["packet"], dtype=np.float32)
    technique_x = np.asarray(artifact.node_features["technique"], dtype=np.float32)
    tactic_ids = np.arange(int(artifact.node_features["tactic"].shape[0]), dtype=np.int64)
    flow_y = np.asarray(artifact.flow_y, dtype=np.int64)
    flow_shard_index = np.zeros((flow_x.shape[0],), dtype=np.int64)
    packet_shard_index = np.zeros((packet_x.shape[0],), dtype=np.int64)

    _write_raw(root / "nodes/flow/features.f32", flow_x, np.float32)
    _write_raw(root / "nodes/flow/labels.i64", flow_y, np.int64)
    _write_raw(root / "nodes/flow/shard_index.i64", flow_shard_index, np.int64)
    if symlink_packet_features and packet_semantic_npy is not None:
        _symlink_packet_features(root / "nodes/packet/features.f32", Path(packet_semantic_npy))
    else:
        _write_raw(root / "nodes/packet/features.f32", packet_x, np.float32)
    _write_raw(root / "nodes/packet/shard_index.i64", packet_shard_index, np.int64)
    _write_raw(root / "nodes/technique/features.f32", technique_x, np.float32)
    _write_raw(root / "nodes/tactic/ids.i64", tactic_ids, np.int64)

    edge_counts: dict[str, int] = {}
    edge_files: dict[str, dict[str, str]] = {}
    for edge_key, edge_index in artifact.edge_index.items():
        edge_name = edge_key_to_name(edge_key)
        attr = artifact.edge_attr.get(edge_key)
        indptr, indices, edge_attr = edge_index_to_csr(
            edge_index=np.asarray(edge_index, dtype=np.int64),
            edge_attr=attr,
            num_src=int(_node_count_for_edge_src(artifact, edge_key[0])),
        )
        edge_dir = root / "edges" / edge_name
        _write_raw(edge_dir / "indptr.i64", indptr, np.int64)
        _write_raw(edge_dir / "indices.i64", indices, np.int64)
        _write_raw(edge_dir / "attr.f32", edge_attr, np.float32)
        edge_counts[edge_name] = int(indices.shape[0])
        edge_files[edge_name] = {
            "indptr": f"edges/{edge_name}/indptr.i64",
            "indices": f"edges/{edge_name}/indices.i64",
            "attr": f"edges/{edge_name}/attr.f32",
        }

    train_ids, val_ids, test_ids = _stratified_split(flow_y, val_ratio, test_ratio, seed)
    _write_raw(root / "splits/train_flow_ids.i64", train_ids, np.int64)
    _write_raw(root / "splits/val_flow_ids.i64", val_ids, np.int64)
    _write_raw(root / "splits/test_flow_ids.i64", test_ids, np.int64)

    flow_mean, flow_std = _flow_stats(flow_x, train_ids)
    class_count_size = int(flow_y.max()) + 1 if flow_y.size else 0
    class_counts = np.bincount(flow_y[train_ids], minlength=class_count_size)
    class_weights = _balanced_class_weights(class_counts)
    metadata = dict(artifact.metadata)
    manifest = {
        "version": "v1",
        "layout": "numpy_memmap_csr",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "node_counts": {
            "flow": int(flow_x.shape[0]),
            "packet": int(packet_x.shape[0]),
            "technique": int(technique_x.shape[0]),
            "tactic": int(tactic_ids.shape[0]),
        },
        "feature_dims": {
            "flow": int(flow_x.shape[1]) if flow_x.ndim == 2 else 0,
            "packet": int(packet_x.shape[1]) if packet_x.ndim == 2 else 0,
            "technique": int(technique_x.shape[1]) if technique_x.ndim == 2 else 0,
        },
        "node_files": {
            "flow": {
                "features": "nodes/flow/features.f32",
                "labels": "nodes/flow/labels.i64",
                "shard_index": "nodes/flow/shard_index.i64",
            },
            "packet": {
                "features": "nodes/packet/features.f32",
                "shard_index": "nodes/packet/shard_index.i64",
            },
            "technique": {"features": "nodes/technique/features.f32"},
            "tactic": {"ids": "nodes/tactic/ids.i64"},
        },
        "edge_counts": edge_counts,
        "edge_files": edge_files,
        "splits": {
            "train": "splits/train_flow_ids.i64",
            "val": "splits/val_flow_ids.i64",
            "test": "splits/test_flow_ids.i64",
        },
        "label_mapping": metadata.get("label_mapping", {}),
        "num_classes": int(flow_y.max()) + 1 if flow_y.size else 0,
        "tactic_shortname_to_idx": metadata.get("tactic_shortname_to_idx", {}),
        "technique_id_to_idx": metadata.get("technique_id_to_idx", {}),
        "flow_feature_names": metadata.get("flow_feature_names", []),
        "flow_feature_stats": {
            "mean": flow_mean.astype(float).tolist(),
            "std": flow_std.astype(float).tolist(),
        },
        "class_counts": class_counts.astype(int).tolist(),
        "class_weights": class_weights.astype(float).tolist(),
        "shards": {
            "flow": [{"shard_id": 0, "range": [0, int(flow_x.shape[0])]}],
            "packet": [{"shard_id": 0, "range": [0, int(packet_x.shape[0])]}],
        },
        "schema": {
            "packet_feature_kind": packet_feature,
            "add_reverse_edges": bool(add_reverse_edges),
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def edge_index_to_csr(
    *,
    edge_index: np.ndarray,
    edge_attr: np.ndarray | None,
    num_src: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    num_edges = int(edge_index.shape[1])
    if num_edges == 0:
        return (
            np.zeros((num_src + 1,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    order = np.argsort(src, kind="stable")
    sorted_src = src[order]
    sorted_dst = dst[order]
    counts = np.bincount(sorted_src, minlength=num_src).astype(np.int64)
    indptr = np.empty((num_src + 1,), dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    if edge_attr is None:
        attr = np.ones((num_edges,), dtype=np.float32)
    else:
        attr_arr = np.asarray(edge_attr, dtype=np.float32)
        if attr_arr.ndim == 2:
            attr_arr = attr_arr[:, 0]
        attr = attr_arr[order].astype(np.float32)
    return indptr, sorted_dst.astype(np.int64), attr


def gather_csr_neighbors(
    indptr: np.ndarray,
    indices: np.ndarray,
    attr: np.ndarray,
    src_ids: np.ndarray,
    fanout: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched neighbor gather over a CSR slice.

    Equivalent to the old per-row loop but vectorized: random subsampling is
    done by assigning a uniform key to every candidate edge and keeping the
    top ``min(degree, fanout)`` keys per row via segment-wise ordering. Output
    order matches the old version (CSR-ascending inside each row).
    """
    src = np.asarray(src_ids, dtype=np.int64).reshape(-1)
    n_src = int(src.shape[0])

    if n_src == 0 or (fanout is not None and int(fanout) == 0):
        return (
            np.zeros((n_src + 1,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    indptr_arr = np.asarray(indptr, dtype=np.int64)
    starts = indptr_arr[src]
    ends = indptr_arr[src + 1]
    degrees = (ends - starts).astype(np.int64)
    total_in = int(degrees.sum())
    if total_in == 0:
        return (
            np.zeros((n_src + 1,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    if fanout is None:
        counts = degrees
    else:
        counts = np.minimum(degrees, int(fanout)).astype(np.int64)

    # Build flat global-edge index for every (row, position-in-row) pair.
    row_of_edge = np.repeat(np.arange(n_src, dtype=np.int64), degrees)
    cum_in = np.empty((n_src + 1,), dtype=np.int64)
    cum_in[0] = 0
    np.cumsum(degrees, out=cum_in[1:])
    offset_in_row = np.arange(total_in, dtype=np.int64) - cum_in[row_of_edge]
    global_edge = np.repeat(starts, degrees) + offset_in_row

    if int(counts.sum()) == total_in:
        # No subsampling: keep every candidate edge.
        selected = global_edge
    else:
        rng = rng or np.random.default_rng()
        keys = rng.random(total_in)
        # Sort by (row asc, key asc) -> keep first counts[r] edges per row,
        # giving an unbiased k-of-n sample without replacement.
        order = np.lexsort((keys, row_of_edge))
        sorted_row = row_of_edge[order]
        pos_in_segment = np.arange(total_in, dtype=np.int64) - cum_in[sorted_row]
        keep = pos_in_segment < counts[sorted_row]
        selected_in_order = order[keep]
        # Restore CSR-ascending order inside each row for stable downstream slicing.
        perm = np.lexsort((global_edge[selected_in_order], row_of_edge[selected_in_order]))
        selected = selected_in_order[perm]

    local_indptr = np.empty((n_src + 1,), dtype=np.int64)
    local_indptr[0] = 0
    np.cumsum(counts, out=local_indptr[1:])

    nbrs = np.asarray(indices[selected], dtype=np.int64)
    weights = np.asarray(attr[selected], dtype=np.float32)
    return local_indptr, nbrs, weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert three-tier NPZ graph to mmap CSR graph store.")
    parser.add_argument("--graph-npz", required=True)
    parser.add_argument("--graph-meta-json", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--packet-feature", choices=["semantic", "payload"], default="semantic")
    parser.add_argument("--no-reverse-edges", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--packet-semantic-npy",
        default=None,
        help="Explicit path to packet_semantic_x.npy sidecar (auto-detected if absent).",
    )
    parser.add_argument(
        "--symlink-packet-features",
        action="store_true",
        help=(
            "Symlink nodes/packet/features.f32 to the source .npy instead of copying. "
            "Saves ~50-60 GB on disk-constrained environments (e.g. Kaggle /tmp). "
            "Requires --packet-semantic-npy and the source must be float32."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_npz_to_on_disk_graph_store(
        graph_npz=args.graph_npz,
        graph_meta_json=args.graph_meta_json,
        output_root=args.output_root,
        packet_feature=args.packet_feature,
        add_reverse_edges=not args.no_reverse_edges,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        packet_semantic_npy=args.packet_semantic_npy,
        symlink_packet_features=args.symlink_packet_features,
    )
    print(f"[OK] Graph store: {args.output_root}")
    print(
        "[OK] Counts -> "
        f"flows={manifest['node_counts']['flow']}, "
        f"packets={manifest['node_counts']['packet']}, "
        f"techniques={manifest['node_counts']['technique']}, "
        f"tactics={manifest['node_counts']['tactic']}"
    )


def _node_count_for_edge_src(artifact: HeteroGraphArtifact, src_type: str) -> int:
    return int(artifact.node_features[src_type].shape[0])


def _ensure_store_dirs(root: Path) -> None:
    for rel_path in [
        "nodes/flow",
        "nodes/packet",
        "nodes/technique",
        "nodes/tactic",
        "edges",
        "splits",
    ]:
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def _write_raw(path: Path, array: np.ndarray, dtype: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array, dtype=dtype).tofile(path)


def _symlink_packet_features(dest: Path, src: Path) -> None:
    """Create a symlink dest -> src instead of copying 50-60 GB of packet features.

    Requires src to already be float32 so the graph store reader can memmap it directly.
    Saves the full copy on disk-constrained environments (e.g. Kaggle /tmp).
    """
    probe = np.load(str(src), mmap_mode="r")
    if probe.dtype != np.float32:
        raise ValueError(
            f"--symlink-packet-features requires float32 source; got {probe.dtype}. "
            "Convert the .npy to float32 first, or omit this flag."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src.resolve())
    print(f"Symlinked packet features: {dest} -> {src.resolve()}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)


def _flow_stats(flow_x: np.ndarray, train_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_ids.size == 0:
        dim = int(flow_x.shape[1]) if flow_x.ndim == 2 else 0
        return np.zeros((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32)
    rows = flow_x[np.asarray(train_ids, dtype=np.int64)]
    mean = rows.mean(axis=0).astype(np.float32)
    std = np.maximum(rows.std(axis=0).astype(np.float32), 1e-6)
    return mean, std


def _balanced_class_weights(class_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=np.float32)
    weights = np.zeros_like(counts, dtype=np.float32)
    nonzero = counts > 0
    if counts.size and np.any(nonzero):
        weights[nonzero] = counts[nonzero].sum() / (float(counts.shape[0]) * counts[nonzero])
    return weights


def _stratified_split(
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
        if n_test >= n:
            n_test = max(0, n - 1)
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


if __name__ == "__main__":
    main()
