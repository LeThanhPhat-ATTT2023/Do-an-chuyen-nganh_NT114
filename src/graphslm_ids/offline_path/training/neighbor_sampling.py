from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from graphslm_ids.offline_path.training.hetero_graph_artifact import HeteroGraphArtifact
from graphslm_ids.offline_path.training.on_disk_graph_store import (
    EdgeKey,
    edge_index_to_csr,
    edge_key_to_name,
    gather_csr_neighbors,
)


class NeighborBackend(Protocol):
    edge_types: list[EdgeKey]
    feature_dims: dict[str, int]
    manifest: dict[str, Any]

    @property
    def num_flows(self) -> int:
        ...

    @property
    def num_tactics(self) -> int:
        ...

    @property
    def num_techniques(self) -> int:
        ...

    def get_flow_features(self, flow_ids: np.ndarray) -> np.ndarray:
        ...

    def get_packet_features(self, packet_ids: np.ndarray) -> np.ndarray:
        ...

    def get_technique_features(self, technique_ids: np.ndarray | None = None) -> np.ndarray:
        ...

    def get_tactic_index(self, tactic_ids: np.ndarray | None = None) -> np.ndarray:
        ...

    def get_flow_labels(self, flow_ids: np.ndarray) -> np.ndarray:
        ...

    def out_neighbors(
        self,
        edge_type: EdgeKey,
        src_ids: np.ndarray,
        fanout: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


@dataclass
class MiniBatchSubgraph:
    node_features: dict[str, np.ndarray]
    edge_index: dict[EdgeKey, np.ndarray]
    edge_attr: dict[EdgeKey, np.ndarray]
    seed_mask: np.ndarray
    seed_labels: np.ndarray
    seed_flow_ids: np.ndarray
    local_to_global: dict[str, np.ndarray]
    stats: dict[str, Any] = field(default_factory=dict)

    def pin_memory(self) -> "MiniBatchSubgraph":
        """Replace numpy arrays with pinned torch tensors in-place.

        Called by PyTorch's DataLoader internal ``_pin_memory_loop`` when
        ``pin_memory=True``. By pinning here (inside the DataLoader's pin
        thread, which runs in the main process) rather than inside the training
        loop, the H2D copy on the next step can run with ``non_blocking=True``
        and overlap with GPU compute. Without this hook, ``MiniBatchSubgraph``
        is treated as an opaque object and PyTorch falls back to a no-op,
        forcing the trainer's ``to_torch_batch`` to do a synchronous pin per
        tensor on the main thread.
        """

        def _pin_dict(d: dict[Any, np.ndarray], cast: Any = None) -> dict[Any, torch.Tensor]:
            out: dict[Any, torch.Tensor] = {}
            for key, value in d.items():
                arr = np.ascontiguousarray(value, dtype=cast) if cast is not None else np.ascontiguousarray(value)
                out[key] = torch.from_numpy(arr).pin_memory()
            return out

        # Mutate in place — DataLoader expects the same object back.
        self.node_features = _pin_dict(self.node_features)  # type: ignore[assignment]
        # Force int64 for edge_index, float32 for edge_attr (downstream expects these dtypes).
        self.edge_index = _pin_dict(self.edge_index, cast=np.int64)  # type: ignore[assignment]
        self.edge_attr = _pin_dict(self.edge_attr, cast=np.float32)  # type: ignore[assignment]
        self.seed_mask = torch.from_numpy(np.ascontiguousarray(self.seed_mask, dtype=bool)).pin_memory()  # type: ignore[assignment]
        self.seed_labels = torch.from_numpy(np.ascontiguousarray(self.seed_labels, dtype=np.int64)).pin_memory()  # type: ignore[assignment]
        # seed_flow_ids + local_to_global stay on CPU as numpy — only logging consumes them.
        return self


class FlowSeedDataset(Dataset):
    def __init__(self, flow_ids: np.ndarray) -> None:
        self.flow_ids = np.asarray(flow_ids, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.flow_ids.shape[0])

    def __getitem__(self, index: int) -> int:
        return int(self.flow_ids[index])


class _LocalMap:
    """Vectorized first-occurrence dedup + sorted lookup for one node type.

    Replaces the legacy ``dict[int, int]`` + ``list[int]`` pair: bulk add returns
    the newly-inserted globals (in first-occurrence order) without any Python
    int iteration, and bulk lookup is one ``np.searchsorted`` over the sorted
    view of every global seen so far. Both scale linearly with neighbourhood
    size — critical when fanouts blow up to hundreds of thousands of packets
    on a 1TB graph.
    """

    __slots__ = ("_chunks", "_count", "_sorted", "_local_of_sorted")

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._count: int = 0
        self._sorted: np.ndarray = np.empty(0, dtype=np.int64)
        self._local_of_sorted: np.ndarray = np.empty(0, dtype=np.int64)

    @property
    def count(self) -> int:
        return self._count

    def add(self, candidates: np.ndarray) -> np.ndarray:
        """Insert candidates, return globals that were newly added in first-occurrence order."""
        arr = np.asarray(candidates, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return np.empty(0, dtype=np.int64)

        uniq_sorted, first_idx = np.unique(arr, return_index=True)

        if self._sorted.size:
            pos = np.searchsorted(self._sorted, uniq_sorted)
            present = np.zeros(uniq_sorted.size, dtype=bool)
            in_bounds = pos < self._sorted.size
            if in_bounds.any():
                cand = np.where(in_bounds, pos, 0)
                present = in_bounds & (self._sorted[cand] == uniq_sorted)
            new_mask_sorted = ~present
        else:
            new_mask_sorted = np.ones(uniq_sorted.size, dtype=bool)

        if not bool(new_mask_sorted.any()):
            return np.empty(0, dtype=np.int64)

        new_sorted = uniq_sorted[new_mask_sorted]
        new_first_idx = first_idx[new_mask_sorted]

        # Order the *new* uniques by their first occurrence in arr so local ids
        # mirror the deterministic ordering the legacy Python loop produced.
        ins_order = np.argsort(new_first_idx, kind="stable")
        new_in_insert_order = new_sorted[ins_order]
        n_new = new_in_insert_order.shape[0]
        first_local = self._count
        new_locals_insert = np.arange(first_local, first_local + n_new, dtype=np.int64)
        self._count += n_new
        self._chunks.append(new_in_insert_order)

        # For the sorted view we need: for each new_sorted[i], its assigned local id.
        new_locals_for_sorted = np.empty(n_new, dtype=np.int64)
        new_locals_for_sorted[ins_order] = new_locals_insert

        if self._sorted.size:
            merged_vals = np.concatenate([self._sorted, new_sorted])
            merged_locs = np.concatenate([self._local_of_sorted, new_locals_for_sorted])
            order = np.argsort(merged_vals, kind="stable")
            self._sorted = merged_vals[order]
            self._local_of_sorted = merged_locs[order]
        else:
            self._sorted = new_sorted
            self._local_of_sorted = new_locals_for_sorted

        return new_in_insert_order

    def lookup(self, query: np.ndarray) -> np.ndarray:
        """Return local indices for each query global id; -1 for missing entries."""
        arr = np.asarray(query, dtype=np.int64).reshape(-1)
        out = np.full(arr.shape, -1, dtype=np.int64)
        if arr.size == 0 or self._sorted.size == 0:
            return out
        pos = np.searchsorted(self._sorted, arr)
        in_bounds = pos < self._sorted.size
        if not in_bounds.any():
            return out
        cand = np.where(in_bounds, pos, 0)
        match = in_bounds & (self._sorted[cand] == arr)
        out[match] = self._local_of_sorted[pos[match]]
        return out

    def globals_array(self) -> np.ndarray:
        if self._count == 0:
            return np.empty(0, dtype=np.int64)
        if len(self._chunks) == 1:
            return self._chunks[0]
        out = np.concatenate(self._chunks)
        self._chunks = [out]
        return out


class InMemoryNeighborBackend:
    """CSR neighbor backend over the legacy in-memory artifact.

    This keeps the baseline NPZ path usable while exercising the same sampler
    and training loop as the mmap graph store.
    """

    def __init__(self, artifact: HeteroGraphArtifact) -> None:
        self.artifact = artifact
        self.manifest = dict(artifact.metadata)
        self.edge_types = list(artifact.edge_index.keys())
        self.feature_dims = {
            "flow": int(artifact.node_features["flow"].shape[1]),
            "packet": int(artifact.node_features["packet"].shape[1]),
            "technique": int(artifact.node_features["technique"].shape[1]),
        }
        self._edge_csr: dict[EdgeKey, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for edge_key, edge_index in artifact.edge_index.items():
            indptr, indices, attr = edge_index_to_csr(
                edge_index=np.asarray(edge_index, dtype=np.int64),
                edge_attr=artifact.edge_attr.get(edge_key),
                num_src=int(artifact.node_features[edge_key[0]].shape[0]),
            )
            self._edge_csr[edge_key] = (indptr, indices, attr)

    @property
    def num_flows(self) -> int:
        return int(self.artifact.node_features["flow"].shape[0])

    @property
    def num_tactics(self) -> int:
        return int(self.artifact.node_features["tactic"].shape[0])

    @property
    def num_techniques(self) -> int:
        return int(self.artifact.node_features["technique"].shape[0])

    def get_flow_features(self, flow_ids: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.artifact.node_features["flow"][np.asarray(flow_ids, dtype=np.int64)],
            dtype=np.float32,
        )

    def get_packet_features(self, packet_ids: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.artifact.node_features["packet"][np.asarray(packet_ids, dtype=np.int64)],
            dtype=np.float32,
        )

    def get_technique_features(self, technique_ids: np.ndarray | None = None) -> np.ndarray:
        if technique_ids is None:
            return np.asarray(self.artifact.node_features["technique"], dtype=np.float32)
        return np.asarray(
            self.artifact.node_features["technique"][np.asarray(technique_ids, dtype=np.int64)],
            dtype=np.float32,
        )

    def get_tactic_index(self, tactic_ids: np.ndarray | None = None) -> np.ndarray:
        all_ids = np.arange(self.num_tactics, dtype=np.int64)
        if tactic_ids is None:
            return all_ids
        return all_ids[np.asarray(tactic_ids, dtype=np.int64)]

    def get_flow_labels(self, flow_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.artifact.flow_y[np.asarray(flow_ids, dtype=np.int64)], dtype=np.int64)

    def out_neighbors(
        self,
        edge_type: EdgeKey,
        src_ids: np.ndarray,
        fanout: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if edge_type not in self._edge_csr:
            src_count = int(np.asarray(src_ids).shape[0])
            return (
                np.zeros((src_count + 1,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
        indptr, indices, attr = self._edge_csr[edge_type]
        return gather_csr_neighbors(indptr, indices, attr, src_ids, fanout=fanout, rng=rng)


class HeteroNeighborSampler:
    def __init__(
        self,
        backend: NeighborBackend,
        *,
        hops: int,
        fanouts: dict[str, int] | None = None,
        reverse_fanouts: dict[str, int] | None = None,
        always_include_all_tactics: bool = True,
        always_include_all_techniques: bool = True,
        flow_feature_stats: dict[str, Any] | None = None,
        standardize_flow_features: bool = True,
        seed: int = 42,
    ) -> None:
        self.backend = backend
        self.hops = int(hops)
        self.fanouts = {str(k): int(v) for k, v in (fanouts or {}).items()}
        self.reverse_fanouts = {str(k): int(v) for k, v in (reverse_fanouts or {}).items()}
        self.always_include_all_tactics = bool(always_include_all_tactics)
        self.always_include_all_techniques = bool(always_include_all_techniques)
        self.flow_feature_stats = dict(flow_feature_stats or {})
        self.standardize_flow_features = bool(standardize_flow_features)
        self.rng = np.random.default_rng(seed)
        # Cache mean/std arrays so we don't re-allocate them on every batch.
        self._mean_arr: np.ndarray | None = None
        self._std_arr: np.ndarray | None = None
        mean = self.flow_feature_stats.get("mean")
        std = self.flow_feature_stats.get("std")
        if mean is not None and std is not None:
            self._mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
            self._std_arr = np.maximum(
                np.asarray(std, dtype=np.float32).reshape(1, -1), 1e-6
            )

    def sample(self, seed_flow_ids: list[int] | np.ndarray) -> MiniBatchSubgraph:
        seeds = _unique_preserve_order(np.asarray(seed_flow_ids, dtype=np.int64).reshape(-1))
        local_maps: dict[str, _LocalMap] = {
            "flow": _LocalMap(),
            "packet": _LocalMap(),
            "technique": _LocalMap(),
            "tactic": _LocalMap(),
        }
        local_maps["flow"].add(seeds)

        # Per edge_type we accumulate three flat numpy arrays (src_global, dst_global, weight).
        # Concatenating happens once at the end, so the K-hop loop stays free of Python-level
        # tuple appends.
        edge_chunks: dict[EdgeKey, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            edge_type: [] for edge_type in self.backend.edge_types
        }
        frontier: dict[str, np.ndarray] = {"flow": seeds.copy()}

        for _ in range(self.hops):
            next_frontier: dict[str, list[np.ndarray]] = {}
            for edge_type in self.backend.edge_types:
                src_type, _, dst_type = edge_type
                src_arr = frontier.get(src_type)
                if src_arr is None or src_arr.size == 0:
                    continue
                fanout = self._fanout(edge_type)
                if fanout == 0:
                    continue
                local_indptr, dst_ids, weights = self.backend.out_neighbors(
                    edge_type,
                    src_arr,
                    fanout=fanout,
                    rng=self.rng,
                )
                if dst_ids.size == 0:
                    continue
                counts_per_src = np.diff(local_indptr).astype(np.int64)
                src_repeated = np.repeat(src_arr.astype(np.int64), counts_per_src)
                edge_chunks[edge_type].append(
                    (src_repeated, dst_ids.astype(np.int64), weights.astype(np.float32))
                )

                local_maps[src_type].add(src_arr)
                newly_added = local_maps[dst_type].add(dst_ids)
                if newly_added.size:
                    next_frontier.setdefault(dst_type, []).append(newly_added)

            if not next_frontier:
                frontier = {}
                break
            frontier = {
                ntype: chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
                for ntype, chunks in next_frontier.items()
            }

        if self.always_include_all_techniques:
            local_maps["technique"].add(
                np.arange(int(self.backend.num_techniques), dtype=np.int64),
            )
        if self.always_include_all_tactics:
            local_maps["tactic"].add(
                np.arange(int(self.backend.num_tactics), dtype=np.int64),
            )
        self._add_static_tactic_edges(edge_chunks, local_maps)

        edge_index: dict[EdgeKey, np.ndarray] = {}
        edge_attr: dict[EdgeKey, np.ndarray] = {}
        for edge_type in self.backend.edge_types:
            chunks = edge_chunks.get(edge_type, [])
            if not chunks:
                edge_index[edge_type] = np.empty((2, 0), dtype=np.int64)
                edge_attr[edge_type] = np.empty((0,), dtype=np.float32)
                continue
            src_type, _, dst_type = edge_type
            if len(chunks) == 1:
                src_global, dst_global, w_global = chunks[0]
            else:
                src_global = np.concatenate([c[0] for c in chunks])
                dst_global = np.concatenate([c[1] for c in chunks])
                w_global = np.concatenate([c[2] for c in chunks])
            src_local = local_maps[src_type].lookup(src_global)
            dst_local = local_maps[dst_type].lookup(dst_global)
            valid = (src_local >= 0) & (dst_local >= 0)
            if not bool(valid.any()):
                edge_index[edge_type] = np.empty((2, 0), dtype=np.int64)
                edge_attr[edge_type] = np.empty((0,), dtype=np.float32)
                continue
            edge_index[edge_type] = np.vstack(
                [src_local[valid].astype(np.int64), dst_local[valid].astype(np.int64)]
            )
            edge_attr[edge_type] = w_global[valid].astype(np.float32)

        flow_ids = local_maps["flow"].globals_array()
        packet_ids = local_maps["packet"].globals_array()
        technique_ids = local_maps["technique"].globals_array()
        tactic_ids = local_maps["tactic"].globals_array()

        flow_x = self.backend.get_flow_features(flow_ids)
        if self.standardize_flow_features:
            flow_x = self._standardize_cached(flow_x)

        # Vectorized seed mask construction.
        seed_local = local_maps["flow"].lookup(seeds)
        seed_mask = np.zeros((flow_ids.shape[0],), dtype=bool)
        valid_seed = seed_local >= 0
        if bool(valid_seed.any()):
            seed_mask[seed_local[valid_seed]] = True

        node_features = {
            "flow": flow_x.astype(np.float32, copy=False),
            "packet": self.backend.get_packet_features(packet_ids).astype(np.float32, copy=False)
            if packet_ids.size
            else np.empty((0, self.backend.feature_dims["packet"]), dtype=np.float32),
            "technique": self.backend.get_technique_features(technique_ids).astype(np.float32, copy=False)
            if technique_ids.size
            else np.empty((0, self.backend.feature_dims["technique"]), dtype=np.float32),
            "tactic": self.backend.get_tactic_index(tactic_ids).reshape(-1, 1).astype(np.int64, copy=False)
            if tactic_ids.size
            else np.empty((0, 1), dtype=np.int64),
        }
        return MiniBatchSubgraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            seed_mask=seed_mask,
            seed_labels=self.backend.get_flow_labels(seeds),
            seed_flow_ids=seeds,
            local_to_global={
                "flow": flow_ids,
                "packet": packet_ids,
                "technique": technique_ids,
                "tactic": tactic_ids,
            },
            stats={
                "nodes": {key: int(value.shape[0]) for key, value in node_features.items()},
                "edges": {edge_key_to_name(key): int(value.shape[1]) for key, value in edge_index.items()},
            },
        )

    def _standardize_cached(self, flow_x: np.ndarray) -> np.ndarray:
        if self._mean_arr is None or self._std_arr is None:
            return flow_x
        if self._mean_arr.shape[1] != flow_x.shape[1] or self._std_arr.shape[1] != flow_x.shape[1]:
            return flow_x
        return ((flow_x - self._mean_arr) / self._std_arr).astype(np.float32, copy=False)

    def _fanout(self, edge_type: EdgeKey) -> int | None:
        edge_name = edge_key_to_name(edge_type)
        relation = edge_type[1]
        if relation.startswith("rev_") and edge_name in self.reverse_fanouts:
            return self.reverse_fanouts[edge_name]
        if relation.startswith("rev_") and relation in self.reverse_fanouts:
            return self.reverse_fanouts[relation]
        if edge_name in self.fanouts:
            return self.fanouts[edge_name]
        if relation in self.fanouts:
            return self.fanouts[relation]
        return None

    def _add_static_tactic_edges(
        self,
        edge_chunks: dict[EdgeKey, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
        local_maps: dict[str, _LocalMap],
    ) -> None:
        edge_type = ("technique", "belongs_to_tactic", "tactic")
        if edge_type not in self.backend.edge_types or local_maps["technique"].count == 0:
            return
        technique_src = local_maps["technique"].globals_array()
        local_indptr, dst_ids, weights = self.backend.out_neighbors(
            edge_type,
            technique_src,
            fanout=None,
            rng=self.rng,
        )
        if dst_ids.size == 0:
            return
        counts_per_src = np.diff(local_indptr).astype(np.int64)
        src_repeated = np.repeat(technique_src, counts_per_src)
        dst_arr = dst_ids.astype(np.int64)
        w_arr = weights.astype(np.float32)
        local_maps["tactic"].add(dst_arr)

        existing_chunks = edge_chunks.get(edge_type, [])
        if existing_chunks:
            # Filter out (src, dst) pairs already produced by the K-hop loop so the
            # static technique→tactic appendix does not duplicate them.
            if len(existing_chunks) == 1:
                existing_src, existing_dst, _ = existing_chunks[0]
            else:
                existing_src = np.concatenate([c[0] for c in existing_chunks])
                existing_dst = np.concatenate([c[1] for c in existing_chunks])
            num_tactics = int(self.backend.num_tactics)
            existing_keys = existing_src * np.int64(num_tactics) + existing_dst
            new_keys = src_repeated * np.int64(num_tactics) + dst_arr
            keep = ~np.isin(new_keys, existing_keys, assume_unique=False)
            if not bool(keep.any()):
                return
            src_repeated = src_repeated[keep]
            dst_arr = dst_arr[keep]
            w_arr = w_arr[keep]
        edge_chunks.setdefault(edge_type, []).append((src_repeated, dst_arr, w_arr))


class NeighborSamplingCollator:
    def __init__(self, sampler: HeteroNeighborSampler) -> None:
        self.sampler = sampler

    def __call__(self, seed_flow_ids: list[int]) -> MiniBatchSubgraph:
        return self.sampler.sample(seed_flow_ids)


# ── Backward-compatible helpers (kept for downstream importers/tests) ────────

def _add_node(
    local_nodes: dict[str, list[int]],
    local_maps: dict[str, dict[int, int]],
    node_type: str,
    node_id: int,
) -> bool:
    if node_id in local_maps[node_type]:
        return False
    local_maps[node_type][node_id] = len(local_nodes[node_type])
    local_nodes[node_type].append(node_id)
    return True


def _add_nodes_bulk(
    local_nodes: dict[str, list[int]],
    local_maps: dict[str, dict[int, int]],
    node_type: str,
    node_ids: np.ndarray,
) -> np.ndarray:
    """Insert a batch of global node ids, returning only the ones that were new.

    Kept on the module for backward compatibility with callers that still use
    Python dicts for the local map. The sampler itself now uses the vectorized
    :class:`_LocalMap`.
    """
    arr = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    uniq, first_idx = np.unique(arr, return_index=True)
    unique_ordered = uniq[np.argsort(first_idx, kind="stable")]
    mapping = local_maps[node_type]
    nodes = local_nodes[node_type]
    new_ids: list[int] = []
    for value in unique_ordered.tolist():
        if value in mapping:
            continue
        mapping[value] = len(nodes)
        nodes.append(value)
        new_ids.append(value)
    return np.asarray(new_ids, dtype=np.int64)


def _remap_global_to_local(global_ids: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Vectorized global→local remap; -1 marks a global id absent from the map.

    Kept for backward compatibility. The sampler now uses ``_LocalMap.lookup``
    directly, which sidesteps the dict entirely.
    """
    arr = np.asarray(global_ids, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    if not mapping:
        return np.full(arr.shape, -1, dtype=np.int64)
    keys_arr = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
    vals_arr = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
    order = np.argsort(keys_arr, kind="stable")
    sorted_keys = keys_arr[order]
    sorted_vals = vals_arr[order]
    pos = np.searchsorted(sorted_keys, arr)
    out = np.full(arr.shape, -1, dtype=np.int64)
    in_bounds = pos < sorted_keys.size
    if in_bounds.any():
        cand = np.where(in_bounds, pos, 0)
        match = in_bounds & (sorted_keys[cand] == arr)
        out[match] = sorted_vals[pos[match]]
    return out


def _unique_preserve_order(values: np.ndarray) -> np.ndarray:
    """Vectorized first-occurrence-preserving unique."""
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    _, first_idx = np.unique(arr, return_index=True)
    return arr[np.sort(first_idx)]


def _standardize(flow_x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        return flow_x
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std_arr = np.maximum(np.asarray(std, dtype=np.float32).reshape(1, -1), 1e-6)
    if mean_arr.shape[1] != flow_x.shape[1] or std_arr.shape[1] != flow_x.shape[1]:
        return flow_x
    return ((flow_x - mean_arr) / std_arr).astype(np.float32)
