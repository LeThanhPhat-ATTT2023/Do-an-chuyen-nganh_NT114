from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
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


class FlowSeedDataset(Dataset):
    def __init__(self, flow_ids: np.ndarray) -> None:
        self.flow_ids = np.asarray(flow_ids, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.flow_ids.shape[0])

    def __getitem__(self, index: int) -> int:
        return int(self.flow_ids[index])


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

    def sample(self, seed_flow_ids: list[int] | np.ndarray) -> MiniBatchSubgraph:
        seeds = _unique_preserve_order(np.asarray(seed_flow_ids, dtype=np.int64).reshape(-1))
        local_nodes: dict[str, list[int]] = {
            "flow": [],
            "packet": [],
            "technique": [],
            "tactic": [],
        }
        local_maps: dict[str, dict[int, int]] = {
            "flow": {},
            "packet": {},
            "technique": {},
            "tactic": {},
        }
        for flow_id in seeds:
            _add_node(local_nodes, local_maps, "flow", int(flow_id))

        edge_rows: dict[EdgeKey, list[tuple[int, int, float]]] = {
            edge_type: [] for edge_type in self.backend.edge_types
        }
        frontier: dict[str, list[int]] = {"flow": [int(x) for x in seeds.tolist()]}

        for _ in range(self.hops):
            next_frontier: dict[str, list[int]] = {}
            for edge_type in self.backend.edge_types:
                src_type, _, dst_type = edge_type
                src_ids = frontier.get(src_type)
                if not src_ids:
                    continue
                fanout = self._fanout(edge_type)
                if fanout == 0:
                    continue
                local_indptr, dst_ids, weights = self.backend.out_neighbors(
                    edge_type,
                    np.asarray(src_ids, dtype=np.int64),
                    fanout=fanout,
                    rng=self.rng,
                )
                for row, src_id in enumerate(src_ids):
                    start = int(local_indptr[row])
                    end = int(local_indptr[row + 1])
                    for dst_id, weight in zip(dst_ids[start:end], weights[start:end]):
                        dst_int = int(dst_id)
                        edge_rows.setdefault(edge_type, []).append((int(src_id), dst_int, float(weight)))
                        _add_node(local_nodes, local_maps, src_type, int(src_id))
                        added = _add_node(local_nodes, local_maps, dst_type, dst_int)
                        if added:
                            next_frontier.setdefault(dst_type, []).append(dst_int)
            frontier = next_frontier
            if not frontier:
                break

        if self.always_include_all_techniques:
            for tech_id in range(int(self.backend.num_techniques)):
                _add_node(local_nodes, local_maps, "technique", tech_id)
        if self.always_include_all_tactics:
            for tactic_id in range(int(self.backend.num_tactics)):
                _add_node(local_nodes, local_maps, "tactic", tactic_id)
        self._add_static_tactic_edges(edge_rows, local_nodes, local_maps)

        edge_index: dict[EdgeKey, np.ndarray] = {}
        edge_attr: dict[EdgeKey, np.ndarray] = {}
        for edge_type in self.backend.edge_types:
            rows = edge_rows.get(edge_type, [])
            src_type, _, dst_type = edge_type
            pairs: list[tuple[int, int]] = []
            weights: list[float] = []
            for src_global, dst_global, weight in rows:
                if src_global not in local_maps[src_type] or dst_global not in local_maps[dst_type]:
                    continue
                pairs.append((local_maps[src_type][src_global], local_maps[dst_type][dst_global]))
                weights.append(weight)
            if pairs:
                edge_index[edge_type] = np.asarray(pairs, dtype=np.int64).T
                edge_attr[edge_type] = np.asarray(weights, dtype=np.float32)
            else:
                edge_index[edge_type] = np.empty((2, 0), dtype=np.int64)
                edge_attr[edge_type] = np.empty((0,), dtype=np.float32)

        flow_ids = np.asarray(local_nodes["flow"], dtype=np.int64)
        packet_ids = np.asarray(local_nodes["packet"], dtype=np.int64)
        technique_ids = np.asarray(local_nodes["technique"], dtype=np.int64)
        tactic_ids = np.asarray(local_nodes["tactic"], dtype=np.int64)

        flow_x = self.backend.get_flow_features(flow_ids)
        if self.standardize_flow_features:
            flow_x = _standardize(flow_x, self.flow_feature_stats)

        seed_mask = np.zeros((flow_ids.shape[0],), dtype=bool)
        for flow_id in seeds:
            local_idx = local_maps["flow"].get(int(flow_id))
            if local_idx is not None:
                seed_mask[local_idx] = True

        node_features = {
            "flow": flow_x.astype(np.float32),
            "packet": self.backend.get_packet_features(packet_ids).astype(np.float32)
            if packet_ids.size
            else np.empty((0, self.backend.feature_dims["packet"]), dtype=np.float32),
            "technique": self.backend.get_technique_features(technique_ids).astype(np.float32)
            if technique_ids.size
            else np.empty((0, self.backend.feature_dims["technique"]), dtype=np.float32),
            "tactic": self.backend.get_tactic_index(tactic_ids).reshape(-1, 1).astype(np.int64)
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
        edge_rows: dict[EdgeKey, list[tuple[int, int, float]]],
        local_nodes: dict[str, list[int]],
        local_maps: dict[str, dict[int, int]],
    ) -> None:
        edge_type = ("technique", "belongs_to_tactic", "tactic")
        if edge_type not in self.backend.edge_types or not local_nodes["technique"]:
            return
        local_indptr, dst_ids, weights = self.backend.out_neighbors(
            edge_type,
            np.asarray(local_nodes["technique"], dtype=np.int64),
            fanout=None,
            rng=self.rng,
        )
        rows = edge_rows.setdefault(edge_type, [])
        existing = {(src, dst) for src, dst, _ in rows}
        for row, src_id in enumerate(local_nodes["technique"]):
            start = int(local_indptr[row])
            end = int(local_indptr[row + 1])
            for dst_id, weight in zip(dst_ids[start:end], weights[start:end]):
                dst_int = int(dst_id)
                _add_node(local_nodes, local_maps, "tactic", dst_int)
                if (int(src_id), dst_int) not in existing:
                    rows.append((int(src_id), dst_int, float(weight)))
                    existing.add((int(src_id), dst_int))


class NeighborSamplingCollator:
    def __init__(self, sampler: HeteroNeighborSampler) -> None:
        self.sampler = sampler

    def __call__(self, seed_flow_ids: list[int]) -> MiniBatchSubgraph:
        return self.sampler.sample(seed_flow_ids)


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


def _unique_preserve_order(values: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values.tolist():
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return np.asarray(ordered, dtype=np.int64)


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
