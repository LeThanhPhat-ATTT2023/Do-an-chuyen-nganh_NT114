from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EdgeKey = tuple[str, str, str]


@dataclass
class Subgraph:
    node_features: dict[str, Any]
    edge_index_dict: dict[EdgeKey, Any]
    edge_weight_dict: dict[EdgeKey, Any]
    packet_local_to_id: dict[int, str]
    flow_local_to_id: dict[int, str]
    technique_local_to_id: dict[int, str]
    tactic_local_to_id: dict[int, str]
    seed_flow_local_idx: int = 0

    def to_snapshot_dict(self) -> dict[str, Any]:
        return {
            "node_features": {key: _to_list(value) for key, value in self.node_features.items()},
            "edge_index": {
                edge_key_to_name(key): _to_list(value)
                for key, value in self.edge_index_dict.items()
            },
            "edge_weight": {
                edge_key_to_name(key): _to_list(value)
                for key, value in self.edge_weight_dict.items()
            },
            "node_ids": {
                "flow": [self.flow_local_to_id[i] for i in sorted(self.flow_local_to_id)],
                "packet": [self.packet_local_to_id[i] for i in sorted(self.packet_local_to_id)],
                "technique": [
                    self.technique_local_to_id[i] for i in sorted(self.technique_local_to_id)
                ],
                "tactic": [self.tactic_local_to_id[i] for i in sorted(self.tactic_local_to_id)],
            },
            "packet_id_map": {node_id: idx for idx, node_id in self.packet_local_to_id.items()},
            "flow_id_map": {node_id: idx for idx, node_id in self.flow_local_to_id.items()},
            "seed_flow_local_idx": self.seed_flow_local_idx,
        }


class SubgraphBuilder:
    """Build HGT-compatible subgraphs from a HotGraphBuffer seed flow."""

    def __init__(
        self,
        buffer: Any,
        hops: int = 3,
        add_reverse_edges: bool = True,
        standardize_flow_features: bool = True,
        flow_feature_stats: dict[str, Any] | None = None,
        packet_feature: str = "semantic",
        protocol_mapping: dict[str, int] | None = None,
    ) -> None:
        self.buffer = buffer
        self.hops = int(hops)
        self.add_reverse_edges = bool(add_reverse_edges)
        self.standardize_flow_features = bool(standardize_flow_features)
        self.flow_feature_stats = flow_feature_stats or {}
        self.packet_feature = packet_feature
        self.protocol_mapping = {str(k).upper(): int(v) for k, v in (protocol_mapping or {}).items()}

    def build(self, seed_flow_id: str) -> Subgraph:
        snapshot = self.buffer.snapshot(seed_flow_id)
        if not snapshot.get("flow"):
            raise KeyError(f"Flow not found in hot graph buffer: {seed_flow_id}")

        flow_ids = [str(seed_flow_id)]
        packet_entries = list(snapshot.get("packets", []))
        packet_ids = [str(packet["packet_id"]) for packet in packet_entries]

        technique_ids = _collect_techniques(snapshot, packet_entries)
        tactic_ids = _collect_tactics(technique_ids, self.buffer.technique_to_tactic)

        flow_x = np.asarray([self._flow_features(snapshot["flow"])], dtype=np.float32)
        if self.standardize_flow_features:
            flow_x = self._standardize_flow(flow_x)

        packet_x = self._packet_features(packet_entries)
        technique_x = self._technique_features(technique_ids)
        tactic_x = np.arange(len(tactic_ids), dtype=np.int64).reshape(-1, 1)

        flow_idx = {flow_id: idx for idx, flow_id in enumerate(flow_ids)}
        packet_idx = {packet_id: idx for idx, packet_id in enumerate(packet_ids)}
        technique_idx = {tech_id: idx for idx, tech_id in enumerate(technique_ids)}
        tactic_idx = {tactic_id: idx for idx, tactic_id in enumerate(tactic_ids)}

        edge_index: dict[EdgeKey, np.ndarray] = {}
        edge_weight: dict[EdgeKey, np.ndarray] = {}

        contains = [(flow_idx[str(seed_flow_id)], packet_idx[pid]) for pid in packet_ids]
        _set_edges(edge_index, edge_weight, ("flow", "contains", "packet"), contains)

        next_edges = [(idx, idx + 1) for idx in range(max(len(packet_ids) - 1, 0))]
        _set_edges(edge_index, edge_weight, ("packet", "next_packet", "packet"), next_edges)

        packet_tech_edges: list[tuple[int, int]] = []
        packet_tech_weights: list[float] = []
        for packet in packet_entries:
            packet_id = str(packet["packet_id"])
            for tech_id, score in _score_pairs(packet.get("mitre_topk", [])):
                if tech_id in technique_idx:
                    packet_tech_edges.append((packet_idx[packet_id], technique_idx[tech_id]))
                    packet_tech_weights.append(float(score))
        _set_edges(
            edge_index,
            edge_weight,
            ("packet", "matches_technique", "technique"),
            packet_tech_edges,
            packet_tech_weights,
        )

        flow_tech_edges: list[tuple[int, int]] = []
        flow_tech_weights: list[float] = []
        for tech_id, score in _score_pairs(snapshot.get("flow_to_mitre", [])):
            if tech_id in technique_idx:
                flow_tech_edges.append((0, technique_idx[tech_id]))
                flow_tech_weights.append(float(score))
        _set_edges(
            edge_index,
            edge_weight,
            ("flow", "matches_technique", "technique"),
            flow_tech_edges,
            flow_tech_weights,
        )

        tactic_edges: list[tuple[int, int]] = []
        for tech_id in technique_ids:
            tactic_id = self.buffer.technique_to_tactic.get(tech_id)
            if tactic_id in tactic_idx:
                tactic_edges.append((technique_idx[tech_id], tactic_idx[tactic_id]))
        _set_edges(edge_index, edge_weight, ("technique", "belongs_to_tactic", "tactic"), tactic_edges)

        if self.add_reverse_edges:
            _add_reverse_edges(edge_index, edge_weight)

        return Subgraph(
            node_features={
                "flow": flow_x,
                "packet": packet_x,
                "technique": technique_x,
                "tactic": tactic_x,
            },
            edge_index_dict=edge_index,
            edge_weight_dict=edge_weight,
            packet_local_to_id={idx: packet_id for packet_id, idx in packet_idx.items()},
            flow_local_to_id={idx: flow_id for flow_id, idx in flow_idx.items()},
            technique_local_to_id={idx: tech_id for tech_id, idx in technique_idx.items()},
            tactic_local_to_id={idx: tactic_id for tactic_id, idx in tactic_idx.items()},
            seed_flow_local_idx=0,
        )

    def to_snapshot_dict(self, sub: Subgraph) -> dict[str, Any]:
        return sub.to_snapshot_dict()

    def _flow_features(self, flow: dict[str, Any]) -> list[float]:
        protocol = str(flow.get("protocol", "OTHER")).upper()
        protocol_id = self.protocol_mapping.get(protocol, self.protocol_mapping.get("OTHER", 0))
        return [
            float(flow.get("packet_count", 0.0)),
            float(flow.get("total_payload_bytes", 0.0)),
            float(flow.get("duration_seconds", 0.0)),
            float(flow.get("src_port", 0.0)),
            float(flow.get("dst_port", 0.0)),
            float(protocol_id),
        ]

    def _standardize_flow(self, flow_x: np.ndarray) -> np.ndarray:
        mean = self.flow_feature_stats.get("mean")
        std = self.flow_feature_stats.get("std")
        if mean is None or std is None:
            return flow_x
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        std_arr = np.maximum(np.asarray(std, dtype=np.float32).reshape(1, -1), 1e-6)
        if mean_arr.shape[1] != flow_x.shape[1] or std_arr.shape[1] != flow_x.shape[1]:
            return flow_x
        return ((flow_x - mean_arr) / std_arr).astype(np.float32)

    def _packet_features(self, packet_entries: list[dict[str, Any]]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for packet in packet_entries:
            embedding = packet.get("embedding")
            if embedding is None:
                embedding = self.buffer.packet_embeddings.get(str(packet.get("packet_id")))
            if embedding is None:
                continue
            rows.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if rows:
            return np.stack(rows, axis=0).astype(np.float32)

        dim = _first_feature_dim(self.buffer.packet_embeddings, default=768)
        return np.empty((0, dim), dtype=np.float32)

    def _technique_features(self, technique_ids: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for tech_id in technique_ids:
            feature = self.buffer.technique_features.get(tech_id)
            if feature is not None:
                rows.append(np.asarray(feature, dtype=np.float32).reshape(-1))
        if rows:
            return np.stack(rows, axis=0).astype(np.float32)
        dim = _first_feature_dim(self.buffer.technique_features, default=768)
        return np.empty((0, dim), dtype=np.float32)


def edge_key_to_name(edge_key: EdgeKey) -> str:
    return "__".join(edge_key)


def _score_pairs(value: Any) -> list[tuple[str, float]]:
    if isinstance(value, dict):
        return [(str(k), float(v)) for k, v in value.items()]
    pairs: list[tuple[str, float]] = []
    if not isinstance(value, (list, tuple)):
        return pairs
    for item in value:
        if isinstance(item, dict):
            tech_id = item.get("technique_id") or item.get("id")
            score = item.get("score") or item.get("cosine") or item.get("value")
            if tech_id is not None and score is not None:
                pairs.append((str(tech_id), float(score)))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), float(item[1])))
    return pairs


def _collect_techniques(snapshot: dict[str, Any], packet_entries: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for tech_id, _ in _score_pairs(snapshot.get("flow_to_mitre", [])):
        seen.setdefault(tech_id, None)
    for packet in packet_entries:
        for tech_id, _ in _score_pairs(packet.get("mitre_topk", [])):
            seen.setdefault(tech_id, None)
    return list(seen.keys())


def _collect_tactics(technique_ids: list[str], technique_to_tactic: dict[str, str]) -> list[str]:
    seen: dict[str, None] = {}
    for tech_id in technique_ids:
        tactic_id = technique_to_tactic.get(tech_id)
        if tactic_id:
            seen.setdefault(str(tactic_id), None)
    return list(seen.keys())


def _set_edges(
    edge_index: dict[EdgeKey, np.ndarray],
    edge_weight: dict[EdgeKey, np.ndarray],
    key: EdgeKey,
    edges: list[tuple[int, int]],
    weights: list[float] | None = None,
) -> None:
    if edges:
        edge_index[key] = np.asarray(edges, dtype=np.int64).T
        weight_values = weights if weights is not None else [1.0] * len(edges)
        edge_weight[key] = np.asarray(weight_values, dtype=np.float32)
    else:
        edge_index[key] = np.empty((2, 0), dtype=np.int64)
        edge_weight[key] = np.empty((0,), dtype=np.float32)


def _add_reverse_edges(edge_index: dict[EdgeKey, np.ndarray], edge_weight: dict[EdgeKey, np.ndarray]) -> None:
    for key, value in list(edge_index.items()):
        src_type, relation, dst_type = key
        reverse_key = (dst_type, f"rev_{relation}", src_type)
        edge_index[reverse_key] = value[[1, 0], :].copy()
        edge_weight[reverse_key] = edge_weight[key].copy()


def _first_feature_dim(features: dict[str, np.ndarray], default: int) -> int:
    for value in features.values():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.size > 0:
            return int(arr.reshape(-1).shape[0])
    return int(default)


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
