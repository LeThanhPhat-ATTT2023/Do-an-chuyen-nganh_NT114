from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from graphslm_ids.offline.preprocessing.payload_features import compute_packet_payload_features


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
    host_local_to_id: dict[int, str] | None = None  # set in build()

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
        cold_store: Any | None = None,
        hops: int = 3,
        add_reverse_edges: bool = True,
        standardize_flow_features: bool = True,
        flow_feature_stats: dict[str, Any] | None = None,
        packet_feature: str = "semantic",
        protocol_mapping: dict[str, int] | None = None,
        tactic_shortname_to_idx: dict[str, int] | None = None,
        always_include_all_tactics: bool = True,
        dlg_top_n_enabled: bool = False,
        dlg_top_n_per_seed: dict[str, int] | None = None,
        dlg_sort_by: str = "semantic_edge_weight",
    ) -> None:
        self.buffer = buffer
        self.cold_store = cold_store
        self.hops = int(hops)
        self.add_reverse_edges = bool(add_reverse_edges)
        self.standardize_flow_features = bool(standardize_flow_features)
        self.flow_feature_stats = flow_feature_stats or {}
        self.packet_feature = packet_feature
        self.protocol_mapping = {str(k).upper(): int(v) for k, v in (protocol_mapping or {}).items()}
        self.tactic_shortname_to_idx = {
            str(k): int(v) for k, v in (tactic_shortname_to_idx or {}).items()
        }
        self.always_include_all_tactics = bool(always_include_all_tactics)
        self.dlg_top_n_enabled = bool(dlg_top_n_enabled)
        self.dlg_top_n_per_seed = {
            str(key): int(value) for key, value in (dlg_top_n_per_seed or {}).items()
        }
        self.dlg_sort_by = str(dlg_sort_by)

    def build(self, seed_flow_id: str) -> Subgraph:
        snapshot = self._snapshot(seed_flow_id)
        if not snapshot.get("flow"):
            raise KeyError(f"Flow not found in hot graph cache or graph store: {seed_flow_id}")

        flow_ids = [str(seed_flow_id)]
        packet_entries = self._select_packet_entries(list(snapshot.get("packets", [])))
        packet_ids = [str(packet["packet_id"]) for packet in packet_entries]

        technique_ids = self._collect_selected_techniques(snapshot, packet_entries)
        technique_to_tactic = self._static_mapping("technique_to_tactic")
        if self.always_include_all_tactics:
            tactic_ids = self._all_tactics(technique_to_tactic)
        else:
            tactic_ids = _collect_tactics(technique_ids, technique_to_tactic)

        flow_x = np.asarray([self._flow_features(snapshot["flow"])], dtype=np.float32)
        if self.standardize_flow_features:
            flow_x = self._standardize_flow(flow_x)

        packet_x = self._packet_features(packet_entries)
        technique_x = self._technique_features(technique_ids)
        tactic_x = self._tactic_features(tactic_ids)
        host_x, host_ids, from_host_pairs, to_host_pairs = self._build_hosts(snapshot["flow"])

        flow_idx = {flow_id: idx for idx, flow_id in enumerate(flow_ids)}
        packet_idx = {packet_id: idx for idx, packet_id in enumerate(packet_ids)}
        technique_idx = {tech_id: idx for idx, tech_id in enumerate(technique_ids)}
        tactic_idx = {tactic_id: idx for idx, tactic_id in enumerate(tactic_ids)}

        edge_index: dict[EdgeKey, np.ndarray] = {}
        edge_weight: dict[EdgeKey, np.ndarray] = {}

        contain = [(flow_idx[str(seed_flow_id)], packet_idx[pid]) for pid in packet_ids]
        _set_edges(edge_index, edge_weight, ("flow", "contain", "packet"), contain)

        next_edges = [
            (idx, idx + 1)
            for idx in range(max(len(packet_ids) - 1, 0))
            if self._packets_are_original_neighbors(packet_entries, idx)
        ]
        next_limit = self._top_n("packet__next_packet__packet")
        if next_limit is not None:
            next_edges = next_edges[:next_limit]
        _set_edges(edge_index, edge_weight, ("packet", "next_packet", "packet"), next_edges)

        # Packet -> technique edges, routed by family to evidence_<family>.
        evidence_by_family: dict[str, tuple[list[tuple[int, int]], list[float]]] = {}
        for packet in packet_entries:
            packet_id = str(packet["packet_id"])
            for tech_id, family, score in _triples(packet.get("mitre_topk", [])):
                if tech_id not in technique_idx:
                    continue
                pairs, weights = evidence_by_family.setdefault(str(family), ([], []))
                pairs.append((packet_idx[packet_id], technique_idx[tech_id]))
                weights.append(float(score))
        for family, (pairs, weights) in evidence_by_family.items():
            _set_edges(edge_index, edge_weight, ("packet", f"evidence_{family}", "technique"),
                       pairs, weights)

        flow_tech_edges: list[tuple[int, int]] = []
        flow_tech_weights: list[float] = []
        for tech_id, _family, score in _triples(snapshot.get("flow_to_mitre", [])):
            if tech_id in technique_idx:
                flow_tech_edges.append((0, technique_idx[tech_id]))
                flow_tech_weights.append(float(score))
        _set_edges(edge_index, edge_weight, ("flow", "flow_technique", "technique"),
                   flow_tech_edges, flow_tech_weights)

        tactic_edges: list[tuple[int, int]] = []
        for tech_id in technique_ids:
            tactic_id = technique_to_tactic.get(tech_id)
            if tactic_id in tactic_idx:
                tactic_edges.append((technique_idx[tech_id], tactic_idx[tactic_id]))
        tactic_limit = self._top_n("technique__technique_tactic__tactic")
        if tactic_limit is not None:
            tactic_edges = tactic_edges[:tactic_limit]
        _set_edges(edge_index, edge_weight, ("technique", "technique_tactic", "tactic"), tactic_edges)

        # has_subtechnique: T1190.001 -> parent T1190 (when both are present).
        subtech_edges = []
        for tech_id in technique_ids:
            if "." in tech_id:
                parent = tech_id.split(".")[0]
                if parent in technique_idx:
                    subtech_edges.append((technique_idx[parent], technique_idx[tech_id]))
        _set_edges(edge_index, edge_weight, ("technique", "has_subtechnique", "technique"),
                   subtech_edges)

        host_bytes = float(snapshot["flow"].get("total_payload_bytes", 0.0))
        _set_edges(edge_index, edge_weight, ("flow", "from_host", "host"), from_host_pairs,
                   [host_bytes] * len(from_host_pairs))
        _set_edges(edge_index, edge_weight, ("flow", "to_host", "host"), to_host_pairs,
                   [host_bytes] * len(to_host_pairs))

        if self.add_reverse_edges:
            _add_reverse_edges(edge_index, edge_weight)

        return Subgraph(
            node_features={
                "flow": flow_x,
                "packet": packet_x,
                "technique": technique_x,
                "tactic": tactic_x,
                "host": host_x,
            },
            edge_index_dict=edge_index,
            edge_weight_dict=edge_weight,
            packet_local_to_id={idx: packet_id for packet_id, idx in packet_idx.items()},
            flow_local_to_id={idx: flow_id for flow_id, idx in flow_idx.items()},
            technique_local_to_id={idx: tech_id for tech_id, idx in technique_idx.items()},
            tactic_local_to_id={idx: tactic_id for tactic_id, idx in tactic_idx.items()},
            seed_flow_local_idx=0,
            host_local_to_id={idx: host_id for idx, host_id in enumerate(host_ids)},
        )

    def to_snapshot_dict(self, sub: Subgraph) -> dict[str, Any]:
        return sub.to_snapshot_dict()

    def _snapshot(self, seed_flow_id: str) -> dict[str, Any]:
        hot_snapshot: dict[str, Any] | None = None
        try:
            hot_snapshot = self.buffer.snapshot(seed_flow_id)
        except (AttributeError, KeyError):
            hot_snapshot = None
        if hot_snapshot and hot_snapshot.get("flow"):
            return hot_snapshot

        if self.cold_store is None:
            return hot_snapshot or {}
        try:
            cold_snapshot = self.cold_store.snapshot(seed_flow_id)
        except (AttributeError, KeyError):
            cold_snapshot = None
        if cold_snapshot is not None:
            return cold_snapshot
        return hot_snapshot or {}

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

    def _build_hosts(self, flow: dict[str, Any]):
        """Two host nodes (src, dst) with 4-d [out_deg, in_deg, fwd_bytes, bwd_bytes]."""
        src_ip = str(flow.get("src_ip", "unknown"))
        dst_ip = str(flow.get("dst_ip", "unknown"))
        total_bytes = float(flow.get("total_payload_bytes", 0.0))
        host_ids = [src_ip, dst_ip]
        host_x = np.asarray(
            [
                [1.0, 0.0, total_bytes, 0.0],  # src host: out-degree 1, fwd bytes
                [0.0, 1.0, 0.0, total_bytes],  # dst host: in-degree 1, bwd bytes
            ],
            dtype=np.float32,
        )
        from_host_pairs = [(0, 0)]  # flow 0 -> src host (index 0)
        to_host_pairs = [(0, 1)]    # flow 0 -> dst host (index 1)
        return host_x, host_ids, from_host_pairs, to_host_pairs

    def _select_packet_entries(self, packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = self._top_n("flow__contains__packet")
        if limit is None or len(packets) <= limit:
            return packets
        indexed = list(enumerate(packets))
        selected = sorted(
            indexed,
            key=lambda item: (
                -self._packet_semantic_score(item[1]),
                float(item[1].get("timestamp", item[0]) or item[0]),
                str(item[1].get("packet_id", "")),
            ),
        )[:limit]
        result: list[dict[str, Any]] = []
        for original_idx, packet in sorted(selected, key=lambda item: item[0]):
            copied = dict(packet)
            copied["_subgraph_original_order"] = original_idx
            result.append(copied)
        return result

    def _select_score_pairs(
        self,
        edge_name: str,
        pairs: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        limit = self._top_n(edge_name)
        if limit is None or len(pairs) <= limit:
            return pairs
        return sorted(pairs, key=lambda item: (-float(item[1]), str(item[0])))[:limit]

    def _top_n(self, edge_name: str) -> int | None:
        if not self.dlg_top_n_enabled:
            return None
        value = self.dlg_top_n_per_seed.get(edge_name)
        if value is None:
            parts = edge_name.split("__")
            if len(parts) == 3:
                value = self.dlg_top_n_per_seed.get(parts[1])
        if value is None or int(value) <= 0:
            return None
        return int(value)

    def _packet_semantic_score(self, packet: dict[str, Any]) -> float:
        pairs = [(t, w) for t, _f, w in _triples(packet.get("mitre_topk", []))]
        if not pairs:
            return 0.0
        return max(float(score) for _, score in pairs)

    def _packets_are_original_neighbors(
        self,
        packet_entries: list[dict[str, Any]],
        left_idx: int,
    ) -> bool:
        left = packet_entries[left_idx].get("_subgraph_original_order", left_idx)
        right = packet_entries[left_idx + 1].get("_subgraph_original_order", left_idx + 1)
        try:
            return int(right) == int(left) + 1
        except (TypeError, ValueError):
            return True

    def _collect_selected_techniques(
        self,
        snapshot: dict[str, Any],
        packet_entries: list[dict[str, Any]],
    ) -> list[str]:
        seen: dict[str, None] = {}
        for tech_id, _family, _w in _triples(snapshot.get("flow_to_mitre", [])):
            seen.setdefault(tech_id, None)
        for packet in packet_entries:
            for tech_id, _family, _w in _triples(packet.get("mitre_topk", [])):
                seen.setdefault(tech_id, None)
        return list(seen.keys())

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
        if self.packet_feature == "ordered_byte":
            rows: list[np.ndarray] = []
            for packet in packet_entries:
                payload = _payload_bytes(packet)
                rows.append(compute_packet_payload_features(payload, len(payload)))
            if rows:
                return np.stack(rows, axis=0).astype(np.float32)
            dim = int(compute_packet_payload_features(b"", 0).shape[0])
            return np.empty((0, dim), dtype=np.float32)

        # Legacy embedding path (kept for the on-disk store / semantic mode).
        rows = []
        packet_embeddings = self._static_mapping("packet_embeddings")
        for packet in packet_entries:
            embedding = packet.get("embedding")
            if embedding is None:
                embedding = packet_embeddings.get(str(packet.get("packet_id")))
            if embedding is None:
                continue
            rows.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if rows:
            return np.stack(rows, axis=0).astype(np.float32)

        dim = _first_feature_dim(packet_embeddings, default=768)
        return np.empty((0, dim), dtype=np.float32)

    def _technique_features(self, technique_ids: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        technique_features = self._static_mapping("technique_features")
        for tech_id in technique_ids:
            feature = technique_features.get(tech_id)
            if feature is not None:
                rows.append(np.asarray(feature, dtype=np.float32).reshape(-1))
        if rows:
            return np.stack(rows, axis=0).astype(np.float32)
        dim = _first_feature_dim(technique_features, default=768)
        return np.empty((0, dim), dtype=np.float32)

    def _tactic_features(self, tactic_ids: list[str]) -> np.ndarray:
        if not tactic_ids:
            return np.empty((0, 1), dtype=np.int64)
        if self.tactic_shortname_to_idx:
            values = [
                self.tactic_shortname_to_idx.get(str(tactic_id), idx)
                for idx, tactic_id in enumerate(tactic_ids)
            ]
            return np.asarray(values, dtype=np.int64).reshape(-1, 1)
        return np.arange(len(tactic_ids), dtype=np.int64).reshape(-1, 1)

    def _static_mapping(self, name: str) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for source in (self.cold_store, self.buffer):
            value = getattr(source, name, None) if source is not None else None
            if isinstance(value, dict):
                merged.update(value)
        return merged

    def _all_tactics(self, technique_to_tactic: dict[str, str]) -> list[str]:
        if self.tactic_shortname_to_idx:
            return [
                tactic_id
                for tactic_id, _ in sorted(
                    self.tactic_shortname_to_idx.items(),
                    key=lambda item: item[1],
                )
            ]
        tactic_metadata = self._static_mapping("tactic_metadata")
        if tactic_metadata:
            return list(tactic_metadata.keys())
        seen: dict[str, None] = {}
        for tactic_id in technique_to_tactic.values():
            if tactic_id:
                seen.setdefault(str(tactic_id), None)
        return list(seen.keys())


def edge_key_to_name(edge_key: EdgeKey) -> str:
    return "__".join(edge_key)


def _payload_bytes(packet: dict[str, Any]) -> bytes:
    """Reconstruct raw payload bytes from a packet entry's stored hex preview."""
    hexs = str(packet.get("payload_preview_hex") or packet.get("payload_hex") or "")
    if not hexs:
        return b""
    try:
        return bytes.fromhex(hexs)
    except ValueError:
        return b""


def _triples(raw: Any) -> list[tuple[str, str, float]]:
    """Coerce mitre_topk entries into (technique, family, weight) triples.

    Accepts both new 3-tuples and legacy 2-tuples (family defaulted to '')."""
    out: list[tuple[str, str, float]] = []
    for item in raw or []:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            out.append((str(item[0]), str(item[1]), float(item[2])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), "", float(item[1])))
    return out


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
