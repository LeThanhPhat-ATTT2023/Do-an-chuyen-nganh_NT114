from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import binascii
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator

import numpy as np

from graphslm_ids.runtime.slow_path.hot_buffer_adapter import HotBufferAdapter
from graphslm_ids.runtime.slow_path.types import GraphContext
from graphslm_ids.utils.io import read_json


EDGE_RELATIONS: tuple[str, ...] = (
    "flow__contains__packet",
    "packet__next_packet__packet",
    "packet__matches_technique__technique",
    "flow__matches_technique__technique",
    "technique__belongs_to_tactic__tactic",
)


class PersistentGraphStore:
    """Append-only sharded graph store used as the runtime/training source of truth.

    The implementation intentionally keeps the on-disk files simple JSONL shards.
    That gives the current codebase a concrete, testable source-of-truth contract
    while preserving the layout needed to replace hot-buffer-only state with
    mmap/parquet shards later.
    """

    source_of_truth = True

    def __init__(
        self,
        root: str | Path,
        *,
        shard_duration_seconds: float = 3600.0,
        max_nodes_per_shard: int = 6000,
        packet_embedding_dim: int = 768,
        payload_length: int = 256,
        mitre_index: Any | None = None,
        schema: dict[str, Any] | None = None,
        drop_after_days: float | None = None,
        disk_quota_gb: float | None = None,
        on_quota_exceeded: str = "drop_oldest",
    ) -> None:
        self.root = Path(root)
        self.shard_duration_seconds = float(shard_duration_seconds)
        self.max_nodes_per_shard = int(max_nodes_per_shard)
        self.packet_embedding_dim = int(packet_embedding_dim)
        self.payload_length = int(payload_length)
        self.schema = dict(schema or {})
        self.drop_after_days = float(drop_after_days) if drop_after_days is not None else None
        self.disk_quota_gb = float(disk_quota_gb) if disk_quota_gb is not None else None
        self.on_quota_exceeded = str(on_quota_exceeded)
        self._lock = threading.RLock()
        self._sealed_shards: set[str] = set()
        self._current_shard: dict[str, Any] | None = None

        self.flow_features: dict[str, dict[str, Any]] = {}
        self.flow_to_packets: dict[str, list[str]] = {}
        self.flow_to_mitre: dict[str, list[tuple[str, float]]] = {}
        self.flow_labels: dict[str, int | str] = {}

        self.packet_metadata: dict[str, dict[str, Any]] = {}
        self.packet_payload_text: dict[str, str] = {}
        self.packet_payload_ascii: dict[str, str] = {}
        self.packet_timestamps: dict[str, float] = {}
        self.packet_len_raw: dict[str, int] = {}
        self.packet_attention: dict[str, float] = {}
        self.packet_counterfactual_drop: dict[str, float] = {}
        self.packet_to_flow: dict[str, str] = {}
        self.packet_to_mitre: dict[str, list[tuple[str, float]]] = {}
        self.packet_embeddings: dict[str, np.ndarray] = {}
        self.packet_shard: dict[str, str] = {}

        self.technique_features: dict[str, np.ndarray] = {}
        self.technique_to_tactic: dict[str, str] = {}
        self.tactic_metadata: dict[str, dict[str, Any]] = {}
        self.mitre_metadata: dict[str, dict[str, Any]] = {}

        self._init_layout()
        self._load_state()
        self._load_static_knowledge()
        if mitre_index is not None:
            self._sync_static_knowledge(mitre_index)
        self._load_index()
        self._write_manifest()

    def append_packet(
        self,
        *,
        packet_id: str,
        flow_id: str,
        embedding: np.ndarray,
        payload_hex: str,
        payload_ascii: str,
        payload_len_raw: int,
        timestamp: float,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        mitre_topk: list[tuple[str, float]],
        flow_label: int | str | None = None,
    ) -> str:
        """Append one packet and derived dynamic graph edges.

        This method is write-through friendly: callers should invoke it before
        updating any RAM-only cache. Each call appends immutable records; the
        in-memory index is only a convenience reader over the append log.
        """

        packet_id = str(packet_id)
        flow_id = str(flow_id)
        timestamp = float(timestamp)
        embedding_arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        topk = [(str(tech_id), float(score)) for tech_id, score in mitre_topk]

        with self._lock:
            shard_id = self._ensure_writable_shard(timestamp)
            previous_packets = list(self.flow_to_packets.get(flow_id, []))
            packet_record = {
                "record_type": "packet",
                "packet_id": packet_id,
                "flow_id": flow_id,
                "embedding": embedding_arr,
                "payload_hex": str(payload_hex),
                "payload_ascii": str(payload_ascii),
                "payload_len_raw": int(payload_len_raw),
                "timestamp": timestamp,
                "src_ip": str(src_ip),
                "dst_ip": str(dst_ip),
                "src_port": int(src_port),
                "dst_port": int(dst_port),
                "protocol": str(protocol).upper(),
                "mitre_topk": list(topk),
                "shard_id": shard_id,
            }
            if flow_label is not None:
                packet_record["flow_label"] = flow_label

            self._append_record(("nodes", "packet", "shards", f"{shard_id}.jsonl"), packet_record)
            self._ingest_packet_record(packet_record, recompute_flow=True)
            if flow_label is not None:
                self.flow_labels[flow_id] = flow_label
                self.flow_features.setdefault(flow_id, {})["label"] = flow_label

            self._append_edge(
                "flow__contains__packet",
                shard_id,
                flow_id,
                packet_id,
                timestamp,
            )
            if previous_packets:
                self._append_edge(
                    "packet__next_packet__packet",
                    shard_id,
                    previous_packets[-1],
                    packet_id,
                    timestamp,
                )
            for tech_id, score in topk:
                self._append_edge(
                    "packet__matches_technique__technique",
                    shard_id,
                    packet_id,
                    tech_id,
                    timestamp,
                    weight=score,
                )

            self._refresh_flow_to_mitre(flow_id)
            flow_record = {
                "record_type": "flow_update",
                "flow_id": flow_id,
                "flow": dict(self.flow_features.get(flow_id, {})),
                "flow_to_mitre": list(self.flow_to_mitre.get(flow_id, [])),
                "timestamp": timestamp,
                "shard_id": shard_id,
            }
            if flow_id in self.flow_labels:
                flow_record["label"] = self.flow_labels[flow_id]
            self._append_record(("nodes", "flow", "shards", f"{shard_id}.jsonl"), flow_record)
            for tech_id, score in self.flow_to_mitre.get(flow_id, []):
                self._append_edge(
                    "flow__matches_technique__technique",
                    shard_id,
                    flow_id,
                    tech_id,
                    timestamp,
                    weight=score,
                )

            self._current_shard["node_count"] = int(self._current_shard.get("node_count", 0)) + 1
            self._save_current_shard()
            return shard_id

    def append_alert_snapshot(self, alert_id: str, flow_id: str, snapshot: dict[str, Any]) -> None:
        """Persist alert metadata without making the snapshot the source of truth."""

        timestamp = time.time()
        with self._lock:
            shard_id = self._ensure_writable_shard(timestamp)
            graph_subgraph = snapshot.get("graph_subgraph", {}) if isinstance(snapshot, dict) else {}
            for packet in snapshot.get("packets", []) if isinstance(snapshot, dict) else []:
                packet_id = str(packet.get("packet_id", ""))
                if not packet_id:
                    continue
                if packet.get("attention_weight") is not None:
                    self.packet_attention[packet_id] = float(packet["attention_weight"])
                if packet.get("counterfactual_drop") is not None:
                    self.packet_counterfactual_drop[packet_id] = float(packet["counterfactual_drop"])
            record = {
                "record_type": "alert_snapshot",
                "alert_id": str(alert_id),
                "flow_id": str(flow_id),
                "timestamp": timestamp,
                "graph_subgraph": graph_subgraph,
                "packet_attention": dict(self.packet_attention),
                "shard_id": shard_id,
            }
            self._append_record(("alerts", "shards", f"{shard_id}.jsonl"), record)

    def save_report(
        self,
        *,
        alert_id: str,
        bundle: Any,
        report: str,
        validation: Any,
        fallback_tier: int,
    ) -> None:
        record = {
            "record_type": "report",
            "alert_id": str(alert_id),
            "timestamp": time.time(),
            "fallback_tier": int(fallback_tier),
            "bundle": _json_safe(_object_to_plain(bundle)),
            "report": str(report),
            "validation": _json_safe(_object_to_plain(validation)),
        }
        with self._lock:
            self._append_record(("state", "reports.jsonl"), record)

    def iter_reports(self, since: float | None = None) -> Iterator[dict[str, Any]]:
        min_ts = float(since) if since is not None else None
        path = self.root / "state" / "reports.jsonl"
        with self._lock:
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("record_type") != "report":
                        continue
                    if min_ts is not None and float(record.get("timestamp", 0.0)) < min_ts:
                        continue
                    yield record

    def seal_current_shard(self, reason: str = "manual") -> str | None:
        with self._lock:
            if self._current_shard is None:
                return None
            shard_id = str(self._current_shard["shard_id"])
            self._seal_shard(shard_id, reason=reason)
            self._current_shard = None
            self._save_current_shard()
            return shard_id

    def sealed_shards(self) -> list[str]:
        with self._lock:
            return sorted(self._sealed_shards)

    def enforce_retention(self, now: float | None = None) -> list[str]:
        """Drop old sealed shards according to retention/quota policy.

        The current writable shard is never deleted. This method is deliberately
        conservative and only removes files inside ``self.root`` with known shard
        filenames.
        """

        dropped: list[str] = []
        now_ts = float(now if now is not None else time.time())
        with self._lock:
            if self.drop_after_days is not None:
                cutoff = now_ts - self.drop_after_days * 86400.0
                for shard_id in self._sealed_shards_by_age():
                    start_ts = self._shard_start_ts(shard_id)
                    if start_ts is not None and start_ts < cutoff:
                        self._drop_sealed_shard(shard_id, reason="drop_after_days")
                        dropped.append(shard_id)

            if self.disk_quota_gb is not None and self.on_quota_exceeded == "drop_oldest":
                quota_bytes = int(self.disk_quota_gb * (1024**3))
                for shard_id in self._sealed_shards_by_age():
                    if self._root_size_bytes() <= quota_bytes:
                        break
                    self._drop_sealed_shard(shard_id, reason="disk_quota")
                    dropped.append(shard_id)
        return dropped

    def get_flow(self, flow_id: str) -> dict[str, Any] | None:
        with self._lock:
            flow = self.flow_features.get(str(flow_id))
            return dict(flow) if flow is not None else None

    def get_packets(self, flow_id: str) -> list[dict[str, Any]]:
        with self._lock:
            packets: list[dict[str, Any]] = []
            for order, packet_id in enumerate(self.flow_to_packets.get(str(flow_id), [])):
                if packet_id not in self.packet_metadata:
                    continue
                record = dict(self.packet_metadata[packet_id])
                record["order_in_flow"] = order
                record["payload_preview_hex"] = self.packet_payload_text.get(packet_id, "")
                record["payload_preview_ascii"] = self.packet_payload_ascii.get(packet_id, "")
                record["payload_len_raw"] = self.packet_len_raw.get(packet_id, 0)
                record["mitre_topk"] = list(self.packet_to_mitre.get(packet_id, []))
                record["attention_weight"] = self.packet_attention.get(packet_id)
                record["counterfactual_drop"] = self.packet_counterfactual_drop.get(packet_id)
                packets.append(record)
            return packets

    def snapshot(self, flow_id: str) -> dict[str, Any]:
        flow_id = str(flow_id)
        with self._lock:
            packet_ids = list(self.flow_to_packets.get(flow_id, []))
            flow = dict(self.flow_features.get(flow_id, {}))
            packets = []
            for packet_id in packet_ids:
                embedding = self.packet_embeddings.get(packet_id)
                packets.append(
                    {
                        "packet_id": packet_id,
                        "metadata": dict(self.packet_metadata.get(packet_id, {})),
                        "payload_hex": self.packet_payload_text.get(packet_id, ""),
                        "payload_ascii": self.packet_payload_ascii.get(packet_id, ""),
                        "timestamp": self.packet_timestamps.get(packet_id, 0.0),
                        "payload_len_raw": self.packet_len_raw.get(packet_id, 0),
                        "embedding": embedding.copy() if embedding is not None else None,
                        "mitre_topk": list(self.packet_to_mitre.get(packet_id, [])),
                        "attention_weight": self.packet_attention.get(packet_id),
                        "counterfactual_drop": self.packet_counterfactual_drop.get(packet_id),
                    }
                )
            return {
                "flow_id": flow_id,
                "flow": flow,
                "packets": packets,
                "flow_to_mitre": list(self.flow_to_mitre.get(flow_id, [])),
                "mitre_metadata": dict(self.mitre_metadata),
            }

    def load_context(self, flow_id: str) -> GraphContext | None:
        return HotBufferAdapter(self, mitre_catalog=self.mitre_metadata).get_context(str(flow_id))

    def out_neighbors(self, edge_type: str | tuple[str, str, str], src_id: str) -> list[str] | None:
        relation = edge_key_to_name(edge_type) if isinstance(edge_type, tuple) else str(edge_type)
        src_id = str(src_id)
        with self._lock:
            if relation == "flow__contains__packet":
                return list(self.flow_to_packets.get(src_id, []))
            if relation == "packet__matches_technique__technique":
                if src_id not in self.packet_to_flow:
                    return None
                return [tech_id for tech_id, _ in self.packet_to_mitre.get(src_id, [])]
            if relation == "flow__matches_technique__technique":
                if src_id not in self.flow_features:
                    return None
                return [tech_id for tech_id, _ in self.flow_to_mitre.get(src_id, [])]
            if relation == "technique__belongs_to_tactic__tactic":
                tactic = self.technique_to_tactic.get(src_id)
                return [tactic] if tactic else []
            if relation == "packet__next_packet__packet":
                flow_id = self.packet_to_flow.get(src_id)
                if flow_id is None:
                    return None
                packets = self.flow_to_packets.get(flow_id, [])
                try:
                    idx = packets.index(src_id)
                except ValueError:
                    return None
                return packets[idx + 1 : idx + 2]
        return None

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "flows": len(self.flow_features),
                "packets": len(self.packet_metadata),
                "techniques": len(self.technique_features),
                "sealed_shards": len(self._sealed_shards),
                "current_shard_node_count": int(
                    (self._current_shard or {}).get("node_count", 0)
                ),
            }

    def to_three_tier_arrays(self, *, sealed_only: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Materialize sealed shards into the legacy NPZ-compatible array schema.

        This is a compatibility bridge for the current full-batch trainer. Large
        deployments should replace this with neighbor sampling directly over the
        shard files, but the training entry point can now consume the same source
        of truth as runtime.
        """

        with self._lock:
            sealed = set(self._sealed_shards)
            packet_ids = [
                packet_id
                for packet_id in self.packet_metadata
                if not sealed_only or self.packet_shard.get(packet_id) in sealed
            ]
            flow_ids = [
                flow_id
                for flow_id, packets in self.flow_to_packets.items()
                if any(packet_id in packet_ids for packet_id in packets)
            ]
            observed_techniques: dict[str, None] = {}
            for flow_id in flow_ids:
                for tech_id, _ in self.flow_to_mitre.get(flow_id, []):
                    observed_techniques.setdefault(tech_id, None)
                for packet_id in self.flow_to_packets.get(flow_id, []):
                    if packet_id not in packet_ids:
                        continue
                    for tech_id, _ in self.packet_to_mitre.get(packet_id, []):
                        observed_techniques.setdefault(tech_id, None)

            technique_ids = list(self.technique_features.keys())
            for tech_id in observed_techniques:
                if tech_id not in self.technique_features:
                    technique_ids.append(tech_id)
            tactic_ids = list(self.tactic_metadata.keys())
            for tech_id in technique_ids:
                tactic_id = self.technique_to_tactic.get(tech_id)
                if tactic_id and tactic_id not in tactic_ids:
                    tactic_ids.append(tactic_id)

            flow_idx = {flow_id: idx for idx, flow_id in enumerate(flow_ids)}
            packet_idx = {packet_id: idx for idx, packet_id in enumerate(packet_ids)}
            technique_idx = {tech_id: idx for idx, tech_id in enumerate(technique_ids)}
            tactic_idx = {tactic_id: idx for idx, tactic_id in enumerate(tactic_ids)}
            protocol_mapping = self._protocol_mapping()

            flow_rows = [
                self._flow_feature_vector(self.flow_features.get(flow_id, {}))
                for flow_id in flow_ids
            ]
            flow_x = (
                np.asarray(flow_rows, dtype=np.float32)
                if flow_rows
                else np.empty((0, 6), dtype=np.float32)
            )
            label_mapping: dict[str, int] = {}
            labels: list[int] = []
            for flow_id in flow_ids:
                raw_label = self.flow_labels.get(flow_id, self.flow_features.get(flow_id, {}).get("label", 0))
                if isinstance(raw_label, str):
                    label_mapping.setdefault(raw_label, len(label_mapping))
                    labels.append(label_mapping[raw_label])
                else:
                    labels.append(int(raw_label or 0))
            flow_y = np.asarray(labels, dtype=np.int64)

            packet_rows = [
                self.packet_embeddings.get(
                    packet_id,
                    np.zeros((self.packet_embedding_dim,), dtype=np.float32),
                )
                for packet_id in packet_ids
            ]
            packet_semantic_x = (
                np.asarray(packet_rows, dtype=np.float32)
                if packet_rows
                else np.empty((0, self.packet_embedding_dim), dtype=np.float32)
            )
            payload_rows = [self._payload_vector(packet_id) for packet_id in packet_ids]
            packet_x = (
                np.asarray(payload_rows, dtype=np.uint8)
                if payload_rows
                else np.empty((0, self.payload_length), dtype=np.uint8)
            )
            technique_dim = (
                packet_semantic_x.shape[1]
                if packet_semantic_x.ndim == 2 and packet_semantic_x.shape[1] > 0
                else self.packet_embedding_dim
            )
            technique_rows = [
                self.technique_features.get(
                    tech_id,
                    np.zeros((technique_dim,), dtype=np.float32),
                )
                for tech_id in technique_ids
            ]
            technique_x = (
                np.asarray(technique_rows, dtype=np.float32)
                if technique_rows
                else np.empty((0, technique_dim), dtype=np.float32)
            )

            contain_edges: list[tuple[int, int]] = []
            link_edges: list[tuple[int, int]] = []
            packet_tech_edges: list[tuple[int, int]] = []
            packet_tech_weights: list[float] = []
            flow_tech_edges: list[tuple[int, int]] = []
            flow_tech_weights: list[float] = []
            technique_tactic_edges: list[tuple[int, int]] = []

            for flow_id in flow_ids:
                local_packets = [
                    packet_id
                    for packet_id in self.flow_to_packets.get(flow_id, [])
                    if packet_id in packet_idx
                ]
                for packet_id in local_packets:
                    contain_edges.append((flow_idx[flow_id], packet_idx[packet_id]))
                    for tech_id, score in self.packet_to_mitre.get(packet_id, []):
                        if tech_id in technique_idx:
                            packet_tech_edges.append((packet_idx[packet_id], technique_idx[tech_id]))
                            packet_tech_weights.append(float(score))
                for left, right in zip(local_packets, local_packets[1:]):
                    link_edges.append((packet_idx[left], packet_idx[right]))
                for tech_id, score in self.flow_to_mitre.get(flow_id, []):
                    if tech_id in technique_idx:
                        flow_tech_edges.append((flow_idx[flow_id], technique_idx[tech_id]))
                        flow_tech_weights.append(float(score))

            for tech_id in technique_ids:
                tactic_id = self.technique_to_tactic.get(tech_id)
                if tactic_id in tactic_idx:
                    technique_tactic_edges.append((technique_idx[tech_id], tactic_idx[tactic_id]))

            arrays = {
                "flow_x": flow_x,
                "flow_y": flow_y,
                "packet_x": packet_x,
                "packet_semantic_x": packet_semantic_x,
                "technique_x": technique_x,
                "contain_edge_index": _edge_index_array(contain_edges),
                "link_edge_index": _edge_index_array(link_edges),
                "link_edge_attr": np.ones((len(link_edges), 1), dtype=np.float32),
                "packet_technique_edge_index": _edge_index_array(packet_tech_edges),
                "packet_technique_edge_attr": np.asarray(packet_tech_weights, dtype=np.float32)[:, None]
                if packet_tech_weights
                else np.empty((0, 1), dtype=np.float32),
                "flow_technique_edge_index": _edge_index_array(flow_tech_edges),
                "flow_technique_edge_attr": np.asarray(flow_tech_weights, dtype=np.float32)[:, None]
                if flow_tech_weights
                else np.empty((0, 1), dtype=np.float32),
                "technique_tactic_edge_index": _edge_index_array(technique_tactic_edges),
                "technique_tactic_edge_attr": np.ones(
                    (len(technique_tactic_edges), 1),
                    dtype=np.float32,
                ),
            }
            metadata = {
                "source": "persistent_graph_store",
                "graph_store_root": str(self.root),
                "sealed_only": bool(sealed_only),
                "sealed_shards": sorted(sealed),
                "num_flows": len(flow_ids),
                "num_packets": len(packet_ids),
                "num_techniques": len(technique_ids),
                "num_tactics": len(tactic_ids),
                "num_tactic_edges": len(technique_tactic_edges),
                "label_mapping": label_mapping,
                "flow_feature_names": [
                    "packet_count",
                    "total_payload_bytes",
                    "duration_seconds",
                    "src_port",
                    "dst_port",
                    "protocol_id",
                ],
                "protocol_mapping": dict(protocol_mapping),
                "technique_id_to_idx": technique_idx,
                "tactic_shortname_to_idx": tactic_idx,
            }
            return arrays, metadata

    def _init_layout(self) -> None:
        for parts in [
            ("nodes", "flow", "shards"),
            ("nodes", "packet", "shards"),
            ("nodes", "technique"),
            ("nodes", "tactic"),
            ("edges", "flow__contains__packet", "shards"),
            ("edges", "packet__next_packet__packet", "shards"),
            ("edges", "packet__matches_technique__technique", "shards"),
            ("edges", "flow__matches_technique__technique", "shards"),
            ("edges", "technique__belongs_to_tactic__tactic"),
            ("alerts", "shards"),
            ("state",),
        ]:
            (self.root.joinpath(*parts)).mkdir(parents=True, exist_ok=True)

    def _write_manifest(self) -> None:
        manifest = {
            "version": "v1",
            "created_or_updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "layout": "append_only_jsonl_shards",
            "schema": {
                "node_types": ["flow", "packet", "technique", "tactic"],
                "edge_relations": list(EDGE_RELATIONS),
                "flow_feature_names": [
                    "packet_count",
                    "total_payload_bytes",
                    "duration_seconds",
                    "src_port",
                    "dst_port",
                    "protocol_id",
                ],
                "packet_feature_kind": "semantic",
                "packet_embedding_dim": self.packet_embedding_dim,
                "payload_length": self.payload_length,
                **self.schema,
            },
            "shard_seal": {
                "by_time_seconds": self.shard_duration_seconds,
                "by_size_nodes": self.max_nodes_per_shard,
            },
            "retention": {
                "drop_after_days": self.drop_after_days,
                "disk_quota_gb": self.disk_quota_gb,
                "on_quota_exceeded": self.on_quota_exceeded,
            },
        }
        self._write_json(self.root / "manifest.json", manifest)

    def _ensure_writable_shard(self, timestamp: float) -> str:
        if self._current_shard is not None and self._current_shard.get("sealed"):
            self._current_shard = None

        if self._current_shard is not None:
            start_ts = float(self._current_shard.get("start_ts", timestamp))
            node_count = int(self._current_shard.get("node_count", 0))
            expired = timestamp - start_ts >= self.shard_duration_seconds
            full = node_count >= self.max_nodes_per_shard
            if not expired and not full:
                return str(self._current_shard["shard_id"])
            self._seal_shard(str(self._current_shard["shard_id"]), reason="seal_policy")

        shard_id = self._make_shard_id(timestamp)
        suffix = 0
        base = shard_id
        while shard_id in self._sealed_shards:
            suffix += 1
            shard_id = f"{base}_{suffix}"
        self._current_shard = {
            "shard_id": shard_id,
            "start_ts": timestamp,
            "node_count": 0,
            "sealed": False,
        }
        self._save_current_shard()
        return shard_id

    def _seal_shard(self, shard_id: str, reason: str) -> None:
        if not shard_id or shard_id in self._sealed_shards:
            return
        self._sealed_shards.add(shard_id)
        if self._current_shard is not None and self._current_shard.get("shard_id") == shard_id:
            self._current_shard["sealed"] = True
            self._current_shard["sealed_at_utc"] = datetime.now(timezone.utc).isoformat()
        record = {
            "shard_id": shard_id,
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": str(reason),
        }
        self._append_record(("state", "seal_log.jsonl"), record)

    def _append_edge(
        self,
        relation: str,
        shard_id: str,
        src: str,
        dst: str,
        timestamp: float,
        weight: float = 1.0,
    ) -> None:
        self._append_record(
            ("edges", relation, "shards", f"{shard_id}.jsonl"),
            {
                "record_type": "edge",
                "relation": relation,
                "src": str(src),
                "dst": str(dst),
                "weight": float(weight),
                "timestamp": float(timestamp),
                "shard_id": str(shard_id),
            },
        )

    def _append_record(self, parts: tuple[str, ...], record: dict[str, Any]) -> None:
        path = self.root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(record), sort_keys=True, ensure_ascii=False) + "\n")

    def _ingest_packet_record(self, record: dict[str, Any], *, recompute_flow: bool) -> None:
        packet_id = str(record["packet_id"])
        flow_id = str(record["flow_id"])
        timestamp = float(record.get("timestamp", 0.0) or 0.0)
        topk = [
            (str(tech_id), float(score))
            for tech_id, score in _score_pairs(record.get("mitre_topk", []))
        ]
        metadata = {
            "packet_id": packet_id,
            "flow_id": flow_id,
            "src_ip": str(record.get("src_ip", "unknown")),
            "dst_ip": str(record.get("dst_ip", "unknown")),
            "src_port": int(record.get("src_port", 0) or 0),
            "dst_port": int(record.get("dst_port", 0) or 0),
            "protocol": str(record.get("protocol", "OTHER")).upper(),
            "timestamp": timestamp,
            "payload_len_raw": int(record.get("payload_len_raw", 0) or 0),
            "mitre_topk": list(topk),
            "shard_id": str(record.get("shard_id", "")),
        }
        self.packet_metadata[packet_id] = metadata
        self.packet_payload_text[packet_id] = str(record.get("payload_hex", ""))
        self.packet_payload_ascii[packet_id] = str(record.get("payload_ascii", ""))
        self.packet_timestamps[packet_id] = timestamp
        self.packet_len_raw[packet_id] = int(record.get("payload_len_raw", 0) or 0)
        self.packet_to_flow[packet_id] = flow_id
        self.packet_to_mitre[packet_id] = topk
        self.packet_embeddings[packet_id] = np.asarray(
            record.get("embedding", np.zeros((self.packet_embedding_dim,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        self.packet_shard[packet_id] = str(record.get("shard_id", ""))
        if "flow_label" in record:
            self.flow_labels[flow_id] = record["flow_label"]

        packets = self.flow_to_packets.setdefault(flow_id, [])
        if packet_id not in packets:
            packets.append(packet_id)
        packets.sort(key=lambda pid: self.packet_timestamps.get(pid, 0.0))
        if recompute_flow:
            self._recompute_flow(flow_id)
            self._refresh_flow_to_mitre(flow_id)

    def _ingest_flow_record(self, record: dict[str, Any]) -> None:
        flow_id = str(record.get("flow_id", ""))
        if not flow_id:
            return
        flow = dict(record.get("flow", {}) or {})
        if flow:
            self.flow_features[flow_id] = flow
        flow_mitre = _score_pairs(record.get("flow_to_mitre", []))
        if flow_mitre:
            self.flow_to_mitre[flow_id] = flow_mitre
        if "label" in record:
            self.flow_labels[flow_id] = record["label"]
            self.flow_features.setdefault(flow_id, {})["label"] = record["label"]

    def _recompute_flow(self, flow_id: str) -> None:
        packet_ids = [pid for pid in self.flow_to_packets.get(flow_id, []) if pid in self.packet_metadata]
        if not packet_ids:
            self.flow_features.pop(flow_id, None)
            return

        first_meta = self.packet_metadata[packet_ids[0]]
        timestamps = [self.packet_timestamps.get(pid, 0.0) for pid in packet_ids]
        lengths = [self.packet_len_raw.get(pid, 0) for pid in packet_ids]
        duration = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
        mean_len = float(sum(lengths) / max(len(lengths), 1))
        flow_stats = {
            "mean_pkt_len": mean_len,
            "min_pkt_len": float(min(lengths, default=0)),
            "max_pkt_len": float(max(lengths, default=0)),
            "std_pkt_len": float(np.std(np.asarray(lengths, dtype=np.float32))) if lengths else 0.0,
        }
        if len(timestamps) >= 2:
            ordered = sorted(timestamps)
            iats = [b - a for a, b in zip(ordered, ordered[1:]) if b >= a]
            if iats:
                flow_stats["mean_iat_ms"] = float(sum(iats) / len(iats) * 1000.0)

        flow = {
            "flow_id": flow_id,
            "src_ip": first_meta.get("src_ip", "unknown"),
            "dst_ip": first_meta.get("dst_ip", "unknown"),
            "src_port": int(first_meta.get("src_port", 0)),
            "dst_port": int(first_meta.get("dst_port", 0)),
            "protocol": str(first_meta.get("protocol", "OTHER")).upper(),
            "packet_count": len(packet_ids),
            "total_payload_bytes": int(sum(lengths)),
            "duration_seconds": float(duration),
            "start_timestamp": float(min(timestamps)),
            "end_timestamp": float(max(timestamps)),
            "packet_ids": list(packet_ids),
            "flow_feature_stats": flow_stats,
            "mitre_topk": list(self.flow_to_mitre.get(flow_id, [])),
        }
        if flow_id in self.flow_labels:
            flow["label"] = self.flow_labels[flow_id]
        self.flow_features[flow_id] = flow

    def _refresh_flow_to_mitre(self, flow_id: str) -> None:
        pooled: dict[str, float] = {}
        for packet_id in self.flow_to_packets.get(flow_id, []):
            for tech_id, score in self.packet_to_mitre.get(packet_id, []):
                pooled[tech_id] = max(pooled.get(tech_id, float("-inf")), float(score))
        self.flow_to_mitre[flow_id] = sorted(
            pooled.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if flow_id in self.flow_features:
            self.flow_features[flow_id]["mitre_topk"] = list(self.flow_to_mitre[flow_id])

    def _load_index(self) -> None:
        with self._lock:
            for path in sorted((self.root / "nodes" / "packet" / "shards").glob("*.jsonl")):
                for record in _iter_jsonl(path):
                    if record.get("record_type") == "packet":
                        self._ingest_packet_record(record, recompute_flow=False)
            for flow_id in list(self.flow_to_packets):
                self._recompute_flow(flow_id)
                self._refresh_flow_to_mitre(flow_id)
            for path in sorted((self.root / "nodes" / "flow" / "shards").glob("*.jsonl")):
                for record in _iter_jsonl(path):
                    if record.get("record_type") == "flow_update":
                        self._ingest_flow_record(record)
            for path in sorted((self.root / "alerts" / "shards").glob("*.jsonl")):
                for record in _iter_jsonl(path):
                    if record.get("record_type") != "alert_snapshot":
                        continue
                    for packet_id, score in dict(record.get("packet_attention", {}) or {}).items():
                        if packet_id in self.packet_metadata:
                            self.packet_attention[str(packet_id)] = float(score)

    def _load_state(self) -> None:
        seal_log = self.root / "state" / "seal_log.jsonl"
        for record in _iter_jsonl(seal_log):
            shard_id = record.get("shard_id")
            if shard_id:
                self._sealed_shards.add(str(shard_id))
        for record in _iter_jsonl(self.root / "state" / "retention_log.jsonl"):
            shard_id = record.get("shard_id")
            if shard_id:
                self._sealed_shards.discard(str(shard_id))

        current_path = self.root / "state" / "current_shard.json"
        if current_path.exists():
            try:
                current = read_json(current_path)
            except json.JSONDecodeError:
                current = None
            if isinstance(current, dict) and current.get("shard_id") and not current.get("sealed"):
                self._current_shard = current

    def _save_current_shard(self) -> None:
        self._write_json(self.root / "state" / "current_shard.json", self._current_shard or {})

    def _drop_sealed_shard(self, shard_id: str, reason: str) -> None:
        if shard_id not in self._sealed_shards:
            return
        for path in self._shard_paths(shard_id):
            if path.exists() and self._is_within_root(path):
                path.unlink()
        self._sealed_shards.discard(shard_id)
        self._drop_shard_from_memory(shard_id)
        self._append_record(
            ("state", "retention_log.jsonl"),
            {
                "shard_id": shard_id,
                "dropped_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": str(reason),
            },
        )

    def _drop_shard_from_memory(self, shard_id: str) -> None:
        packet_ids = [pid for pid, sid in self.packet_shard.items() if sid == shard_id]
        affected_flows = {self.packet_to_flow.get(pid) for pid in packet_ids}
        for packet_id in packet_ids:
            self.packet_metadata.pop(packet_id, None)
            self.packet_payload_text.pop(packet_id, None)
            self.packet_payload_ascii.pop(packet_id, None)
            self.packet_timestamps.pop(packet_id, None)
            self.packet_len_raw.pop(packet_id, None)
            self.packet_attention.pop(packet_id, None)
            self.packet_counterfactual_drop.pop(packet_id, None)
            self.packet_to_mitre.pop(packet_id, None)
            self.packet_embeddings.pop(packet_id, None)
            self.packet_shard.pop(packet_id, None)
            flow_id = self.packet_to_flow.pop(packet_id, None)
            if flow_id in self.flow_to_packets:
                self.flow_to_packets[flow_id] = [
                    existing for existing in self.flow_to_packets[flow_id] if existing != packet_id
                ]
        for flow_id in affected_flows:
            if not flow_id:
                continue
            if self.flow_to_packets.get(flow_id):
                self._recompute_flow(flow_id)
                self._refresh_flow_to_mitre(flow_id)
            else:
                self.flow_to_packets.pop(flow_id, None)
                self.flow_features.pop(flow_id, None)
                self.flow_to_mitre.pop(flow_id, None)
                self.flow_labels.pop(flow_id, None)

    def _shard_paths(self, shard_id: str) -> list[Path]:
        filename = f"{shard_id}.jsonl"
        paths = [
            self.root / "nodes" / "flow" / "shards" / filename,
            self.root / "nodes" / "packet" / "shards" / filename,
            self.root / "alerts" / "shards" / filename,
        ]
        for relation in EDGE_RELATIONS:
            if relation == "technique__belongs_to_tactic__tactic":
                continue
            paths.append(self.root / "edges" / relation / "shards" / filename)
        return paths

    def _sealed_shards_by_age(self) -> list[str]:
        return sorted(
            self._sealed_shards,
            key=lambda shard_id: self._shard_start_ts(shard_id) or float("inf"),
        )

    def _shard_start_ts(self, shard_id: str) -> float | None:
        stem = shard_id
        if stem.startswith("shard_"):
            stem = stem[len("shard_") :]
        stem = stem.split("_", 1)[0]
        try:
            dt = datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return float(dt.timestamp())

    def _root_size_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += int(path.stat().st_size)
        return total

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root.resolve())
            return True
        except ValueError:
            return False

    def _sync_static_knowledge(self, mitre_index: Any) -> None:
        technique_features = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in getattr(mitre_index, "technique_features", {}).items()
        }
        technique_to_tactic = dict(getattr(mitre_index, "technique_to_tactic", {}))
        tactic_metadata = dict(getattr(mitre_index, "tactic_metadata", {}))
        technique_metadata = dict(getattr(mitre_index, "technique_metadata", {}))

        if technique_features:
            self.technique_features = technique_features
        if technique_to_tactic:
            self.technique_to_tactic = {str(k): str(v) for k, v in technique_to_tactic.items()}
        if tactic_metadata:
            self.tactic_metadata = {
                str(k): _json_safe(v) if isinstance(v, dict) else {"name": str(v)}
                for k, v in tactic_metadata.items()
            }
        if technique_metadata:
            self.mitre_metadata = {
                str(k): _json_safe(v) if isinstance(v, dict) else {"technique_name": str(v)}
                for k, v in technique_metadata.items()
            }
        self._write_static_knowledge()

    def _write_static_knowledge(self) -> None:
        self._write_jsonl(
            self.root / "nodes" / "technique" / "features.jsonl",
            [
                {
                    "technique_id": tech_id,
                    "embedding": feature,
                    "metadata": self.mitre_metadata.get(tech_id, {}),
                }
                for tech_id, feature in self.technique_features.items()
            ],
        )
        self._write_jsonl(
            self.root / "nodes" / "tactic" / "metadata.jsonl",
            [
                {"tactic_id": tactic_id, "metadata": metadata}
                for tactic_id, metadata in self.tactic_metadata.items()
            ],
        )
        self._write_jsonl(
            self.root / "edges" / "technique__belongs_to_tactic__tactic" / "static.jsonl",
            [
                {
                    "relation": "technique__belongs_to_tactic__tactic",
                    "src": tech_id,
                    "dst": tactic_id,
                    "weight": 1.0,
                }
                for tech_id, tactic_id in self.technique_to_tactic.items()
            ],
        )

    def _load_static_knowledge(self) -> None:
        for record in _iter_jsonl(self.root / "nodes" / "technique" / "features.jsonl"):
            tech_id = str(record.get("technique_id", ""))
            if not tech_id:
                continue
            self.technique_features[tech_id] = np.asarray(record.get("embedding", []), dtype=np.float32)
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                self.mitre_metadata[tech_id] = metadata
        for record in _iter_jsonl(self.root / "nodes" / "tactic" / "metadata.jsonl"):
            tactic_id = str(record.get("tactic_id", ""))
            if tactic_id:
                self.tactic_metadata[tactic_id] = dict(record.get("metadata", {}) or {})
        for record in _iter_jsonl(
            self.root / "edges" / "technique__belongs_to_tactic__tactic" / "static.jsonl"
        ):
            src = record.get("src")
            dst = record.get("dst")
            if src is not None and dst is not None:
                self.technique_to_tactic[str(src)] = str(dst)

    def _flow_feature_vector(self, flow: dict[str, Any]) -> list[float]:
        protocol = str(flow.get("protocol", "OTHER")).upper()
        protocol_mapping = self._protocol_mapping()
        protocol_id = int(protocol_mapping.get(protocol, protocol_mapping.get("OTHER", 0)))
        return [
            float(flow.get("packet_count", 0.0)),
            float(flow.get("total_payload_bytes", 0.0)),
            float(flow.get("duration_seconds", 0.0)),
            float(flow.get("src_port", 0.0)),
            float(flow.get("dst_port", 0.0)),
            float(protocol_id),
        ]

    def _protocol_mapping(self) -> dict[str, int]:
        raw = self.schema.get("protocol_mapping", {})
        if isinstance(raw, dict) and raw:
            return {str(key).upper(): int(value) for key, value in raw.items()}
        return {"OTHER": 0, "TCP": 1, "UDP": 2, "ICMP": 3}

    def _payload_vector(self, packet_id: str) -> np.ndarray:
        raw = _hex_to_bytes(self.packet_payload_text.get(packet_id, ""))
        vector = np.zeros((self.payload_length,), dtype=np.uint8)
        if raw:
            clipped = raw[: self.payload_length]
            vector[: len(clipped)] = np.frombuffer(clipped, dtype=np.uint8)
        return vector

    def _make_shard_id(self, timestamp: float) -> str:
        dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        return f"shard_{dt.strftime('%Y%m%dT%H%M%SZ')}"

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, sort_keys=True, ensure_ascii=False, indent=2)

    def _write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_json_safe(record), sort_keys=True, ensure_ascii=False) + "\n")


def edge_key_to_name(edge_key: str | tuple[str, str, str]) -> str:
    if isinstance(edge_key, tuple):
        return "__".join(edge_key)
    return str(edge_key)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _score_pairs(value: Any) -> list[tuple[str, float]]:
    if isinstance(value, dict):
        return [(str(k), float(v)) for k, v in value.items()]
    pairs: list[tuple[str, float]] = []
    if not isinstance(value, (list, tuple)):
        return pairs
    for item in value:
        if isinstance(item, dict):
            tech_id = item.get("technique_id") or item.get("id")
            score = item.get("score")
            if score is None:
                score = item.get("cosine")
            if score is None:
                score = item.get("value")
            if tech_id is not None and score is not None:
                pairs.append((str(tech_id), float(score)))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), float(item[1])))
    return pairs


def _edge_index_array(edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def _hex_to_bytes(value: str) -> bytes:
    cleaned = str(value or "").replace(" ", "").strip()
    if len(cleaned) % 2 != 0:
        cleaned = cleaned[:-1]
    if not cleaned:
        return b""
    try:
        return binascii.unhexlify(cleaned)
    except (binascii.Error, ValueError):
        return b""


def _object_to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
