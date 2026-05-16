from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any

import numpy as np


@dataclass
class _Event:
    timestamp: float
    packet_id: str
    flow_id: str


class HotGraphBuffer:
    """Thread-safe hot graph state shared by online detection and slow-path hydration."""

    def __init__(
        self,
        ttl_seconds: float = 60.0,
        max_events: int = 100_000,
        max_packets_per_flow: int = 64,
        max_techniques_per_node: int = 5,
        mitre_index: Any | None = None,
    ) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self.max_events = int(max_events)
        self.max_packets_per_flow = int(max_packets_per_flow)
        self.max_techniques_per_node = int(max_techniques_per_node)
        self._lock = threading.RLock()
        self._event_queue: deque[_Event] = deque()

        self.flow_features: dict[str, dict[str, Any]] = {}
        self.flow_to_packets: dict[str, list[str]] = {}
        self.flow_to_mitre: dict[str, list[tuple[str, float]]] = {}

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

        self.technique_features: dict[str, np.ndarray] = {}
        self.technique_to_tactic: dict[str, str] = {}
        self.tactic_metadata: dict[str, dict[str, Any]] = {}
        self.mitre_metadata: dict[str, dict[str, Any]] = {}

        if mitre_index is not None:
            self.technique_features = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in getattr(mitre_index, "technique_features", {}).items()
            }
            self.technique_to_tactic = dict(getattr(mitre_index, "technique_to_tactic", {}))
            self.tactic_metadata = dict(getattr(mitre_index, "tactic_metadata", {}))
            self.mitre_metadata = dict(getattr(mitre_index, "technique_metadata", {}))

    def add_packet(
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
    ) -> None:
        packet_id = str(packet_id)
        flow_id = str(flow_id)
        timestamp = float(timestamp)
        with self._lock:
            self._event_queue.append(_Event(timestamp, packet_id, flow_id))
            topk = [
                (str(tech_id), float(score))
                for tech_id, score in mitre_topk[: self.max_techniques_per_node]
            ]
            metadata = {
                "packet_id": packet_id,
                "flow_id": flow_id,
                "src_ip": str(src_ip),
                "dst_ip": str(dst_ip),
                "src_port": int(src_port),
                "dst_port": int(dst_port),
                "protocol": str(protocol).upper(),
                "timestamp": timestamp,
                "payload_len_raw": int(payload_len_raw),
                "mitre_topk": list(topk),
            }
            self.packet_metadata[packet_id] = metadata
            self.packet_payload_text[packet_id] = str(payload_hex)
            self.packet_payload_ascii[packet_id] = str(payload_ascii)
            self.packet_timestamps[packet_id] = timestamp
            self.packet_len_raw[packet_id] = int(payload_len_raw)
            self.packet_to_flow[packet_id] = flow_id
            self.packet_to_mitre[packet_id] = topk
            self.packet_embeddings[packet_id] = np.asarray(embedding, dtype=np.float32).copy()

            packet_ids = self.flow_to_packets.setdefault(flow_id, [])
            packet_ids.append(packet_id)
            while len(packet_ids) > self.max_packets_per_flow:
                self._purge_packet(packet_ids.pop(0))

            self._recompute_flow(flow_id)
            self._refresh_flow_to_mitre(flow_id)
            self._enforce_max_events()

    def update_attention(self, packet_attention: dict[str, float]) -> None:
        with self._lock:
            for packet_id, score in packet_attention.items():
                if packet_id in self.packet_metadata:
                    self.packet_attention[str(packet_id)] = float(score)

    def update_counterfactual(self, packet_cf: dict[str, float]) -> None:
        with self._lock:
            for packet_id, score in packet_cf.items():
                if packet_id in self.packet_metadata:
                    self.packet_counterfactual_drop[str(packet_id)] = float(score)

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

    def evict_expired(self, now: float) -> list[str]:
        evicted_flows: list[str] = []
        with self._lock:
            while self._event_queue and (float(now) - self._event_queue[0].timestamp) > self.ttl_seconds:
                event = self._event_queue.popleft()
                if event.packet_id not in self.packet_to_flow:
                    continue
                flow_id = self.packet_to_flow.get(event.packet_id, event.flow_id)
                self._purge_packet(event.packet_id)
                if not self.flow_to_packets.get(flow_id):
                    self._purge_flow(flow_id)
                    evicted_flows.append(flow_id)
                else:
                    self._recompute_flow(flow_id)
                    self._refresh_flow_to_mitre(flow_id)
        return evicted_flows

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

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "ttl_seconds": self.ttl_seconds,
                "flows": len(self.flow_features),
                "packets": len(self.packet_metadata),
                "events": len(self._event_queue),
                "techniques": len(self.technique_features),
            }

    def __contains__(self, flow_id: str) -> bool:
        with self._lock:
            return str(flow_id) in self.flow_features

    def _enforce_max_events(self) -> None:
        while len(self._event_queue) > self.max_events:
            event = self._event_queue.popleft()
            if event.packet_id not in self.packet_to_flow:
                continue
            flow_id = self.packet_to_flow[event.packet_id]
            self._purge_packet(event.packet_id)
            if not self.flow_to_packets.get(flow_id):
                self._purge_flow(flow_id)
            else:
                self._recompute_flow(flow_id)
                self._refresh_flow_to_mitre(flow_id)

    def _purge_packet(self, packet_id: str) -> None:
        flow_id = self.packet_to_flow.pop(packet_id, None)
        if flow_id is not None and flow_id in self.flow_to_packets:
            self.flow_to_packets[flow_id] = [
                existing for existing in self.flow_to_packets[flow_id] if existing != packet_id
            ]
        self.packet_metadata.pop(packet_id, None)
        self.packet_payload_text.pop(packet_id, None)
        self.packet_payload_ascii.pop(packet_id, None)
        self.packet_timestamps.pop(packet_id, None)
        self.packet_len_raw.pop(packet_id, None)
        self.packet_attention.pop(packet_id, None)
        self.packet_counterfactual_drop.pop(packet_id, None)
        self.packet_to_mitre.pop(packet_id, None)
        self.packet_embeddings.pop(packet_id, None)

    def _purge_flow(self, flow_id: str) -> None:
        for packet_id in list(self.flow_to_packets.get(flow_id, [])):
            self._purge_packet(packet_id)
        self.flow_to_packets.pop(flow_id, None)
        self.flow_features.pop(flow_id, None)
        self.flow_to_mitre.pop(flow_id, None)

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

        self.flow_features[flow_id] = {
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

    def _refresh_flow_to_mitre(self, flow_id: str) -> None:
        pooled: dict[str, float] = {}
        for packet_id in self.flow_to_packets.get(flow_id, []):
            for tech_id, score in self.packet_to_mitre.get(packet_id, []):
                pooled[tech_id] = max(pooled.get(tech_id, float("-inf")), float(score))
        top = sorted(pooled.items(), key=lambda item: item[1], reverse=True)[
            : self.max_techniques_per_node
        ]
        self.flow_to_mitre[flow_id] = top
        if flow_id in self.flow_features:
            self.flow_features[flow_id]["mitre_topk"] = list(top)
