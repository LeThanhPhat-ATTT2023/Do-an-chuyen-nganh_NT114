from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator

import numpy as np

from graphslm_ids.runtime.slow_path.hot_buffer_adapter import HotBufferAdapter
from graphslm_ids.runtime.slow_path.types import GraphContext


class ColdStore:
    """Append-only JSONL context and report store for slow-path fallback."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._latest_snapshot_by_flow: dict[str, dict[str, Any]] = {}
        self._load_index()

    def append_alert_snapshot(self, alert_id: str, flow_id: str, snapshot: dict[str, Any]) -> None:
        record = {
            "type": "alert_snapshot",
            "alert_id": str(alert_id),
            "flow_id": str(flow_id),
            "ts": time.time(),
            "snapshot": _json_safe(snapshot),
        }
        with self._lock:
            self._append(record)
            self._latest_snapshot_by_flow[str(flow_id)] = record

    def load_context(self, flow_id: str) -> GraphContext | None:
        with self._lock:
            record = self._latest_snapshot_by_flow.get(str(flow_id))
        if record is None:
            return None
        source = _SnapshotSource(record.get("snapshot", {}))
        return HotBufferAdapter(source).get_context(str(flow_id))

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
            "type": "report",
            "alert_id": str(alert_id),
            "ts": time.time(),
            "fallback_tier": int(fallback_tier),
            "bundle": _object_to_plain(bundle),
            "report": str(report),
            "validation": _object_to_plain(validation),
        }
        with self._lock:
            self._append(record)

    def iter_reports(self, since: float | None = None) -> Iterator[dict[str, Any]]:
        min_ts = float(since) if since is not None else None
        with self._lock:
            if not self.path.exists():
                return
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("type") != "report":
                        continue
                    if min_ts is not None and float(record.get("ts", 0.0)) < min_ts:
                        continue
                    yield record

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def _load_index(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") == "alert_snapshot":
                    flow_id = str(record.get("flow_id", ""))
                    if flow_id:
                        self._latest_snapshot_by_flow[flow_id] = record


class _SnapshotSource:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.flow_id = str(snapshot.get("flow_id", ""))
        flow = dict(snapshot.get("flow", {}) or {})
        packets = list(snapshot.get("packets", []) or [])

        self.flow_features = {self.flow_id: flow}
        self.flow_to_packets = {
            self.flow_id: [str(packet.get("packet_id")) for packet in packets if packet.get("packet_id")]
        }
        self.flow_to_mitre = {self.flow_id: list(snapshot.get("flow_to_mitre", []) or [])}
        self.mitre_metadata = dict(snapshot.get("mitre_metadata", {}) or {})

        self.packet_metadata: dict[str, dict[str, Any]] = {}
        self.packet_payload_text: dict[str, str] = {}
        self.packet_payload_ascii: dict[str, str] = {}
        self.packet_timestamps: dict[str, float] = {}
        self.packet_len_raw: dict[str, int] = {}
        self.packet_attention: dict[str, float] = {}
        self.packet_counterfactual_drop: dict[str, float] = {}
        self.packet_to_mitre: dict[str, list[Any]] = {}

        for order, packet in enumerate(packets):
            packet_id = str(packet.get("packet_id"))
            if not packet_id:
                continue
            metadata = dict(packet.get("metadata", {}) or {})
            metadata.setdefault("packet_id", packet_id)
            metadata.setdefault("flow_id", self.flow_id)
            metadata.setdefault("order_in_flow", order)
            metadata.setdefault("timestamp", packet.get("timestamp", 0.0))
            metadata.setdefault("payload_len_raw", packet.get("payload_len_raw", 0))
            metadata.setdefault("mitre_topk", packet.get("mitre_topk", []))
            self.packet_metadata[packet_id] = metadata
            self.packet_payload_text[packet_id] = str(packet.get("payload_hex", ""))
            self.packet_payload_ascii[packet_id] = str(packet.get("payload_ascii", ""))
            self.packet_timestamps[packet_id] = float(packet.get("timestamp", 0.0) or 0.0)
            self.packet_len_raw[packet_id] = int(packet.get("payload_len_raw", 0) or 0)
            if packet.get("attention_weight") is not None:
                self.packet_attention[packet_id] = float(packet["attention_weight"])
            if packet.get("counterfactual_drop") is not None:
                self.packet_counterfactual_drop[packet_id] = float(packet["counterfactual_drop"])
            self.packet_to_mitre[packet_id] = list(packet.get("mitre_topk", []) or [])

    def get_flow(self, flow_id: str) -> dict[str, Any] | None:
        return self.flow_features.get(str(flow_id))

    def get_packets(self, flow_id: str) -> list[dict[str, Any]]:
        packets = []
        for packet_id in self.flow_to_packets.get(str(flow_id), []):
            record = dict(self.packet_metadata.get(packet_id, {}))
            record["payload_preview_hex"] = self.packet_payload_text.get(packet_id, "")
            record["payload_preview_ascii"] = self.packet_payload_ascii.get(packet_id, "")
            record["mitre_topk"] = list(self.packet_to_mitre.get(packet_id, []))
            record["attention_weight"] = self.packet_attention.get(packet_id)
            record["counterfactual_drop"] = self.packet_counterfactual_drop.get(packet_id)
            packets.append(record)
        return packets


def _object_to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return _json_safe(value)


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
