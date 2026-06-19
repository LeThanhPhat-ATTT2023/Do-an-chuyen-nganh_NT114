from __future__ import annotations

import binascii
import math
from typing import Any, Iterable, Mapping, Sequence

from graphslm_ids.runtime.slow_path.types import (
    FlowContext, GraphContext, MitreEdge, MitreMetadata, PacketContext,
)


class HotBufferAdapter:
    """Adapt a hot graph buffer snapshot into a GraphContext."""

    def __init__(
        self,
        hot_buffer: Any,
        mitre_catalog: Mapping[str, MitreMetadata] | Mapping[str, Mapping[str, Any]] | None = None,
        payload_preview_bytes: int = 64,
        max_packets: int | None = None,
        include_empty_payload: bool = True,
    ) -> None:
        self.hot_buffer = hot_buffer
        self.mitre_catalog = mitre_catalog
        self.payload_preview_bytes = max(int(payload_preview_bytes), 0)
        self.max_packets = max_packets
        self.include_empty_payload = include_empty_payload
        self._mitre_catalog_cache: dict[str, MitreMetadata] | None = None

    def get_context(self, flow_id: str) -> GraphContext | None:
        flow_record = self._resolve_flow_record(flow_id)
        packet_records, packet_to_mitre = self._resolve_packet_records(flow_id, flow_record)
        if flow_record is None and not packet_records:
            return None

        packet_contexts = self._build_packet_contexts(packet_records, packet_to_mitre)
        if not self.include_empty_payload:
            packet_contexts = [pkt for pkt in packet_contexts if pkt.payload_len_raw > 0]

        packet_contexts = sorted(packet_contexts, key=lambda pkt: pkt.order_in_flow)
        if self.max_packets is not None:
            packet_contexts = packet_contexts[: self.max_packets]

        flow_context = self._build_flow_context(flow_id, flow_record, packet_contexts, packet_records)
        flow_mitre_evidence = self._resolve_flow_mitre_evidence(
            flow_id, flow_record, packet_records, packet_contexts
        )
        mitre_metadata = self._resolve_mitre_metadata(flow_mitre_evidence, packet_contexts)

        return GraphContext(
            flow=flow_context,
            packets=packet_contexts,
            mitre_metadata=mitre_metadata,
            flow_mitre_evidence=flow_mitre_evidence,
        )

    def _resolve_flow_record(self, flow_id: str) -> Mapping[str, Any] | None:
        flow_record = self._call_if_present("get_flow", flow_id)
        if flow_record is None:
            flow_record = self._call_if_present("get_flow_features", flow_id)
        if flow_record is None:
            flow_map = self._first_mapping("flow_features", "flow_metadata", "flows", "flow_stats")
            flow_record = self._lookup_mapping(flow_map, flow_id)
        if flow_record is None:
            flow_map = self._first_mapping("flow_records", "flow_store")
            flow_record = self._lookup_mapping(flow_map, flow_id)
        return self._coerce_mapping(flow_record)

    def _resolve_packet_records(
        self, flow_id: str, flow_record: Mapping[str, Any] | None
    ) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
        packet_records = self._call_if_present("get_packets", flow_id)
        if isinstance(packet_records, Mapping):
            packet_records = list(packet_records.values())
        elif isinstance(packet_records, Iterable) and not isinstance(
            packet_records, (str, bytes, bytearray)
        ):
            packet_records = list(packet_records)
        else:
            packet_records = None

        packet_to_mitre = self._first_mapping("packet_to_mitre", "packet_mitre", "packet_to_technique")
        if not isinstance(packet_to_mitre, Mapping):
            packet_to_mitre = {}

        if packet_records is not None:
            coerced = [self._coerce_mapping(record) or {} for record in packet_records]
            return coerced, packet_to_mitre

        raw_packets = self._get_field(flow_record, "packets")
        if isinstance(raw_packets, Sequence) and not isinstance(raw_packets, (str, bytes, bytearray)):
            sample = next((item for item in raw_packets if item is not None), None)
            if isinstance(sample, Mapping) or hasattr(sample, "__dict__"):
                coerced = [self._coerce_mapping(record) or {} for record in raw_packets]
                return coerced, packet_to_mitre

        packet_ids = self._resolve_flow_packet_ids(flow_id, flow_record)
        packet_metadata = self._first_mapping("packet_metadata", "packet_meta", "packet_records")
        packet_payload_text = self._first_mapping(
            "packet_payload_text", "packet_payload_hex", "packet_payloads"
        )
        packet_payload_bytes = self._first_mapping("packet_payload_bytes", "packet_payload_raw")
        packet_payload_ascii = self._first_mapping("packet_payload_ascii", "packet_payload_text_ascii")
        packet_timestamps = self._first_mapping("packet_timestamps", "packet_times")
        packet_len_raw = self._first_mapping("packet_len_raw", "packet_lengths")
        packet_order_map = self._first_mapping("packet_order_in_flow", "packet_order")
        packet_attention = self._first_mapping("packet_attention", "packet_attention_weight")
        packet_cf = self._first_mapping("packet_counterfactual_drop", "packet_cf_drop")

        records: list[Mapping[str, Any]] = []
        if packet_ids is None:
            return records, packet_to_mitre

        for idx, packet_id in enumerate(packet_ids):
            record = dict(self._lookup_mapping(packet_metadata, packet_id) or {})
            record.setdefault("packet_id", packet_id)
            record.setdefault("order_in_flow", self._lookup_mapping(packet_order_map, packet_id))
            record.setdefault("timestamp", self._lookup_mapping(packet_timestamps, packet_id))
            record.setdefault("payload_len_raw", self._lookup_mapping(packet_len_raw, packet_id))
            if packet_payload_text and packet_id in packet_payload_text:
                record.setdefault("payload_preview_hex", packet_payload_text[packet_id])
            if packet_payload_bytes and packet_id in packet_payload_bytes:
                record.setdefault("payload_bytes", packet_payload_bytes[packet_id])
            if packet_payload_ascii and packet_id in packet_payload_ascii:
                record.setdefault("payload_preview_ascii", packet_payload_ascii[packet_id])
            if packet_attention and packet_id in packet_attention:
                record.setdefault("attention_weight", packet_attention[packet_id])
            if packet_cf and packet_id in packet_cf:
                record.setdefault("counterfactual_drop", packet_cf[packet_id])
            record.setdefault("_order_from_list", idx)
            records.append(record)

        return records, packet_to_mitre

    def _resolve_flow_packet_ids(
        self, flow_id: str, flow_record: Mapping[str, Any] | None
    ) -> Sequence[str] | None:
        flow_to_packets = self._first_mapping("flow_to_packets", "flow_packets", "flow_packet_ids")
        packet_ids = self._lookup_mapping(flow_to_packets, flow_id)
        if packet_ids is None and flow_record is not None:
            packet_ids = self._get_field(flow_record, "packet_ids", "packets")
        if packet_ids is None:
            return None
        if isinstance(packet_ids, Sequence) and not isinstance(packet_ids, (str, bytes, bytearray)):
            return [str(packet_id) for packet_id in packet_ids]
        return None

    def _build_packet_contexts(
        self,
        packet_records: Sequence[Mapping[str, Any]],
        packet_to_mitre: Mapping[str, Any],
    ) -> list[PacketContext]:
        packet_contexts: list[PacketContext] = []

        for idx, record in enumerate(packet_records):
            packet_id = self._get_field(record, "packet_id", "id")
            packet_id = str(packet_id) if packet_id is not None else f"pkt_{idx:09d}"

            order_in_flow = self._get_field(record, "order_in_flow", "packet_order_in_flow", "order")
            if order_in_flow is None:
                order_in_flow = self._get_field(record, "_order_from_list")
            order_in_flow = int(order_in_flow) if order_in_flow is not None else idx

            timestamp = self._safe_float(self._get_field(record, "timestamp", "time"))
            payload_len_raw = self._safe_int(
                self._get_field(record, "payload_len_raw", "payload_len", "payload_length")
            )

            payload_hex = self._get_field(
                record, "payload_preview_hex", "payload_hex", "payload_text"
            )
            payload_ascii = self._get_field(record, "payload_preview_ascii", "payload_ascii")
            payload_bytes = self._get_field(record, "payload_bytes", "payload_raw", "payload")
            payload_hex, payload_ascii, payload_len_raw = self._finalize_payload_preview(
                payload_hex,
                payload_ascii,
                payload_bytes,
                payload_len_raw,
            )

            mitre_evidence = self._coerce_mitre_evidence(
                self._get_field(record, "mitre_topk", "mitre_cosine_scores", "mitre_scores"),
                self._get_field(record, "mitre_provenance"),
            )
            if not mitre_evidence:
                mitre_evidence = self._coerce_mitre_evidence(packet_to_mitre.get(packet_id))

            attention_weight = self._safe_float(self._get_field(record, "attention_weight"))
            counterfactual_drop = self._safe_float(self._get_field(record, "counterfactual_drop"))

            packet_contexts.append(
                PacketContext(
                    packet_id=packet_id,
                    order_in_flow=order_in_flow,
                    timestamp=timestamp,
                    payload_len_raw=payload_len_raw,
                    payload_preview_hex=payload_hex,
                    payload_preview_ascii=payload_ascii,
                    mitre_evidence=mitre_evidence,
                    attention_weight=attention_weight,
                    counterfactual_drop=counterfactual_drop,
                )
            )

        return packet_contexts

    def _build_flow_context(
        self,
        flow_id: str,
        flow_record: Mapping[str, Any] | None,
        packet_contexts: Sequence[PacketContext],
        packet_records: Sequence[Mapping[str, Any]],
    ) -> FlowContext:
        src_ip = self._get_flow_field(flow_record, packet_records, "src_ip", default="unknown")
        dst_ip = self._get_flow_field(flow_record, packet_records, "dst_ip", default="unknown")
        src_port = self._safe_int(
            self._get_flow_field(flow_record, packet_records, "src_port", default=0)
        )
        dst_port = self._safe_int(
            self._get_flow_field(flow_record, packet_records, "dst_port", default=0)
        )
        protocol = self._get_flow_field(flow_record, packet_records, "protocol", default="unknown")

        duration_seconds = self._resolve_duration(flow_record, packet_contexts)
        packet_count = self._safe_int(self._get_field(flow_record, "packet_count"))
        if packet_count <= 0:
            packet_count = len(packet_contexts)

        total_payload_bytes = self._safe_int(self._get_field(flow_record, "total_payload_bytes"))
        if total_payload_bytes <= 0:
            total_payload_bytes = sum(packet.payload_len_raw for packet in packet_contexts)

        flow_feature_stats = self._resolve_flow_feature_stats(flow_record, packet_contexts)

        return FlowContext(
            flow_id=str(flow_id),
            src_ip=str(src_ip),
            dst_ip=str(dst_ip),
            src_port=src_port,
            dst_port=dst_port,
            protocol=str(protocol),
            duration_seconds=duration_seconds,
            packet_count=packet_count,
            total_payload_bytes=total_payload_bytes,
            flow_feature_stats=flow_feature_stats,
        )

    def _resolve_duration(
        self, flow_record: Mapping[str, Any] | None, packet_contexts: Sequence[PacketContext]
    ) -> float:
        duration = self._safe_float(
            self._get_field(flow_record, "duration_seconds", "duration", "duration_sec")
        )
        if duration > 0:
            return duration

        start_ts = self._safe_float(
            self._get_field(flow_record, "start_timestamp", "start_time")
        )
        end_ts = self._safe_float(self._get_field(flow_record, "end_timestamp", "end_time"))
        if end_ts > 0 and end_ts >= start_ts:
            return end_ts - start_ts

        timestamps = [pkt.timestamp for pkt in packet_contexts if pkt.timestamp > 0]
        if len(timestamps) < 2:
            return 0.0
        return max(timestamps) - min(timestamps)

    def _resolve_flow_feature_stats(
        self, flow_record: Mapping[str, Any] | None, packet_contexts: Sequence[PacketContext]
    ) -> dict[str, float]:
        feature_stats = self._coerce_feature_stats(flow_record)
        if feature_stats:
            return feature_stats

        return self._basic_flow_feature_stats(packet_contexts)

    def _basic_flow_feature_stats(self, packet_contexts: Sequence[PacketContext]) -> dict[str, float]:
        lengths = [pkt.payload_len_raw for pkt in packet_contexts if pkt.payload_len_raw > 0]
        stats: dict[str, float] = {}
        if lengths:
            mean_len = sum(lengths) / len(lengths)
            stats["mean_pkt_len"] = mean_len
            stats["min_pkt_len"] = float(min(lengths))
            stats["max_pkt_len"] = float(max(lengths))
            stats["std_pkt_len"] = self._stddev(lengths, mean_len)

        timestamps = [pkt.timestamp for pkt in packet_contexts if pkt.timestamp > 0]
        if len(timestamps) >= 2:
            timestamps.sort()
            iats = [b - a for a, b in zip(timestamps, timestamps[1:]) if b >= a]
            if iats:
                stats["mean_iat_ms"] = (sum(iats) / len(iats)) * 1000.0

        return stats

    def _resolve_flow_mitre_evidence(
        self,
        flow_id: str,
        flow_record: Mapping[str, Any] | None,
        packet_records: Sequence[Mapping[str, Any]],
        packet_contexts: Sequence[PacketContext],
    ) -> dict[str, "MitreEdge"]:
        flow_scores = self._coerce_mitre_evidence(
            self._get_field(flow_record, "flow_mitre_scores", "mitre_scores", "mitre_topk")
        )
        if flow_scores:
            return flow_scores

        flow_to_mitre = self._first_mapping("flow_to_mitre", "flow_mitre", "flow_to_technique")
        if flow_to_mitre:
            flow_scores = self._coerce_mitre_evidence(flow_to_mitre.get(flow_id))
            if flow_scores:
                return flow_scores

        aggregated: dict[str, MitreEdge] = {}
        for packet in packet_contexts:
            for tech_id, edge in packet.mitre_evidence.items():
                if tech_id not in aggregated or edge.weight > aggregated[tech_id].weight:
                    aggregated[tech_id] = edge
        return aggregated

    def _resolve_mitre_metadata(
        self, flow_mitre_scores: Mapping[str, float], packet_contexts: Sequence[PacketContext]
    ) -> dict[str, MitreMetadata]:
        technique_ids: set[str] = set(flow_mitre_scores.keys())
        for packet in packet_contexts:
            technique_ids.update(packet.mitre_evidence.keys())

        catalog = self._get_mitre_catalog()
        metadata: dict[str, MitreMetadata] = {}
        for technique_id in technique_ids:
            entry = catalog.get(technique_id)
            if entry is None:
                entry = self._build_mitre_metadata_from_hot_buffer(technique_id)
            if entry is not None:
                metadata[technique_id] = entry
        return metadata

    def _get_mitre_catalog(self) -> dict[str, MitreMetadata]:
        if self._mitre_catalog_cache is not None:
            return self._mitre_catalog_cache

        catalog: dict[str, MitreMetadata] = {}
        raw_catalog = self.mitre_catalog
        if raw_catalog is None:
            raw_catalog = self._first_mapping(
                "mitre_metadata",
                "technique_metadata",
                "technique_catalog",
                "mitre_catalog",
            )

        if isinstance(raw_catalog, Mapping):
            for tech_id, raw in raw_catalog.items():
                meta = self._coerce_mitre_metadata(tech_id, raw)
                if meta is not None:
                    catalog[meta.technique_id] = meta

        self._mitre_catalog_cache = catalog
        return catalog

    def _build_mitre_metadata_from_hot_buffer(self, technique_id: str) -> MitreMetadata | None:
        technique_to_tactic = self._first_mapping("technique_to_tactic", "technique_tactic")
        tactic_metadata = self._first_mapping("tactic_metadata", "mitre_tactics", "tactics")
        tactic_id = None
        if technique_to_tactic and technique_id in technique_to_tactic:
            tactic_id = str(technique_to_tactic[technique_id])

        tactic_name = None
        if tactic_metadata:
            raw_tactic = tactic_metadata.get(tactic_id) if tactic_id else None
            if raw_tactic is None and tactic_id:
                raw_tactic = tactic_metadata.get(tactic_id.lower())
            if isinstance(raw_tactic, Mapping):
                tactic_name = str(
                    raw_tactic.get("shortname")
                    or raw_tactic.get("tactic")
                    or raw_tactic.get("name")
                    or tactic_id
                )

        if tactic_id or tactic_name:
            return MitreMetadata(
                technique_id=technique_id,
                technique_name=technique_id,
                tactic=tactic_name or tactic_id or "unknown",
                tactic_id=tactic_id or "",
            )
        return None

    def _coerce_mitre_metadata(self, technique_id: str, raw: Any) -> MitreMetadata | None:
        if isinstance(raw, MitreMetadata):
            return raw
        if not isinstance(raw, Mapping):
            return None

        tactic_id = raw.get("tactic_id") or raw.get("tacticId")
        tactic_name = raw.get("tactic") or raw.get("tactic_name") or raw.get("tactic_shortname")
        if not tactic_name and tactic_id:
            tactic_name = str(tactic_id)

        technique_name = raw.get("technique_name") or raw.get("name") or technique_id
        source = raw.get("source", "msee_ensemble")
        mapping_caution = raw.get(
            "mapping_caution",
            "This link is from the v3 MSEE ensemble (PMI + procedure matcher), "
            "a statistical evidence vote, not forensic proof.",
        )

        return MitreMetadata(
            technique_id=str(raw.get("technique_id") or technique_id),
            technique_name=str(technique_name),
            tactic=str(tactic_name or "unknown"),
            tactic_id=str(tactic_id or ""),
            source=str(source),
            mapping_caution=str(mapping_caution),
        )

    def _coerce_mitre_evidence(
        self, value: Any, provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, "MitreEdge"]:
        prov = dict(provenance or {})
        out: dict[str, MitreEdge] = {}
        if isinstance(value, Mapping):
            # Legacy ``{technique_id: weight}`` form (no family/provenance).
            for tech, weight in value.items():
                if weight is None:
                    continue
                p = prov.get(str(tech), {})
                out[str(tech)] = MitreEdge(
                    family="", weight=float(weight),
                    source=str(p.get("source", "")),
                    tokens=[str(t) for t in p.get("tokens", [])],
                    literals=[str(lit) for lit in p.get("literals", [])],
                )
            return out
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                    if len(item) == 3:
                        tech, family, weight = str(item[0]), str(item[1]), float(item[2])
                    elif len(item) == 2:
                        tech, family, weight = str(item[0]), "", float(item[1])
                    else:
                        continue
                    p = prov.get(tech, {})
                    out[tech] = MitreEdge(
                        family=family, weight=weight,
                        source=str(p.get("source", "")),
                        tokens=[str(t) for t in p.get("tokens", [])],
                        literals=[str(lit) for lit in p.get("literals", [])],
                    )
        return out

    def _coerce_mitre_scores(self, value: Any) -> dict[str, float]:
        if isinstance(value, Mapping):
            return {str(k): float(v) for k, v in value.items() if v is not None}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            scores: dict[str, float] = {}
            for item in value:
                if isinstance(item, Mapping):
                    tech_id = item.get("technique_id") or item.get("id")
                    score = item.get("score") or item.get("cosine") or item.get("value")
                    if tech_id is not None and score is not None:
                        scores[str(tech_id)] = float(score)
                elif isinstance(item, Sequence) and len(item) == 2:
                    tech_id, score = item
                    scores[str(tech_id)] = float(score)
            return scores
        return {}

    def _finalize_payload_preview(
        self,
        payload_hex: Any,
        payload_ascii: Any,
        payload_bytes: Any,
        payload_len_raw: int,
    ) -> tuple[str, str, int]:
        hex_text = self._coerce_hex(payload_hex)
        ascii_text = self._coerce_ascii(payload_ascii)

        raw_bytes = self._coerce_bytes(payload_bytes)
        if raw_bytes is None and hex_text:
            raw_bytes = self._hex_to_bytes(hex_text)
        if raw_bytes is None and ascii_text:
            raw_bytes = ascii_text.encode("latin-1", errors="replace")

        if raw_bytes is not None:
            if payload_len_raw <= 0:
                payload_len_raw = len(raw_bytes)
            if not hex_text:
                hex_text = binascii.hexlify(raw_bytes).decode("ascii")
            if not ascii_text:
                ascii_text = self._bytes_to_ascii(raw_bytes)

        hex_text = self._truncate_hex(hex_text)
        ascii_text = self._truncate_ascii(ascii_text)
        return hex_text, ascii_text, max(int(payload_len_raw), 0)

    def _truncate_hex(self, value: str) -> str:
        if not value:
            return ""
        if self.payload_preview_bytes <= 0:
            return ""
        max_chars = self.payload_preview_bytes * 2
        return value[:max_chars]

    def _truncate_ascii(self, value: str) -> str:
        if not value:
            return ""
        if self.payload_preview_bytes <= 0:
            return ""
        return value[: self.payload_preview_bytes]

    def _bytes_to_ascii(self, raw: bytes) -> str:
        return "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in raw)

    def _hex_to_bytes(self, value: str) -> bytes | None:
        cleaned = value.replace(" ", "").strip()
        if len(cleaned) % 2 != 0:
            cleaned = cleaned[:-1]
        if not cleaned:
            return None
        try:
            return binascii.unhexlify(cleaned)
        except (binascii.Error, ValueError):
            return None

    def _coerce_hex(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return binascii.hexlify(value).decode("ascii")
        text = str(value).strip().replace(" ", "")
        if self._looks_like_hex(text):
            return text
        return ""

    def _coerce_ascii(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    def _coerce_bytes(self, value: Any) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            try:
                return bytes(int(item) & 0xFF for item in value)
            except (TypeError, ValueError):
                return None
        if hasattr(value, "tobytes"):
            try:
                return value.tobytes()
            except Exception:
                return None
        return None

    def _looks_like_hex(self, value: str) -> bool:
        if len(value) % 2 != 0:
            return False
        for ch in value:
            if ch not in "0123456789abcdefABCDEF":
                return False
        return True

    def _resolve_flow_packet_meta(self, packet_records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        for record in packet_records:
            if not isinstance(record, Mapping):
                continue
            if "src_ip" in record or "dst_ip" in record:
                return record
        return {}

    def _get_flow_field(
        self,
        flow_record: Mapping[str, Any] | None,
        packet_records: Sequence[Mapping[str, Any]],
        field: str,
        default: Any,
    ) -> Any:
        value = self._get_field(flow_record, field)
        if value is not None:
            return value
        packet_meta = self._resolve_flow_packet_meta(packet_records)
        return self._get_field(packet_meta, field, default=default)

    def _coerce_feature_stats(self, flow_record: Mapping[str, Any] | None) -> dict[str, float]:
        if flow_record is None:
            return {}

        raw_stats = self._get_field(
            flow_record, "flow_feature_stats", "flow_stats", "flow_features"
        )
        if isinstance(raw_stats, Mapping):
            return {
                str(key): float(val)
                for key, val in raw_stats.items()
                if val is not None and self._is_number(val)
            }

        if isinstance(raw_stats, Sequence) and not isinstance(raw_stats, (str, bytes, bytearray)):
            feature_names = self._get_field(flow_record, "flow_feature_names", "feature_names")
            if not feature_names:
                feature_names = self._first_mapping("flow_feature_names", "feature_names")
            if isinstance(feature_names, Sequence) and len(feature_names) == len(raw_stats):
                return {
                    str(name): float(val)
                    for name, val in zip(feature_names, raw_stats)
                    if val is not None and self._is_number(val)
                }

        return {}

    def _stddev(self, values: Sequence[float], mean: float) -> float:
        if not values:
            return 0.0
        variance = sum((val - mean) ** 2 for val in values) / len(values)
        return math.sqrt(variance)

    def _is_number(self, value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _lookup_mapping(self, mapping: Any, key: Any) -> Any:
        if isinstance(mapping, Mapping):
            return mapping.get(key)
        return None

    def _first_mapping(self, *names: str) -> Mapping[str, Any] | None:
        for name in names:
            value = self._read_attr_or_item(name)
            if isinstance(value, Mapping):
                return value
        return None

    def _read_attr_or_item(self, name: str) -> Any:
        if self.hot_buffer is None:
            return None
        if hasattr(self.hot_buffer, name):
            return getattr(self.hot_buffer, name)
        if isinstance(self.hot_buffer, Mapping) and name in self.hot_buffer:
            return self.hot_buffer[name]
        return None

    def _call_if_present(self, method: str, *args: Any) -> Any:
        if self.hot_buffer is None:
            return None
        func = getattr(self.hot_buffer, method, None)
        if callable(func):
            try:
                return func(*args)
            except Exception:
                return None
        return None

    def _get_field(self, record: Any, *fields: str, default: Any = None) -> Any:
        if record is None:
            return default
        for field in fields:
            if isinstance(record, Mapping) and field in record:
                return record[field]
            if hasattr(record, field):
                return getattr(record, field)
        return default

    def _coerce_mapping(self, value: Any) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        if value is None:
            return None
        if hasattr(value, "__dict__"):
            return value.__dict__
        return None
