from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(val) for key, val in value.items() if val is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value if item is not None]
    return value


@dataclass
class AlertEvidence:
    evidence_id: str
    alert_id: str
    flow_id: str
    predicted_label: str
    confidence: float
    top_classes: list[dict[str, Any]]
    alert_threshold: float
    trigger_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "alert_id": self.alert_id,
            "flow_id": self.flow_id,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "top_classes": self.top_classes,
            "alert_threshold": self.alert_threshold,
            "trigger_reason": self.trigger_reason,
        }


@dataclass
class FlowEvidence:
    evidence_id: str
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    duration_seconds: float
    packet_count: int
    total_payload_bytes: int
    flow_feature_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "evidence_id": self.evidence_id,
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "duration_seconds": self.duration_seconds,
            "packet_count": self.packet_count,
            "total_payload_bytes": self.total_payload_bytes,
        }
        if self.flow_feature_stats:
            payload["flow_feature_stats"] = self.flow_feature_stats
        return payload


@dataclass
class PacketEvidence:
    evidence_id: str
    packet_id: str
    order_in_flow: int
    timestamp: float
    payload_len_raw: int
    payload_preview_hex: str
    payload_preview_ascii: str
    linked_techniques: list[str]
    importance_score: float
    importance_sources: dict[str, float]
    importance_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "packet_id": self.packet_id,
            "order_in_flow": self.order_in_flow,
            "timestamp": self.timestamp,
            "payload_len_raw": self.payload_len_raw,
            "payload_preview_hex": self.payload_preview_hex,
            "payload_preview_ascii": self.payload_preview_ascii,
            "linked_techniques": self.linked_techniques,
            "importance_score": self.importance_score,
            "importance_sources": self.importance_sources,
            "importance_reason": self.importance_reason,
        }


@dataclass
class MitreEvidence:
    evidence_id: str
    technique_id: str
    technique_name: str
    tactic: str
    tactic_id: str
    family: str
    evidence_weight: float
    source: str
    matched_tokens: list[str]
    matched_literals: list[str]
    supporting_packet_count: int
    matched_from: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "tactic": self.tactic,
            "tactic_id": self.tactic_id,
            "family": self.family,
            "evidence_weight": self.evidence_weight,
            "source": self.source,
            "matched_tokens": self.matched_tokens,
            "matched_literals": self.matched_literals,
            "supporting_packet_count": self.supporting_packet_count,
            "matched_from": self.matched_from,
        }


@dataclass
class GraphPathEvidence:
    evidence_id: str
    path_nodes: list[dict[str, str]]
    path_edges: list[str]
    path_score: float
    attention_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "path_nodes": self.path_nodes,
            "path_edges": self.path_edges,
            "path_score": self.path_score,
            "attention_weight": self.attention_weight,
        }


@dataclass
class CounterfactualEvidence:
    evidence_id: str
    masked_element_id: str
    masked_element_type: str
    confidence_before: float
    confidence_after: float
    confidence_drop: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "masked_element_id": self.masked_element_id,
            "masked_element_type": self.masked_element_type,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "confidence_drop": self.confidence_drop,
            "interpretation": self.interpretation,
        }


@dataclass
class BundleStats:
    num_packets_in_flow: int
    num_packets_in_evidence: int
    num_techniques_in_evidence: int
    num_paths_in_evidence: int
    total_tokens_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_packets_in_flow": self.num_packets_in_flow,
            "num_packets_in_evidence": self.num_packets_in_evidence,
            "num_techniques_in_evidence": self.num_techniques_in_evidence,
            "num_paths_in_evidence": self.num_paths_in_evidence,
            "total_tokens_estimate": self.total_tokens_estimate,
        }


@dataclass
class EvidenceBundle:
    bundle_version: str
    alert: AlertEvidence
    flow_evidence: FlowEvidence
    packet_evidence: list[PacketEvidence]
    mitre_evidence: list[MitreEvidence]
    graph_paths: list[GraphPathEvidence]
    counterfactual_evidence: list[CounterfactualEvidence]
    limitations: list[str]
    bundle_stats: BundleStats

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "bundle_version": self.bundle_version,
            "alert": self.alert.to_dict(),
            "flow_evidence": self.flow_evidence.to_dict(),
            "packet_evidence": [packet.to_dict() for packet in self.packet_evidence],
            "mitre_evidence": [tech.to_dict() for tech in self.mitre_evidence],
            "graph_paths": [path.to_dict() for path in self.graph_paths],
            "counterfactual_evidence": [
                item.to_dict() for item in self.counterfactual_evidence
            ],
            "limitations": self.limitations,
            "bundle_stats": self.bundle_stats.to_dict(),
        }
        return _drop_none(payload)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
