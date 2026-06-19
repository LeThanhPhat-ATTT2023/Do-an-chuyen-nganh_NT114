from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlowPathJob:
    alert_id: str
    flow_id: str
    predicted_label: str
    confidence: float
    subgraph_snapshot: Any | None = None
    hgt_attention: Any | None = None
    timestamp: float | None = None
    top_classes: list[dict[str, Any]] | None = None
    alert_threshold: float | None = None
    predicted_label_idx: int | None = None


@dataclass
class FlowContext:
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


@dataclass
class MitreEdge:
    """One v3 evidence edge: family-routed technique with provenance."""
    family: str
    weight: float
    source: str = ""
    tokens: list[str] = field(default_factory=list)
    literals: list[str] = field(default_factory=list)


@dataclass
class PacketContext:
    packet_id: str
    order_in_flow: int
    timestamp: float
    payload_len_raw: int
    payload_preview_hex: str
    payload_preview_ascii: str
    mitre_evidence: dict[str, "MitreEdge"] = field(default_factory=dict)
    attention_weight: float | None = None
    counterfactual_drop: float | None = None


@dataclass
class MitreMetadata:
    technique_id: str
    technique_name: str
    tactic: str
    tactic_id: str
    source: str = "msee_ensemble"
    mapping_caution: str = (
        "This link is from the v3 MSEE ensemble (PMI + procedure matcher), "
        "a statistical evidence vote, not forensic proof."
    )


@dataclass
class GraphContext:
    flow: FlowContext
    packets: list[PacketContext]
    mitre_metadata: dict[str, MitreMetadata]
    flow_mitre_evidence: dict[str, "MitreEdge"] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
