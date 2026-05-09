from __future__ import annotations

from graphslm_ids.slow_path.evidence_bundle import EvidenceBundle
from graphslm_ids.slow_path.types import SlowPathJob


def render_template(bundle: EvidenceBundle, template_tag: str = "TEMPLATE FALLBACK") -> str:
    lines: list[str] = []
    alert = bundle.alert
    flow = bundle.flow_evidence

    lines.append(f"# XAI Report - {alert.alert_id}")
    lines.append(f"[{template_tag}]")
    lines.append("")
    lines.append("## 1. Alert Summary")
    lines.append(
        "The HGT classifier flagged flow {flow_id} as **{label}** with confidence {conf:.2f} "
        "(threshold: {thresh:.2f}). [E_ALERT]".format(
            flow_id=alert.flow_id,
            label=alert.predicted_label,
            conf=alert.confidence,
            thresh=alert.alert_threshold,
        )
    )
    lines.append(
        "Network flow: {src_ip}:{src_port} -> {dst_ip}:{dst_port}, protocol {protocol}, "
        "{packet_count} packets, {total_bytes} bytes, duration {duration:.2f}s. [E_FLOW_001]".format(
            src_ip=flow.src_ip,
            src_port=flow.src_port,
            dst_ip=flow.dst_ip,
            dst_port=flow.dst_port,
            protocol=flow.protocol,
            packet_count=flow.packet_count,
            total_bytes=flow.total_payload_bytes,
            duration=flow.duration_seconds,
        )
    )
    lines.append("")

    lines.append("## 2. Key Evidence")
    if bundle.packet_evidence:
        for packet in bundle.packet_evidence:
            lines.append(
                "- **{packet_id}** (order {order}): importance score {score:.2f} "
                "(counterfactual drop {cf:.2f}, attention weight {attn:.2f}). [{evidence}]".format(
                    packet_id=packet.packet_id,
                    order=packet.order_in_flow,
                    score=packet.importance_score,
                    cf=packet.importance_sources.get("counterfactual_drop", 0.0),
                    attn=packet.importance_sources.get("hgt_attention_weight", 0.0),
                    evidence=packet.evidence_id,
                )
            )
    else:
        lines.append("- No packet evidence available. [E_ALERT]")
    lines.append("")

    lines.append("## 3. Graph-Based Explanation")
    if bundle.graph_paths:
        for idx, path in enumerate(bundle.graph_paths, start=1):
            node_chain = " -> ".join(node["id"] for node in path.path_nodes)
            lines.append(
                "Path {index}: {nodes}, path score {score:.2f}. [{evidence}]".format(
                    index=idx,
                    nodes=node_chain,
                    score=path.path_score,
                    evidence=path.evidence_id,
                )
            )
    else:
        lines.append("No graph paths available. [E_ALERT]")
    lines.append("")

    lines.append("## 4. MITRE ATT&CK Interpretation")
    if bundle.mitre_evidence:
        for tech in bundle.mitre_evidence:
            lines.append(
                "- **{tech_id}** ({tech_name}), tactic: {tactic}, cosine score: {score:.2f}. "
                "This link is based on embedding cosine similarity and should be treated as "
                "semantic indicator, not deterministic signature matching. [{evidence}]".format(
                    tech_id=tech.technique_id,
                    tech_name=tech.technique_name,
                    tactic=tech.tactic,
                    score=tech.cosine_score,
                    evidence=tech.evidence_id,
                )
            )
    else:
        lines.append("No MITRE evidence available. [E_ALERT]")
    lines.append("")

    lines.append("## 5. Confidence and Limitations")
    lines.append(f"HGT confidence: {alert.confidence:.2f}. [E_ALERT]")
    for limitation in bundle.limitations:
        citation = "" if "[E_" in limitation else " [E_ALERT]"
        lines.append(f"- {limitation}{citation}")
    lines.append("")

    lines.append("## 6. Recommended Analyst Actions")
    lines.append(
        "- Review all flows from source IP {src_ip} within the past 10 minutes. [E_FLOW_001]".format(
            src_ip=flow.src_ip
        )
    )
    lines.append(
        "- Inspect {dst_ip}:{dst_port} service logs for access anomalies. [E_FLOW_001]".format(
            dst_ip=flow.dst_ip,
            dst_port=flow.dst_port,
        )
    )
    if bundle.mitre_evidence:
        lines.append(
            "- Cross-reference mapped MITRE techniques against SIEM for corroborating events. "
            "[{evidence}]".format(evidence=bundle.mitre_evidence[0].evidence_id)
        )
    lines.append("- Preserve packet capture for forensic analysis. [E_ALERT]")

    return "\n".join(lines)


def render_minimal(job: SlowPathJob) -> str:
    lines = [f"# XAI Report - {job.alert_id}", "[TEMPLATE FALLBACK]"]
    lines.append("")
    lines.append("## 1. Alert Summary")
    lines.append(
        "The HGT classifier flagged flow {flow_id} as **{label}** with confidence {conf:.2f}. [E_ALERT]".format(
            flow_id=job.flow_id,
            label=job.predicted_label,
            conf=job.confidence,
        )
    )
    lines.append("")
    lines.append("## 2. Key Evidence")
    lines.append("- Evidence bundle unavailable due to timeout. [E_ALERT]")
    return "\n".join(lines)
