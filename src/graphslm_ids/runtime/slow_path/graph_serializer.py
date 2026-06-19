"""Serialize the HGT explanation subgraph (EvidenceBundle) into deterministic
graph-text for the SLM. Citation handles are the existing evidence_ids, so the
verifier and the SLM share one citation scheme. Encoding follows the
'Talk like a Graph' (ICLR'24) finding that the encoding function matters: an
explicit node-table + typed edge-list + salience block, not free prose.
"""
from __future__ import annotations

from graphslm_ids.runtime.slow_path.evidence_bundle import EvidenceBundle, PacketEvidence


def _packet_attn(pkt: PacketEvidence) -> float:
    return float(pkt.importance_sources.get("hgt_attention_weight", 0.0))


def _packet_cf(pkt: PacketEvidence) -> float:
    return float(pkt.importance_sources.get("counterfactual_drop", 0.0))


def serialize_bundle(bundle: EvidenceBundle) -> str:
    """Return a deterministic graph-text rendering of the subgraph.

    Ordering is stable: packets by (-attention, evidence_id); techniques by
    (-cosine, evidence_id); paths by (-path_score, evidence_id). Every number a
    report may cite is present here so claims are verifiable against this text.
    """
    alert = bundle.alert
    flow = bundle.flow_evidence
    packets = sorted(bundle.packet_evidence, key=lambda p: (-_packet_attn(p), p.evidence_id))
    techs = sorted(bundle.mitre_evidence, key=lambda t: (-t.evidence_weight, t.evidence_id))
    paths = sorted(bundle.graph_paths, key=lambda p: (-p.path_score, p.evidence_id))

    lines: list[str] = []
    # Decision header
    lines.append(f"## ALERT {alert.alert_id} — HGT decision")
    top2 = ""
    if len(alert.top_classes) > 1:
        second = alert.top_classes[1]
        top2 = f" ; top2={second.get('label')} {float(second.get('prob', 0.0)):.2f}"
    lines.append(
        f"pred={alert.predicted_label} conf={alert.confidence:.2f}{top2} "
        f"; threshold={alert.alert_threshold:.2f} [E_ALERT]"
    )
    lines.append("")

    # Node table
    lines.append("## NODES")
    lines.append(
        f"flow [{flow.evidence_id}] id={flow.flow_id} proto={flow.protocol} "
        f"src={flow.src_ip}:{flow.src_port} dst={flow.dst_ip}:{flow.dst_port} "
        f"dur={flow.duration_seconds:.1f}s pkts={flow.packet_count}"
    )
    for pkt in packets:
        lines.append(
            f"pkt [{pkt.evidence_id}] id={pkt.packet_id} order={pkt.order_in_flow} "
            f"attn={_packet_attn(pkt):.2f} cf_drop={_packet_cf(pkt):.2f} "
            f'payload_ascii="{pkt.payload_preview_ascii}"'
        )
    for tech in techs:
        toks = ",".join(tech.matched_tokens[:5])
        lits = ",".join(tech.matched_literals[:5])
        prov = f' tokens="{toks}"' if toks else ""
        prov += f' literals="{lits}"' if lits else ""
        lines.append(
            f"tech [{tech.evidence_id}] id={tech.technique_id} family={tech.family} "
            f"w={tech.evidence_weight:.2f} src={tech.source}{prov} "
            f'name="{tech.technique_name}"'
        )
    seen_tactics: set[str] = set()
    for tech in techs:
        if tech.tactic_id in seen_tactics:
            continue
        seen_tactics.add(tech.tactic_id)
        lines.append(f"tactic id={tech.tactic_id} name=\"{tech.tactic}\"")
    lines.append("")

    # Host nodes (v3 schema): flow endpoints.
    lines.append(f"host id={flow.src_ip} role=src")
    lines.append(f"host id={flow.dst_ip} role=dst")
    lines.append("")

    # Edge list
    lines.append("## EDGES")
    for path in paths:
        nodes = path.path_nodes
        edges = path.path_edges
        for i in range(min(len(edges), max(len(nodes) - 1, 0))):
            src = nodes[i].get("id")
            dst = nodes[i + 1].get("id")
            prov = ""
            if edges[i].startswith("evidence_"):
                match = next((t for t in techs if t.technique_id == dst), None)
                if match is not None:
                    prov = f" src={match.source} w={match.evidence_weight:.2f}"
            lines.append(f"{src} -{edges[i]}-> {dst}{prov} [{path.evidence_id}]")
    lines.append("")

    # Salience block
    lines.append("## SALIENCE (top packets by HGT attention)")
    for rank, pkt in enumerate(packets, start=1):
        lines.append(
            f"{rank}) [{pkt.evidence_id}] {pkt.packet_id} attn={_packet_attn(pkt):.2f} "
            f"cf_drop={_packet_cf(pkt):.2f}"
        )

    return "\n".join(lines)
