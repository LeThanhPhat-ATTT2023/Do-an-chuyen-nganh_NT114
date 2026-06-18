from graphslm_ids.runtime.slow_path.evidence_bundle import (
    AlertEvidence, BundleStats, CounterfactualEvidence, EvidenceBundle,
    FlowEvidence, GraphPathEvidence, MitreEvidence, PacketEvidence,
)


def make_bundle() -> EvidenceBundle:
    alert = AlertEvidence(
        evidence_id="E_ALERT", alert_id="A_001", flow_id="flow_42",
        predicted_label="SqlInjection", confidence=0.88,
        top_classes=[{"label": "SqlInjection", "prob": 0.88}, {"label": "Benign", "prob": 0.07}],
        alert_threshold=0.70, trigger_reason="HGT SqlInjection probability 0.88 exceeds threshold 0.70",
    )
    flow = FlowEvidence(
        evidence_id="E_FLOW_001", flow_id="flow_42", src_ip="10.0.0.5", dst_ip="10.0.0.9",
        src_port=44321, dst_port=80, protocol="TCP", duration_seconds=2.1,
        packet_count=14, total_payload_bytes=1850, flow_feature_stats={"r_psh": 0.5},
    )
    packet = PacketEvidence(
        evidence_id="E_PKT_001", packet_id="pkt_flow_42_00000003", order_in_flow=3,
        timestamp=1718000000.0, payload_len_raw=120,
        payload_preview_hex="474554202f3f713d31", payload_preview_ascii="GET /?q=1' OR 1=1--",
        linked_techniques=["E_TECH_001"], importance_score=0.91,
        importance_sources={"counterfactual_drop": 0.31, "hgt_attention_weight": 0.82, "combined_score": 0.91},
        importance_reason="Packet received high attention weight in the HGT model",
        mitre_max_cosine=0.71,
    )
    tech = MitreEvidence(
        evidence_id="E_TECH_001", technique_id="T1190", technique_name="Exploit Public-Facing Application",
        tactic="Initial Access", tactic_id="TA0001", cosine_score=0.71,
        matched_from=["pkt_flow_42_00000003"], supporting_packet_count=1,
        mapping_type="pmi+procedure", mapping_caution="Mapping uses embedding cosine similarity, semantic only.",
    )
    path = GraphPathEvidence(
        evidence_id="E_PATH_001",
        path_nodes=[{"id": "flow_42", "type": "flow"}, {"id": "pkt_flow_42_00000003", "type": "packet"},
                    {"id": "T1190", "type": "technique"}, {"id": "TA0001", "type": "tactic"}],
        path_edges=["contains", "matches_technique", "belongs_to_tactic"],
        path_score=0.58, attention_weight=0.82,
    )
    cf = CounterfactualEvidence(
        evidence_id="E_CF_001", masked_element_id="pkt_flow_42_00000003", masked_element_type="packet",
        confidence_before=0.88, confidence_after=0.57, confidence_drop=0.31,
        interpretation="Removing pkt_flow_42_00000003 reduced malicious confidence by 0.31.",
    )
    return EvidenceBundle(
        bundle_version="1.0", alert=alert, flow_evidence=flow, packet_evidence=[packet],
        mitre_evidence=[tech], graph_paths=[path], counterfactual_evidence=[cf],
        limitations=["MITRE mapping uses embedding cosine similarity, not deterministic signature matching."],
        bundle_stats=BundleStats(14, 1, 1, 1, 0),
    )


from graphslm_ids.runtime.slow_path.graph_serializer import serialize_bundle


def test_serialization_has_four_sections_and_handles():
    text = serialize_bundle(make_bundle())
    assert "## ALERT A_001 — HGT decision" in text
    assert "pred=SqlInjection conf=0.88" in text
    assert "## NODES" in text and "## EDGES" in text and "## SALIENCE" in text
    assert "[E_FLOW_001]" in text and "[E_PKT_001]" in text and "[E_TECH_001]" in text
    assert "attn=0.82" in text and "cf_drop=0.31" in text
    assert "T1190" in text and "TA0001" in text
    assert "matches_technique" in text and "pmi+procedure" in text


def test_serialization_is_deterministic():
    a = serialize_bundle(make_bundle())
    b = serialize_bundle(make_bundle())
    assert a == b


def test_salience_orders_packets_by_attention():
    bundle = make_bundle()
    from graphslm_ids.runtime.slow_path.evidence_bundle import PacketEvidence
    bundle.packet_evidence.append(PacketEvidence(
        evidence_id="E_PKT_002", packet_id="pkt_flow_42_00000007", order_in_flow=7,
        timestamp=1718000001.0, payload_len_raw=40, payload_preview_hex="00", payload_preview_ascii="x",
        linked_techniques=[], importance_score=0.2,
        importance_sources={"counterfactual_drop": 0.0, "hgt_attention_weight": 0.10, "combined_score": 0.2},
        importance_reason="low", mitre_max_cosine=0.0,
    ))
    text = serialize_bundle(bundle)
    salience = text.split("## SALIENCE")[1]
    assert salience.index("E_PKT_001") < salience.index("E_PKT_002")
