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
from graphslm_ids.runtime.slow_path.graph_verifier import (
    GraphVerifier, NullNliScorer, VerifierConfig,
)


GOOD_REPORT = (
    "# XAI Report - A_001\n"
    "## 1. Alert Summary\n"
    "The HGT classifier flagged flow flow_42 as SqlInjection with confidence 0.88. [E_ALERT]\n"
    "## 2. Key Evidence\n"
    "- Packet pkt_flow_42_00000003 received attention weight 0.82 in the HGT model. [E_PKT_001]\n"
    "## 4. MITRE ATT&CK Interpretation\n"
    "- Technique T1190 is linked via embedding cosine similarity. [E_TECH_001]\n"
)


def _verifier():
    return GraphVerifier(VerifierConfig(enable_nli=False), nli_scorer=NullNliScorer())


def test_good_report_passes():
    bundle = make_bundle()
    rec = _verifier().verify(GOOD_REPORT, bundle, serialize_bundle(bundle))
    assert rec.citation_grounding_rate == 1.0
    assert rec.hallucination_rate == 0.0
    assert rec.overall_pass is True


def test_fake_packet_id_is_contradicted():
    bundle = make_bundle()
    bad = GOOD_REPORT.replace("pkt_flow_42_00000003", "pkt_flow_42_99999999")
    rec = _verifier().verify(bad, bundle, serialize_bundle(bundle))
    assert rec.hallucination_rate > 0.0
    assert any(v.label == "contradicted" for v in rec.claim_verdicts)
    assert rec.overall_pass is False


def test_wrong_number_fails_numeric_check():
    bundle = make_bundle()
    bad = GOOD_REPORT.replace("confidence 0.88", "confidence 0.10")
    rec = _verifier().verify(bad, bundle, serialize_bundle(bundle))
    assert rec.numeric_accuracy < 1.0
    assert rec.overall_pass is False


def test_uncited_key_claim_lowers_grounding():
    bundle = make_bundle()
    bad = GOOD_REPORT.replace(" [E_TECH_001]", "")
    rec = _verifier().verify(bad, bundle, serialize_bundle(bundle))
    assert rec.citation_grounding_rate < 1.0


def test_nli_tier_marks_unsupported_when_scorer_low():
    bundle = make_bundle()

    class LowScorer:
        name = "low"
        def entail(self, premise, hypothesis):
            return 0.0

    verifier = GraphVerifier(VerifierConfig(enable_nli=True, nli_threshold=0.5), nli_scorer=LowScorer())
    rec = verifier.verify(GOOD_REPORT, bundle, serialize_bundle(bundle))
    # citations + numbers are fine, but NLI says not entailed -> unsupported
    assert any(v.label == "unsupported" for v in rec.claim_verdicts)
    assert rec.factual_consistency == 0.0
