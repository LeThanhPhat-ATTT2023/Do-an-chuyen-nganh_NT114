from graphslm_ids.runtime.slow_path.evidence_bundle import (
    AlertEvidence, BundleStats, CounterfactualEvidence, EvidenceBundle,
    FlowEvidence, GraphPathEvidence, MitreEvidence, PacketEvidence,
)
from graphslm_ids.runtime.slow_path.graph_verifier import NullNliScorer, VerifierConfig
from graphslm_ids.runtime.slow_path.report_generator import ReportGenerator
from graphslm_ids.runtime.slow_path.slow_path_worker import SlowPathWorker, SlowPathConfig
from graphslm_ids.runtime.slow_path.types import SlowPathJob


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


class StubHydrator:
    def __init__(self, context):
        self._context = context
    def hydrate(self, flow_id, hot_buffer, store):
        return self._context


class StubBuilder:
    def __init__(self, bundle):
        self._bundle = bundle
    def build(self, job, context, max_cf_packets=10):
        return self._bundle


class PassRanker:
    def rank_and_truncate(self, bundle, **kw):
        return bundle


def _job():
    return SlowPathJob(alert_id="A_001", flow_id="flow_42",
                       predicted_label="SqlInjection", confidence=0.88)


def _worker(slm_callable, verifier_cfg=None):
    bundle = make_bundle()
    return SlowPathWorker(
        config=SlowPathConfig(),
        context_hydrator=StubHydrator(context=object()),
        evidence_builder=StubBuilder(bundle),
        evidence_ranker=PassRanker(),
        report_generator=ReportGenerator(slm_callable=slm_callable),
        verifier_config=verifier_cfg or VerifierConfig(enable_nli=False),
        nli_scorer=NullNliScorer(),
    )


def test_clean_report_passes_tier1():
    good = (
        "# XAI Report - A_001\n"
        "Flow flow_42 flagged as SqlInjection with confidence 0.88. [E_ALERT]\n"
        "Packet pkt_flow_42_00000003 had attention weight 0.82. [E_PKT_001]\n"
        "Technique T1190 linked via embedding cosine similarity. [E_TECH_001]\n"
    )
    worker = _worker(lambda s, u: good)
    result = worker.process_job(_job())
    assert result.fallback_tier == 1
    assert result.validation.overall_pass is True


def test_repair_loop_is_bounded_then_falls_back():
    calls = {"n": 0}
    def always_bad(system, user):
        calls["n"] += 1
        return ("# XAI Report - A_001\n"
                "Packet pkt_flow_42_99999999 was malicious. [E_PKT_001]\n")  # fake id
    worker = _worker(always_bad)
    result = worker.process_job(_job())
    # tier1 + N repair attempts are bounded; final report is the template fallback
    assert result.fallback_tier == 3
    assert calls["n"] <= 1 + worker.config.max_repair_attempts
    assert "TEMPLATE FALLBACK" in result.report
