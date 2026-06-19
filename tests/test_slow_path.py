from __future__ import annotations

import queue

from graphslm_ids.runtime.slow_path import (
    EvidenceBuilder,
    FlowContext,
    GraphContext,
    HotBufferAdapter,
    MitreMetadata,
    PacketContext,
    ReportGenerator,
    ReportGeneratorConfig,
    ReportValidator,
    SlowPathConfig,
    SlowPathJob,
    SlowPathWorker,
)
from graphslm_ids.runtime.slow_path.types import MitreEdge


def _sample_context(with_mitre: bool = True) -> GraphContext:
    packets = [
        PacketContext(
            packet_id="pkt_a",
            order_in_flow=0,
            timestamp=1.0,
            payload_len_raw=4,
            payload_preview_hex="47455420",
            payload_preview_ascii="GET ",
            mitre_evidence={"T1190": MitreEdge("injection", 0.86, "pmi", ["t:get"], [])} if with_mitre else {},
        ),
        PacketContext(
            packet_id="pkt_b",
            order_in_flow=1,
            timestamp=2.0,
            payload_len_raw=4,
            payload_preview_hex="2f61646d",
            payload_preview_ascii="/adm",
            mitre_evidence={"T1190": MitreEdge("injection", 0.74, "pmi", ["t:adm"], [])} if with_mitre else {},
        ),
    ]
    metadata = {}
    if with_mitre:
        metadata["T1190"] = MitreMetadata(
            technique_id="T1190",
            technique_name="Exploit Public-Facing Application",
            tactic="initial-access",
            tactic_id="TA0001",
        )
    return GraphContext(
        flow=FlowContext(
            flow_id="flow_1",
            src_ip="192.168.1.10",
            dst_ip="10.0.0.5",
            src_port=51522,
            dst_port=80,
            protocol="TCP",
            duration_seconds=1.0,
            packet_count=2,
            total_payload_bytes=8,
        ),
        packets=packets,
        mitre_metadata=metadata,
        flow_mitre_evidence={"T1190": MitreEdge("injection", 0.86, "pmi", ["t:get"], [])} if with_mitre else {},
    )


def _sample_job(**overrides: object) -> SlowPathJob:
    payload = {
        "alert_id": "alert_1",
        "flow_id": "flow_1",
        "predicted_label": "malicious",
        "confidence": 0.88,
        "predicted_label_idx": 1,
    }
    payload.update(overrides)
    return SlowPathJob(**payload)


class _ContextStore:
    def __init__(self, context: GraphContext) -> None:
        self.context = context
        self.saved_reports = []

    def get_context(self, flow_id: str) -> GraphContext | None:
        if flow_id == self.context.flow.flow_id:
            return self.context
        return None

    def load_context(self, flow_id: str) -> GraphContext | None:
        return self.get_context(flow_id)

    def save_report(self, **kwargs: object) -> None:
        self.saved_reports.append(kwargs)


def test_edge_attention_is_aggregated_to_packet_evidence() -> None:
    snapshot = {
        "node_ids": {"packet": ["pkt_a", "pkt_b"]},
        "edge_index": {
            ("packet", "matches_technique", "technique"): [[0, 1], [0, 0]],
        },
    }
    job = _sample_job(
        subgraph_snapshot=snapshot,
        hgt_attention={("packet", "matches_technique", "technique"): [0.25, 0.75]},
    )

    bundle = EvidenceBuilder().build(job, _sample_context())

    weights = {
        packet.packet_id: packet.importance_sources["hgt_attention_weight"]
        for packet in bundle.packet_evidence
    }
    assert weights == {"pkt_a": 0.25, "pkt_b": 0.75}


def test_validator_does_not_require_mitre_caution_without_mitre_evidence() -> None:
    bundle = EvidenceBuilder().build(_sample_job(), _sample_context(with_mitre=False))
    report = (
        "# XAI Report - alert_1\n"
        "The HGT classifier flagged flow flow_1 with confidence 0.88. [E_ALERT]\n"
        "Network flow: 192.168.1.10:51522 -> 10.0.0.5:80, protocol TCP. [E_FLOW_001]"
    )

    result = ReportValidator().validate(report, bundle)

    assert result.overall_pass
    assert result.mitre_caution_present


def test_worker_uses_tier3_when_slm_backend_is_unavailable() -> None:
    worker = SlowPathWorker(
        report_generator=ReportGenerator(ReportGeneratorConfig(backend=None)),
    )

    result = worker.process_job(_sample_job(), hot_buffer=_ContextStore(_sample_context()))

    assert result.fallback_tier == 3
    assert "[TEMPLATE FALLBACK]" in result.report
    # VG²R: the tier-3 template is the degraded fallback. It is graded by the
    # graph verifier (a FaithfulnessRecord), which may not clear the strict
    # by-construction gate (cgr==1.0, hr==0.0) — the tier itself signals that.
    # The fallback must still be safe and carry a faithfulness record.
    assert result.validation.repair_tier == 3
    assert result.validation.safety_pass


def test_worker_queue_consumes_job_persists_report_and_marks_done() -> None:
    store = _ContextStore(_sample_context())
    worker = SlowPathWorker(
        config=SlowPathConfig(queue_timeout=0.01),
        report_generator=ReportGenerator(ReportGeneratorConfig(backend=None)),
        cold_store=store,
    )
    slow_queue: queue.Queue[SlowPathJob] = queue.Queue()
    slow_queue.put(_sample_job())

    results = worker.run_queue(slow_queue, hot_buffer=store, max_jobs=1)

    assert len(results) == 1
    assert results[0].fallback_tier == 3
    assert slow_queue.unfinished_tasks == 0
    assert len(store.saved_reports) == 1


def test_hot_buffer_adapter_builds_context_from_mapping() -> None:
    hot_buffer = {
        "flows": {
            "flow_1": {
                "src_ip": "192.168.1.10",
                "dst_ip": "10.0.0.5",
                "src_port": 51522,
                "dst_port": 80,
                "protocol": "TCP",
                "packet_ids": ["pkt_a"],
            }
        },
        "packet_payloads": {"pkt_a": "47455420"},
        "packet_to_mitre": {"pkt_a": {"T1190": 0.86}},
        "mitre_metadata": {
            "T1190": {
                "technique_name": "Exploit Public-Facing Application",
                "tactic": "initial-access",
                "tactic_id": "TA0001",
            }
        },
    }

    context = HotBufferAdapter(hot_buffer).get_context("flow_1")

    assert context is not None
    assert context.flow.flow_id == "flow_1"
    assert context.packets[0].payload_preview_ascii == "GET "
    assert context.mitre_metadata["T1190"].tactic_id == "TA0001"
