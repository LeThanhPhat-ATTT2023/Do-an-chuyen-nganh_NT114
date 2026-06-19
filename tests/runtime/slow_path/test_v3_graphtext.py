import pandas as pd

from graphslm_ids.offline.preprocessing.ensemble import (
    build_pmi_lookup_from_table,
    lookup_pmi_per_packet,
    lookup_pmi_per_packet_with_tokens,
)


def _lookup():
    df = pd.DataFrame([
        {"token": "t:select", "technique": "T1190", "family": "injection", "weight": 0.9},
    ])
    return build_pmi_lookup_from_table(df)


def test_pmi_with_tokens_matches_base_weights():
    lk = _lookup()
    payload = b"... select ..."
    base = lookup_pmi_per_packet(payload, lk)
    prov = lookup_pmi_per_packet_with_tokens(payload, lk)
    # same techniques + same (family, weight)
    assert set(prov) == set(base)
    for tech, (family, weight, tokens) in prov.items():
        assert (family, weight) == base[tech]
        assert "t:select" in tokens


def test_pmi_with_tokens_empty_payload():
    assert lookup_pmi_per_packet_with_tokens(b"", _lookup()) == {}


from graphslm_ids.runtime.fast_path.edge_assigner import RuntimeEdgeAssigner


class _StubProc:
    def weight_per_technique(self, payload: bytes):
        return {"T1059": 0.9} if b"cmd.exe" in payload else {}

    def match(self, payload: bytes):
        return {"T1059": ["cmd.exe"]} if b"cmd.exe" in payload else {}


def _assigner():
    df = pd.DataFrame([
        {"token": "t:select", "technique": "T1190", "family": "injection", "weight": 0.9},
    ])
    return RuntimeEdgeAssigner.from_components(
        pmi_table=df, procedure_matcher=_StubProc(),
        technique_family_map={"T1059": "command_exec"}, tau_edge=0.4,
    )


def test_assign_packet_default_still_triples():
    edges = _assigner().assign_packet(b"... select ... cmd.exe")
    assert ("T1190", "injection", pytest.approx(0.495, abs=1e-3)) in edges
    assert any(e[0] == "T1059" for e in edges)


def test_assign_packet_returns_provenance():
    edges, prov = _assigner().assign_packet(
        b"... select ... cmd.exe", return_provenance=True)
    assert any(e[0] == "T1190" for e in edges)
    assert "t:select" in prov["T1190"]["tokens"]
    assert prov["T1190"]["source"] == "pmi"
    assert "cmd.exe" in prov["T1059"]["literals"]
    assert prov["T1059"]["source"] == "procedure"


import pytest  # noqa: E402


import numpy as np  # noqa: E402
from graphslm_ids.runtime.fast_path.hot_graph_buffer import HotGraphBuffer  # noqa: E402


def test_hot_buffer_round_trips_provenance():
    buf = HotGraphBuffer(ttl_seconds=999)
    buf.add_packet(
        packet_id="p1", flow_id="f1", embedding=np.zeros(1, dtype=np.float32),
        payload_hex="6162", payload_ascii="ab", payload_len_raw=2, timestamp=0.0,
        src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1, dst_port=80, protocol="TCP",
        mitre_topk=[("T1190", "injection", 0.9)],
        mitre_provenance={"T1190": {"source": "pmi", "tokens": ["t:select"], "literals": []}},
    )
    pkts = buf.get_packets("f1")
    assert pkts[0]["mitre_provenance"]["T1190"]["tokens"] == ["t:select"]


from graphslm_ids.runtime.slow_path.types import (  # noqa: E402
    GraphContext, MitreEdge, PacketContext,
)


def test_mitre_edge_defaults():
    e = MitreEdge(family="injection", weight=0.9)
    assert e.source == "" and e.tokens == [] and e.literals == []


def test_packet_context_has_mitre_evidence():
    pc = PacketContext(
        packet_id="p1", order_in_flow=0, timestamp=0.0, payload_len_raw=2,
        payload_preview_hex="6162", payload_preview_ascii="ab",
        mitre_evidence={"T1190": MitreEdge("injection", 0.9, "pmi", ["t:select"], [])},
    )
    assert pc.mitre_evidence["T1190"].family == "injection"


from graphslm_ids.runtime.slow_path.hot_buffer_adapter import HotBufferAdapter  # noqa: E402


def test_adapter_builds_mitre_evidence_from_triples():
    adapter = HotBufferAdapter(hot_buffer=None, mitre_catalog={})
    record = {
        "packet_id": "p1", "order_in_flow": 0, "timestamp": 0.0,
        "payload_preview_hex": "6162", "payload_preview_ascii": "ab", "payload_len_raw": 2,
        "mitre_topk": [("T1190", "injection", 0.9)],
        "mitre_provenance": {"T1190": {"source": "pmi", "tokens": ["t:select"], "literals": []}},
    }
    pcs = adapter._build_packet_contexts([record], {})
    edge = pcs[0].mitre_evidence["T1190"]
    assert edge.family == "injection" and edge.weight == 0.9
    assert edge.source == "pmi" and edge.tokens == ["t:select"]


from graphslm_ids.runtime.slow_path.evidence_bundle import MitreEvidence  # noqa: E402


def test_mitre_evidence_v3_fields():
    m = MitreEvidence(
        evidence_id="E_TECH_001", technique_id="T1190", technique_name="X",
        tactic="Initial Access", tactic_id="TA0001", family="injection",
        evidence_weight=0.9, source="pmi", matched_tokens=["t:select"],
        matched_literals=[], supporting_packet_count=1, matched_from=["p1"],
    )
    d = m.to_dict()
    assert d["family"] == "injection" and d["evidence_weight"] == 0.9
    assert "cosine_score" not in d and "mapping_type" not in d


from graphslm_ids.runtime.slow_path.evidence_builder import EvidenceBuilder  # noqa: E402
from graphslm_ids.runtime.slow_path.types import (  # noqa: E402
    FlowContext, MitreMetadata, SlowPathJob,
)


def _ctx():
    flow = FlowContext(flow_id="f1", src_ip="10.0.0.1", dst_ip="10.0.0.2",
                       src_port=1, dst_port=80, protocol="TCP",
                       duration_seconds=0.0, packet_count=1, total_payload_bytes=2)
    pkt = PacketContext(packet_id="p1", order_in_flow=0, timestamp=0.0,
                        payload_len_raw=2, payload_preview_hex="6162",
                        payload_preview_ascii="ab",
                        mitre_evidence={"T1190": MitreEdge("injection", 0.9, "pmi", ["t:select"], [])})
    meta = {"T1190": MitreMetadata("T1190", "Exploit Public-Facing App", "Initial Access", "TA0001")}
    return GraphContext(flow=flow, packets=[pkt], mitre_metadata=meta,
                        flow_mitre_evidence={})


def test_builder_emits_v3_mitre_evidence():
    job = SlowPathJob(alert_id="A1", flow_id="f1", predicted_label="SqlInjection",
                      confidence=0.9, alert_threshold=0.7)
    bundle = EvidenceBuilder().build(job, _ctx())
    assert bundle.mitre_evidence
    me = bundle.mitre_evidence[0]
    assert me.family == "injection" and me.source == "pmi"
    assert me.matched_tokens == ["t:select"]
    if bundle.graph_paths:
        assert bundle.graph_paths[0].path_edges == ["contain", "evidence_injection", "technique_tactic"]


from graphslm_ids.runtime.slow_path.graph_serializer import serialize_bundle  # noqa: E402


def test_serializer_v3_graphtext():
    job = SlowPathJob(alert_id="A1", flow_id="f1", predicted_label="SqlInjection",
                      confidence=0.9, alert_threshold=0.7)
    bundle = EvidenceBuilder().build(job, _ctx())
    text = serialize_bundle(bundle)
    assert "evidence_injection" in text
    assert "w=0.90" in text or "w=0.9" in text
    assert "src=pmi" in text
    assert "cosine=" not in text and "matches_technique" not in text
