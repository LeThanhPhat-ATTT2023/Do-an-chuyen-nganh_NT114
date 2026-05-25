"""Tests for the v2 grounded-explanation builder."""
from __future__ import annotations

from graphslm_ids.runtime.slow_path.grounded_report import (
    EvidenceEdge,
    build_grounded_explanation,
)


def test_explanation_cites_every_active_technique() -> None:
    out = build_grounded_explanation(
        flow_id="f0",
        flow_features={
            "fan_dst_port": 1200,
            "r_syn": 0.95,
            "plen_pay_mean": 0,
        },
        activated_techniques=[("T1046", 0.92), ("T1018", 0.7)],
        evidence_edges=[
            EvidenceEdge(
                signature_id="flow:syn_fanout",
                feature="fan_dst_port",
                value=1200,
                technique="T1046",
            ),
            EvidenceEdge(
                signature_id="flow:syn_fanout",
                feature="r_syn",
                value=0.95,
                technique="T1046",
            ),
            EvidenceEdge(
                signature_id="flow:host_fanout",
                feature="fan_dst_ip",
                value=200,
                technique="T1018",
            ),
        ],
        predicted_class="Recon-PortScan",
    )
    nl = out["natural_language"]
    # Both activated techniques must appear by id in the prose.
    assert "T1046" in nl and "T1018" in nl
    # Predicted class must appear once.
    assert "Recon-PortScan" in nl
    # Every sentence carries traceable evidence (the summary sentence carries
    # all evidence, the per-technique sentences carry their own).
    sentences = out["sentences"]
    assert len(sentences) >= 3  # summary + 2 techniques
    # The T1046 sentence cites both fan_dst_port and r_syn.
    tech_sentences = [s for s in sentences if "T1046" in s["text"]]
    assert tech_sentences, "no sentence references T1046"
    features_cited = {ev["feature"] for ev in tech_sentences[0]["evidence"]}
    assert {"fan_dst_port", "r_syn"} <= features_cited


def test_dict_evidence_accepted_alongside_dataclass() -> None:
    out = build_grounded_explanation(
        flow_id="f1",
        flow_features={"r_psh": 0.5},
        activated_techniques=[("T1190", 0.85)],
        evidence_edges=[
            {
                "signature_id": "sqli_union_select",
                "feature": "payload_token",
                "value": "UNION SELECT",
                "technique": "T1190",
            }
        ],
        predicted_class="SqlInjection",
    )
    assert "T1190" in out["natural_language"]
    assert out["sentences"][1]["evidence"][0]["signature_id"] == "sqli_union_select"


def test_unmatched_technique_still_emits_sentence() -> None:
    # Active technique with no evidence rows: still cited so the explanation
    # is consistent with the model output.
    out = build_grounded_explanation(
        flow_id="f2",
        flow_features={},
        activated_techniques=[("T1041", 0.6)],
        evidence_edges=[],
        predicted_class="Benign",
    )
    assert "T1041" in out["natural_language"]
    # No evidence dicts attached to the T1041 sentence.
    tech_sentence = [s for s in out["sentences"] if "T1041" in s["text"]][0]
    assert tech_sentence["evidence"] == []


def test_payload_dict_payload_serializes() -> None:
    """The returned payload must be JSON-serialisable (no dataclasses leak through)."""
    import json

    out = build_grounded_explanation(
        flow_id="f3",
        flow_features={},
        activated_techniques=[("T1046", 0.9)],
        evidence_edges=[
            EvidenceEdge(
                signature_id="flow:syn_fanout",
                feature="fan_dst_port",
                value=1200,
                technique="T1046",
            )
        ],
        predicted_class="Recon-PortScan",
    )
    encoded = json.dumps(out, ensure_ascii=False)
    assert "T1046" in encoded
