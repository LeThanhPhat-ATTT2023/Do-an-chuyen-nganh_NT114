"""Evidence-grounded explanation builder for the v2 SLM slow path.

Where v1's XAI was a post-hoc LLM narrative (XG-NID-style, prone to confabulation),
v2's explanation is **structurally grounded**: every sentence traces back to a
concrete evidence edge between a flow/packet and a MITRE ATT&CK technique. The
LLM (if any) downstream may add prose flourish, but it cannot remove the
grounding metadata, and the explanation can be verified by simply reading the
evidence edges out of the graph artifact.

This module is intentionally pure: no LLM call, no I/O. It composes a
structured payload (sentences + evidence) plus a plain Vietnamese-language
string. Faithfulness is a property of the data, not of any generation step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EvidenceEdge:
    """One piece of evidence supporting the activation of a technique node."""

    signature_id: str
    feature: str
    value: Any
    technique: str


@dataclass(frozen=True)
class GroundedSentence:
    """A claim made by the explanation, paired with the evidence that justifies it."""

    text: str
    evidence: list[EvidenceEdge]


# Optional friendly names for the MITRE techniques we touch most often. Keeps
# the prose readable without forcing the SLM to look up everything.
_TECHNIQUE_NAMES: dict[str, str] = {
    "T1018": "Remote System Discovery",
    "T1041": "Exfiltration Over C2 Channel",
    "T1046": "Network Service Scanning",
    "T1059.004": "Unix Shell command injection",
    "T1059.007": "JavaScript / browser execution",
    "T1190": "Exploit Public-Facing Application (SQLi)",
    "T1498": "Network Denial of Service",
    "T1499": "Endpoint Denial of Service",
    "T1505.003": "Web Shell",
}


def _technique_name(tech_id: str) -> str:
    return _TECHNIQUE_NAMES.get(tech_id, tech_id)


def _format_value(value: Any) -> str:
    """Render an evidence value compactly for a single sentence."""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def build_grounded_explanation(
    flow_id: str,
    flow_features: Mapping[str, Any],
    activated_techniques: Sequence[tuple[str, float]],
    evidence_edges: Sequence[Mapping[str, Any] | EvidenceEdge],
    predicted_class: str,
) -> dict[str, Any]:
    """Compose a grounded explanation payload for a single classified flow.

    Args:
      flow_id: identifier of the flow being explained.
      flow_features: dict of flow feature name -> value (the same numbers fed
        into the HGT). Only features cited in evidence_edges need to be present.
      activated_techniques: list of ``(technique_id, weight)`` ordered by
        importance (the SLM caller decides the order; typically by edge weight).
      evidence_edges: list of either :class:`EvidenceEdge` instances or dicts
        with the same keys (``signature_id``, ``feature``, ``value``,
        ``technique``). One entry per piece of evidence the HGT received.
      predicted_class: the model's top-1 class label.

    Returns a dict with two top-level keys:

      * ``sentences``: ``[{"text": "...", "evidence": [...]}]``
      * ``natural_language``: a single Vietnamese-language paragraph composed
        from the sentences in order.

    Both share the same structured evidence so an auditor can verify any claim.
    """
    # Normalise evidence list into EvidenceEdge instances we can group on.
    norm_evidence: list[EvidenceEdge] = []
    for ev in evidence_edges:
        if isinstance(ev, EvidenceEdge):
            norm_evidence.append(ev)
            continue
        norm_evidence.append(
            EvidenceEdge(
                signature_id=str(ev.get("signature_id", "")),
                feature=str(ev.get("feature", "")),
                value=ev.get("value"),
                technique=str(ev.get("technique", "")),
            )
        )

    # Group evidence by technique so we can attribute multiple features to the
    # same activated technique node in one sentence.
    by_tech: dict[str, list[EvidenceEdge]] = {}
    for ev in norm_evidence:
        by_tech.setdefault(ev.technique, []).append(ev)

    sentences: list[GroundedSentence] = []

    # Sentence 1: prediction summary (no evidence required, but we attach the
    # full evidence list so the auditor can read everything from one place).
    sentences.append(
        GroundedSentence(
            text=(
                f"Flow {flow_id} được phân loại là **{predicted_class}** "
                f"dựa trên bằng chứng có cấu trúc từ đồ thị tri thức ATT&CK."
            ),
            evidence=list(norm_evidence),
        )
    )

    # Sentences 2..N: one per activated technique, listing the evidence.
    for tech_id, weight in activated_techniques:
        tech_name = _technique_name(tech_id)
        tech_evidence = by_tech.get(tech_id, [])
        if tech_evidence:
            evidence_strs = ", ".join(
                f"{ev.feature}={_format_value(ev.value)}" for ev in tech_evidence
            )
            text = (
                f"Kỹ thuật **{tech_id} ({tech_name})** được kích hoạt với độ "
                f"tin cậy {weight:.2f}, dựa trên bằng chứng: {evidence_strs}."
            )
        else:
            # Technique active but no concrete evidence ref'd -- still emit a
            # sentence so the explanation reflects the activation faithfully.
            text = (
                f"Kỹ thuật **{tech_id} ({tech_name})** được kích hoạt với độ "
                f"tin cậy {weight:.2f}."
            )
        sentences.append(GroundedSentence(text=text, evidence=tech_evidence))

    return {
        "flow_id": flow_id,
        "predicted_class": predicted_class,
        "activated_techniques": [
            {"technique_id": t, "weight": float(w)} for t, w in activated_techniques
        ],
        "sentences": [
            {
                "text": s.text,
                "evidence": [
                    {
                        "signature_id": ev.signature_id,
                        "feature": ev.feature,
                        "value": ev.value,
                        "technique": ev.technique,
                    }
                    for ev in s.evidence
                ],
            }
            for s in sentences
        ],
        "natural_language": " ".join(s.text for s in sentences),
    }
