"""Graph-grounded verifier for VG²R reports. Three tiers:
  1. symbolic — cited handles/entities must exist in the subgraph;
  2. numeric  — quantitative claims must match graph values within tolerance;
  3. NLI      — qualitative claims must be entailed by the graph-text.
The symbolic+numeric tiers are a hard gate; NLI (injected, pretrained) only
judges qualitative claims. No model is trained here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Protocol

from graphslm_ids.runtime.slow_path.evidence_bundle import EvidenceBundle

# Reuse the existing claim/citation/safety helpers — one definition.
from graphslm_ids.runtime.slow_path.report_validator import (
    _check_safety, _collect_evidence_ids, _collect_valid_entities,
    _has_valid_citation, _is_key_claim,
)


# Sentence splitter for claim grounding. Unlike the report-validator's splitter,
# it splits ONLY on sentence-ending punctuation, NOT after "]". SLMs frequently
# cite mid-sentence ("the flow [E_FLOW_001] triggered ..."); splitting at "]"
# would orphan the rest of the sentence into a spurious uncited claim.
_EVIDENCE_REF = r"(?:E_[A-Z]+_\d{3}|E_ALERT|E_FLOW_\d{3})"
_PROTECT_CITE = re.compile(rf"([.!?])\s+(\[{_EVIDENCE_REF}\])")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_claims(text: str) -> list[str]:
    # Protect "claim. [E_ID]" so the trailing citation (and the decimal point in
    # values like "0.88. [E_ALERT]") stays attached to its sentence.
    protected = _PROTECT_CITE.sub(r"\1@@CITE@@\2", text.strip())
    parts = _SENT_SPLIT.split(protected)
    return [part.replace("@@CITE@@", " ").strip() for part in parts if part.strip()]


class NliScorer(Protocol):
    name: str
    def entail(self, premise: str, hypothesis: str) -> float: ...


class NullNliScorer:
    name = "null"
    def entail(self, premise: str, hypothesis: str) -> float:  # noqa: D401
        return 1.0


@dataclass
class ClaimVerdict:
    text: str
    label: str  # "supported" | "unsupported" | "contradicted"
    citations: list[str] = field(default_factory=list)
    reason: str = ""
    nli_score: float | None = None


@dataclass
class FaithfulnessRecord:
    citation_grounding_rate: float
    hallucination_rate: float
    numeric_accuracy: float
    factual_consistency: float
    safety_pass: bool
    claim_verdicts: list[ClaimVerdict]
    nli_model: str
    repair_tier: int
    overall_pass: bool


@dataclass
class VerifierConfig:
    numeric_tolerance: float = 0.01
    nli_threshold: float = 0.5
    cgr_threshold: float = 1.0
    max_hallucination_rate: float = 0.0
    enable_nli: bool = True


_CITE_RE = re.compile(r"\[(E_[A-Z]+_\d{3}|E_ALERT|E_FLOW_\d{3})\]")
_NUM_NEAR = re.compile(
    r"\b(confidence|probability|attention weight|attention|cf_drop|counterfactual drop|drop|port)\b\D{0,12}(\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def _bundle_numeric_values(bundle: EvidenceBundle) -> list[float]:
    """All numbers a faithful report may legitimately cite."""
    vals: list[float] = [
        round(bundle.alert.confidence, 4),
        round(bundle.alert.alert_threshold, 4),
        float(bundle.flow_evidence.src_port),
        float(bundle.flow_evidence.dst_port),
    ]
    for cls in bundle.alert.top_classes:
        vals.append(round(float(cls.get("prob", 0.0)), 4))
    for pkt in bundle.packet_evidence:
        vals.append(round(float(pkt.importance_sources.get("hgt_attention_weight", 0.0)), 4))
        vals.append(round(float(pkt.importance_sources.get("counterfactual_drop", 0.0)), 4))
    for tech in bundle.mitre_evidence:
        vals.append(round(float(tech.cosine_score), 4))
    for cf in bundle.counterfactual_evidence:
        vals.append(round(float(cf.confidence_drop), 4))
    return vals


class GraphVerifier:
    def __init__(self, config: VerifierConfig | None = None, nli_scorer: NliScorer | None = None) -> None:
        self.config = config or VerifierConfig()
        self.nli_scorer = nli_scorer or NullNliScorer()

    def verify(
        self,
        report: str,
        bundle: EvidenceBundle,
        graph_text: str,
        repair_tier: int = 1,
    ) -> FaithfulnessRecord:
        evidence_ids = _collect_evidence_ids(bundle)
        valid_entities = _collect_valid_entities(bundle)
        valid_numbers = _bundle_numeric_values(bundle)
        sentences = _split_claims(report)
        key_claims = [s for s in sentences if _is_key_claim(s)]

        verdicts: list[ClaimVerdict] = []
        cited_count = 0
        numeric_checked = 0
        numeric_ok = 0
        nli_scores: list[float] = []

        for claim in key_claims:
            citations = _CITE_RE.findall(claim)
            has_citation = _has_valid_citation(claim, evidence_ids)
            if has_citation:
                cited_count += 1

            # Tier 1: symbolic — referenced entities must exist.
            bad_entity = self._first_unknown_entity(claim, valid_entities)
            if bad_entity is not None:
                verdicts.append(ClaimVerdict(claim, "contradicted", citations,
                                             f"unknown entity {bad_entity}"))
                continue

            # Tier 2: numeric — cited numbers must match graph values.
            num_pass, n_checked, n_ok = self._numeric_ok(claim, valid_numbers)
            numeric_checked += n_checked
            numeric_ok += n_ok
            if not num_pass:
                verdicts.append(ClaimVerdict(claim, "contradicted", citations, "numeric mismatch"))
                continue

            if not has_citation:
                verdicts.append(ClaimVerdict(claim, "unsupported", citations, "missing citation"))
                continue

            # Tier 3: NLI — qualitative entailment by the graph-text.
            if self.config.enable_nli:
                score = float(self.nli_scorer.entail(graph_text, claim))
                nli_scores.append(score)
                if score < self.config.nli_threshold:
                    verdicts.append(ClaimVerdict(claim, "unsupported", citations,
                                                 "not entailed by graph", score))
                    continue
                verdicts.append(ClaimVerdict(claim, "supported", citations, "ok", score))
            else:
                verdicts.append(ClaimVerdict(claim, "supported", citations, "ok"))

        total = len(key_claims) or 1
        cgr = cited_count / total
        bad = sum(1 for v in verdicts if v.label in ("unsupported", "contradicted"))
        hr = bad / total
        num_acc = (numeric_ok / numeric_checked) if numeric_checked else 1.0
        fcs = (sum(nli_scores) / len(nli_scores)) if nli_scores else (1.0 if not self.config.enable_nli else 0.0)
        safety_pass = _check_safety(report)

        overall = (
            cgr >= self.config.cgr_threshold
            and hr <= self.config.max_hallucination_rate
            and safety_pass
        )
        return FaithfulnessRecord(
            citation_grounding_rate=cgr, hallucination_rate=hr, numeric_accuracy=num_acc,
            factual_consistency=fcs, safety_pass=safety_pass, claim_verdicts=verdicts,
            nli_model=self.nli_scorer.name, repair_tier=repair_tier, overall_pass=overall,
        )

    def _first_unknown_entity(self, claim: str, valid_entities: set[str]) -> str | None:
        candidates: list[str] = []
        candidates += re.findall(r"\bpkt_\w+", claim)
        candidates += re.findall(r"\bflow_\w+", claim)
        candidates += re.findall(r"\bT\d{4}(?:\.\d{3})?\b", claim)
        candidates += re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", claim)
        for cand in candidates:
            if cand not in valid_entities:
                return cand
        return None

    def _numeric_ok(self, claim: str, valid_numbers: list[float]) -> tuple[bool, int, int]:
        checked = 0
        ok = 0
        tol = self.config.numeric_tolerance
        for _kw, raw in _NUM_NEAR.findall(claim):
            value = float(raw)
            checked += 1
            # Accept either the literal value or its percentage form (88 == 0.88),
            # since SLMs often render a graph fraction 0.88 as "88%".
            if any(
                abs(value - v) <= tol or abs(value / 100.0 - v) <= tol
                for v in valid_numbers
            ):
                ok += 1
        return (ok == checked, checked, ok)
