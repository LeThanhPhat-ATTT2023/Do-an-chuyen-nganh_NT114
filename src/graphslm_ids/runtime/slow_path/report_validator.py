from __future__ import annotations

from dataclasses import dataclass
import re

from graphslm_ids.runtime.slow_path.evidence_bundle import EvidenceBundle


@dataclass
class ValidationResult:
    grounding_rate: float
    hallucinated_entity_count: int
    mitre_caution_present: bool
    safety_pass: bool
    unsupported_claim_count: int
    overall_pass: bool


@dataclass
class ValidatorConfig:
    grounding_threshold: float = 0.60
    max_hallucinated_entities: int = 2
    max_unsupported_claims: int = 0
    require_mitre_caution: bool = True


class ReportValidator:
    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self.config = config or ValidatorConfig()

    def validate(self, report: str, bundle: EvidenceBundle) -> ValidationResult:
        evidence_ids = _collect_evidence_ids(bundle)
        sentences = _split_sentences(report)

        key_claims = [sentence for sentence in sentences if _is_key_claim(sentence)]
        cited_claims = [
            sentence for sentence in key_claims if _has_valid_citation(sentence, evidence_ids)
        ]
        grounding_rate = 1.0
        if key_claims:
            grounding_rate = len(cited_claims) / len(key_claims)

        hallucinated_entity_count = _count_hallucinated_entities(report, bundle)
        mitre_caution_required = self.config.require_mitre_caution and _requires_mitre_caution(bundle)
        mitre_caution_present = True
        if mitre_caution_required:
            mitre_caution_present = _has_mitre_caution(report)
        unsupported_claim_count = _count_unsupported_claims(sentences, evidence_ids)
        safety_pass = _check_safety(report)

        overall_pass = (
            grounding_rate >= self.config.grounding_threshold
            and hallucinated_entity_count <= self.config.max_hallucinated_entities
            and unsupported_claim_count <= self.config.max_unsupported_claims
            and safety_pass
        )
        if mitre_caution_required:
            overall_pass = overall_pass and mitre_caution_present

        return ValidationResult(
            grounding_rate=grounding_rate,
            hallucinated_entity_count=hallucinated_entity_count,
            mitre_caution_present=mitre_caution_present,
            safety_pass=safety_pass,
            unsupported_claim_count=unsupported_claim_count,
            overall_pass=overall_pass,
        )


def _split_sentences(text: str) -> list[str]:
    evidence_ref = r"(?:E_[A-Z]+_\d{3}|E_ALERT|E_FLOW_\d{3})"
    protected = re.sub(rf"([.!?])\s+(\[{evidence_ref}\])", r"\1@@CITE@@\2", text.strip())
    parts = re.split(r"(?:(?<=[.!?])|(?<=\]))\s+", protected)
    return [part.replace("@@CITE@@", " ").strip() for part in parts if part.strip()]


def _is_key_claim(sentence: str) -> bool:
    claim_keywords = (
        "indicates",
        "suggests",
        "shows",
        "linked",
        "associated",
        "confidence",
        "probability",
        "technique",
        "tactic",
    )
    if any(keyword in sentence.lower() for keyword in claim_keywords):
        return True
    if re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", sentence):
        return True
    if re.search(r"T\d{4}(?:\.\d{3})?", sentence):
        return True
    if re.search(r"pkt_\w+", sentence):
        return True
    if re.search(r"flow_\w+", sentence):
        return True
    return False


def _has_valid_citation(sentence: str, evidence_ids: set[str]) -> bool:
    matches = re.findall(r"\[(E_[A-Z]+_\d{3}|E_ALERT|E_FLOW_\d{3})\]", sentence)
    if not matches:
        return False
    return any(match in evidence_ids for match in matches)


def _collect_evidence_ids(bundle: EvidenceBundle) -> set[str]:
    ids = {bundle.alert.evidence_id, bundle.flow_evidence.evidence_id}
    ids.update(item.evidence_id for item in bundle.packet_evidence)
    ids.update(item.evidence_id for item in bundle.mitre_evidence)
    ids.update(item.evidence_id for item in bundle.graph_paths)
    ids.update(item.evidence_id for item in bundle.counterfactual_evidence)
    return ids


def _collect_valid_entities(bundle: EvidenceBundle) -> set[str]:
    entities = {
        bundle.flow_evidence.src_ip,
        bundle.flow_evidence.dst_ip,
        str(bundle.flow_evidence.src_port),
        str(bundle.flow_evidence.dst_port),
        bundle.flow_evidence.flow_id,
    }
    for packet in bundle.packet_evidence:
        entities.add(packet.packet_id)
    for tech in bundle.mitre_evidence:
        entities.add(tech.technique_id)
        entities.add(tech.tactic_id)
        entities.add(tech.tactic)
    return entities


def _count_hallucinated_entities(report: str, bundle: EvidenceBundle) -> int:
    valid_entities = _collect_valid_entities(bundle)
    extracted = set()
    extracted.update(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", report))
    extracted.update(re.findall(r"T\d{4}(?:\.\d{3})?", report))
    extracted.update(re.findall(r"pkt_\w+", report))
    extracted.update(re.findall(r"flow_\w+", report))
    extracted.update(re.findall(r":(\d{2,5})\b", report))
    extracted.update(re.findall(r"\bport\s+(\d{2,5})\b", report, flags=re.IGNORECASE))

    hallucinated = [entity for entity in extracted if entity not in valid_entities]
    return len(hallucinated)


def _requires_mitre_caution(bundle: EvidenceBundle) -> bool:
    return any(
        "embedding" in tech.mapping_type.lower()
        or "cosine" in tech.mapping_type.lower()
        or "semantic" in tech.mapping_caution.lower()
        for tech in bundle.mitre_evidence
    )


def _has_mitre_caution(report: str) -> bool:
    keywords = (
        "embedding similarity",
        "cosine similarity",
        "semantic similarity",
        "not deterministic",
        "not a signature",
    )
    report_lower = report.lower()
    return any(keyword in report_lower for keyword in keywords)


def _count_unsupported_claims(sentences: list[str], evidence_ids: set[str]) -> int:
    patterns = (
        "confirms that",
        "proves that",
        "is definitely",
        "is certainly",
    )
    count = 0
    for sentence in sentences:
        lower_sentence = sentence.lower()
        has_unsupported_pattern = any(pattern in lower_sentence for pattern in patterns)
        has_unsupported_pattern = has_unsupported_pattern or bool(
            re.search(r"\bis (?:a|an) [a-z0-9 _-]+ attack\b", lower_sentence)
        )
        if not has_unsupported_pattern:
            continue
        if not _has_valid_citation(sentence, evidence_ids):
            count += 1
    return count


def _check_safety(report: str) -> bool:
    blocklist = (
        "exploit code",
        "shellcode",
        "reverse shell",
        "payload delivery",
        "step by step",
        "how to",
        "to bypass",
    )
    report_lower = report.lower()
    if any(pattern in report_lower for pattern in blocklist):
        return False
    if re.search(r"\b[0-9a-f]{64,}\b", report_lower):
        return False
    return True
