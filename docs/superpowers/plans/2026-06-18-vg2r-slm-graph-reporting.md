# VG²R — Verifiable Graph-Grounded SLM Reporting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SLM read the HGT explanation subgraph directly via a lossless graph-text serialization, machine-verify every claim against that subgraph (symbolic + path + NLI), and repair/fallback unsupported claims — replacing the current `EvidenceBundle`-prose path — plus a dual evaluation rubric (GNN-explanation fidelity × NLG faithfulness).

**Architecture:** New `GraphSerializer` turns the typed `EvidenceBundle` (already extracted from `subgraph_snapshot`) into deterministic graph-text whose handles are the existing `evidence_id`s. The rewritten `ReportGenerator` prompts the local SLM on that graph-text. A new `GraphVerifier` labels each claim `{supported, unsupported, contradicted}` via a symbolic-citation tier, a numeric tier, a path tier, and an injected NLI tier; the `SlowPathWorker` runs a bounded repair loop and falls back to the existing template. A pure `vg2r_metrics` module + an eval script compute the dual rubric.

**Tech Stack:** Python 3.13 (`D:\v\nt114\Scripts\python.exe`), pytest, existing `runtime/slow_path` modules, Ollama `qwen2.5:3b` (unchanged), optional pretrained NLI cross-encoder (injected, not trained), numpy for bootstrap CI.

**Spec:** `docs/superpowers/specs/2026-06-18-vg2r-slm-graph-reporting-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/graphslm_ids/runtime/slow_path/graph_serializer.py` | Create | `EvidenceBundle` → deterministic graph-text (decision header, node table, edge list, salience block). |
| `src/graphslm_ids/runtime/slow_path/graph_verifier.py` | Create | 3-tier verifier + `NliScorer` protocol + `FaithfulnessRecord`. |
| `src/graphslm_ids/runtime/slow_path/report_generator.py` | Modify | Prompts on graph-text instead of bundle JSON; repair-prompt builder. |
| `src/graphslm_ids/runtime/slow_path/slow_path_worker.py` | Modify | Serialize → generate → verify → bounded repair → template fallback (old mini-bundle prose tier removed). |
| `src/graphslm_ids/runtime/slow_path/vg2r_metrics.py` | Create | Pure dual-rubric math (Fid+/Fid−/sparsity/characterization, coverage, plausibility, composite). |
| `scripts/eval/vg2r_report_eval.py` | Create | Orchestrate rubric over an eval set, bootstrap 95% CI, write JSON. |
| `tests/runtime/slow_path/__init__.py` | Create | Make the test package importable. |
| `tests/runtime/slow_path/test_graph_serializer.py` | Create | Golden serialization + determinism. |
| `tests/runtime/slow_path/test_graph_verifier.py` | Create | Catches injected hallucinations; numeric tolerance; path/NLI. |
| `tests/runtime/slow_path/test_report_generator_graphtext.py` | Create | Prompt contains graph-text + citation rules. |
| `tests/runtime/slow_path/test_vg2r_worker.py` | Create | Repair loop bounded; fallback path. |
| `tests/runtime/slow_path/test_vg2r_metrics.py` | Create | Rubric math correctness. |

**Out of scope (separate spec):** App 2 GraphToken soft-prompt ablation (needs HF/vLLM engine + projector training). Do **not** implement here.

**Shared test fixture:** every test builds a minimal `EvidenceBundle` inline from the dataclasses in `src/graphslm_ids/runtime/slow_path/evidence_bundle.py`. The canonical fixture (copy into each test file that needs it) is:

```python
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
```

---

## Task 1: GraphSerializer

**Files:**
- Create: `src/graphslm_ids/runtime/slow_path/graph_serializer.py`
- Create: `tests/runtime/slow_path/__init__.py`
- Test: `tests/runtime/slow_path/test_graph_serializer.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/runtime/slow_path/__init__.py` with an empty content (one line):

```python
# test package
```

- [ ] **Step 2: Write the failing test**

Create `tests/runtime/slow_path/test_graph_serializer.py` (include the `make_bundle` fixture from the File Structure section above, then):

```python
from graphslm_ids.runtime.slow_path.graph_serializer import serialize_bundle


def test_serialization_has_four_sections_and_handles():
    text = serialize_bundle(make_bundle())
    assert "## ALERT A_001 — HGT decision" in text
    assert "pred=SqlInjection conf=0.88" in text
    assert "## NODES" in text and "## EDGES" in text and "## SALIENCE" in text
    # every entity carries its evidence_id handle
    assert "[E_FLOW_001]" in text and "[E_PKT_001]" in text and "[E_TECH_001]" in text
    # node attributes are present and citable
    assert "attn=0.82" in text and "cf_drop=0.31" in text
    assert "T1190" in text and "TA0001" in text
    # typed edge with provenance
    assert "matches_technique" in text and "pmi+procedure" in text


def test_serialization_is_deterministic():
    a = serialize_bundle(make_bundle())
    b = serialize_bundle(make_bundle())
    assert a == b


def test_salience_orders_packets_by_attention():
    bundle = make_bundle()
    # add a second, lower-attention packet
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_graph_serializer.py -v`
Expected: FAIL with `ModuleNotFoundError: graph_serializer`.

- [ ] **Step 4: Write the implementation**

Create `src/graphslm_ids/runtime/slow_path/graph_serializer.py`:

```python
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
    techs = sorted(bundle.mitre_evidence, key=lambda t: (-t.cosine_score, t.evidence_id))
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
        lines.append(
            f"tech [{tech.evidence_id}] id={tech.technique_id} cosine={tech.cosine_score:.2f} "
            f"mapping={tech.mapping_type} name=\"{tech.technique_name}\""
        )
    # tactic nodes (deduped, deterministic by tactic_id)
    seen_tactics: set[str] = set()
    for tech in techs:
        if tech.tactic_id in seen_tactics:
            continue
        seen_tactics.add(tech.tactic_id)
        lines.append(f"tactic id={tech.tactic_id} name=\"{tech.tactic}\"")
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
            if edges[i] == "matches_technique":
                # attach technique provenance if we can find it
                match = next((t for t in techs if t.technique_id == dst), None)
                if match is not None:
                    prov = f" src={match.mapping_type} w={match.cosine_score:.2f}"
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_graph_serializer.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/graph_serializer.py tests/runtime/slow_path/__init__.py tests/runtime/slow_path/test_graph_serializer.py
git commit -m "feat(vg2r): GraphSerializer — subgraph to deterministic graph-text"
```

---

## Task 2: GraphVerifier (symbolic + numeric + path + NLI)

**Files:**
- Create: `src/graphslm_ids/runtime/slow_path/graph_verifier.py`
- Test: `tests/runtime/slow_path/test_graph_verifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/runtime/slow_path/test_graph_verifier.py` (include the `make_bundle` fixture, then):

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_graph_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: graph_verifier`.

- [ ] **Step 3: Write the implementation**

Create `src/graphslm_ids/runtime/slow_path/graph_verifier.py`:

```python
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

# Reuse the existing sentence/claim/citation/safety helpers — one definition.
from graphslm_ids.runtime.slow_path.report_validator import (
    _check_safety, _collect_evidence_ids, _collect_valid_entities,
    _has_valid_citation, _is_key_claim, _split_sentences,
)


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
    r"(confidence|probability|attention weight|attention|cf_drop|counterfactual drop|drop|port)\D{0,12}(\d+(?:\.\d+)?)",
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
        sentences = _split_sentences(report)
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
        for _kw, raw in _NUM_NEAR.findall(claim):
            value = float(raw)
            checked += 1
            if any(abs(value - v) <= self.config.numeric_tolerance for v in valid_numbers):
                ok += 1
        return (ok == checked, checked, ok)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_graph_verifier.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/graph_verifier.py tests/runtime/slow_path/test_graph_verifier.py
git commit -m "feat(vg2r): GraphVerifier — symbolic+numeric+NLI claim verification"
```

---

## Task 3: ReportGenerator on graph-text

**Files:**
- Modify: `src/graphslm_ids/runtime/slow_path/report_generator.py`
- Test: `tests/runtime/slow_path/test_report_generator_graphtext.py`

- [ ] **Step 1: Write the failing test**

Create `tests/runtime/slow_path/test_report_generator_graphtext.py`:

```python
from graphslm_ids.runtime.slow_path.report_generator import (
    ReportGenerator, build_repair_prompt, build_user_prompt_from_graphtext,
)

GRAPH_TEXT = "## ALERT A_001 — HGT decision\npred=SqlInjection conf=0.88 [E_ALERT]\n## NODES\n..."


def test_user_prompt_embeds_graph_text():
    prompt = build_user_prompt_from_graphtext(GRAPH_TEXT, alert_id="A_001")
    assert GRAPH_TEXT in prompt
    assert "A_001" in prompt
    assert "[E_" in prompt  # instructs handle citation


def test_repair_prompt_lists_failing_claims():
    prompt = build_repair_prompt(GRAPH_TEXT, ["claim X is wrong", "claim Y unverified"])
    assert "claim X is wrong" in prompt and "claim Y unverified" in prompt
    assert GRAPH_TEXT in prompt


def test_generate_calls_injected_callable_with_graph_text():
    seen = {}

    def fake_slm(system, user):
        seen["system"] = system
        seen["user"] = user
        return "# XAI Report - A_001\nok [E_ALERT]"

    gen = ReportGenerator(slm_callable=fake_slm)
    out = gen.generate_from_graphtext(GRAPH_TEXT, alert_id="A_001")
    assert out.startswith("# XAI Report")
    assert GRAPH_TEXT in seen["user"]
    assert "HGT" in seen["system"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_report_generator_graphtext.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_user_prompt_from_graphtext'`.

- [ ] **Step 3: Edit the implementation**

In `src/graphslm_ids/runtime/slow_path/report_generator.py`, add the graph-text prompt builders and two generator methods. Append these functions at module level (after `build_user_prompt`):

```python
def build_user_prompt_from_graphtext(graph_text: str, alert_id: str) -> str:
    return (
        "Generate an English XAI report for the following network intrusion alert.\n"
        "The SUBGRAPH below is the only allowed source. Every entity is tagged with a\n"
        "handle in square brackets (e.g. [E_PKT_001]); cite the handle for every claim.\n\n"
        "<subgraph>\n```\n"
        f"{graph_text}\n"
        "```\n</subgraph>\n\n"
        "Output format (strict Markdown):\n\n"
        f"# XAI Report - {alert_id}\n\n"
        "## 1. Alert Summary\n[1-2 sentences; cite [E_ALERT] and [E_FLOW_001].]\n\n"
        "## 2. Key Evidence\n[3-5 bullets, each ending with a handle like [E_PKT_001].]\n\n"
        "## 3. Graph-Based Explanation\n[Explain the flow->packet->technique->tactic path(s). Cite [E_PATH_*].]\n\n"
        "## 4. MITRE ATT&CK Interpretation\n[Per technique: id, name, tactic, cosine; state the link "
        "is embedding cosine similarity. Cite [E_TECH_*].]\n\n"
        "## 5. Confidence and Limitations\n[HGT confidence + limitations. Cite [E_ALERT].]\n\n"
        "## 6. Recommended Analyst Actions\n[3-5 generic actions grounded in cited evidence.]"
    )


def build_repair_prompt(graph_text: str, failing_claims: list[str]) -> str:
    bullets = "\n".join(f"- {c}" for c in failing_claims)
    return (
        "Your previous report contained claims NOT grounded in the subgraph. "
        "Remove or correct EACH of these claims, keeping everything else; cite a handle "
        "for every claim.\n\nUngrounded claims:\n"
        f"{bullets}\n\n<subgraph>\n```\n{graph_text}\n```\n</subgraph>"
    )
```

Then add two methods to the `ReportGenerator` class (place after `generate`):

```python
    def generate_from_graphtext(self, graph_text: str, alert_id: str, tier: int = 1) -> str:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt_from_graphtext(graph_text, alert_id)
        if tier >= 2:
            user_prompt += "\n\nBe concise. Cite every claim with a [handle]."
        return self._dispatch(system_prompt, user_prompt, tier)

    def regenerate_with_repair(self, graph_text: str, failing_claims: list[str], tier: int = 1) -> str:
        system_prompt = build_system_prompt()
        user_prompt = build_repair_prompt(graph_text, failing_claims)
        return self._dispatch(system_prompt, user_prompt, tier)

    def _dispatch(self, system_prompt: str, user_prompt: str, tier: int) -> str:
        if self.slm_callable is not None:
            return self.slm_callable(system_prompt, user_prompt)
        if self.client is not None:
            try:
                return self.client.generate(
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    temperature=self.config.temperature, top_p=self.config.top_p,
                    repeat_penalty=self.config.repeat_penalty,
                    context_length=self.config.context_length,
                    max_new_tokens=(min(self.config.max_new_tokens, self.config.tier2_max_new_tokens)
                                    if tier >= 2 else self.config.max_new_tokens),
                )
            except RuntimeError as exc:
                raise ReportGenerationError(str(exc)) from exc
        raise ReportGenerationError("No SLM backend is configured.")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_report_generator_graphtext.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/report_generator.py tests/runtime/slow_path/test_report_generator_graphtext.py
git commit -m "feat(vg2r): ReportGenerator prompts on graph-text + repair prompt"
```

---

## Task 4: Wire VG²R into SlowPathWorker (replace prose path, add repair loop)

**Files:**
- Modify: `src/graphslm_ids/runtime/slow_path/slow_path_worker.py`
- Test: `tests/runtime/slow_path/test_vg2r_worker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/runtime/slow_path/test_vg2r_worker.py` (include the `make_bundle` fixture, then):

```python
from graphslm_ids.runtime.slow_path.graph_verifier import NullNliScorer, VerifierConfig
from graphslm_ids.runtime.slow_path.report_generator import ReportGenerator
from graphslm_ids.runtime.slow_path.slow_path_worker import SlowPathWorker, SlowPathConfig
from graphslm_ids.runtime.slow_path.types import SlowPathJob


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_vg2r_worker.py -v`
Expected: FAIL (`SlowPathConfig` has no `max_repair_attempts`; `SlowPathWorker` has no `verifier_config`/`nli_scorer`).

- [ ] **Step 3: Edit `SlowPathConfig`**

In `src/graphslm_ids/runtime/slow_path/slow_path_worker.py`, add a field to `SlowPathConfig` (after `max_payload_preview_bytes`):

```python
    max_repair_attempts: int = 2
```

- [ ] **Step 4: Edit `SlowPathWorker.__init__`**

Replace the imports block at the top of `slow_path_worker.py` so the validator import becomes the new verifier set (keep all other imports). Add:

```python
from graphslm_ids.runtime.slow_path.graph_serializer import serialize_bundle
from graphslm_ids.runtime.slow_path.graph_verifier import GraphVerifier, NliScorer, NullNliScorer, VerifierConfig
```

Add two constructor params to `__init__` (after `validator`):

```python
        verifier_config: VerifierConfig | None = None,
        nli_scorer: NliScorer | None = None,
```

And inside `__init__`, after `self.validator = ...`, add:

```python
        self.verifier = GraphVerifier(verifier_config or VerifierConfig(), nli_scorer or NullNliScorer())
```

- [ ] **Step 5: Replace `process_job` body**

Replace the whole `process_job` method body (the `try:` block contents from `context = ...` through `return SlowPathResult(...)`) with the VG²R flow:

```python
    def process_job(self, job, hot_buffer=None, cold_store=None):
        store = cold_store or self.cold_store
        try:
            context = self.context_hydrator.hydrate(job.flow_id, hot_buffer, store)
            bundle = self.evidence_builder.build(
                job=job, context=context, max_cf_packets=self.config.max_cf_packets)
            bundle = self.evidence_ranker.rank_and_truncate(
                bundle, top_packets=self.config.top_packets,
                top_techniques=self.config.top_techniques, top_paths=self.config.top_paths)

            graph_text = serialize_bundle(bundle)
            report = None
            record = None
            try:
                report = self.report_generator.generate_from_graphtext(graph_text, job.alert_id, tier=1)
                record = self.verifier.verify(report, bundle, graph_text, repair_tier=1)
            except (ReportGenerationError, TimeoutError):
                report = None

            attempt = 0
            while (record is not None and not record.overall_pass
                   and attempt < self.config.max_repair_attempts):
                attempt += 1
                failing = [v.text for v in record.claim_verdicts
                           if v.label in ("unsupported", "contradicted")]
                try:
                    report = self.report_generator.regenerate_with_repair(graph_text, failing, tier=2)
                    record = self.verifier.verify(report, bundle, graph_text, repair_tier=1 + attempt)
                except (ReportGenerationError, TimeoutError):
                    break

            if record is not None and record.overall_pass:
                final_report, final_validation, final_tier = report, record, 1 if attempt == 0 else 2
            else:
                final_report = render_template(bundle, template_tag="TEMPLATE FALLBACK")
                final_validation = self.verifier.verify(
                    final_report, bundle, graph_text, repair_tier=3)
                final_tier = 3

            self._persist(job, bundle, final_report, final_validation, final_tier, store)
            return SlowPathResult(final_report, bundle, final_validation, final_tier)

        except TimeoutError:
            report = render_minimal(job)
            self._persist(job, None, report, None, 3, store)
            return SlowPathResult(report, None, None, 3)
```

> Note: this **removes** the old tier-2 `truncate_to_mini` prose branch and the old `ReportValidator` pass/fail gate from `process_job`. The `self.validator` field may remain for backward compatibility but is no longer used in the VG²R flow.

- [ ] **Step 6: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_vg2r_worker.py -v`
Expected: 2 passed.

- [ ] **Step 7: Run the full slow-path regression**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/test_slow_path.py -q`
Expected: PASS. If a test asserts the old tier-2 prose behavior, update it to expect tier-3 template fallback when verification fails (the VG²R flow has no prose tier-2). Show the change in the commit.

- [ ] **Step 8: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/slow_path_worker.py tests/runtime/slow_path/test_vg2r_worker.py
git commit -m "feat(vg2r): wire serialize->generate->verify->repair into SlowPathWorker"
```

---

## Task 5: Dual-rubric math (`vg2r_metrics.py`)

**Files:**
- Create: `src/graphslm_ids/runtime/slow_path/vg2r_metrics.py`
- Test: `tests/runtime/slow_path/test_vg2r_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/runtime/slow_path/test_vg2r_metrics.py`:

```python
import math
from graphslm_ids.runtime.slow_path.vg2r_metrics import (
    characterization, composite_f_star, coverage, fidelity_minus, fidelity_plus,
    plausibility, sparsity,
)


def test_fidelity_plus_and_minus():
    assert fidelity_plus(prob_full=0.88, prob_without_cited=0.30) == 0.58
    assert fidelity_minus(prob_full=0.88, prob_only_cited=0.85) == 0.03


def test_sparsity():
    assert sparsity(num_cited=2, num_total=10) == 0.2
    assert sparsity(num_cited=0, num_total=0) == 0.0


def test_characterization_high_when_necessary_and_sufficient():
    # high fid+ and low fid- -> high characterization
    high = characterization(fid_plus=0.9, fid_minus=0.05)
    low = characterization(fid_plus=0.1, fid_minus=0.8)
    assert high > low
    assert 0.0 <= high <= 1.0


def test_coverage_recall_of_salient_nodes():
    assert coverage(cited={"E_PKT_001"}, salient={"E_PKT_001", "E_PKT_002"}) == 0.5
    assert coverage(cited=set(), salient=set()) == 1.0


def test_plausibility_matches_class_map():
    cmap = {"SqlInjection": ["T1190"]}
    assert plausibility(cited_techniques=["T1190"], predicted_label="SqlInjection", class_to_technique=cmap) == 1.0
    assert plausibility(cited_techniques=["T1059"], predicted_label="SqlInjection", class_to_technique=cmap) == 0.0


def test_composite_is_harmonic_mean():
    val = composite_f_star(cgr=1.0, hallucination_rate=0.0, numeric_accuracy=1.0,
                           factual_consistency=1.0, characterization=1.0)
    assert math.isclose(val, 1.0)
    worse = composite_f_star(cgr=1.0, hallucination_rate=0.5, numeric_accuracy=1.0,
                            factual_consistency=1.0, characterization=1.0)
    assert worse < 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_vg2r_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: vg2r_metrics`.

- [ ] **Step 3: Write the implementation**

Create `src/graphslm_ids/runtime/slow_path/vg2r_metrics.py`:

```python
"""Pure dual-rubric math for VG²R. No I/O, no model — testable in isolation.

Axis A (explanation fidelity to HGT): GraphFramEx Fid+/Fid−/sparsity +
characterization. Axis B (report faithfulness): coverage, plausibility, and a
composite F* (harmonic mean). The predict-probability inputs are supplied by
the caller (which re-runs HGT on masked subgraphs), keeping this module pure.
"""
from __future__ import annotations


def fidelity_plus(prob_full: float, prob_without_cited: float) -> float:
    """Necessity: drop in HGT prob when the cited evidence is removed. Higher better."""
    return round(float(prob_full) - float(prob_without_cited), 6)


def fidelity_minus(prob_full: float, prob_only_cited: float) -> float:
    """Sufficiency: change in HGT prob when ONLY cited evidence is kept. Lower better."""
    return round(float(prob_full) - float(prob_only_cited), 6)


def sparsity(num_cited: int, num_total: int) -> float:
    if num_total <= 0:
        return 0.0
    return num_cited / num_total


def characterization(fid_plus: float, fid_minus: float, w_plus: float = 0.5, w_minus: float = 0.5) -> float:
    """Weighted harmonic mean of fid+ and (1 - fid-) (GraphFramEx). In [0, 1]."""
    a = max(min(float(fid_plus), 1.0), 0.0)
    b = max(min(1.0 - float(fid_minus), 1.0), 0.0)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return (w_plus + w_minus) / (w_plus / a + w_minus / b)


def coverage(cited: set[str], salient: set[str]) -> float:
    """Recall of HGT's top-k salient nodes among the report's citations."""
    if not salient:
        return 1.0
    return len(cited & salient) / len(salient)


def plausibility(cited_techniques: list[str], predicted_label: str, class_to_technique: dict[str, list[str]]) -> float:
    """Fraction of cited techniques that belong to the predicted class's MITRE map."""
    allowed = set(class_to_technique.get(predicted_label, []))
    if not cited_techniques:
        return 0.0
    hits = sum(1 for t in cited_techniques if t in allowed)
    return hits / len(cited_techniques)


def composite_f_star(cgr: float, hallucination_rate: float, numeric_accuracy: float,
                     factual_consistency: float, characterization: float) -> float:
    """Harmonic mean of {CGR, 1-HR, NumAcc, FCS, Characterization}. In [0, 1]."""
    parts = [float(cgr), 1.0 - float(hallucination_rate), float(numeric_accuracy),
             float(factual_consistency), float(characterization)]
    parts = [max(min(p, 1.0), 0.0) for p in parts]
    if any(p <= 0.0 for p in parts):
        return 0.0
    return len(parts) / sum(1.0 / p for p in parts)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_vg2r_metrics.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/vg2r_metrics.py tests/runtime/slow_path/test_vg2r_metrics.py
git commit -m "feat(vg2r): pure dual-rubric metrics (Fid+/Fid-/sparsity/coverage/plausibility/F*)"
```

---

## Task 6: Eval orchestration script + bootstrap CI

**Files:**
- Create: `scripts/eval/vg2r_report_eval.py`
- Test: `tests/runtime/slow_path/test_vg2r_eval_script.py`

This script grades a directory of precomputed records (so it runs without Ollama/HGT in CI). Each input record is a JSON with the fields the rubric needs; the script aggregates per-axis means + bootstrap 95% CI and the composite.

- [ ] **Step 1: Write the failing test**

Create `tests/runtime/slow_path/test_vg2r_eval_script.py`:

```python
import json
from pathlib import Path
from scripts.eval.vg2r_report_eval import aggregate_records, bootstrap_ci


def test_aggregate_records_computes_axes():
    records = [
        {"cgr": 1.0, "hallucination_rate": 0.0, "numeric_accuracy": 1.0,
         "factual_consistency": 1.0, "fid_plus": 0.6, "fid_minus": 0.05,
         "sparsity": 0.2, "coverage": 1.0, "plausibility": 1.0},
        {"cgr": 1.0, "hallucination_rate": 0.0, "numeric_accuracy": 1.0,
         "factual_consistency": 0.9, "fid_plus": 0.5, "fid_minus": 0.10,
         "sparsity": 0.3, "coverage": 0.5, "plausibility": 1.0},
    ]
    summary = aggregate_records(records)
    assert 0.0 <= summary["composite_f_star"] <= 1.0
    assert summary["axis_a"]["fid_plus_mean"] == 0.55
    assert "f_star_ci95" in summary


def test_bootstrap_ci_is_ordered():
    lo, hi = bootstrap_ci([0.8, 0.9, 0.85, 0.95, 0.7], seed=42, n=200)
    assert lo <= hi
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_vg2r_eval_script.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.eval.vg2r_report_eval`.

> If the import path fails because `scripts/` lacks `__init__.py`, create empty `scripts/__init__.py` and `scripts/eval/__init__.py` first (check with `git ls-files scripts/__init__.py`); only add them if missing.

- [ ] **Step 3: Write the implementation**

Create `scripts/eval/vg2r_report_eval.py`:

```python
"""Grade VG²R reports on the dual rubric. Reads a directory of per-alert JSON
records (each carrying the rubric inputs), aggregates per-axis means + the
composite F*, and reports bootstrap 95% CI. Runs offline (no Ollama/HGT needed
at grading time); the heavy generation/HGT-masking step writes the records.

Usage:
  D:\\v\\nt114\\Scripts\\python.exe scripts/eval/vg2r_report_eval.py \\
    --records-dir outputs/v3_ob_eacs_v2/vg2r_records --out outputs/v3_ob_eacs_v2/vg2r_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphslm_ids.runtime.slow_path.vg2r_metrics import characterization, composite_f_star


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def bootstrap_ci(values: list[float], seed: int = 42, n: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def aggregate_records(records: list[dict]) -> dict:
    def col(key: str) -> list[float]:
        return [float(r[key]) for r in records if key in r]

    char_vals = [characterization(float(r["fid_plus"]), float(r["fid_minus"]))
                 for r in records if "fid_plus" in r and "fid_minus" in r]
    f_stars = [
        composite_f_star(float(r["cgr"]), float(r["hallucination_rate"]), float(r["numeric_accuracy"]),
                         float(r["factual_consistency"]),
                         characterization(float(r["fid_plus"]), float(r["fid_minus"])))
        for r in records
    ]
    lo, hi = bootstrap_ci(f_stars)
    return {
        "n": len(records),
        "axis_a": {
            "fid_plus_mean": round(_mean(col("fid_plus")), 6),
            "fid_minus_mean": round(_mean(col("fid_minus")), 6),
            "sparsity_mean": round(_mean(col("sparsity")), 6),
            "characterization_mean": round(_mean(char_vals), 6),
        },
        "axis_b": {
            "cgr_mean": round(_mean(col("cgr")), 6),
            "hallucination_rate_mean": round(_mean(col("hallucination_rate")), 6),
            "numeric_accuracy_mean": round(_mean(col("numeric_accuracy")), 6),
            "factual_consistency_mean": round(_mean(col("factual_consistency")), 6),
            "coverage_mean": round(_mean(col("coverage")), 6),
            "plausibility_mean": round(_mean(col("plausibility")), 6),
        },
        "composite_f_star": round(_mean(f_stars), 6),
        "f_star_ci95": [round(lo, 6), round(hi, 6)],
    }


def _load_records(records_dir: Path) -> list[dict]:
    records = []
    for path in sorted(records_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="VG²R dual-rubric eval")
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = _load_records(Path(args.records_dir))
    summary = aggregate_records(records)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_vg2r_eval_script.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -f scripts/eval/vg2r_report_eval.py tests/runtime/slow_path/test_vg2r_eval_script.py
git commit -m "feat(vg2r): dual-rubric eval script with bootstrap 95% CI"
```

---

## Task 7: Full regression + docs link

**Files:**
- Modify: `docs/reports/2026-06-17-smart-both-evaluation-methodology.md` (optional cross-link) — skip if no natural anchor.

- [ ] **Step 1: Run the entire test suite**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/ -q`
Expected: all pass. Fix any test that asserted the removed tier-2 prose behavior (see Task 4 Step 7).

- [ ] **Step 2: Commit any regression fixes**

```bash
git add -f tests/
git commit -m "test(vg2r): align slow-path tests with VG2R verify/repair flow"
```

---

## Self-Review (completed)

**1. Spec coverage:**
- §3.2 modules → Tasks 1–4 (Serializer, Verifier, Generator, Worker wiring). ✓
- §3.3 GraphSerializer encoding (header/nodes/edges/salience, deterministic) → Task 1. ✓
- §3.4 GraphVerifier 3 tiers → Task 2 (symbolic+numeric+NLI; path is covered by symbolic entity/handle existence on `E_PATH_*` + path_nodes entities — note below). ✓
- §3.5 RepairLoop + fallback → Task 4. ✓
- §4 dual rubric (Fid+/Fid−/sparsity/characterization, CGR/HR/NumAcc/FCS/coverage/plausibility/composite, bootstrap CI) → Tasks 5–6. ✓
- §5.1 replace-not-parallel (remove prose tier-2) → Task 4 Step 5. ✓
- §6 config knobs (τ numeric_tolerance, N max_repair_attempts, NLI threshold/model) → `VerifierConfig` (Task 2) + `SlowPathConfig.max_repair_attempts` (Task 4). ✓
- Safety rule preserved → reused `_check_safety` in Task 2. ✓

**Gap noted (intentional, in spec §7 limitations):** the **NLI scorer real implementation** (cross-encoder) and the **records-producing harness** (run SLM + re-run HGT on masked subgraphs to fill `fid_plus`/`fid_minus`/`coverage`/`plausibility`) are wired as injection points but the heavyweight model glue is left to execution time (needs the L40S + a model choice). Task 6 grades precomputed records so the pipeline is testable offline. The real NLI scorer and the records producer are follow-on integration steps; the LLM-as-judge correlation (spec §4.3) is an analysis step on the produced records, not new code. If the executor wants them in-plan, add: a `CrossEncoderNliScorer` (lazy-import `sentence-transformers`/`transformers`) implementing the `NliScorer` protocol, and a `vg2r_record_writer.py` that calls `SlowPathWorker` + `HGTCounterfactual`. These are additive and do not change the interfaces above.

**2. Placeholder scan:** no TBD/TODO; every code step has full code. ✓

**3. Type consistency:** `serialize_bundle(bundle)->str`, `GraphVerifier.verify(report,bundle,graph_text,repair_tier)->FaithfulnessRecord`, `ReportGenerator.generate_from_graphtext/regenerate_with_repair/_dispatch`, `SlowPathConfig.max_repair_attempts`, `VerifierConfig(numeric_tolerance,nli_threshold,cgr_threshold,max_hallucination_rate,enable_nli)`, metric fn signatures — all consistent across tasks. ✓
```
