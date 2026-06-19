# SLM Graph-Reading Realignment to v3 — Design

**Date:** 2026-06-19
**Status:** Design (awaiting user review)
**Scope decision:** Family A (graph-as-text, no training, keep Ollama) at provenance level **A2 (full token-level provenance)**.

## 1. Problem

The slow-path SLM is supposed to read the HGT evidence subgraph and produce a
grounded XAI report. Today it reads a **graph that no longer exists**:

- The runtime hot buffer now emits `mitre_topk` as v3 triples
  `(technique, family, weight)` from the **MSEE ensemble** (PMI + procedure),
  produced by `RuntimeEdgeAssigner` (see the 2026-06-18 runtime-v3 realignment).
- But `HotBufferAdapter` coerces `mitre_topk` into
  `PacketContext.mitre_cosine_scores: dict[str, float]`
  (`hot_buffer_adapter.py:180,196`), **dropping `family`** and mislabelling the
  source as `mapping_type="embedding_cosine_similarity"` (`types.py:54`,
  `hot_buffer_adapter.py:401`).
- Everything downstream — `EvidenceBundle.MitreEvidence` (`cosine_score`,
  `mapping_type`, `mapping_caution`, `mitre_max_cosine`), `serialize_bundle`
  (`cosine=…`, `matches_technique` edges), `graph_verifier`, and
  `DEFAULT_LIMITATIONS` ("MITRE mapping uses embedding cosine similarity") —
  describes the **v1 SecureBERT cosine scheme that CLAUDE.md explicitly calls
  "semantically meaningless" and that v3 removed.**

This is the same *runtime drift* fixed in the fast path, now in the slow path.
The SLM cannot read the real graph because the real graph's evidence (family
routing, PMI/procedure weights, token-level provenance, host nodes) is destroyed
before it reaches the serializer.

## 2. Goal & non-goals

**Goal:** the SLM reads a **faithful structural encoding of the v3 evidence
subgraph** — typed `evidence_<family>` edges, PMI/procedure weights, the matched
tokens/literals (token-level provenance), host nodes, and HGT-attention salience
— with citation handles the verifier shares.

**Non-goals (explicit, evidence-based):**
- No projector / soft-prompt / embedding injection. That is the *other* research
  family (GraphToken arXiv 2402.05862; LLaGA ICML'24 arXiv 2402.08170; GraphGPT
  SIGIR'24), which keeps the LLM frozen but **requires training a projector** and
  an `inputs_embeds` runtime. The user chose **not** to train. Recorded here so
  the decision is auditable.
- No change to the SLM (qwen2.5:3b stays off-the-shelf, frozen, via Ollama).
- No change to the VG²R control flow (generate → verify → bounded repair →
  template fallback). Only the **data fields** it reads change.

## 3. Scientific grounding

- Fatemi, Halcrow, Perozzi. *Talk like a Graph: Encoding Graphs for LLMs.*
  ICLR 2024. — A frozen off-the-shelf LLM reasons over a graph when the graph is
  given as a structured encoding (node table + typed edge list); **the encoding
  function materially changes accuracy**. This is the basis for reading the graph
  as structured text rather than prose, and for making that text v3-faithful.
- The token-level provenance requirement is the project's own stated novelty
  (CLAUDE.md: "each edge carries token-level provenance suitable for SOC audit"),
  which the current cosine encoding cannot express.

## 4. Architecture & data flow

Bottom-up realignment; each layer keeps its single responsibility and a typed
interface to the next.

```
HotGraphBuffer (mitre_topk: (tech, family, weight) [+provenance])
   │
   ▼  HotBufferAdapter._build_packets / _resolve_flow_mitre_scores
PacketContext.evidence{tech: (family, weight, [tokens], [literals])}
GraphContext.flow_evidence{...}
   │
   ▼  EvidenceBuilder._build_mitre / _build_packets / _build_paths
EvidenceBundle.MitreEvidence(family, evidence_weight, source, matched_tokens,
                             matched_literals, supporting_packet_count)
   │
   ▼  serialize_bundle
graph-text:  evidence_<family> w=0.83 src=pmi+proc tokens="select,union" [E_TECH_001]
   │
   ▼  ReportGenerator (Ollama, frozen)  →  GraphVerifier (claims ↔ v3 fields)
grounded report
```

### 4.1 Provenance source (A2)

`RuntimeEdgeAssigner.assign_packet` is extended to return, alongside each
`(technique, family, weight)` edge, the **evidence that fired it**:
- **PMI tokens:** a new provenance-returning variant of `lookup_pmi_per_packet`
  in `ensemble.py` that also returns `{technique: [matched_tokens]}` (the
  function already tokenizes and accumulates per token; it just discards the
  token identity today — we keep it).
- **Procedure literals:** `ProcedureMatcher.match()` **already** returns
  `{technique: [matched_patterns]}` (`procedure_matcher.py:172`); the assigner
  calls `match()` and keeps the literals (it currently calls only
  `weight_per_technique`, which collapses them).

The new edge record is therefore `(technique, family, weight, source, tokens,
literals)` where `source ∈ {"pmi", "procedure", "pmi+procedure", "flow"}`.

To stay backward compatible with the fast-path graph builder (which consumes
`(tech, family, weight)` triples), the provenance is carried as an **optional
parallel structure** keyed by `(packet_id, technique)`, not by widening the
existing triple tuple the SubgraphBuilder relies on.

## 5. Component changes

| File | Change |
|---|---|
| `offline/preprocessing/ensemble.py` | Add `lookup_pmi_per_packet_with_tokens(payload, lookup) -> dict[tech, (family, weight, [tokens])]`. Keep the existing function (no behaviour change to the offline build). |
| `runtime/fast_path/edge_assigner.py` | `assign_packet` gains `return_provenance: bool = False`. When set, returns `(edges, provenance)` where `provenance[(tech)] = {source, tokens, literals}`. Default path unchanged (fast path keeps emitting plain triples). |
| `runtime/pipeline/runtime_pipeline.py` | When the assigner runs in `on_packet`, also stash per-packet provenance into the hot buffer (new optional field on the packet metadata), so the slow path can hydrate it. |
| `runtime/fast_path/hot_graph_buffer.py` | Store optional `mitre_provenance` per packet alongside `mitre_topk`; expose in `get_packets`/`snapshot`. |
| `runtime/slow_path/types.py` | `PacketContext`: replace `mitre_cosine_scores: dict[str,float]` semantics with `mitre_evidence: dict[str, MitreEdge]` where `MitreEdge=(family, weight, source, tokens, literals)`; drop the `mapping_type="embedding_cosine_similarity"` default. Keep a compatibility accessor so existing call sites degrade gracefully. |
| `runtime/slow_path/hot_buffer_adapter.py` | Parse v3 triples + provenance instead of coercing to `dict[str,float]`; populate `mitre_evidence`. |
| `runtime/slow_path/evidence_bundle.py` | `MitreEvidence`: replace `cosine_score`/`mapping_type`/`mapping_caution`/`mitre_max_cosine` with `family: str`, `evidence_weight: float`, `source: str`, `matched_tokens: list[str]`, `matched_literals: list[str]`. Keep `technique_id/name/tactic/tactic_id/supporting_packet_count/matched_from`. |
| `runtime/slow_path/evidence_builder.py` | `_build_mitre` aggregates v3 edges (max weight per technique, union of tokens/literals, source set); `_build_paths` uses v3 edge names (`contain`, `evidence_<family>`, `technique_tactic`); `_build_packets.linked_techniques` unchanged; rewrite `DEFAULT_LIMITATIONS` (remove the false cosine line, state PMI+procedure ensemble + that weights are statistical, not proof). |
| `runtime/slow_path/graph_serializer.py` | NODES: add a host block; tech rows print `family`, `w=<weight>`, `src=<source>`, and `tokens="…"`/`literals="…"` instead of `cosine=`/`mapping=`. EDGES: print `pkt -evidence_<family>-> tech  w=… src=… [E_…]` and `tech -technique_tactic-> tactic`. SALIENCE block unchanged (HGT attention). |
| `runtime/slow_path/graph_verifier.py` | Numeric/grounding checks read `evidence_weight` and `family` from the bundle instead of `cosine_score`; citation scheme unchanged. |

## 6. VG²R unchanged

`SlowPathWorker.process_job` keeps the exact control flow: serialize → tier-1
generate → verify → bounded repair (tier-2) → template fallback (tier-3). Only
`serialize_bundle` output and the verifier's field names change. The citation
handles (`E_ALERT`, `E_PKT_*`, `E_TECH_*`, `E_PATH_*`) are preserved so the
generator/verifier contract is stable.

## 7. Testing

- **Unit (offline):** `lookup_pmi_per_packet_with_tokens` returns the same
  `(family, weight)` as the existing function plus the correct token list (parity
  test against `lookup_pmi_per_packet`).
- **Unit (assigner):** `assign_packet(..., return_provenance=True)` returns
  literals from `ProcedureMatcher.match` and tokens from the PMI variant; default
  call still returns plain triples (fast-path regression guard).
- **Unit (adapter):** v3 `mitre_topk` triples + provenance hydrate into
  `PacketContext.mitre_evidence` with family preserved; a legacy 2-tuple input
  still parses (degraded: empty family/provenance).
- **Unit (bundle/serializer):** `MitreEvidence` has no cosine fields; serialized
  graph-text contains `evidence_<family>`, `w=`, `src=`, `tokens=`/`literals=`,
  a host block, and **no** `cosine=`/`matches_technique`.
- **Unit (verifier):** a report citing the v3 weight passes; a report citing a
  wrong weight/family is flagged unsupported.
- **Integration:** end-to-end `process_job` on a synthetic v3 bundle yields a
  passing tier-1 or tier-2 report whose claims are all v3-grounded.
- **Regression:** full `pytest tests/ -q` stays green (the fast-path triple
  contract is unchanged).

## 8. Known gaps (intentional)

- `burst_neighbor` (flow→flow) is still not produced online (single seed flow) —
  inherited from the fast-path realignment; out of scope here.
- Counterfactual evidence is unchanged (packet-masking ΔConfidence); it is
  orthogonal to the cosine→v3 change.
- Provenance is only as rich as the online assigner sees per packet; PMI tokens
  are payload byte-n-gram/text tokens (the same tokenizer the offline build
  uses), not human-curated rule names.

## 9. References (verified 2026-06-18 via web search)

- Fatemi et al., *Talk like a Graph*, ICLR 2024.
- Perozzi et al., *Let Your Graph Do the Talking* (GraphToken), arXiv 2402.05862 — LLM frozen, GNN+projection trained.
- Chen et al., *LLaGA*, ICML 2024, arXiv 2402.08170 — "no modifications to the LLM parameters", projector trained.
- Tang et al., *GraphGPT*, SIGIR 2024, arXiv 2310.13023.
- Liu et al., *Can we Soft Prompt LLMs for Graph Learning Tasks?*, arXiv 2402.10359 — frozen LLM + embedding projector.
- *Enhancing Small Language Models for Graph Tasks Through Graph Encoder Integration*, MDPI Appl. Sci. 2025.
