# SLM Graph-Text v3 Realignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the slow-path SLM read the **v3 evidence subgraph faithfully** (family-typed evidence edges, PMI/procedure weights, token-level provenance, host nodes) instead of the dead v1 cosine scheme — with no SLM training, keeping Ollama and the VG²R control flow.

**Architecture:** Bottom-up realignment. Extend the offline PMI lookup + `RuntimeEdgeAssigner` to surface matched tokens/literals (provenance). Carry `(technique, family, weight)` + provenance through the hot buffer → `HotBufferAdapter` → `GraphContext`/`PacketContext` → `EvidenceBundle` → `serialize_bundle` → `GraphVerifier`. Drop all `cosine`/`mapping_type` fields. VG²R generate→verify→repair→template is unchanged; only the data fields change.

**Tech Stack:** Python 3.13 (`D:\v\nt114\Scripts\python.exe`), pytest, numpy/pandas, existing `runtime/slow_path` + `runtime/fast_path` + `offline/preprocessing` modules.

**Spec:** `docs/superpowers/specs/2026-06-19-slm-graphtext-v3-realign-design.md`

---

## Background the implementer MUST know

The v3 runtime emits `mitre_topk` as triples `(technique, family, weight)` from
`RuntimeEdgeAssigner` (PMI + procedure ensemble). The slow path still expects the
old v1 cosine scheme and **drops the family**:

- `HotBufferAdapter._coerce_mitre_scores` only handles `len(item)==2` tuples, so
  v3 triples are skipped entirely (`hot_buffer_adapter.py:416-428`).
- `MitreMetadata.mapping_type` defaults to `"embedding_cosine_similarity"`
  (`types.py:54`); `MitreEvidence` carries `cosine_score`/`mapping_type`/
  `mapping_caution`/`mitre_max_cosine` (`evidence_bundle.py:103-128`).
- `serialize_bundle` prints `cosine=…` and `matches_technique`
  (`graph_serializer.py:59-85`); `graph_verifier` reads `tech.cosine_score`
  (`graph_verifier.py:103`).

The offline evidence functions already exist:
- `ProcedureMatcher.match(payload) -> dict[tech, [literals]]`
  (`offline/preprocessing/procedure_matcher.py:149-176`) — literals available.
- `lookup_pmi_per_packet(payload, lookup) -> dict[tech, (family, weight)]`
  (`offline/preprocessing/ensemble.py:82-129`) — tokenizes internally but drops
  token identity (Task 1 adds a provenance variant).
- `aggregate_evidence(pmi_hits, proc_hits, flow, family_map, tau_edge)` returns
  `[(tech, family, weight)]` (`ensemble.py:132+`).

**Shared data contract (consistent across all tasks):**
- Per-packet provenance: `dict[str, dict]` keyed by `technique_id`, value
  `{"source": str, "tokens": list[str], "literals": list[str]}` where
  `source ∈ {"pmi", "procedure", "pmi+procedure", "flow"}`.
- `types.MitreEdge` dataclass: `family: str`, `weight: float`, `source: str = ""`,
  `tokens: list[str] = []`, `literals: list[str] = []`.
- `PacketContext.mitre_evidence: dict[str, MitreEdge]` (replaces
  `mitre_cosine_scores`). `GraphContext.flow_mitre_evidence: dict[str, MitreEdge]`
  (replaces `flow_mitre_scores`).
- `EvidenceBundle.MitreEvidence` fields: `family`, `evidence_weight`, `source`,
  `matched_tokens`, `matched_literals` (replace the cosine fields).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/graphslm_ids/offline/preprocessing/ensemble.py` | Modify | Add `lookup_pmi_per_packet_with_tokens` (provenance variant; offline build untouched). |
| `src/graphslm_ids/runtime/fast_path/edge_assigner.py` | Modify | `assign_packet(..., return_provenance=False)` optionally returns `(edges, provenance)`. |
| `src/graphslm_ids/runtime/fast_path/hot_graph_buffer.py` | Modify | Store + expose optional per-packet `mitre_provenance`. |
| `src/graphslm_ids/runtime/pipeline/runtime_pipeline.py` | Modify | Request provenance from the assigner; pass it into `add_packet`. |
| `src/graphslm_ids/runtime/slow_path/types.py` | Modify | `MitreEdge` dataclass; `PacketContext.mitre_evidence`; `GraphContext.flow_mitre_evidence`; drop cosine default on `MitreMetadata`. |
| `src/graphslm_ids/runtime/slow_path/hot_buffer_adapter.py` | Modify | Parse v3 triples + provenance into `MitreEdge`. |
| `src/graphslm_ids/runtime/slow_path/evidence_bundle.py` | Modify | `MitreEvidence` v3 fields. |
| `src/graphslm_ids/runtime/slow_path/evidence_builder.py` | Modify | Build v3 MitreEvidence; v3 path edge names; fix `DEFAULT_LIMITATIONS`. |
| `src/graphslm_ids/runtime/slow_path/graph_serializer.py` | Modify | Render `evidence_<family>` + weight + source + provenance + host block. |
| `src/graphslm_ids/runtime/slow_path/graph_verifier.py` | Modify | Numeric values read `evidence_weight`. |
| `tests/runtime/slow_path/test_v3_graphtext.py` | Create | All new behaviour (one file, grouped). |

---

## Task 1: PMI token-level provenance variant

**Files:**
- Modify: `src/graphslm_ids/offline/preprocessing/ensemble.py`
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

- [ ] **Step 1: Write the failing test**

Create `tests/runtime/slow_path/test_v3_graphtext.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k pmi_with_tokens -v`
Expected: FAIL — `ImportError: cannot import name 'lookup_pmi_per_packet_with_tokens'`.

- [ ] **Step 3: Implement the provenance variant**

In `ensemble.py`, add after `lookup_pmi_per_packet` (it ends near line 129):

```python
def lookup_pmi_per_packet_with_tokens(
    payload: bytes,
    pmi_lookup: PmiLookup,
) -> dict[str, tuple[str, float, list[str]]]:
    """Same as :func:`lookup_pmi_per_packet` but also returns the matched tokens.

    Returns ``{technique_id: (family, summed_weight_clipped, [matched_tokens])}``.
    Token order is the first-seen order; duplicates are de-duplicated. Empty
    payload / no tokens / no hits all return ``{}``.
    """
    if not payload:
        return {}
    tokens = tokenize_payload(payload)
    if not tokens:
        return {}
    accum: dict[str, list] = {}  # tech -> [family, weight, [tokens]]
    for tok in tokens:
        rows = pmi_lookup.get(tok)
        if not rows:
            continue
        for tech, family, w in rows:
            existing = accum.get(tech)
            if existing is None:
                accum[tech] = [family, w, [tok]]
            else:
                existing[1] += w
                if tok not in existing[2]:
                    existing[2].append(tok)
    out: dict[str, tuple[str, float, list[str]]] = {}
    for tech, (family, w, toks) in accum.items():
        if w > 1.0:
            w = 1.0
        out[tech] = (family, float(w), list(toks))
    return out
```

Note: `tokenize_payload` is already imported at the top of `ensemble.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k pmi_with_tokens -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/offline/preprocessing/ensemble.py tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "feat(ensemble): PMI per-packet lookup variant returning matched tokens"
```

---

## Task 2: RuntimeEdgeAssigner returns provenance

**Files:**
- Modify: `src/graphslm_ids/runtime/fast_path/edge_assigner.py`
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/runtime/slow_path/test_v3_graphtext.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k assign_packet -v`
Expected: FAIL — `assign_packet() got an unexpected keyword argument 'return_provenance'`.

- [ ] **Step 3: Edit `edge_assigner.py`**

Replace the imports block so the provenance PMI variant is available:

```python
from graphslm_ids.offline.preprocessing.ensemble import (
    aggregate_evidence,
    build_pmi_lookup_from_table,
    lookup_pmi_per_packet,
    lookup_pmi_per_packet_with_tokens,
)
```

Replace the whole `assign_packet` method with:

```python
    def assign_packet(
        self,
        payload: bytes,
        flow_consensus: dict[str, float] | None = None,
        return_provenance: bool = False,
    ):
        if not payload:
            return ([], {}) if return_provenance else []
        proc_hits = self._proc.weight_per_technique(payload)
        if not return_provenance:
            pmi_hits = lookup_pmi_per_packet(payload, self._pmi_lookup)
            return aggregate_evidence(
                pmi_hits, proc_hits, flow_consensus or {}, self._family, tau_edge=self._tau
            )

        pmi_tok = lookup_pmi_per_packet_with_tokens(payload, self._pmi_lookup)
        pmi_hits = {tech: (fam, w) for tech, (fam, w, _toks) in pmi_tok.items()}
        edges = aggregate_evidence(
            pmi_hits, proc_hits, flow_consensus or {}, self._family, tau_edge=self._tau
        )
        proc_literals = self._proc.match(payload) if hasattr(self._proc, "match") else {}
        provenance: dict[str, dict] = {}
        for tech, _family, _w in edges:
            in_pmi = tech in pmi_tok
            in_proc = tech in proc_hits
            source = "pmi+procedure" if (in_pmi and in_proc) else ("pmi" if in_pmi else "procedure")
            provenance[tech] = {
                "source": source,
                "tokens": list(pmi_tok.get(tech, ("", 0.0, []))[2]),
                "literals": list(proc_literals.get(tech, [])),
            }
        return edges, provenance
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k assign_packet -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/fast_path/edge_assigner.py tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "feat(runtime): RuntimeEdgeAssigner optional token/literal provenance"
```

---

## Task 3: hot buffer + pipeline carry provenance

**Files:**
- Modify: `src/graphslm_ids/runtime/fast_path/hot_graph_buffer.py`
- Modify: `src/graphslm_ids/runtime/pipeline/runtime_pipeline.py`
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

The hot buffer stores an optional `mitre_provenance` per packet and exposes it in
`get_packets`. The pipeline asks the assigner for provenance and passes it in.

- [ ] **Step 1: Write the failing test**

Append to `tests/runtime/slow_path/test_v3_graphtext.py`:

```python
import numpy as np
from graphslm_ids.runtime.fast_path.hot_graph_buffer import HotGraphBuffer


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k provenance -v`
Expected: FAIL — `add_packet() got an unexpected keyword argument 'mitre_provenance'`.

- [ ] **Step 3: Edit `hot_graph_buffer.py`**

In `__init__`, beside the other packet dicts (near `self.packet_to_mitre = {}`), add:

```python
        self.packet_to_provenance: dict[str, dict] = {}
```

Change the `add_packet` signature to accept the new optional argument (add after
the `mitre_topk` parameter):

```python
        mitre_topk: list[tuple[str, str, float]],
        mitre_provenance: dict[str, dict] | None = None,
```

Inside `add_packet`, right after `self.packet_to_mitre[packet_id] = topk`, add:

```python
            self.packet_to_provenance[packet_id] = dict(mitre_provenance or {})
```

In `_purge_packet` (wherever per-packet dicts are popped), add a safe pop:

```python
        self.packet_to_provenance.pop(packet_id, None)
```

In `get_packets`, where the record dict is assembled (after
`record["mitre_topk"] = ...`), add:

```python
                record["mitre_provenance"] = dict(self.packet_to_provenance.get(packet_id, {}))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k provenance -v`
Expected: 1 passed.

- [ ] **Step 5: Wire the pipeline**

In `runtime_pipeline.py` `on_packet`, replace the assigner call block:

```python
        embedding: Any = None
        payload_bytes = bytes.fromhex(extracted.hex_64) if extracted.hex_64 else b""
        mitre_topk: list[Any] = []
        mitre_provenance: dict[str, dict] = {}
        if self.edge_assigner is not None:
            mitre_topk, mitre_provenance = self.edge_assigner.assign_packet(
                payload_bytes, return_provenance=True
            )
```

Add `mitre_provenance=mitre_provenance,` to **both** `add_packet` calls (the
`graph_store.append_packet` call does NOT take it — leave that one unchanged; only
`self.hot_buffer.add_packet(...)` gets the new kwarg).

- [ ] **Step 6: Run regression + commit**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/ -q`
Expected: all pass.

```bash
git add -f src/graphslm_ids/runtime/fast_path/hot_graph_buffer.py src/graphslm_ids/runtime/pipeline/runtime_pipeline.py tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "feat(runtime): carry MITRE provenance through hot buffer + pipeline"
```

---

## Task 4: v3 context types

**Files:**
- Modify: `src/graphslm_ids/runtime/slow_path/types.py`
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from graphslm_ids.runtime.slow_path.types import MitreEdge, PacketContext, GraphContext


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k "mitre_edge or mitre_evidence" -v`
Expected: FAIL — `cannot import name 'MitreEdge'`.

- [ ] **Step 3: Edit `types.py`**

Add the `MitreEdge` dataclass (after `FlowContext`):

```python
@dataclass
class MitreEdge:
    """One v3 evidence edge: family-routed technique with provenance."""
    family: str
    weight: float
    source: str = ""
    tokens: list[str] = field(default_factory=list)
    literals: list[str] = field(default_factory=list)
```

Change `PacketContext.mitre_cosine_scores` to:

```python
    mitre_evidence: dict[str, "MitreEdge"] = field(default_factory=dict)
```

Change `GraphContext.flow_mitre_scores` to:

```python
    flow_mitre_evidence: dict[str, "MitreEdge"] = field(default_factory=dict)
```

Change `MitreMetadata` defaults (remove the cosine wording):

```python
@dataclass
class MitreMetadata:
    technique_id: str
    technique_name: str
    tactic: str
    tactic_id: str
    source: str = "msee_ensemble"
    mapping_caution: str = (
        "This link is from the v3 MSEE ensemble (PMI + procedure matcher), "
        "a statistical evidence vote, not forensic proof."
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k "mitre_edge or mitre_evidence" -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/types.py tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "feat(slow-path): v3 MitreEdge + mitre_evidence context types"
```

---

## Task 5: adapter parses v3 triples + provenance

**Files:**
- Modify: `src/graphslm_ids/runtime/slow_path/hot_buffer_adapter.py`
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

`_coerce_mitre_scores` (returns `dict[str,float]`) is replaced by
`_coerce_mitre_evidence` (returns `dict[str, MitreEdge]`) that understands v3
triples `(tech, family, weight)` and a sibling `mitre_provenance` record. All
call sites that read `.mitre_cosine_scores`/`flow_mitre_scores` switch to the new
fields.

- [ ] **Step 1: Write the failing test**

Append:

```python
from graphslm_ids.runtime.slow_path.hot_buffer_adapter import HotBufferAdapter


def test_adapter_builds_mitre_evidence_from_triples():
    adapter = HotBufferAdapter(buffer=None, mitre_catalog={})
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k adapter -v`
Expected: FAIL — `AttributeError: 'PacketContext' object has no attribute 'mitre_evidence'` is already fixed (Task 4), so it fails on `_build_packet_contexts` still setting `mitre_cosine_scores`.

- [ ] **Step 3: Edit `hot_buffer_adapter.py`**

Import the new type at the top:

```python
from graphslm_ids.runtime.slow_path.types import (
    FlowContext, GraphContext, MitreEdge, MitreMetadata, PacketContext,
)
```

Add a new coercion method next to `_coerce_mitre_scores`:

```python
    def _coerce_mitre_evidence(
        self, value: Any, provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, "MitreEdge"]:
        prov = dict(provenance or {})
        out: dict[str, MitreEdge] = {}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                    if len(item) == 3:
                        tech, family, weight = str(item[0]), str(item[1]), float(item[2])
                    elif len(item) == 2:
                        tech, family, weight = str(item[0]), "", float(item[1])
                    else:
                        continue
                    p = prov.get(tech, {})
                    out[tech] = MitreEdge(
                        family=family, weight=weight,
                        source=str(p.get("source", "")),
                        tokens=[str(t) for t in p.get("tokens", [])],
                        literals=[str(l) for l in p.get("literals", [])],
                    )
        return out
```

In `_build_packet_contexts`, replace the `mitre_scores = self._coerce_mitre_scores(...)`
block (lines ~179-183) and the `PacketContext(...)` kwarg with:

```python
            mitre_evidence = self._coerce_mitre_evidence(
                self._get_field(record, "mitre_topk", "mitre_cosine_scores", "mitre_scores"),
                self._get_field(record, "mitre_provenance"),
            )
            if not mitre_evidence:
                mitre_evidence = self._coerce_mitre_evidence(packet_to_mitre.get(packet_id))
```

and in the appended `PacketContext(...)`:

```python
                    mitre_evidence=mitre_evidence,
```

In `_resolve_flow_mitre_scores` rename to `_resolve_flow_mitre_evidence` returning
`dict[str, MitreEdge]`; replace its body's packet aggregation loop:

```python
        aggregated: dict[str, MitreEdge] = {}
        for packet in packet_contexts:
            for tech_id, edge in packet.mitre_evidence.items():
                if tech_id not in aggregated or edge.weight > aggregated[tech_id].weight:
                    aggregated[tech_id] = edge
        return aggregated
```

and its first two lookups use `_coerce_mitre_evidence(...)` instead of
`_coerce_mitre_scores(...)`.

In `_resolve_mitre_metadata`, change the two `.mitre_cosine_scores.keys()` reads to
`.mitre_evidence.keys()`, and `flow_mitre_scores.keys()` stays valid (it is now a
`dict[str, MitreEdge]`, `.keys()` still returns technique ids).

In `get_context`, rename the local `flow_mitre_scores` → `flow_mitre_evidence`, call
`_resolve_flow_mitre_evidence(...)`, and pass `flow_mitre_evidence=flow_mitre_evidence`
to the `GraphContext(...)` constructor.

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k adapter -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/hot_buffer_adapter.py tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "feat(slow-path): adapter parses v3 evidence triples + provenance"
```

---

## Task 6: EvidenceBundle MitreEvidence v3 fields

**Files:**
- Modify: `src/graphslm_ids/runtime/slow_path/evidence_bundle.py`
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from graphslm_ids.runtime.slow_path.evidence_bundle import MitreEvidence


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k mitre_evidence_v3 -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'family'`.

- [ ] **Step 3: Edit `evidence_bundle.py`**

Replace the `MitreEvidence` dataclass entirely:

```python
@dataclass
class MitreEvidence:
    evidence_id: str
    technique_id: str
    technique_name: str
    tactic: str
    tactic_id: str
    family: str
    evidence_weight: float
    source: str
    matched_tokens: list[str]
    matched_literals: list[str]
    supporting_packet_count: int
    matched_from: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "tactic": self.tactic,
            "tactic_id": self.tactic_id,
            "family": self.family,
            "evidence_weight": self.evidence_weight,
            "source": self.source,
            "matched_tokens": self.matched_tokens,
            "matched_literals": self.matched_literals,
            "supporting_packet_count": self.supporting_packet_count,
            "matched_from": self.matched_from,
        }
```

Also remove `mitre_max_cosine` from `PacketEvidence` (delete the field at line 85
and the `to_dict` never emitted it, so no `to_dict` change needed).

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k mitre_evidence_v3 -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/evidence_bundle.py tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "feat(slow-path): EvidenceBundle MitreEvidence v3 fields (family/weight/provenance)"
```

---

## Task 7: EvidenceBuilder builds v3 evidence

**Files:**
- Modify: `src/graphslm_ids/runtime/slow_path/evidence_builder.py`
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from graphslm_ids.runtime.slow_path.evidence_builder import EvidenceBuilder
from graphslm_ids.runtime.slow_path.types import FlowContext, MitreMetadata
from graphslm_ids.runtime.slow_path.types import SlowPathJob


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
    # paths use v3 edge names
    if bundle.graph_paths:
        assert bundle.graph_paths[0].path_edges == ["contain", "evidence_injection", "technique_tactic"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k builder_emits -v`
Expected: FAIL — builder still reads `mitre_cosine_scores` / builds cosine fields.

- [ ] **Step 3: Edit `evidence_builder.py`**

Replace `DEFAULT_LIMITATIONS` (lines 19-24):

```python
DEFAULT_LIMITATIONS = [
    "MITRE mapping is from the v3 MSEE ensemble (PMI token votes + procedure "
    "literal matches), a statistical evidence vote, not a deterministic signature.",
    "Payload preview is truncated. Full payload content is not available.",
    "HGT confidence is a probabilistic model output, not forensic proof.",
    "Counterfactual scores are approximations computed by zeroing packet embeddings.",
]
```

Replace `_build_packets`'s mitre block: change
`mitre_scores = packet.mitre_cosine_scores` / `mitre_max = max(...)` and the
`linked_techniques=sorted(packet.mitre_cosine_scores.keys())` /
`mitre_max_cosine=mitre_max` to use `packet.mitre_evidence`:

```python
            mitre_edges = source.mitre_evidence if source is not None else {}
```

where `source = packet_map.get(...)` — but `_build_packets` iterates
`context.packets` directly; use `packet.mitre_evidence`. Concretely, replace the
two lines that compute `mitre_scores`/`mitre_max` with nothing (delete them), set:

```python
                    linked_techniques=sorted(packet.mitre_evidence.keys()),
```

and remove the `mitre_max_cosine=mitre_max,` kwarg from the `PacketEvidence(...)`.

Replace `_build_mitre` entirely:

```python
    def _build_mitre(
        self,
        context: GraphContext,
        packet_evidence: list[PacketEvidence],
    ) -> list[MitreEvidence]:
        candidate: dict[str, dict[str, object]] = {}

        def _acc(tech_id: str, edge, origin: str) -> None:
            entry = candidate.setdefault(tech_id, {
                "family": edge.family, "weight": float(edge.weight),
                "source": set(), "tokens": set(), "literals": set(),
                "supporting": set(), "matched_from": set(),
            })
            entry["weight"] = max(float(entry["weight"]), float(edge.weight))
            if edge.family:
                entry["family"] = edge.family
            if edge.source:
                entry["source"].add(edge.source)
            entry["tokens"].update(edge.tokens)
            entry["literals"].update(edge.literals)
            entry["matched_from"].add(origin)

        for packet in context.packets:
            for tech_id, edge in packet.mitre_evidence.items():
                _acc(tech_id, edge, packet.packet_id)
                candidate[tech_id]["supporting"].add(packet.packet_id)
        for tech_id, edge in context.flow_mitre_evidence.items():
            _acc(tech_id, edge, context.flow.flow_id)

        evidence: list[MitreEvidence] = []
        for idx, (tech_id, data) in enumerate(candidate.items(), start=1):
            metadata = context.mitre_metadata.get(tech_id)
            if metadata is None:
                continue
            evidence.append(MitreEvidence(
                evidence_id=f"E_TECH_{idx:03d}",
                technique_id=metadata.technique_id,
                technique_name=metadata.technique_name,
                tactic=metadata.tactic,
                tactic_id=metadata.tactic_id,
                family=str(data["family"]),
                evidence_weight=float(data["weight"]),
                source="+".join(sorted(data["source"])) if data["source"] else "",
                matched_tokens=sorted(str(t) for t in data["tokens"]),
                matched_literals=sorted(str(l) for l in data["literals"]),
                supporting_packet_count=len(data["supporting"]),
                matched_from=sorted(str(m) for m in data["matched_from"]),
            ))
        return evidence
```

In `_link_packet_to_mitre`, `packet.linked_techniques` is already a list of
technique ids; it stays valid (no change).

Replace `_build_paths`'s per-technique loop body to use `mitre_evidence` and v3
edge names:

```python
        for packet in packet_evidence:
            source = packet_map.get(packet.packet_id)
            if source is None:
                continue
            for technique_id, edge in source.mitre_evidence.items():
                mitre_item = mitre_map.get(technique_id)
                if mitre_item is None:
                    continue
                attention = packet.importance_sources.get("hgt_attention_weight", 0.0)
                path_score = float(edge.weight) * float(attention)
                if not math.isfinite(path_score):
                    path_score = 0.0
                paths.append(GraphPathEvidence(
                    evidence_id=f"E_PATH_{path_idx:03d}",
                    path_nodes=[
                        {"id": context.flow.flow_id, "type": "flow"},
                        {"id": packet.packet_id, "type": "packet"},
                        {"id": mitre_item.technique_id, "type": "technique"},
                        {"id": mitre_item.tactic_id, "type": "tactic"},
                    ],
                    path_edges=["contain", f"evidence_{edge.family}", "technique_tactic"],
                    path_score=path_score,
                    attention_weight=float(attention),
                ))
                path_idx += 1
```

(`mitre_map` is built from `mitre_evidence` by `technique_id`; that line is
unchanged.) Also update `_build_mitre`'s usage in `build()` — signature unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k builder_emits -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/evidence_builder.py tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "feat(slow-path): EvidenceBuilder emits v3 family/weight/provenance + v3 path edges"
```

---

## Task 8: serializer renders v3 graph-text

**Files:**
- Modify: `src/graphslm_ids/runtime/slow_path/graph_serializer.py`
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from graphslm_ids.runtime.slow_path.graph_serializer import serialize_bundle


def test_serializer_v3_graphtext():
    job = SlowPathJob(alert_id="A1", flow_id="f1", predicted_label="SqlInjection",
                      confidence=0.9, alert_threshold=0.7)
    bundle = EvidenceBuilder().build(job, _ctx())
    text = serialize_bundle(bundle)
    assert "evidence_injection" in text
    assert "w=0.90" in text or "w=0.9" in text
    assert "src=pmi" in text
    assert "cosine=" not in text and "matches_technique" not in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k serializer_v3 -v`
Expected: FAIL — current serializer prints `cosine=`.

- [ ] **Step 3: Edit `graph_serializer.py`**

Replace the tech sort key and tech node lines (the block that sorts by
`t.cosine_score` and prints `cosine=…`):

```python
    techs = sorted(bundle.mitre_evidence, key=lambda t: (-t.evidence_weight, t.evidence_id))
```

Tech node rows:

```python
    for tech in techs:
        toks = ",".join(tech.matched_tokens[:5])
        lits = ",".join(tech.matched_literals[:5])
        prov = f' tokens="{toks}"' if toks else ""
        prov += f' literals="{lits}"' if lits else ""
        lines.append(
            f"tech [{tech.evidence_id}] id={tech.technique_id} family={tech.family} "
            f"w={tech.evidence_weight:.2f} src={tech.source}{prov} "
            f'name="{tech.technique_name}"'
        )
```

Edge-list provenance block: replace the `if edges[i] == "matches_technique":`
branch with a family-aware lookup:

```python
            if edges[i].startswith("evidence_"):
                match = next((t for t in techs if t.technique_id == dst), None)
                if match is not None:
                    prov = f" src={match.source} w={match.evidence_weight:.2f}"
```

Add a host block right before the EDGES section (hosts come from the flow’s
src/dst ip):

```python
    lines.append(f"host id={flow.src_ip} role=src")
    lines.append(f"host id={flow.dst_ip} role=dst")
    lines.append("")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k serializer_v3 -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/graph_serializer.py tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "feat(slow-path): serializer renders v3 evidence edges + provenance + host"
```

---

## Task 9: verifier reads v3 weight

**Files:**
- Modify: `src/graphslm_ids/runtime/slow_path/graph_verifier.py`
- Test: covered by the integration test (Task 10).

- [ ] **Step 1: Edit `graph_verifier.py`**

In `_bundle_numeric_values`, change the mitre loop (line ~102-103):

```python
    for tech in bundle.mitre_evidence:
        vals.append(round(float(tech.evidence_weight), 4))
```

- [ ] **Step 2: Run regression**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/ -q`
Expected: all pass (no references to `cosine_score` remain).

- [ ] **Step 3: Commit**

```bash
git add -f src/graphslm_ids/runtime/slow_path/graph_verifier.py
git commit -m "fix(slow-path): verifier numeric values read v3 evidence_weight"
```

---

## Task 10: end-to-end VG²R on a v3 bundle

**Files:**
- Test: `tests/runtime/slow_path/test_v3_graphtext.py`

Confirm the full VG²R path still grades a v3 bundle with a stub SLM (no Ollama).

- [ ] **Step 1: Write the test**

Append:

```python
from graphslm_ids.runtime.slow_path.graph_verifier import GraphVerifier, VerifierConfig


def test_verifier_passes_v3_grounded_claim():
    job = SlowPathJob(alert_id="A1", flow_id="f1", predicted_label="SqlInjection",
                      confidence=0.9, alert_threshold=0.7)
    bundle = EvidenceBuilder().build(job, _ctx())
    graph_text = serialize_bundle(bundle)
    # A claim that cites the real v3 weight and family, with a citation handle.
    report = "Packet evidence supports T1190 via injection family w=0.90 [E_TECH_001]."
    record = GraphVerifier(VerifierConfig()).verify(report, bundle, graph_text, repair_tier=1)
    assert record.numeric_accuracy == 1.0
```

- [ ] **Step 2: Run the test**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/runtime/slow_path/test_v3_graphtext.py -k verifier_passes -v`
Expected: PASS. If `VerifierConfig`/`GraphVerifier` names differ, open
`graph_verifier.py` and use the actual exported class names (do not invent).

- [ ] **Step 3: Full regression**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -f tests/runtime/slow_path/test_v3_graphtext.py
git commit -m "test(slow-path): end-to-end VG2R grades a v3-grounded report"
```

---

## Self-Review

**1. Spec coverage:** §4 data flow → Tasks 1-8. §4.1 provenance (PMI tokens +
procedure literals) → Tasks 1-2. §5 component table → Task per file (ensemble T1,
assigner T2, hot buffer+pipeline T3, types T4, adapter T5, bundle T6, builder T7,
serializer T8, verifier T9). §6 VG²R unchanged → Task 10 exercises it. §7 testing
→ each task is TDD; T10 integration + full regression. ✓

**2. Placeholder scan:** every code step shows real code; the only "use actual
names" note is Task 10 step 2 (verifier class names), which is a safety guard, not
a placeholder — the edit itself (T9) is concrete. ✓

**3. Type consistency:** `MitreEdge(family, weight, source, tokens, literals)`
defined T4, used T5/T7; `mitre_evidence`/`flow_mitre_evidence` names consistent
T4→T5→T7; `MitreEvidence(family, evidence_weight, source, matched_tokens,
matched_literals, supporting_packet_count, matched_from)` defined T6, built T7,
read T8/T9; `assign_packet(..., return_provenance)` defined T2, called T3;
`mitre_provenance` kwarg consistent T2→T3→T5. Path edges `["contain",
"evidence_<family>", "technique_tactic"]` match v3 schema. ✓

**Known gaps (intentional, per spec §8):** `burst_neighbor` not produced;
counterfactual evidence unchanged; provenance limited to what the online assigner
sees per packet.
