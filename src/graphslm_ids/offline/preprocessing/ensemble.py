"""Per-packet evidence aggregation across the three v3 sources.

Inputs per packet:

* ``w_pmi`` (Source 1) — PMI + L1-LR table looked up by token presence.
* ``w_proc`` (Source 2) — MITRE procedure-pattern matcher (Aho-Corasick).
* ``w_flow`` (Source 3) — flow-level signature consensus (v2 wrap).

Aggregation rules (exactly the spec §4.6 formulas — do NOT tune these here;
they're calibrated against ``tau_edge = 0.4``):

    n_voters = (w_pmi > 0.1) + (w_proc > 0.1)
    if n_voters == 0:
        skip
    base = 0.55 * w_pmi + 0.45 * w_proc
    if w_flow > 0.1:    base *= 1.20    # flow corroboration
    if n_voters == 2:   base *= 1.30    # multi-source agreement
    final = min(1.0, base)
    if final >= tau_edge:
        emit (technique, family, final)

Why these particular constants?

* 0.55 / 0.45 — PMI evidence is denser (most packets get some token weight)
  but noisier; procedure matcher is sparser but high-precision. The slight PMI
  lean reflects that PMI alone is enough to fire many true-positive edges
  while procedure alone is rare.
* 1.20 / 1.30 — multiplicative boosts (NOT additive) so the cap at 1.0 acts
  as a soft saturating ceiling. A 1.30 multi-source boost on a 0.4 base
  becomes 0.52; that's just over τ and emits an edge.
* τ_edge = 0.4 — tuned so a single high-confidence procedure hit
  (weight=0.4) alone does NOT emit an edge; it requires PMI corroboration.

Memory: streaming. The aggregator never materialises more than one packet's
worth of evidence at a time.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd

from graphslm_ids.offline.preprocessing.tokenizer import tokenize_payload

# Type alias for the PMI lookup: token -> list of (technique, family, weight).
# A token may evidence multiple techniques (different rows in pmi_table.parquet
# for the same token) so the value side is a list, not a single tuple.
PmiLookup = dict[str, list[tuple[str, str, float]]]


def build_pmi_lookup_from_table(pmi_table: pd.DataFrame) -> PmiLookup:
    """Convert a ``pmi_table.parquet`` frame to a fast per-token lookup.

    Args:
        pmi_table: must have columns ``token``, ``technique``, ``family``,
            ``weight`` (output of :func:`pmi_learner.project_to_techniques`).

    Returns:
        ``{token: [(technique, family, weight), ...]}``. Empty input -> empty dict.
    """
    out: PmiLookup = {}
    if pmi_table is None or len(pmi_table) == 0:
        return out
    required = {"token", "technique", "family", "weight"}
    missing = required - set(pmi_table.columns)
    if missing:
        raise ValueError(f"pmi_table missing columns: {sorted(missing)}")
    # itertuples is fastest for this conversion; ~5x faster than iterrows.
    for r in pmi_table.itertuples(index=False):
        tok = str(r.token)
        tech = str(r.technique)
        family = str(r.family)
        try:
            w = float(r.weight)
        except (TypeError, ValueError):
            continue
        out.setdefault(tok, []).append((tech, family, w))
    return out


def lookup_pmi_per_packet(
    payload: bytes,
    pmi_lookup: PmiLookup,
) -> dict[str, tuple[str, float]]:
    """For one packet, aggregate PMI weights per technique.

    Tokenize the payload, look up each token in ``pmi_lookup``, and accumulate
    weights per technique. The final per-technique weight is clipped to ``1.0``
    (the negative side of the L1-LR coefficients can pull the sum below zero;
    we keep that signed value because the ensemble vote threshold operates on
    the positive evidence — a strongly-negative entry simply won't be > 0.1).

    Args:
        payload: raw payload bytes (may be empty).
        pmi_lookup: output of :func:`build_pmi_lookup_from_table`.

    Returns:
        ``{technique_id: (family, summed_weight)}``. Empty payload, no
        tokens, or no PMI hits all return ``{}``.
    """
    if not payload:
        return {}
    tokens = tokenize_payload(payload)
    if not tokens:
        return {}

    # tech -> [family_str, summed_weight]. Family is constant per technique
    # but we store it alongside the weight so callers don't need a second
    # lookup against ``technique_family``.
    accum: dict[str, list] = {}
    for tok in tokens:
        rows = pmi_lookup.get(tok)
        if not rows:
            continue
        for tech, family, w in rows:
            existing = accum.get(tech)
            if existing is None:
                accum[tech] = [family, w]
            else:
                existing[1] += w
    out: dict[str, tuple[str, float]] = {}
    for tech, (family, w) in accum.items():
        # Soft saturating clip at 1.0 (negative side preserved so multi-token
        # cancellations don't get misread as positive evidence).
        if w > 1.0:
            w = 1.0
        out[tech] = (family, float(w))
    return out


def aggregate_evidence(
    packet_pmi_hits: dict[str, tuple[str, float]],
    packet_proc_hits: dict[str, float],
    flow_consensus: dict[str, float],
    technique_family: dict[str, str],
    tau_edge: float = 0.4,
) -> list[tuple[str, str, float]]:
    """Combine the three evidence sources for one packet (spec §4.6 formula).

    Args:
        packet_pmi_hits: output of :func:`lookup_pmi_per_packet`.
        packet_proc_hits: output of
            :func:`procedure_matcher.ProcedureMatcher.weight_per_technique`.
        flow_consensus: per-flow consensus map for THIS packet's parent flow
            (output of :func:`flow_consensus.flow_consensus_hits` looked up
            by ``flow_id``). May be ``{}`` if the flow had no signature fire.
        technique_family: fallback ``technique -> family`` map for techniques
            that come from the procedure matcher (which doesn't carry a family
            field). PMI hits already include family.
        tau_edge: minimum final weight to emit an edge. Default ``0.4``.

    Returns:
        Sorted list of ``(technique, family, final_weight)`` tuples. Sorted
        by ``(technique, family)`` so the output is deterministic across
        runs even though the inputs are dicts.
    """
    if not packet_pmi_hits and not packet_proc_hits:
        return []

    candidate_techs = set(packet_pmi_hits) | set(packet_proc_hits)
    edges: list[tuple[str, str, float]] = []

    for tech in candidate_techs:
        pmi_entry = packet_pmi_hits.get(tech)
        if pmi_entry is not None:
            family_pmi, w_pmi = pmi_entry
        else:
            family_pmi, w_pmi = None, 0.0

        w_proc = float(packet_proc_hits.get(tech, 0.0))
        w_flow = float(flow_consensus.get(tech, 0.0))

        n_voters = int(w_pmi > 0.1) + int(w_proc > 0.1)
        if n_voters == 0:
            continue

        base = 0.55 * w_pmi + 0.45 * w_proc
        if w_flow > 0.1:
            base *= 1.20
        if n_voters == 2:
            base *= 1.30
        final = min(1.0, base)
        if final < tau_edge:
            continue

        # Resolve family: prefer the family the PMI table carried (it was
        # derived from class_technique_map.csv which is the authoritative
        # routing for the v3 ensemble). Fall back to the global
        # ``technique_family`` map for proc-only hits.
        family = family_pmi if family_pmi is not None else technique_family.get(tech)
        if family is None:
            # Technique has no family entry anywhere -> can't route to a
            # typed edge type. Drop silently.
            continue
        edges.append((tech, family, float(final)))

    edges.sort(key=lambda e: (e[0], e[1]))
    return edges


def iter_packet_evidence(
    packet_rows: Iterable[dict],
    pmi_lookup: PmiLookup,
    procedure_matcher,
    flow_consensus_map: dict[str, dict[str, float]],
    technique_family: dict[str, str],
    tau_edge: float = 0.4,
):
    """Convenience generator: yield per-packet edge lists for the graph builder.

    Each input row must have ``payload`` (bytes-like) and ``flow_id`` (str).
    Yields ``(packet_idx, [(tech, family, weight), ...])`` for every packet,
    INCLUDING ones with no edges (caller can decide to skip empties).

    Streaming: holds at most one packet's evidence in memory at a time.
    """
    for packet_idx, row in enumerate(packet_rows):
        payload = row.get("payload", b"")
        if isinstance(payload, (bytearray, memoryview)):
            payload = bytes(payload)
        elif not isinstance(payload, bytes):
            try:
                payload = bytes(payload)
            except (TypeError, ValueError):
                payload = b""
        flow_id = str(row.get("flow_id", ""))
        if not payload:
            yield packet_idx, []
            continue

        pmi_hits = lookup_pmi_per_packet(payload, pmi_lookup)
        proc_hits = procedure_matcher.weight_per_technique(payload)
        flow_hits = flow_consensus_map.get(flow_id, {})
        edges = aggregate_evidence(
            pmi_hits, proc_hits, flow_hits, technique_family, tau_edge=tau_edge
        )
        yield packet_idx, edges


__all__ = [
    "PmiLookup",
    "build_pmi_lookup_from_table",
    "lookup_pmi_per_packet",
    "aggregate_evidence",
    "iter_packet_evidence",
]


# Hint to keep ``defaultdict`` import live for readers — used in tests that
# stub the consensus map.
_ = defaultdict
