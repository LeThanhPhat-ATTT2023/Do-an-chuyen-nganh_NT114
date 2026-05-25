"""Unit tests for the 3-source v3 evidence ensemble.

Locks in the spec §4.6 aggregation formula:

    n_voters = (w_pmi > 0.1) + (w_proc > 0.1)
    base = 0.55 * w_pmi + 0.45 * w_proc
    if w_flow > 0.1:    base *= 1.20
    if n_voters == 2:   base *= 1.30
    final = min(1.0, base);  emit iff final >= tau_edge

Plus the auxiliary lookups: PMI table -> per-token list, per-packet PMI sum
clipping, and the streaming iterator helper.
"""
from __future__ import annotations

import pandas as pd

from graphslm_ids.offline.preprocessing.ensemble import (
    aggregate_evidence,
    build_pmi_lookup_from_table,
    iter_packet_evidence,
    lookup_pmi_per_packet,
)


def test_build_pmi_lookup_from_empty_table() -> None:
    assert build_pmi_lookup_from_table(pd.DataFrame()) == {}


def test_build_pmi_lookup_groups_per_token() -> None:
    df = pd.DataFrame(
        [
            {"token": "t:union", "technique": "T1190", "family": "injection", "weight": 0.8},
            {"token": "t:union", "technique": "T1189", "family": "injection", "weight": 0.5},
            {"token": "t:cat", "technique": "T1059", "family": "command_exec", "weight": 0.7},
        ]
    )
    lookup = build_pmi_lookup_from_table(df)
    assert set(lookup) == {"t:union", "t:cat"}
    assert len(lookup["t:union"]) == 2
    techniques = {tech for tech, _, _ in lookup["t:union"]}
    assert techniques == {"T1190", "T1189"}


def test_lookup_pmi_per_packet_clips_at_one() -> None:
    # Two highly-overlapping tokens both point at the same technique with
    # weight 0.9 each -> sum 1.8 -> must clip to 1.0.
    df = pd.DataFrame(
        [
            {"token": "t:union", "technique": "T1190", "family": "injection", "weight": 0.9},
            {"token": "t:select", "technique": "T1190", "family": "injection", "weight": 0.9},
        ]
    )
    lookup = build_pmi_lookup_from_table(df)
    hits = lookup_pmi_per_packet(b"select union", lookup)
    assert "T1190" in hits
    family, weight = hits["T1190"]
    assert family == "injection"
    assert weight == 1.0


def test_lookup_pmi_per_packet_empty_payload_returns_empty() -> None:
    lookup = build_pmi_lookup_from_table(
        pd.DataFrame(
            [{"token": "t:x", "technique": "T1", "family": "injection", "weight": 1.0}]
        )
    )
    assert lookup_pmi_per_packet(b"", lookup) == {}


def test_aggregate_evidence_below_threshold_emits_nothing() -> None:
    # Single voter (PMI only) with low weight cannot clear tau=0.4.
    out = aggregate_evidence(
        packet_pmi_hits={"T1190": ("injection", 0.3)},
        packet_proc_hits={},
        flow_consensus={},
        technique_family={"T1190": "injection"},
        tau_edge=0.4,
    )
    # 0.55 * 0.3 = 0.165 < 0.4 -> dropped.
    assert out == []


def test_aggregate_evidence_single_pmi_voter_above_threshold() -> None:
    # 0.55 * 0.8 = 0.44 -> just over tau.
    out = aggregate_evidence(
        packet_pmi_hits={"T1190": ("injection", 0.8)},
        packet_proc_hits={},
        flow_consensus={},
        technique_family={"T1190": "injection"},
        tau_edge=0.4,
    )
    assert len(out) == 1
    tech, family, weight = out[0]
    assert tech == "T1190"
    assert family == "injection"
    assert abs(weight - 0.44) < 1e-6


def test_aggregate_evidence_two_voters_triggers_multi_source_boost() -> None:
    # base = 0.55*0.5 + 0.45*0.5 = 0.5; n_voters=2 -> base *= 1.30 -> 0.65.
    out = aggregate_evidence(
        packet_pmi_hits={"T1190": ("injection", 0.5)},
        packet_proc_hits={"T1190": 0.5},
        flow_consensus={},
        technique_family={"T1190": "injection"},
        tau_edge=0.4,
    )
    assert len(out) == 1
    assert abs(out[0][2] - 0.65) < 1e-6


def test_aggregate_evidence_flow_consensus_boost() -> None:
    # base = 0.55*0.7 + 0 = 0.385; flow boost 1.20 -> 0.462; emit.
    out = aggregate_evidence(
        packet_pmi_hits={"T1059": ("command_exec", 0.7)},
        packet_proc_hits={},
        flow_consensus={"T1059": 0.5},
        technique_family={"T1059": "command_exec"},
        tau_edge=0.4,
    )
    assert len(out) == 1
    assert abs(out[0][2] - 0.55 * 0.7 * 1.20) < 1e-6


def test_aggregate_evidence_caps_final_at_one() -> None:
    # Pathological inputs that would push past 1.0 (PMI=1, proc=1, flow boost,
    # multi-source boost). base = 1.0 * 1.20 * 1.30 = 1.56 -> capped at 1.0.
    out = aggregate_evidence(
        packet_pmi_hits={"T1": ("injection", 1.0)},
        packet_proc_hits={"T1": 1.0},
        flow_consensus={"T1": 1.0},
        technique_family={"T1": "injection"},
        tau_edge=0.4,
    )
    assert len(out) == 1
    assert out[0][2] == 1.0


def test_aggregate_evidence_drops_unknown_family() -> None:
    # Procedure-only hit on a technique with no family entry -> can't route.
    out = aggregate_evidence(
        packet_pmi_hits={},
        packet_proc_hits={"T9999": 0.9},
        flow_consensus={},
        technique_family={},  # no family for T9999
        tau_edge=0.4,
    )
    assert out == []


def test_aggregate_evidence_proc_only_uses_global_family_map() -> None:
    out = aggregate_evidence(
        packet_pmi_hits={},
        packet_proc_hits={"T1059": 0.9},
        flow_consensus={},
        technique_family={"T1059": "command_exec"},
        tau_edge=0.4,
    )
    # 0.45 * 0.9 = 0.405 -> just over tau, emit with command_exec family.
    assert len(out) == 1
    assert out[0][1] == "command_exec"


def test_aggregate_evidence_output_is_sorted_deterministically() -> None:
    out = aggregate_evidence(
        packet_pmi_hits={
            "T1190": ("injection", 0.9),
            "T1059": ("command_exec", 0.9),
        },
        packet_proc_hits={},
        flow_consensus={},
        technique_family={"T1190": "injection", "T1059": "command_exec"},
        tau_edge=0.4,
    )
    assert [t for t, _, _ in out] == sorted(t for t, _, _ in out)


class _StubMatcher:
    """Procedure-matcher stub: per-payload weight is keyed by payload bytes."""

    def __init__(self, table: dict[bytes, dict[str, float]]) -> None:
        self._table = table

    def weight_per_technique(self, payload: bytes) -> dict[str, float]:
        return dict(self._table.get(payload, {}))


def test_iter_packet_evidence_yields_per_packet_edge_lists() -> None:
    pmi_table = pd.DataFrame(
        [{"token": "t:union", "technique": "T1190", "family": "injection", "weight": 0.9}]
    )
    pmi_lookup = build_pmi_lookup_from_table(pmi_table)
    proc = _StubMatcher({b"select union": {"T1190": 0.9}})
    flow_map: dict[str, dict[str, float]] = {}

    rows = [
        {"payload": b"select union", "flow_id": "f0"},
        {"payload": b"", "flow_id": "f1"},  # empty payload -> empty edge list
        {"payload": b"unrelated", "flow_id": "f2"},
    ]
    out = list(
        iter_packet_evidence(
            rows,
            pmi_lookup,
            proc,
            flow_map,
            technique_family={"T1190": "injection"},
            tau_edge=0.4,
        )
    )
    assert len(out) == 3
    # First packet: both PMI and proc fire -> multi-source boost emits an edge.
    p0_idx, p0_edges = out[0]
    assert p0_idx == 0
    assert len(p0_edges) == 1 and p0_edges[0][0] == "T1190"
    # Second packet (empty): no edges.
    assert out[1] == (1, [])
    # Third packet (unrelated bytes): no PMI tokens hit -> no edges.
    assert out[2][1] == []
