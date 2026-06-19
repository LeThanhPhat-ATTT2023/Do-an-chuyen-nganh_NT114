"""Online MITRE technique-edge assignment that reuses the offline v3 MSEE
ensemble (PMI lookup + procedure matcher + aggregate_evidence). No new model —
the exact same deterministic functions the graph builder uses offline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from graphslm_ids.offline.preprocessing.ensemble import (
    aggregate_evidence,
    build_pmi_lookup_from_table,
    lookup_pmi_per_packet,
    lookup_pmi_per_packet_with_tokens,
)
from graphslm_ids.offline.preprocessing.procedure_matcher import ProcedureMatcher


class RuntimeEdgeAssigner:
    """Assign ``(technique_id, family, weight)`` evidence edges for one packet."""

    def __init__(
        self,
        pmi_table_path: str | Path,
        stix_json_path: str | Path,
        technique_family_map: dict[str, str],
        tau_edge: float = 0.4,
    ) -> None:
        pmi_table = pd.read_parquet(pmi_table_path)
        self._pmi_lookup = build_pmi_lookup_from_table(pmi_table)
        self._proc = ProcedureMatcher(Path(stix_json_path))
        self._family = dict(technique_family_map)
        self._tau = float(tau_edge)

    @classmethod
    def from_components(
        cls,
        pmi_table: pd.DataFrame,
        procedure_matcher: Any,
        technique_family_map: dict[str, str],
        tau_edge: float = 0.4,
    ) -> "RuntimeEdgeAssigner":
        """Build directly from in-memory components (for tests / injection)."""
        obj = cls.__new__(cls)
        obj._pmi_lookup = build_pmi_lookup_from_table(pmi_table)
        obj._proc = procedure_matcher
        obj._family = dict(technique_family_map)
        obj._tau = float(tau_edge)
        return obj

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
