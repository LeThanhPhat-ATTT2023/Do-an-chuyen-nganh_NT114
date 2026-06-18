import pandas as pd
import pytest

from graphslm_ids.runtime.fast_path.edge_assigner import RuntimeEdgeAssigner


class StubProc:
    """Stand-in for ProcedureMatcher.weight_per_technique.

    Weight 0.9 is deliberate: a proc-only hit needs base = 0.45*w >= tau (0.4),
    i.e. w >= ~0.889, to emit an edge. The aggregation counts voters per
    technique, not across techniques, so a 0.8 weight here would be silently
    dropped by design (lone proc hit below tau)."""
    def weight_per_technique(self, payload: bytes) -> dict[str, float]:
        return {"T1059": 0.9} if b"cmd.exe" in payload else {}


def _assigner():
    # PMI table: text token "t:select" -> T1190 (injection family), weight 0.9.
    # tokenize_payload prefixes text tokens with "t:", so the lookup key must
    # carry that prefix to match what the runtime ensemble actually produces.
    pmi_df = pd.DataFrame(
        [{"token": "t:select", "technique": "T1190", "family": "injection", "weight": 0.9}]
    )
    a = RuntimeEdgeAssigner.from_components(
        pmi_table=pmi_df,
        procedure_matcher=StubProc(),
        technique_family_map={"T1059": "command_exec"},
        tau_edge=0.4,
    )
    return a


def test_pmi_only_below_tau_emits_no_edge():
    a = _assigner()
    # only PMI (1 voter), base = 0.55*0.9 = 0.495 >= 0.4 -> emits injection edge
    edges = a.assign_packet(b"... select ...", flow_consensus={})
    assert ("T1190", "injection", pytest.approx(0.495, abs=1e-3)) in edges


def test_pmi_and_proc_each_emit_own_family_edge():
    a = _assigner()
    # payload triggers pmi(T1190, injection) AND proc(T1059, command_exec);
    # they are distinct techniques so each emits its own family-routed edge.
    edges = a.assign_packet(b"select ... cmd.exe", flow_consensus={})
    techs = {e[0]: e for e in edges}
    assert "T1190" in techs and techs["T1190"][1] == "injection"
    assert "T1059" in techs and techs["T1059"][1] == "command_exec"


def test_empty_payload_returns_empty():
    a = _assigner()
    assert a.assign_packet(b"", flow_consensus={}) == []
