import numpy as np
import pandas as pd

from graphslm_ids.runtime.fast_path.subgraph_builder import SubgraphBuilder
from graphslm_ids.runtime.fast_path.edge_assigner import RuntimeEdgeAssigner


class StubProc:
    def weight_per_technique(self, payload: bytes):
        return {"T1190": 0.9} if b"OR 1=1" in payload else {}


def test_subgraph_build_runs_with_assigned_evidence():
    # Assign evidence for a SQLi-looking payload, feed it through a hot buffer,
    # and confirm the builder emits the v3 evidence edge for that family.
    from graphslm_ids.runtime.fast_path.hot_graph_buffer import HotGraphBuffer
    buf = HotGraphBuffer(ttl_seconds=999)
    pmi_df = pd.DataFrame([{"token": "t:select", "technique": "T1190",
                            "family": "injection", "weight": 0.9}])
    assigner = RuntimeEdgeAssigner.from_components(
        pmi_table=pmi_df, procedure_matcher=StubProc(),
        technique_family_map={"T1190": "injection"})
    payload = b"GET /?q=1 select OR 1=1"
    topk = assigner.assign_packet(payload)
    assert topk and topk[0][1] == "injection"

    buf.add_packet(packet_id="p1", flow_id="f1", embedding=np.zeros(1, dtype=np.float32),
                   payload_hex=payload.hex(), payload_ascii="", payload_len_raw=len(payload),
                   timestamp=0.0, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1,
                   dst_port=80, protocol="TCP", mitre_topk=topk)

    builder = SubgraphBuilder(buf, packet_feature="ordered_byte",
                              tactic_shortname_to_idx={"TA0001": 0}, always_include_all_tactics=False)
    # technique_to_tactic comes from the buffer's mitre_index (none here) -> inject directly
    buf.technique_to_tactic = {"T1190": "TA0001"}
    buf.technique_features = {"T1190": np.zeros(768, dtype=np.float32)}
    sub = builder.build("f1")
    assert ("packet", "evidence_injection", "technique") in sub.edge_index_dict
    assert "host" in sub.node_features
