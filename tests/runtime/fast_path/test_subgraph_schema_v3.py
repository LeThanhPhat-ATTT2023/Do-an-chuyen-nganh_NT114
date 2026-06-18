import numpy as np

from graphslm_ids.offline.preprocessing.payload_features import compute_packet_payload_features
from graphslm_ids.runtime.fast_path.subgraph_builder import SubgraphBuilder


class StubBuffer:
    """Minimal hot-buffer stand-in exposing snapshot() + static maps."""
    technique_features = {}
    technique_to_tactic = {}
    tactic_metadata = {}

    def __init__(self, snapshot):
        self._snap = snapshot

    def snapshot(self, flow_id):
        return self._snap


def _snapshot_one_packet(payload: bytes):
    hexs = payload.hex()
    return {
        "flow_id": "flow_1",
        "flow": {
            "flow_id": "flow_1", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
            "src_port": 1234, "dst_port": 80, "protocol": "TCP",
            "packet_count": 1, "total_payload_bytes": len(payload),
            "duration_seconds": 0.0,
        },
        "packets": [{
            "packet_id": "pkt_1", "payload_preview_hex": hexs, "payload_preview_ascii": "",
            "payload_len_raw": len(payload), "timestamp": 0.0,
            "mitre_topk": [], "attention_weight": None, "counterfactual_drop": None,
        }],
        "flow_to_mitre": [],
    }


def test_ordered_byte_packet_features_match_offline():
    payload = b"GET /?q=1 OR 1=1"
    builder = SubgraphBuilder(StubBuffer(_snapshot_one_packet(payload)), packet_feature="ordered_byte")
    sub = builder.build("flow_1")
    got = np.asarray(sub.node_features["packet"], dtype=np.float32)
    expected = compute_packet_payload_features(payload, len(payload)).reshape(1, -1)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-6)


def test_host_nodes_and_edges_built():
    builder = SubgraphBuilder(StubBuffer(_snapshot_one_packet(b"abc")), packet_feature="ordered_byte")
    sub = builder.build("flow_1")
    assert "host" in sub.node_features
    host_x = np.asarray(sub.node_features["host"])
    assert host_x.shape[0] == 2 and host_x.shape[1] == 4  # src + dst host, 4-d
    assert ("flow", "from_host", "host") in sub.edge_index_dict
    assert ("flow", "to_host", "host") in sub.edge_index_dict


def _snapshot_with_evidence():
    snap = _snapshot_one_packet(b"select")
    # mitre_topk now carries (tech, family, weight)
    snap["packets"][0]["mitre_topk"] = [("T1190", "injection", 0.9)]
    return snap


class StubBufferTech(StubBuffer):
    technique_to_tactic = {"T1190": "TA0001"}
    tactic_metadata = {"TA0001": {"tactic_id": "TA0001"}}
    technique_features = {"T1190": np.zeros(768, dtype=np.float32)}


def test_v3_edge_type_names_emitted():
    builder = SubgraphBuilder(
        StubBufferTech(_snapshot_with_evidence()),
        packet_feature="ordered_byte", always_include_all_tactics=False,
    )
    sub = builder.build("flow_1")
    keys = set(sub.edge_index_dict.keys())
    assert ("flow", "contain", "packet") in keys
    assert ("packet", "evidence_injection", "technique") in keys
    assert ("technique", "technique_tactic", "tactic") in keys
    # old names must NOT appear
    assert ("packet", "matches_technique", "technique") not in keys
    assert ("technique", "belongs_to_tactic", "tactic") not in keys
