from __future__ import annotations

import queue

import numpy as np

from graphslm_ids.fast_path import (
    AlertDispatcher,
    FlowTracker,
    HotGraphBuffer,
    MitreIndex,
    PayloadExtractor,
    PolicyEngine,
    SubgraphBuilder,
)
from graphslm_ids.runtime import ColdStore, PipelineConfig
from graphslm_ids.slow_path import HotBufferAdapter


def _mitre_index() -> MitreIndex:
    embeddings = np.eye(2, 4, dtype=np.float32)
    return MitreIndex.from_arrays(
        embeddings,
        [
            {"technique_id": "T1001", "name": "Technique A"},
            {"technique_id": "T1002", "name": "Technique B"},
        ],
        [
            {"technique_id": "T1001", "tactic_shortname": "initial-access"},
            {"technique_id": "T1002", "tactic_shortname": "execution"},
        ],
    )


def _buffer_with_packet() -> HotGraphBuffer:
    buffer = HotGraphBuffer(mitre_index=_mitre_index(), ttl_seconds=10.0)
    buffer.add_packet(
        packet_id="pkt_1",
        flow_id="flow_1",
        embedding=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        payload_hex="47455420",
        payload_ascii="GET ",
        payload_len_raw=4,
        timestamp=1.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.5",
        src_port=51522,
        dst_port=80,
        protocol="TCP",
        mitre_topk=[("T1001", 0.91)],
    )
    return buffer


def test_flow_tracker_reuses_flow_until_idle_timeout() -> None:
    tracker = FlowTracker(idle_timeout_seconds=5.0, flow_id_prefix="flow")
    packet = {
        "src_ip": "1.1.1.1",
        "dst_ip": "2.2.2.2",
        "src_port": 1234,
        "dst_port": 80,
        "protocol": "TCP",
    }

    first = tracker.update(packet, now=10.0)
    second = tracker.update(packet, now=12.0)
    third = tracker.update(packet, now=20.0)

    assert first.flow_id == second.flow_id
    assert third.flow_id != first.flow_id
    assert third.packet_count == 1


def test_payload_extractor_accepts_packet_like_dict() -> None:
    extracted = PayloadExtractor(payload_length=8).extract(
        {
            "src_ip": "1.1.1.1",
            "dst_ip": "2.2.2.2",
            "src_port": 1111,
            "dst_port": 2222,
            "protocol": "udp",
            "timestamp": 3.0,
            "payload": b"hello world",
        }
    )

    assert extracted.payload_u8.shape == (8,)
    assert extracted.payload_u8[:5].tolist() == [104, 101, 108, 108, 111]
    assert extracted.raw_len == 11
    assert extracted.protocol == "UDP"


def test_hot_graph_buffer_matches_slow_path_adapter_contract() -> None:
    buffer = _buffer_with_packet()
    buffer.update_attention({"pkt_1": 0.7})

    context = HotBufferAdapter(buffer).get_context("flow_1")

    assert context is not None
    assert context.flow.flow_id == "flow_1"
    assert context.flow.packet_count == 1
    assert context.packets[0].payload_preview_ascii == "GET "
    assert context.packets[0].attention_weight == 0.7
    assert context.mitre_metadata["T1001"].tactic_id == "initial-access"


def test_subgraph_builder_produces_hgt_snapshot() -> None:
    buffer = _buffer_with_packet()
    subgraph = SubgraphBuilder(buffer, protocol_mapping={"TCP": 1}).build("flow_1")
    snapshot = subgraph.to_snapshot_dict()

    assert subgraph.node_features["flow"].shape == (1, 6)
    assert subgraph.node_features["packet"].shape == (1, 4)
    assert subgraph.node_features["technique"].shape == (1, 4)
    assert ("flow", "contains", "packet") in subgraph.edge_index_dict
    assert snapshot["node_ids"]["packet"] == ["pkt_1"]
    assert "packet__matches_technique__technique" in snapshot["edge_index"]


def test_policy_dispatcher_and_cold_store_round_trip(tmp_path) -> None:
    buffer = _buffer_with_packet()
    subgraph = SubgraphBuilder(buffer, protocol_mapping={"TCP": 1}).build("flow_1")
    policy = PolicyEngine({"benign": 0, "malicious": 1}, alert_threshold=0.5)
    decision = policy.decide(type("Output", (), {"logits": np.asarray([0.0, 2.0])})())
    cold_store = ColdStore(tmp_path / "events.jsonl")
    slow_queue: queue.Queue = queue.Queue()
    dispatcher = AlertDispatcher(slow_queue, cold_store, alert_id_prefix="test")

    alert_id = dispatcher.dispatch(
        decision,
        "flow_1",
        buffer,
        subgraph,
        {"pkt_1": 0.8},
        timestamp=10.0,
    )
    job = slow_queue.get_nowait()
    context = cold_store.load_context("flow_1")

    assert alert_id is not None
    assert job.alert_id == alert_id
    assert job.predicted_label == "malicious"
    assert job.hgt_attention["packet_attention"] == {"pkt_1": 0.8}
    assert context is not None
    assert context.packets[0].mitre_cosine_scores == {"T1001": 0.91}


def test_pipeline_config_reads_existing_example() -> None:
    cfg = PipelineConfig.from_yaml("configs/pipeline.example.yaml")

    assert cfg.fast_path.payload_length == 256
    assert cfg.hgt.checkpoint.endswith("hgt_flow_best.pt")
    assert cfg.slow_path.queue_max_size == 1000
    assert cfg.cold_store_path == "data/runtime/events.jsonl"
