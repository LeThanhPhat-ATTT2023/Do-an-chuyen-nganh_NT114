from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

from graphslm_ids.fast_path import (
    AlertDispatcher,
    FlowTracker,
    HGTRuntime,
    HotGraphBuffer,
    MitreIndex,
    PayloadExtractor,
    PolicyEngine,
    StudentRuntime,
    SubgraphBuilder,
)
from graphslm_ids.runtime.cold_store import ColdStore
from graphslm_ids.runtime.counterfactual import HGTCounterfactual
from graphslm_ids.runtime.pipeline_config import PipelineConfig
from graphslm_ids.slow_path.hot_buffer_adapter import HotBufferAdapter
from graphslm_ids.slow_path.report_generator import ReportGenerator
from graphslm_ids.slow_path.report_validator import ReportValidator
from graphslm_ids.slow_path.slow_path_worker import SlowPathWorker


@dataclass
class DetectionResult:
    flow_id: str
    label: str
    confidence: float
    alert_id: str | None


class FastPathPipeline:
    """End-to-end online pipeline from packet ingestion to slow-path alert dispatch."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.flow_tracker = FlowTracker(cfg.hot_graph.ttl_seconds)
        self.payload_extractor = PayloadExtractor(cfg.fast_path.payload_length)
        self.student_runtime = StudentRuntime(cfg.fast_path.student_onnx)
        self.mitre_index = MitreIndex(
            cfg.fast_path.mitre_embeddings,
            cfg.fast_path.techniques_csv,
            cfg.fast_path.technique_tactic_csv,
        )
        self.hot_buffer = HotGraphBuffer(
            ttl_seconds=cfg.hot_graph.ttl_seconds,
            max_events=cfg.hot_graph.max_events,
            max_packets_per_flow=cfg.hot_graph.max_packets_per_flow,
            max_techniques_per_node=cfg.hot_graph.max_techniques_per_node,
            mitre_index=self.mitre_index,
        )
        self.hgt_runtime = HGTRuntime(
            cfg.hgt.checkpoint,
            cfg.hgt.graph_meta_json,
            device=cfg.hgt.device,
        )
        meta = _load_json(cfg.hgt.graph_meta_json)
        self.subgraph_builder = SubgraphBuilder(
            self.hot_buffer,
            hops=cfg.hgt.num_layers,
            add_reverse_edges=cfg.hgt.add_reverse_edges,
            standardize_flow_features=cfg.hgt.standardize_flow_features,
            flow_feature_stats=self.hgt_runtime.flow_feature_stats,
            packet_feature=cfg.hgt.packet_feature,
            protocol_mapping=meta.get("protocol_mapping", {}),
        )
        self.policy = PolicyEngine(
            self.hgt_runtime.label_to_index,
            alert_threshold=cfg.policy.alert_threshold,
            alert_labels=cfg.policy.alert_labels,
        )
        self.cold_store = ColdStore(cfg.cold_store_path)
        self.slow_queue: queue.Queue[Any] = queue.Queue(
            maxsize=int(getattr(cfg.slow_path, "queue_max_size", 1000))
        )
        self.dispatcher = AlertDispatcher(self.slow_queue, self.cold_store)
        self.counterfactual = HGTCounterfactual(
            self.hgt_runtime,
            max_cf_packets=cfg.slow_path.max_cf_packets,
        )
        self.hot_source = HotBufferAdapter(self.hot_buffer, mitre_catalog=self.hot_buffer.mitre_metadata)
        self.slow_worker = SlowPathWorker(
            config=cfg.slow_path,
            counterfactual_model=self.counterfactual,
            label_to_index=self.hgt_runtime.label_to_index,
            cold_store=self.cold_store,
            report_generator=ReportGenerator(cfg.slm),
            validator=ReportValidator(cfg.validator),
        )
        self._packet_counter = 0

    def on_packet(self, raw_pkt: Any) -> DetectionResult:
        now = time.time()
        flow_state = self.flow_tracker.update(raw_pkt, now)
        extracted = self.payload_extractor.extract(raw_pkt)
        embedding = self.student_runtime.embed(extracted.payload_u8)
        mitre_topk = self.mitre_index.topk(
            embedding,
            k=self.cfg.fast_path.mitre_topk,
            threshold=self.cfg.fast_path.mitre_threshold,
        )

        packet_id = self._make_packet_id(flow_state.flow_id)
        self.hot_buffer.add_packet(
            packet_id=packet_id,
            flow_id=flow_state.flow_id,
            embedding=embedding,
            payload_hex=extracted.hex_64,
            payload_ascii=extracted.ascii_64,
            payload_len_raw=extracted.raw_len,
            timestamp=extracted.timestamp or now,
            src_ip=extracted.src_ip,
            dst_ip=extracted.dst_ip,
            src_port=extracted.src_port,
            dst_port=extracted.dst_port,
            protocol=extracted.protocol,
            mitre_topk=mitre_topk,
        )

        subgraph = self.subgraph_builder.build(flow_state.flow_id)
        hgt_output = self.hgt_runtime.infer(subgraph)
        packet_attention = self.hgt_runtime.aggregate_packet_attention(
            subgraph,
            hgt_output.edge_attention,
        )
        self.hot_buffer.update_attention(packet_attention)

        decision = self.policy.decide(hgt_output)
        alert_id = self.dispatcher.dispatch(
            decision,
            flow_state.flow_id,
            self.hot_buffer,
            subgraph,
            packet_attention,
            now,
        )

        self._housekeeping(now)
        return DetectionResult(
            flow_id=flow_state.flow_id,
            label=decision.label,
            confidence=decision.confidence,
            alert_id=alert_id,
        )

    def start_slow_worker(self, daemon: bool = True) -> threading.Thread:
        thread = threading.Thread(
            target=self.slow_worker.run_queue,
            kwargs={
                "slow_path_queue": self.slow_queue,
                "hot_buffer": self.hot_source,
                "cold_store": self.cold_store,
            },
            daemon=daemon,
        )
        thread.start()
        return thread

    def _make_packet_id(self, flow_id: str) -> str:
        self._packet_counter += 1
        return f"pkt_{flow_id}_{self._packet_counter:08d}"

    def _housekeeping(self, now: float) -> None:
        for flow_id in self.flow_tracker.evict_idle(now):
            if flow_id in self.hot_buffer:
                self.cold_store.append_alert_snapshot(
                    alert_id=f"snap_{flow_id}",
                    flow_id=flow_id,
                    snapshot=self.hot_buffer.snapshot(flow_id),
                )
        self.hot_buffer.evict_expired(now)


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
