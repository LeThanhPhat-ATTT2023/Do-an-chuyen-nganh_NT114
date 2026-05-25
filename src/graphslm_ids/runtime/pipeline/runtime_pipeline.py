from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
from typing import Any

from graphslm_ids.utils.io import read_json

from graphslm_ids.runtime.fast_path import (
    AlertDispatcher,
    FlowTracker,
    HGTRuntime,
    HotGraphBuffer,
    MitreIndex,
    PayloadExtractor,
    PolicyEngine,
    SigcEdgeFilter,
    SigcEdgeFilterConfig,
    SubgraphBuilder,
)
from graphslm_ids.runtime.pipeline.cold_store import ColdStore
from graphslm_ids.runtime.pipeline.counterfactual import HGTCounterfactual
from graphslm_ids.runtime.pipeline.graph_store import PersistentGraphStore
from graphslm_ids.runtime.pipeline.pipeline_config import PipelineConfig
from graphslm_ids.runtime.slow_path.hot_buffer_adapter import HotBufferAdapter
from graphslm_ids.runtime.slow_path.report_generator import ReportGenerator
from graphslm_ids.runtime.slow_path.report_validator import ReportValidator
from graphslm_ids.runtime.slow_path.slow_path_worker import SlowPathWorker


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
        self.mitre_index = MitreIndex(
            cfg.fast_path.mitre_embeddings,
            cfg.fast_path.techniques_csv,
            cfg.fast_path.technique_tactic_csv,
        )
        meta = read_json(Path(cfg.hgt.graph_meta_json)) if Path(cfg.hgt.graph_meta_json).exists() else {}
        self.graph_store = (
            PersistentGraphStore(
                cfg.graph_store.root,
                shard_duration_seconds=cfg.graph_store.shard_seal_by_time_seconds,
                max_nodes_per_shard=cfg.graph_store.shard_seal_by_size_nodes,
                payload_length=cfg.fast_path.payload_length,
                mitre_index=self.mitre_index,
                drop_after_days=cfg.graph_store.drop_after_days,
                disk_quota_gb=cfg.graph_store.disk_quota_gb,
                on_quota_exceeded=cfg.graph_store.on_quota_exceeded,
                schema={
                    "add_reverse_edges": cfg.hgt.add_reverse_edges,
                    "packet_feature_kind": cfg.hgt.packet_feature,
                    "use_semantic_edge_weights": cfg.hgt.use_semantic_edge_weights,
                    "protocol_mapping": meta.get("protocol_mapping", {}),
                },
            )
            if cfg.graph_store.enabled
            else None
        )
        self.hot_buffer = HotGraphBuffer(
            ttl_seconds=cfg.hot_graph.ttl_seconds,
            max_events=cfg.hot_graph.max_events,
            max_packets_per_flow=cfg.hot_graph.max_packets_per_flow,
            max_techniques_per_node=cfg.hot_graph.max_techniques_per_node,
            mitre_index=self.mitre_index,
        )
        self.sigc_filter = SigcEdgeFilter(
            SigcEdgeFilterConfig(
                enabled=cfg.preprocessor.sigc.enabled,
                alpha=cfg.preprocessor.sigc.alpha,
                min_score_to_keep_edge=cfg.preprocessor.sigc.min_score_to_keep_edge,
                apply_to_edge_types=cfg.preprocessor.sigc.apply_to_edge_types,
            )
        )
        self.hgt_runtime = HGTRuntime(
            cfg.hgt.checkpoint,
            cfg.hgt.graph_meta_json,
            device=cfg.hgt.device,
        )
        self.subgraph_builder = SubgraphBuilder(
            self.hot_buffer,
            cold_store=self.graph_store,
            hops=cfg.hgt.num_layers,
            add_reverse_edges=cfg.hgt.add_reverse_edges,
            standardize_flow_features=cfg.hgt.standardize_flow_features,
            flow_feature_stats=self.hgt_runtime.flow_feature_stats,
            packet_feature=cfg.hgt.packet_feature,
            protocol_mapping=meta.get("protocol_mapping", {}),
            tactic_shortname_to_idx=meta.get("tactic_shortname_to_idx", {}),
            always_include_all_tactics=True,
            dlg_top_n_enabled=cfg.subgraph_builder.dlg_ids_top_n.enabled,
            dlg_top_n_per_seed=cfg.subgraph_builder.dlg_ids_top_n.top_n_per_seed,
            dlg_sort_by=cfg.subgraph_builder.dlg_ids_top_n.sort_by,
        )
        self.policy = PolicyEngine(
            self.hgt_runtime.label_to_index,
            alert_threshold=cfg.policy.alert_threshold,
            alert_labels=cfg.policy.alert_labels,
            benign_labels=cfg.policy.benign_labels,
        )
        self.cold_store = self.graph_store or ColdStore(cfg.cold_store_path)
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
        # TODO(v3): replace with PMI-based technique assignment from v3 ensemble;
        # student CNN embedding removed — mitre_topk now empty until v3 runtime
        # edge-assignment is wired in.
        embedding: Any = None
        mitre_topk: list[Any] = []

        packet_id = self._make_packet_id(flow_state.flow_id)
        if self.graph_store is not None:
            self.graph_store.append_packet(
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
                "hot_buffer": None if self.graph_store is not None else self.hot_source,
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
            if self.graph_store is None and flow_id in self.hot_buffer:
                self.cold_store.append_alert_snapshot(
                    alert_id=f"snap_{flow_id}",
                    flow_id=flow_id,
                    snapshot=self.hot_buffer.snapshot(flow_id),
                )
        self.hot_buffer.evict_expired(now)
        if self.graph_store is not None:
            self.graph_store.enforce_retention(now)


