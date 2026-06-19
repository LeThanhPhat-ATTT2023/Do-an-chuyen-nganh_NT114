from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence
from graphslm_ids.runtime.slow_path.evidence_bundle import (
    AlertEvidence,
    BundleStats,
    CounterfactualEvidence,
    EvidenceBundle,
    FlowEvidence,
    GraphPathEvidence,
    MitreEvidence,
    PacketEvidence,
)
from graphslm_ids.runtime.slow_path.types import GraphContext, SlowPathJob


DEFAULT_LIMITATIONS = [
    "MITRE mapping is from the v3 MSEE ensemble (PMI token votes + procedure "
    "literal matches), a statistical evidence vote, not a deterministic signature.",
    "Payload preview is truncated. Full payload content is not available.",
    "HGT confidence is a probabilistic model output, not forensic proof.",
    "Counterfactual scores are approximations computed by zeroing packet embeddings.",
]


@dataclass
class EvidenceBuilderConfig:
    bundle_version: str = "1.0"
    alert_threshold: float = 0.70
    max_payload_preview_bytes: int = 64


class EvidenceBuilder:
    def __init__(
        self,
        config: EvidenceBuilderConfig | None = None,
        counterfactual_model: Any | None = None,
        label_to_index: Mapping[str, int] | None = None,
    ) -> None:
        self.config = config or EvidenceBuilderConfig()
        self.counterfactual_model = counterfactual_model
        self.label_to_index = dict(label_to_index or {})

    def build(
        self,
        job: SlowPathJob,
        context: GraphContext,
        max_cf_packets: int = 10,
    ) -> EvidenceBundle:
        alert = self._build_alert(job)
        flow = self._build_flow(context)
        packet_evidence = self._build_packets(context)
        self._apply_job_attention(job, packet_evidence)
        mitre_evidence = self._build_mitre(context, packet_evidence)
        self._link_packet_to_mitre(packet_evidence, mitre_evidence)
        graph_paths = self._build_paths(context, packet_evidence, mitre_evidence)
        counterfactuals = self._build_counterfactuals(job, context, packet_evidence, max_cf_packets)
        limitations = context.limitations or DEFAULT_LIMITATIONS

        bundle = EvidenceBundle(
            bundle_version=self.config.bundle_version,
            alert=alert,
            flow_evidence=flow,
            packet_evidence=packet_evidence,
            mitre_evidence=mitre_evidence,
            graph_paths=graph_paths,
            counterfactual_evidence=counterfactuals,
            limitations=limitations,
            bundle_stats=BundleStats(
                num_packets_in_flow=flow.packet_count,
                num_packets_in_evidence=len(packet_evidence),
                num_techniques_in_evidence=len(mitre_evidence),
                num_paths_in_evidence=len(graph_paths),
                total_tokens_estimate=0,
            ),
        )

        bundle.bundle_stats.total_tokens_estimate = self._estimate_tokens(bundle)
        return bundle

    def _build_alert(self, job: SlowPathJob) -> AlertEvidence:
        alert_threshold = (
            job.alert_threshold if job.alert_threshold is not None else self.config.alert_threshold
        )
        top_classes = job.top_classes or [{"label": job.predicted_label, "prob": job.confidence}]
        trigger_reason = (
            f"HGT {job.predicted_label} probability {job.confidence:.2f} "
            f"exceeds threshold {alert_threshold:.2f}"
        )
        return AlertEvidence(
            evidence_id="E_ALERT",
            alert_id=job.alert_id,
            flow_id=job.flow_id,
            predicted_label=job.predicted_label,
            confidence=job.confidence,
            top_classes=top_classes,
            alert_threshold=alert_threshold,
            trigger_reason=trigger_reason,
        )

    def _build_flow(self, context: GraphContext) -> FlowEvidence:
        flow = context.flow
        return FlowEvidence(
            evidence_id="E_FLOW_001",
            flow_id=flow.flow_id,
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            src_port=flow.src_port,
            dst_port=flow.dst_port,
            protocol=flow.protocol,
            duration_seconds=flow.duration_seconds,
            packet_count=flow.packet_count,
            total_payload_bytes=flow.total_payload_bytes,
            flow_feature_stats=flow.flow_feature_stats,
        )

    def _build_packets(self, context: GraphContext) -> list[PacketEvidence]:
        packet_evidence: list[PacketEvidence] = []
        for idx, packet in enumerate(context.packets, start=1):
            payload_hex, payload_ascii = self._truncate_payload_preview(
                packet.payload_preview_hex,
                packet.payload_preview_ascii,
            )
            cf_drop = packet.counterfactual_drop or 0.0
            attn = packet.attention_weight or 0.0

            if cf_drop > 0:
                reason = f"Masking this packet reduced malicious confidence by {cf_drop:.2f}"
            elif attn > 0:
                reason = "Packet received high attention weight in the HGT model"
            else:
                reason = "Packet selected as part of the alert evidence set"

            packet_evidence.append(
                PacketEvidence(
                    evidence_id=f"E_PKT_{idx:03d}",
                    packet_id=packet.packet_id,
                    order_in_flow=packet.order_in_flow,
                    timestamp=packet.timestamp,
                    payload_len_raw=packet.payload_len_raw,
                    payload_preview_hex=payload_hex,
                    payload_preview_ascii=payload_ascii,
                    linked_techniques=sorted(packet.mitre_evidence.keys()),
                    importance_score=0.0,
                    importance_sources={
                        "counterfactual_drop": cf_drop,
                        "hgt_attention_weight": attn,
                        "combined_score": 0.0,
                    },
                    importance_reason=reason,
                )
            )
        return packet_evidence

    def _apply_job_attention(
        self,
        job: SlowPathJob,
        packet_evidence: list[PacketEvidence],
    ) -> None:
        attention_map = self._coerce_packet_attention_map(job.hgt_attention)
        if not attention_map:
            attention_map = self._aggregate_edge_attention_to_packets(
                job.hgt_attention,
                job.subgraph_snapshot,
            )
        if not attention_map:
            attention_map = self._compute_gradient_packet_saliency(job)
        if not attention_map:
            return

        for packet in packet_evidence:
            if packet.packet_id not in attention_map:
                continue
            attn = float(attention_map[packet.packet_id])
            packet.importance_sources["hgt_attention_weight"] = attn
            if attn > 0.0 and packet.importance_sources.get("counterfactual_drop", 0.0) <= 0.0:
                packet.importance_reason = "Packet received high attention weight in the HGT model"

    def _coerce_packet_attention_map(self, attention: Any) -> dict[str, float]:
        if not isinstance(attention, Mapping):
            return {}

        raw_packet_map = attention.get("packet_attention") or attention.get("packet_scores")
        if isinstance(raw_packet_map, Mapping):
            return {
                str(key): val
                for key, raw_val in raw_packet_map.items()
                if (val := self._to_float(raw_val)) is not None
            }

        packet_map: dict[str, float] = {}
        for key, raw_val in attention.items():
            if key in {"packet_attention", "packet_scores"}:
                continue
            if self._normalize_edge_key(key) is not None:
                continue
            value = self._to_float(raw_val)
            if value is not None:
                packet_map[str(key)] = value
        return packet_map

    def _aggregate_edge_attention_to_packets(
        self,
        attention: Any,
        snapshot: Any,
    ) -> dict[str, float]:
        if not isinstance(attention, Mapping) or snapshot is None:
            return {}

        edge_index = self._snapshot_get(snapshot, "edge_index")
        if not isinstance(edge_index, Mapping):
            return {}

        node_ids = (
            self._snapshot_get(snapshot, "node_ids")
            or self._snapshot_get(snapshot, "node_id_map")
            or self._snapshot_get(snapshot, "node_index")
        )
        packet_id_to_idx = self._resolve_node_id_map(node_ids, "packet")
        if not packet_id_to_idx:
            packet_id_to_idx = self._resolve_flat_id_map(snapshot, "packet_id_map")
        idx_to_packet_id = {idx: packet_id for packet_id, idx in packet_id_to_idx.items()}
        if not idx_to_packet_id:
            return {}

        packet_scores: dict[str, float] = {}
        for raw_key, raw_attention in attention.items():
            edge_key = self._normalize_edge_key(raw_key)
            if edge_key is None:
                continue
            src_type, _, dst_type = edge_key
            if src_type != "packet" and dst_type != "packet":
                continue

            raw_edge_index = self._edge_index_for_key(edge_index, edge_key)
            src_indices, dst_indices = self._edge_index_to_rows(raw_edge_index)
            attention_values = self._attention_to_values(raw_attention)
            if not src_indices or not attention_values:
                continue

            packet_indices = src_indices if src_type == "packet" else dst_indices
            for packet_idx, score in zip(packet_indices, attention_values):
                packet_id = idx_to_packet_id.get(packet_idx)
                if packet_id is None:
                    continue
                packet_scores[packet_id] = max(packet_scores.get(packet_id, 0.0), score)
        return packet_scores

    def _compute_gradient_packet_saliency(self, job: SlowPathJob) -> dict[str, float]:
        if not job.subgraph_snapshot or self.counterfactual_model is None:
            return {}

        target_idx = self._resolve_target_index(job)
        if target_idx is None:
            return {}

        try:
            import torch
        except ImportError:
            return {}

        snapshot = job.subgraph_snapshot
        node_features = self._snapshot_get(snapshot, "node_features")
        edge_index = self._snapshot_get(snapshot, "edge_index")
        if not isinstance(node_features, Mapping) or not isinstance(edge_index, Mapping):
            return {}

        node_ids = (
            self._snapshot_get(snapshot, "node_ids")
            or self._snapshot_get(snapshot, "node_id_map")
            or self._snapshot_get(snapshot, "node_index")
        )
        packet_id_to_idx = self._resolve_node_id_map(node_ids, "packet")
        if not packet_id_to_idx:
            packet_id_to_idx = self._resolve_flat_id_map(snapshot, "packet_id_map")
        if not packet_id_to_idx:
            return {}

        flow_id_to_idx = self._resolve_node_id_map(node_ids, "flow")
        flow_idx = int(flow_id_to_idx.get(job.flow_id, 0)) if flow_id_to_idx else 0

        model = self.counterfactual_model
        model.eval()
        device = self._model_device(model, torch)

        node_tensors: dict[str, torch.Tensor] = {}
        for node_type, features in node_features.items():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            node_tensors[str(node_type)] = tensor

        required_nodes = {"flow", "packet", "technique", "tactic"}
        if not required_nodes.issubset(node_tensors.keys()):
            return {}

        edge_index_dict = self._prepare_edge_index_dict(model, dict(edge_index), device)
        if not edge_index_dict:
            return {}

        edge_weight = self._snapshot_get(snapshot, "edge_weight")
        if edge_weight is None:
            edge_weight = self._snapshot_get(snapshot, "edge_attr")
        edge_weight_dict = self._prepare_edge_weight_dict(model, edge_weight, device)

        packet_x = node_tensors["packet"].clone().detach().requires_grad_(True)
        node_tensors = dict(node_tensors)
        node_tensors["packet"] = packet_x

        try:
            try:
                model.zero_grad(set_to_none=True)
            except TypeError:
                model.zero_grad()
            logits = model(node_tensors, edge_index_dict, edge_weight_dict=edge_weight_dict)
            if isinstance(logits, tuple):
                logits = logits[0]
            if flow_idx >= logits.shape[0]:
                flow_idx = 0
            if target_idx >= logits.shape[1]:
                return {}
            score = logits[flow_idx, target_idx]
            score.backward()
        except Exception:
            return {}

        if packet_x.grad is None:
            return {}

        saliency = packet_x.grad.detach().abs().mean(dim=-1).cpu().tolist()
        return {
            packet_id: float(saliency[idx])
            for packet_id, idx in packet_id_to_idx.items()
            if idx < len(saliency)
        }

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
        for idx, (technique_id, data) in enumerate(candidate.items(), start=1):
            metadata = context.mitre_metadata.get(technique_id)
            if metadata is None:
                continue
            evidence.append(
                MitreEvidence(
                    evidence_id=f"E_TECH_{idx:03d}",
                    technique_id=metadata.technique_id,
                    technique_name=metadata.technique_name,
                    tactic=metadata.tactic,
                    tactic_id=metadata.tactic_id,
                    family=str(data["family"]),
                    evidence_weight=float(data["weight"]),
                    source="+".join(sorted(data["source"])) if data["source"] else "",
                    matched_tokens=sorted(str(t) for t in data["tokens"]),
                    matched_literals=sorted(str(lit) for lit in data["literals"]),
                    supporting_packet_count=len(data["supporting"]),
                    matched_from=sorted(str(m) for m in data["matched_from"]),
                )
            )
        return evidence

    def _link_packet_to_mitre(
        self,
        packet_evidence: list[PacketEvidence],
        mitre_evidence: list[MitreEvidence],
    ) -> None:
        evidence_by_tech = {item.technique_id: item.evidence_id for item in mitre_evidence}
        for packet in packet_evidence:
            linked = [
                evidence_by_tech[tech_id]
                for tech_id in packet.linked_techniques
                if tech_id in evidence_by_tech
            ]
            packet.linked_techniques = linked

    def _build_paths(
        self,
        context: GraphContext,
        packet_evidence: list[PacketEvidence],
        mitre_evidence: list[MitreEvidence],
    ) -> list[GraphPathEvidence]:
        packet_map = {packet.packet_id: packet for packet in context.packets}
        mitre_map = {item.technique_id: item for item in mitre_evidence}
        paths: list[GraphPathEvidence] = []
        path_idx = 1

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
                if math.isfinite(path_score) is False:
                    path_score = 0.0

                paths.append(
                    GraphPathEvidence(
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
                    )
                )
                path_idx += 1
        return paths

    def _build_counterfactuals(
        self,
        job: SlowPathJob,
        context: GraphContext,
        packet_evidence: list[PacketEvidence],
        max_cf_packets: int,
    ) -> list[CounterfactualEvidence]:
        counterfactuals: list[CounterfactualEvidence] = []
        if not packet_evidence or max_cf_packets <= 0:
            return counterfactuals

        candidates = sorted(
            packet_evidence,
            key=lambda pkt: pkt.importance_sources.get("hgt_attention_weight", 0.0),
            reverse=True,
        )
        candidates = candidates[:max_cf_packets]
        if not candidates:
            return counterfactuals

        drop_map, base_confidence = self._compute_counterfactual_drop(job, candidates)
        for packet in candidates:
            if packet.packet_id in drop_map:
                drop = float(drop_map[packet.packet_id])
                packet.importance_sources["counterfactual_drop"] = drop
                if drop > 0:
                    packet.importance_reason = (
                        f"Masking this packet reduced malicious confidence by {drop:.2f}"
                    )
                elif drop < 0:
                    packet.importance_reason = (
                        f"Masking this packet increased malicious confidence by {abs(drop):.2f}"
                    )

        for idx, packet in enumerate(candidates, start=1):
            if packet.packet_id in drop_map:
                drop = float(drop_map[packet.packet_id])
            else:
                drop = packet.importance_sources.get("counterfactual_drop")
                if drop is None or float(drop) == 0.0:
                    continue
                drop = float(drop)

            before = float(base_confidence)
            after = min(max(before - drop, 0.0), 1.0)
            if drop >= 0:
                interpretation = (
                    f"Removing {packet.packet_id} reduced malicious confidence by {drop:.2f}."
                )
            else:
                interpretation = (
                    f"Removing {packet.packet_id} increased malicious confidence by {abs(drop):.2f}."
                )
            counterfactuals.append(
                CounterfactualEvidence(
                    evidence_id=f"E_CF_{idx:03d}",
                    masked_element_id=packet.packet_id,
                    masked_element_type="packet",
                    confidence_before=before,
                    confidence_after=after,
                    confidence_drop=drop,
                    interpretation=interpretation,
                )
            )
        return counterfactuals

    def _compute_counterfactual_drop(
        self,
        job: SlowPathJob,
        candidates: list[PacketEvidence],
    ) -> tuple[dict[str, float], float]:
        if not job.subgraph_snapshot or self.counterfactual_model is None:
            return {}, float(job.confidence)

        target_idx = self._resolve_target_index(job)
        if target_idx is None:
            return {}, float(job.confidence)

        try:
            import torch
        except ImportError:
            return {}, float(job.confidence)

        snapshot = job.subgraph_snapshot
        node_features = self._snapshot_get(snapshot, "node_features")
        edge_index = self._snapshot_get(snapshot, "edge_index")
        if not isinstance(node_features, dict) or not isinstance(edge_index, dict):
            return {}, float(job.confidence)

        node_ids = (
            self._snapshot_get(snapshot, "node_ids")
            or self._snapshot_get(snapshot, "node_id_map")
            or self._snapshot_get(snapshot, "node_index")
        )
        packet_id_to_idx = self._resolve_node_id_map(node_ids, "packet")
        if not packet_id_to_idx:
            packet_id_to_idx = self._resolve_flat_id_map(snapshot, "packet_id_map")
        if not packet_id_to_idx:
            return {}, float(job.confidence)

        flow_id_to_idx = self._resolve_node_id_map(node_ids, "flow")
        flow_idx = 0
        if flow_id_to_idx and job.flow_id in flow_id_to_idx:
            flow_idx = int(flow_id_to_idx[job.flow_id])

        model = self.counterfactual_model
        model.eval()
        device = self._model_device(model, torch)

        node_tensors: dict[str, torch.Tensor] = {}
        for node_type, features in node_features.items():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            node_tensors[str(node_type)] = tensor

        required_nodes = {"flow", "packet", "technique", "tactic"}
        if not required_nodes.issubset(node_tensors.keys()):
            return {}, float(job.confidence)

        edge_index_dict = self._prepare_edge_index_dict(model, edge_index, device)
        if not edge_index_dict:
            return {}, float(job.confidence)

        edge_weight = self._snapshot_get(snapshot, "edge_weight")
        if edge_weight is None:
            edge_weight = self._snapshot_get(snapshot, "edge_attr")
        edge_weight_dict = self._prepare_edge_weight_dict(model, edge_weight, device)

        with torch.no_grad():
            base_logits = model(node_tensors, edge_index_dict, edge_weight_dict=edge_weight_dict)
            base_probs = torch.softmax(base_logits, dim=-1)
            if flow_idx >= base_probs.shape[0]:
                flow_idx = 0
            if target_idx >= base_probs.shape[1]:
                return {}, float(job.confidence)
            base_conf = float(base_probs[flow_idx, target_idx].item())

        drop_map: dict[str, float] = {}
        packet_x = node_tensors.get("packet")
        if packet_x is None:
            return {}, base_conf

        for packet in candidates:
            packet_idx = packet_id_to_idx.get(packet.packet_id)
            if packet_idx is None:
                continue
            masked_packet_x = packet_x.clone()
            masked_packet_x[int(packet_idx)] = torch.zeros_like(masked_packet_x[int(packet_idx)])
            masked_nodes = dict(node_tensors)
            masked_nodes["packet"] = masked_packet_x
            with torch.no_grad():
                logits = model(masked_nodes, edge_index_dict, edge_weight_dict=edge_weight_dict)
                probs = torch.softmax(logits, dim=-1)
                masked_conf = float(probs[flow_idx, target_idx].item())
            drop_map[packet.packet_id] = base_conf - masked_conf

        return drop_map, base_conf

    def _resolve_target_index(self, job: SlowPathJob) -> int | None:
        predicted_idx = getattr(job, "predicted_label_idx", None)
        if isinstance(predicted_idx, int):
            return predicted_idx
        if self.label_to_index and job.predicted_label in self.label_to_index:
            return int(self.label_to_index[job.predicted_label])
        try:
            return int(job.predicted_label)
        except (TypeError, ValueError):
            return None

    def _snapshot_get(self, snapshot: Any, key: str) -> Any:
        if isinstance(snapshot, dict):
            return snapshot.get(key)
        return getattr(snapshot, key, None)

    def _resolve_node_id_map(
        self,
        node_ids: Any,
        node_type: str,
    ) -> dict[str, int]:
        if not isinstance(node_ids, dict):
            return {}
        payload = node_ids.get(node_type) or node_ids.get(f"{node_type}_ids")
        if isinstance(payload, dict):
            return {str(key): int(val) for key, val in payload.items()}
        if isinstance(payload, (list, tuple)):
            return {str(value): idx for idx, value in enumerate(payload)}
        return {}

    def _resolve_flat_id_map(self, snapshot: Any, key: str) -> dict[str, int]:
        payload = self._snapshot_get(snapshot, key)
        if not isinstance(payload, dict):
            return {}
        return {str(node_id): int(idx) for node_id, idx in payload.items()}

    def _prepare_edge_index_dict(
        self,
        model: Any,
        edge_index: dict[Any, Any],
        device: Any,
    ) -> dict[tuple[str, str, str], Any]:
        try:
            import torch
        except ImportError:
            return {}

        edge_index_dict: dict[tuple[str, str, str], Any] = {}
        for edge_type in getattr(model, "edge_types", []):
            edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)

        for raw_key, value in edge_index.items():
            edge_key = self._normalize_edge_key(raw_key)
            if edge_key is None:
                continue
            edge_index_dict[edge_key] = torch.as_tensor(value, dtype=torch.long, device=device)
        return edge_index_dict

    def _prepare_edge_weight_dict(
        self,
        model: Any,
        edge_weight: Any,
        device: Any,
    ) -> dict[tuple[str, str, str], Any] | None:
        if not isinstance(edge_weight, dict):
            return None

        try:
            import torch
        except ImportError:
            return None

        edge_weight_dict: dict[tuple[str, str, str], Any] = {}
        for raw_key, value in edge_weight.items():
            edge_key = self._normalize_edge_key(raw_key)
            if edge_key is None:
                continue
            edge_weight_dict[edge_key] = torch.as_tensor(value, dtype=torch.float32, device=device)

        if not edge_weight_dict:
            return None
        return edge_weight_dict

    def _normalize_edge_key(self, raw_key: Any) -> tuple[str, str, str] | None:
        if isinstance(raw_key, tuple) and len(raw_key) == 3:
            return tuple(str(part) for part in raw_key)
        if isinstance(raw_key, list) and len(raw_key) == 3:
            return (str(raw_key[0]), str(raw_key[1]), str(raw_key[2]))
        if isinstance(raw_key, str):
            if raw_key.count("__") == 2:
                parts = raw_key.split("__")
                return (parts[0], parts[1], parts[2])
            cleaned = raw_key.strip().strip("()")
            parts = [part.strip().strip("'\"") for part in cleaned.split(",") if part.strip()]
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
        return None

    def _edge_index_for_key(
        self,
        edge_index: Mapping[Any, Any],
        edge_key: tuple[str, str, str],
    ) -> Any:
        if edge_key in edge_index:
            return edge_index[edge_key]
        edge_name = "__".join(edge_key)
        if edge_name in edge_index:
            return edge_index[edge_name]
        edge_text = str(edge_key)
        if edge_text in edge_index:
            return edge_index[edge_text]
        for raw_key, value in edge_index.items():
            if self._normalize_edge_key(raw_key) == edge_key:
                return value
        return None

    def _edge_index_to_rows(self, value: Any) -> tuple[list[int], list[int]]:
        if value is None:
            return [], []
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return [], []
        if len(value) == 2 and all(isinstance(row, Sequence) for row in value):
            return [int(idx) for idx in value[0]], [int(idx) for idx in value[1]]
        if value and all(isinstance(row, Sequence) and len(row) == 2 for row in value):
            return [int(row[0]) for row in value], [int(row[1]) for row in value]
        return [], []

    def _attention_to_values(self, value: Any) -> list[float]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            tensor = value.detach()
            if getattr(tensor, "ndim", 0) > 1:
                tensor = tensor.mean(dim=1)
            return [float(item) for item in tensor.cpu().reshape(-1).tolist()]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            numeric = self._to_float(value)
            return [] if numeric is None else [numeric]
        values: list[float] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                row = [self._to_float(raw) for raw in item]
                row = [raw for raw in row if raw is not None]
                if row:
                    values.append(float(sum(row) / len(row)))
                continue
            numeric = self._to_float(item)
            if numeric is not None:
                values.append(numeric)
        return values

    def _to_float(self, value: Any) -> float | None:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _model_device(self, model: Any, torch_module: Any) -> Any:
        if hasattr(model, "parameters"):
            try:
                return next(model.parameters()).device
            except (StopIteration, TypeError):
                pass
        return torch_module.device("cpu")

    def _truncate_payload_preview(self, payload_hex: str, payload_ascii: str) -> tuple[str, str]:
        limit = max(int(self.config.max_payload_preview_bytes), 0)
        hex_chars = limit * 2
        truncated_hex = payload_hex[:hex_chars]
        truncated_ascii = payload_ascii[:limit]
        return truncated_hex, truncated_ascii

    def _estimate_tokens(self, bundle: EvidenceBundle) -> int:
        raw_json = bundle.to_json(indent=2)
        return max(int(len(raw_json) / 4), 1)
