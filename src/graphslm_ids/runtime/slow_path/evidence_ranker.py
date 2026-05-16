from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

from graphslm_ids.runtime.slow_path.evidence_bundle import EvidenceBundle


@dataclass
class RankerConfig:
    alpha: float = 0.40
    beta: float = 0.30
    gamma: float = 0.20
    delta: float = 0.10


class EvidenceRanker:
    def __init__(self, config: RankerConfig | None = None) -> None:
        self.config = config or RankerConfig()

    def rank_and_truncate(
        self,
        bundle: EvidenceBundle,
        top_packets: int = 5,
        top_techniques: int = 5,
        top_paths: int = 5,
    ) -> EvidenceBundle:
        self._score_packets(bundle)
        technique_scores = self._score_techniques(bundle)
        bundle.packet_evidence.sort(key=lambda item: item.importance_score, reverse=True)
        if top_packets > 0:
            bundle.packet_evidence = bundle.packet_evidence[:top_packets]
        bundle.mitre_evidence.sort(
            key=lambda item: technique_scores.get(item.technique_id, 0.0),
            reverse=True,
        )
        if top_techniques > 0:
            bundle.mitre_evidence = bundle.mitre_evidence[:top_techniques]

        surviving_packet_ids = {item.packet_id for item in bundle.packet_evidence}
        surviving_technique_ids = {item.technique_id for item in bundle.mitre_evidence}
        surviving_tech_evidence_ids = {item.evidence_id for item in bundle.mitre_evidence}
        for packet in bundle.packet_evidence:
            packet.linked_techniques = [
                eid for eid in packet.linked_techniques if eid in surviving_tech_evidence_ids
            ]

        bundle.graph_paths.sort(key=lambda item: item.path_score, reverse=True)
        bundle.graph_paths = [
            path
            for path in bundle.graph_paths
            if _path_refs_surviving_evidence(path, surviving_packet_ids, surviving_technique_ids)
        ]
        if top_paths > 0:
            bundle.graph_paths = bundle.graph_paths[:top_paths]

        bundle.counterfactual_evidence = [
            item
            for item in bundle.counterfactual_evidence
            if item.masked_element_id in surviving_packet_ids
        ]

        bundle.bundle_stats.num_packets_in_evidence = len(bundle.packet_evidence)
        bundle.bundle_stats.num_techniques_in_evidence = len(bundle.mitre_evidence)
        bundle.bundle_stats.num_paths_in_evidence = len(bundle.graph_paths)
        bundle.bundle_stats.total_tokens_estimate = _estimate_tokens(bundle)
        return bundle

    def truncate_to_mini(self, bundle: EvidenceBundle) -> EvidenceBundle:
        mini = copy.deepcopy(bundle)
        mini.packet_evidence = mini.packet_evidence[:1]
        mini.mitre_evidence = mini.mitre_evidence[:1]
        mini.graph_paths = mini.graph_paths[:1]
        mini.counterfactual_evidence = mini.counterfactual_evidence[:1]
        mini.bundle_stats.num_packets_in_evidence = len(mini.packet_evidence)
        mini.bundle_stats.num_techniques_in_evidence = len(mini.mitre_evidence)
        mini.bundle_stats.num_paths_in_evidence = len(mini.graph_paths)
        mini.bundle_stats.total_tokens_estimate = _estimate_tokens(mini)
        return mini

    def _score_packets(self, bundle: EvidenceBundle) -> None:
        packets = bundle.packet_evidence
        if not packets:
            return

        total_packets = bundle.flow_evidence.packet_count or len(packets)
        denom = max(total_packets - 1, 1)

        cf_values = [p.importance_sources.get("counterfactual_drop", 0.0) for p in packets]
        attn_values = [p.importance_sources.get("hgt_attention_weight", 0.0) for p in packets]
        cos_values = [p.mitre_max_cosine for p in packets]
        temp_values = [1.0 - (p.order_in_flow / denom) for p in packets]

        cf_norm = _min_max_norm(cf_values)
        attn_norm = _min_max_norm(attn_values)
        cos_norm = _min_max_norm(cos_values)
        temp_norm = _min_max_norm(temp_values)

        for idx, packet in enumerate(packets):
            score = (
                self.config.alpha * cf_norm[idx]
                + self.config.beta * attn_norm[idx]
                + self.config.gamma * cos_norm[idx]
                + self.config.delta * temp_norm[idx]
            )
            packet.importance_score = float(score)
            packet.importance_sources["combined_score"] = float(score)

    def _score_techniques(self, bundle: EvidenceBundle) -> dict[str, float]:
        if not bundle.mitre_evidence:
            return {}

        cosine_vals = [tech.cosine_score for tech in bundle.mitre_evidence]
        support_vals = [tech.supporting_packet_count for tech in bundle.mitre_evidence]
        cf_vals = [self._sum_cf_drop(bundle, tech.technique_id) for tech in bundle.mitre_evidence]

        cosine_norm = _min_max_norm(cosine_vals)
        support_norm = _min_max_norm(support_vals)
        cf_norm = _min_max_norm(cf_vals)

        scores: dict[str, float] = {}
        for idx, tech in enumerate(bundle.mitre_evidence):
            score = 0.50 * cosine_norm[idx] + 0.30 * support_norm[idx] + 0.20 * cf_norm[idx]
            scores[tech.technique_id] = float(score)
        return scores

    def _sum_cf_drop(self, bundle: EvidenceBundle, technique_id: str) -> float:
        packet_ids = set()
        for tech in bundle.mitre_evidence:
            if tech.technique_id != technique_id:
                continue
            packet_ids.update([item for item in tech.matched_from if item.startswith("pkt_")])

        total = 0.0
        for packet in bundle.packet_evidence:
            if packet.packet_id in packet_ids:
                total += float(packet.importance_sources.get("counterfactual_drop", 0.0))
        return total


def _min_max_norm(values: Iterable[float]) -> list[float]:
    values = list(values)
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val <= 1e-9:
        return [0.0 for _ in values]
    return [(val - min_val) / (max_val - min_val) for val in values]


def _path_refs_surviving_evidence(
    path: object,
    packet_ids: set[str],
    technique_ids: set[str],
) -> bool:
    path_nodes = getattr(path, "path_nodes", [])
    seen_packet = False
    seen_technique = False
    for node in path_nodes:
        node_id = node.get("id") if isinstance(node, dict) else None
        node_type = node.get("type") if isinstance(node, dict) else None
        if node_type == "packet":
            seen_packet = node_id in packet_ids
        elif node_type == "technique":
            seen_technique = node_id in technique_ids
    return seen_packet and seen_technique


def _estimate_tokens(bundle: EvidenceBundle) -> int:
    return max(int(len(bundle.to_json(indent=2)) / 4), 1)
