from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class SigcEdgeFilterConfig:
    enabled: bool = False
    alpha: float = 0.4
    min_score_to_keep_edge: float = 0.15
    apply_to_edge_types: tuple[str, ...] = (
        "flow__contains__packet",
        "packet__next_packet__packet",
        "packet__matches_technique__technique",
    )


class SigcEdgeFilter:
    """Lightweight Structural Importance Graph Compression filter.

    The filter is intentionally cheap enough for online preprocessing. It keeps
    the existing MITRE cosine score as the semantic component and multiplies it
    by a local degree term plus hop-distance decay before edges enter the
    persistent store or hot buffer.
    """

    def __init__(self, config: SigcEdgeFilterConfig | None = None) -> None:
        self.config = config or SigcEdgeFilterConfig()
        self._enabled_edge_types = {str(item) for item in self.config.apply_to_edge_types}

    def local_contribution_score(
        self,
        edge_type: str,
        *,
        semantic_weight: float = 1.0,
        src_out_degree: int = 1,
        dst_in_degree: int = 0,
        hop_distance: int = 1,
    ) -> float:
        if not self.config.enabled or str(edge_type) not in self._enabled_edge_types:
            return float(semantic_weight)
        src_degree = max(int(src_out_degree), 0) + 1
        dst_degree = max(int(dst_in_degree), 0) + 1
        degree_term = min(1.0, max(0.0, src_degree / dst_degree))
        hop_decay = exp(-float(self.config.alpha) * max(int(hop_distance), 0))
        return float(semantic_weight) * degree_term * hop_decay

    def keep(
        self,
        edge_type: str,
        *,
        semantic_weight: float = 1.0,
        src_out_degree: int = 1,
        dst_in_degree: int = 0,
        hop_distance: int = 1,
    ) -> bool:
        if not self.config.enabled or str(edge_type) not in self._enabled_edge_types:
            return True
        score = self.local_contribution_score(
            edge_type,
            semantic_weight=semantic_weight,
            src_out_degree=src_out_degree,
            dst_in_degree=dst_in_degree,
            hop_distance=hop_distance,
        )
        return score >= float(self.config.min_score_to_keep_edge)

    def filter_mitre_topk(self, topk: list[tuple[str, float]]) -> list[tuple[str, float]]:
        edge_type = "packet__matches_technique__technique"
        if not self.config.enabled or edge_type not in self._enabled_edge_types:
            return [(str(tech_id), float(score)) for tech_id, score in topk]
        src_out_degree = max(len(topk), 1)
        kept: list[tuple[str, float]] = []
        for tech_id, score in topk:
            if self.keep(
                edge_type,
                semantic_weight=float(score),
                src_out_degree=src_out_degree,
                dst_in_degree=0,
                hop_distance=1,
            ):
                kept.append((str(tech_id), float(score)))
        return kept
