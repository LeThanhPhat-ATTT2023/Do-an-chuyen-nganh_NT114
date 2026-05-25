"""Build evidence-grounded MITRE technique edges from flow features and packet payloads.

Replaces the v1 cosine-threshold edges with deterministic, interpretable
evidence: every emitted edge is justified by either a flow-signature firing on
real flow statistics or a payload-signature firing on actual byte content.

Returned arrays use the same shape conventions as the v1 graph artifact so the
HGT loader can consume both without per-edge code paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.v2.signatures import (
    match_flow_signatures,
    match_payload_signatures,
)


def _dedupe_clip(pairs: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate (src, dst) pairs by summing weights, clipped to 1.0."""
    if not pairs:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 1), dtype=np.float32),
        )
    acc: dict[tuple[int, int], float] = {}
    for s, d, w in pairs:
        key = (s, d)
        acc[key] = min(1.0, acc.get(key, 0.0) + float(w))
    src = np.array([k[0] for k in acc.keys()], dtype=np.int64)
    dst = np.array([k[1] for k in acc.keys()], dtype=np.int64)
    weights = np.array(list(acc.values()), dtype=np.float32)[:, None]
    edge_index = np.vstack([src, dst])
    return edge_index, weights


def build_evidence_edges(
    flow_features: pd.DataFrame,
    payload_matrix: np.ndarray,
    packet_payload_idx: dict[str, np.ndarray],
    technique_id_to_idx: dict[str, int],
    packet_subset_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Produce evidence-grounded edges for flow->technique and packet->technique.

    Args:
      flow_features: per-flow feature frame indexed by flow_id. Row order
        defines the integer flow index used in the returned edge_index.
      payload_matrix: ``(N_packets, payload_length)`` uint8 array.
      packet_payload_idx: ``flow_id -> int64 array of row indices into
        payload_matrix``. Row indices align with payload_matrix rows AND with
        the integer packet index used in the returned edge_index.
      technique_id_to_idx: maps MITRE technique id strings to integer node
        positions in the graph artifact's technique array.
      packet_subset_indices: optional subset of payload_matrix rows to evaluate
        (default: all rows). Use for large datasets where evaluating every
        packet is too slow.

    Returns a dict with four arrays mirroring the v1 graph artifact's edges.
    """
    flow_id_to_idx = {fid: i for i, fid in enumerate(flow_features.index.tolist())}

    # ---- flow -> technique --------------------------------------------------
    flow_edges: list[tuple[int, int, float]] = []
    for fid, row in flow_features.iterrows():
        f_idx = flow_id_to_idx[fid]
        for tech_id, weight in match_flow_signatures(row):
            t_idx = technique_id_to_idx.get(tech_id)
            if t_idx is None:
                continue
            flow_edges.append((f_idx, int(t_idx), weight))
    ft_edge_index, ft_edge_attr = _dedupe_clip(flow_edges)

    # ---- packet -> technique ------------------------------------------------
    packet_edges: list[tuple[int, int, float]] = []
    if packet_subset_indices is None:
        rows_to_scan = range(int(payload_matrix.shape[0]))
    else:
        rows_to_scan = list(map(int, packet_subset_indices))
    for p_idx in rows_to_scan:
        raw = bytes(np.asarray(payload_matrix[p_idx], dtype=np.uint8).ravel())
        for tech_id, weight in match_payload_signatures(raw):
            t_idx = technique_id_to_idx.get(tech_id)
            if t_idx is None:
                continue
            packet_edges.append((p_idx, int(t_idx), weight))
    pt_edge_index, pt_edge_attr = _dedupe_clip(packet_edges)

    return {
        "flow_technique_edge_index": ft_edge_index,
        "flow_technique_edge_attr": ft_edge_attr,
        "packet_technique_edge_index": pt_edge_index,
        "packet_technique_edge_attr": pt_edge_attr,
    }
