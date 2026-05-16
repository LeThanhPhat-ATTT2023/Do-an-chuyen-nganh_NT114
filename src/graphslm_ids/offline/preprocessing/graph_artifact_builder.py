from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.graph_csv_builder import build_graph_csv_tables


@dataclass
class GraphArtifact:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _encode_series(values: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    unique_values = sorted({str(value) for value in values.tolist()})
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    encoded = np.asarray([mapping[str(value)] for value in values.tolist()], dtype=np.int64)
    return encoded, mapping


def build_graph_artifact(
    metadata: pd.DataFrame,
    payload_matrix: np.ndarray,
    flow_timeout_seconds: float = 30.0,
    max_packets_per_flow: int = 20,
) -> GraphArtifact:
    """Build a compact graph artifact directly from packet metadata and payload matrix."""
    if payload_matrix.ndim != 2:
        raise ValueError("payload_matrix must be 2D.")

    tables = build_graph_csv_tables(
        metadata,
        flow_timeout_seconds=flow_timeout_seconds,
        max_packets_per_flow=max_packets_per_flow,
    )

    flow_nodes = tables.flow_nodes.reset_index(drop=True)
    packet_nodes = tables.packet_nodes.reset_index(drop=True)
    contain_edges = tables.contain_edges.reset_index(drop=True)
    link_edges = tables.link_edges.reset_index(drop=True)

    flow_id_to_idx = {flow_id: idx for idx, flow_id in enumerate(flow_nodes["flow_id"].tolist())}
    packet_id_to_idx = {
        packet_id: idx for idx, packet_id in enumerate(packet_nodes["packet_node_id"].tolist())
    }

    packet_payload_indices = packet_nodes["payload_row_index"].astype(int).to_numpy(dtype=np.int64)
    if packet_payload_indices.size and packet_payload_indices.max() >= payload_matrix.shape[0]:
        raise ValueError("payload_row_index exceeds payload_matrix rows.")

    packet_x = payload_matrix[packet_payload_indices] if packet_payload_indices.size else np.empty((0, payload_matrix.shape[1]), dtype=payload_matrix.dtype)

    flow_protocol_encoded, flow_protocol_map = _encode_series(flow_nodes["protocol"]) if not flow_nodes.empty else (np.empty((0,), dtype=np.int64), {})
    flow_label_encoded, flow_label_map = _encode_series(flow_nodes["label"]) if not flow_nodes.empty else (np.empty((0,), dtype=np.int64), {})

    if flow_nodes.empty:
        flow_x = np.empty((0, 6), dtype=np.float32)
    else:
        flow_x = np.column_stack(
            [
                flow_nodes["packet_count"].astype(float).to_numpy(),
                flow_nodes["total_payload_bytes"].astype(float).to_numpy(),
                flow_nodes["duration_seconds"].astype(float).to_numpy(),
                flow_nodes["src_port"].astype(float).to_numpy(),
                flow_nodes["dst_port"].astype(float).to_numpy(),
                flow_protocol_encoded.astype(float),
            ]
        ).astype(np.float32)

    if contain_edges.empty:
        contain_edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        contain_edge_index = np.vstack(
            [
                contain_edges["flow_id"].map(flow_id_to_idx).astype(int).to_numpy(dtype=np.int64),
                contain_edges["packet_node_id"].map(packet_id_to_idx).astype(int).to_numpy(dtype=np.int64),
            ]
        )

    if link_edges.empty:
        link_edge_index = np.empty((2, 0), dtype=np.int64)
        link_edge_attr = np.empty((0, 1), dtype=np.float32)
    else:
        link_edge_index = np.vstack(
            [
                link_edges["src_packet_node_id"].map(packet_id_to_idx).astype(int).to_numpy(dtype=np.int64),
                link_edges["dst_packet_node_id"].map(packet_id_to_idx).astype(int).to_numpy(dtype=np.int64),
            ]
        )
        link_edge_attr = link_edges[["delta_time_seconds"]].astype(float).to_numpy(dtype=np.float32)

    arrays = {
        "flow_x": flow_x,
        "packet_x": packet_x,
        "flow_y": flow_label_encoded.astype(np.int64),
        "contain_edge_index": contain_edge_index,
        "link_edge_index": link_edge_index,
        "link_edge_attr": link_edge_attr,
        "packet_payload_row_index": packet_payload_indices,
    }

    meta = {
        "num_flows": int(flow_nodes.shape[0]),
        "num_packets": int(packet_nodes.shape[0]),
        "num_contain_edges": int(contain_edge_index.shape[1]),
        "num_link_edges": int(link_edge_index.shape[1]),
        "payload_length": int(payload_matrix.shape[1]),
        "flow_feature_names": [
            "packet_count",
            "total_payload_bytes",
            "duration_seconds",
            "src_port",
            "dst_port",
            "protocol_id",
        ],
        "label_mapping": flow_label_map,
        "protocol_mapping": flow_protocol_map,
    }

    return GraphArtifact(arrays=arrays, metadata=meta)
