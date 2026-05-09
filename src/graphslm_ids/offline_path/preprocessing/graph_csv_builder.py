from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GraphCsvTables:
    flow_nodes: pd.DataFrame
    packet_nodes: pd.DataFrame
    contain_edges: pd.DataFrame
    link_edges: pd.DataFrame


def build_graph_csv_tables(
    metadata: pd.DataFrame,
    flow_timeout_seconds: float = 30.0,
    max_packets_per_flow: int = 20,
) -> GraphCsvTables:
    """Split packet-level metadata into review-friendly flow/packet node and edge CSV tables."""
    required_columns = {
        "pcap_file",
        "packet_index",
        "timestamp",
        "label",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "payload_len_raw",
    }

    missing_columns = sorted(required_columns.difference(metadata.columns))
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {missing_columns}")
    if max_packets_per_flow < 1:
        raise ValueError("max_packets_per_flow must be >= 1")

    if metadata.empty:
        return GraphCsvTables(
            flow_nodes=pd.DataFrame(
                columns=[
                    "flow_id",
                    "pcap_file",
                    "label",
                    "src_ip",
                    "dst_ip",
                    "src_port",
                    "dst_port",
                    "protocol",
                    "start_timestamp",
                    "end_timestamp",
                    "duration_seconds",
                    "packet_count",
                    "total_payload_bytes",
                ]
            ),
            packet_nodes=pd.DataFrame(
                columns=[
                    "packet_node_id",
                    "flow_id",
                    "payload_row_index",
                    "packet_index",
                    "timestamp",
                    "label",
                    "src_ip",
                    "dst_ip",
                    "src_port",
                    "dst_port",
                    "protocol",
                    "payload_len_raw",
                    "packet_order_in_flow",
                ]
            ),
            contain_edges=pd.DataFrame(
                columns=["contain_edge_id", "flow_id", "packet_node_id", "packet_order_in_flow"]
            ),
            link_edges=pd.DataFrame(
                columns=[
                    "link_edge_id",
                    "flow_id",
                    "src_packet_node_id",
                    "dst_packet_node_id",
                    "delta_time_seconds",
                ]
            ),
        )

    metadata_work = metadata.copy().reset_index(drop=False)
    metadata_work.rename(columns={"index": "payload_row_index"}, inplace=True)
    metadata_work["timestamp"] = pd.to_numeric(metadata_work["timestamp"], errors="coerce").fillna(0.0)
    metadata_work["payload_len_raw"] = pd.to_numeric(metadata_work["payload_len_raw"], errors="coerce").fillna(0)
    metadata_work.sort_values(
        by=["pcap_file", "timestamp", "packet_index"],
        inplace=True,
        kind="mergesort",
    )

    active_flows: dict[tuple[str, str, str, int, int, str, str], dict[str, object]] = {}
    flow_aggregates: dict[str, dict[str, object]] = {}

    packet_rows: list[dict[str, object]] = []
    contain_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []

    flow_counter = 0
    packet_counter = 0
    contain_counter = 0
    link_counter = 0

    for row in metadata_work.itertuples(index=False):
        flow_key = (
            str(row.pcap_file),
            str(row.src_ip),
            str(row.dst_ip),
            int(row.src_port),
            int(row.dst_port),
            str(row.protocol),
            str(row.label),
        )
        timestamp = float(row.timestamp)

        flow_state = active_flows.get(flow_key)
        should_rollover = False
        if flow_state is not None:
            idle_gap = timestamp - float(flow_state["last_timestamp"])
            packet_count = int(flow_state["packet_count"])
            should_rollover = idle_gap > flow_timeout_seconds or packet_count >= max_packets_per_flow

        if flow_state is None or should_rollover:
            flow_counter += 1
            flow_id = f"flow_{flow_counter:08d}"
            flow_state = {
                "flow_id": flow_id,
                "last_timestamp": timestamp,
                "packet_count": 0,
                "last_packet_node_id": None,
            }
            active_flows[flow_key] = flow_state
            flow_aggregates[flow_id] = {
                "flow_id": flow_id,
                "pcap_file": str(row.pcap_file),
                "label": str(row.label),
                "src_ip": str(row.src_ip),
                "dst_ip": str(row.dst_ip),
                "src_port": int(row.src_port),
                "dst_port": int(row.dst_port),
                "protocol": str(row.protocol),
                "start_timestamp": timestamp,
                "end_timestamp": timestamp,
                "packet_count": 0,
                "total_payload_bytes": 0,
            }

        flow_id = str(flow_state["flow_id"])
        packet_order = int(flow_state["packet_count"])

        packet_counter += 1
        packet_node_id = f"pkt_{packet_counter:09d}"

        packet_rows.append(
            {
                "packet_node_id": packet_node_id,
                "flow_id": flow_id,
                "payload_row_index": int(row.payload_row_index),
                "packet_index": int(row.packet_index),
                "timestamp": timestamp,
                "label": str(row.label),
                "src_ip": str(row.src_ip),
                "dst_ip": str(row.dst_ip),
                "src_port": int(row.src_port),
                "dst_port": int(row.dst_port),
                "protocol": str(row.protocol),
                "payload_len_raw": int(row.payload_len_raw),
                "packet_order_in_flow": packet_order,
            }
        )

        contain_counter += 1
        contain_rows.append(
            {
                "contain_edge_id": f"contain_{contain_counter:09d}",
                "flow_id": flow_id,
                "packet_node_id": packet_node_id,
                "packet_order_in_flow": packet_order,
            }
        )

        previous_packet_node_id = flow_state["last_packet_node_id"]
        if previous_packet_node_id is not None:
            link_counter += 1
            delta_time = timestamp - float(flow_state["last_timestamp"])
            link_rows.append(
                {
                    "link_edge_id": f"link_{link_counter:09d}",
                    "flow_id": flow_id,
                    "src_packet_node_id": str(previous_packet_node_id),
                    "dst_packet_node_id": packet_node_id,
                    "delta_time_seconds": float(delta_time),
                }
            )

        aggregate = flow_aggregates[flow_id]
        aggregate["end_timestamp"] = timestamp
        aggregate["packet_count"] = int(aggregate["packet_count"]) + 1
        aggregate["total_payload_bytes"] = int(aggregate["total_payload_bytes"]) + int(row.payload_len_raw)

        flow_state["packet_count"] = int(flow_state["packet_count"]) + 1
        flow_state["last_timestamp"] = timestamp
        flow_state["last_packet_node_id"] = packet_node_id

    flow_nodes = pd.DataFrame(flow_aggregates.values())
    if not flow_nodes.empty:
        flow_nodes["duration_seconds"] = flow_nodes["end_timestamp"] - flow_nodes["start_timestamp"]
        flow_nodes = flow_nodes[
            [
                "flow_id",
                "pcap_file",
                "label",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "protocol",
                "start_timestamp",
                "end_timestamp",
                "duration_seconds",
                "packet_count",
                "total_payload_bytes",
            ]
        ]

    packet_nodes = pd.DataFrame(packet_rows)
    contain_edges = pd.DataFrame(contain_rows)
    link_edges = pd.DataFrame(link_rows)

    return GraphCsvTables(
        flow_nodes=flow_nodes,
        packet_nodes=packet_nodes,
        contain_edges=contain_edges,
        link_edges=link_edges,
    )
