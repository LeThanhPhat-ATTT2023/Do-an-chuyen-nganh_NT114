from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GraphCsvTables:
    flow_nodes: pd.DataFrame
    packet_nodes: pd.DataFrame
    contain_edges: pd.DataFrame
    link_edges: pd.DataFrame


_KEY_COLS = ["pcap_file", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]

_FLOW_NODE_COLS = [
    "flow_id", "pcap_file", "label", "src_ip", "dst_ip",
    "src_port", "dst_port", "protocol",
    "start_timestamp", "end_timestamp", "duration_seconds",
    "packet_count", "total_payload_bytes",
]
_PACKET_NODE_COLS = [
    "packet_node_id", "flow_id", "payload_row_index", "packet_index",
    "timestamp", "label", "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol", "payload_len_raw", "packet_order_in_flow",
]


def _empty_tables() -> GraphCsvTables:
    return GraphCsvTables(
        flow_nodes=pd.DataFrame(columns=_FLOW_NODE_COLS),
        packet_nodes=pd.DataFrame(columns=_PACKET_NODE_COLS),
        contain_edges=pd.DataFrame(
            columns=["contain_edge_id", "flow_id", "packet_node_id", "packet_order_in_flow"]
        ),
        link_edges=pd.DataFrame(
            columns=["link_edge_id", "flow_id", "src_packet_node_id", "dst_packet_node_id", "delta_time_seconds"]
        ),
    )


def build_graph_csv_tables(
    metadata: pd.DataFrame,
    flow_timeout_seconds: float = 30.0,
    max_packets_per_flow: int = 20,
) -> GraphCsvTables:
    """Split packet-level metadata into flow/packet node and edge tables.

    Vectorized implementation: uses pandas groupby + cumsum + factorize to assign
    flow IDs without a Python-level row loop.  Runs in O(N log N) pandas time
    instead of O(N) Python iterations, which is orders of magnitude faster for
    large PCAP datasets.
    """
    required_columns = {
        "pcap_file", "packet_index", "timestamp", "label",
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "payload_len_raw",
    }
    missing = sorted(required_columns.difference(metadata.columns))
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    if max_packets_per_flow < 1:
        raise ValueError("max_packets_per_flow must be >= 1")
    if metadata.empty:
        return _empty_tables()

    # ── prepare work copy ─────────────────────────────────────────────────────
    df = metadata.copy().reset_index(drop=False)
    df.rename(columns={"index": "payload_row_index"}, inplace=True)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
    df["payload_len_raw"] = pd.to_numeric(df["payload_len_raw"], errors="coerce").fillna(0).astype(np.int64)
    df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).astype(np.int32)
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(np.int32)

    df.sort_values(
        by=_KEY_COLS + ["timestamp", "packet_index"],
        inplace=True,
        kind="mergesort",
    )
    df.reset_index(drop=True, inplace=True)

    # ── flow boundary detection ───────────────────────────────────────────────
    # prev_ts is NaN for the first packet of every new key-group
    prev_ts = df.groupby(_KEY_COLS, sort=False)["timestamp"].shift(1)
    time_gap = df["timestamp"] - prev_ts
    # A new flow segment starts when: first packet in key-group OR timeout exceeded
    df["_break"] = (prev_ts.isna() | (time_gap > flow_timeout_seconds)).astype(np.int8)
    # Cumulative breaks within each key-group → timeout-segment index
    df["_seg"] = df.groupby(_KEY_COLS, sort=False)["_break"].cumsum()

    # ── max-packets-per-flow chunking ─────────────────────────────────────────
    seg_cols = _KEY_COLS + ["_seg"]
    df["_chunk"] = df.groupby(seg_cols, sort=False).cumcount() // max_packets_per_flow

    # ── assign unique flow IDs ────────────────────────────────────────────────
    # Build a compact string key for every (5-tuple + segment + chunk) combination
    flow_key_cols = _KEY_COLS + ["_seg", "_chunk"]
    sep = "\x00"
    key_str: pd.Series = df[flow_key_cols[0]].astype(str)
    for col in flow_key_cols[1:]:
        key_str = key_str + sep + df[col].astype(str)

    # factorize: assigns integers in first-occurrence order (stable)
    flow_int, _ = pd.factorize(key_str, sort=False)
    df["flow_id"] = "flow_" + pd.Series(flow_int + 1).astype(str).str.zfill(8).values

    # ── packet ordering within flows ──────────────────────────────────────────
    df["packet_order_in_flow"] = df.groupby("flow_id", sort=False).cumcount()

    # ── packet node IDs ───────────────────────────────────────────────────────
    df["packet_node_id"] = "pkt_" + (df.index + 1).astype(str).str.zfill(9)

    # ── flow_nodes ────────────────────────────────────────────────────────────
    flow_agg = (
        df.groupby("flow_id", sort=False)
        .agg(
            pcap_file=("pcap_file", "first"),
            label=("label", "first"),
            src_ip=("src_ip", "first"),
            dst_ip=("dst_ip", "first"),
            src_port=("src_port", "first"),
            dst_port=("dst_port", "first"),
            protocol=("protocol", "first"),
            start_timestamp=("timestamp", "min"),
            end_timestamp=("timestamp", "max"),
            packet_count=("payload_row_index", "count"),
            total_payload_bytes=("payload_len_raw", "sum"),
        )
        .reset_index()
    )
    flow_agg["duration_seconds"] = flow_agg["end_timestamp"] - flow_agg["start_timestamp"]
    flow_nodes = flow_agg[_FLOW_NODE_COLS]

    # ── packet_nodes ──────────────────────────────────────────────────────────
    packet_nodes = df[_PACKET_NODE_COLS].copy()

    # ── contain_edges ─────────────────────────────────────────────────────────
    contain_edges = df[["flow_id", "packet_node_id", "packet_order_in_flow"]].copy()
    contain_edges.insert(
        0,
        "contain_edge_id",
        "contain_" + (df.index + 1).astype(str).str.zfill(9).values,
    )

    # ── link_edges (consecutive packets within the same flow) ─────────────────
    prev_pkt_id = df.groupby("flow_id", sort=False)["packet_node_id"].shift(1)
    prev_ts_flow = df.groupby("flow_id", sort=False)["timestamp"].shift(1)
    link_mask = prev_pkt_id.notna()
    n_links = int(link_mask.sum())
    if n_links > 0:
        link_edges = pd.DataFrame(
            {
                "link_edge_id": "link_" + pd.Series(np.arange(1, n_links + 1), dtype=str).str.zfill(9).values,
                "flow_id": df["flow_id"].values[link_mask],
                "src_packet_node_id": prev_pkt_id.values[link_mask],
                "dst_packet_node_id": df["packet_node_id"].values[link_mask],
                "delta_time_seconds": (df["timestamp"] - prev_ts_flow).values[link_mask].astype(float),
            }
        )
    else:
        link_edges = pd.DataFrame(
            columns=["link_edge_id", "flow_id", "src_packet_node_id", "dst_packet_node_id", "delta_time_seconds"]
        )

    return GraphCsvTables(
        flow_nodes=flow_nodes,
        packet_nodes=packet_nodes,
        contain_edges=contain_edges,
        link_edges=link_edges,
    )
