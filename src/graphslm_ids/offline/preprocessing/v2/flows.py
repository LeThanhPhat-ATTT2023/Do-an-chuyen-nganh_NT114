"""Bidirectional flow assembly + CICFlowMeter-style features (with TCP flags + fan-out).

Replaces the v1 flow logic which (a) chunked every flow to <=20 packets — destroying
duration / IAT statistics — and (b) only produced 6 coarse flow features. v2:
  * Bidirectional 5-tuple keyed flows with 30-second idle-timeout segmentation. No
    packet cap; full flow is one row.
  * ~80-dim flow feature vector covering CICFlowMeter-style statistics, including
    TCP flag counts/ratios (the v1 dataset never recorded these) and per-source
    fan-out (the macro signature of scans, computed across the input batch).

The feature names returned in ``meta["feature_names"]`` are the persistent contract
read by the v2 graph builder and the HGT loader.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# TCP flag bit masks (mirrors the dpkt constants we use in the extractor).
_FIN = 0x01
_SYN = 0x02
_RST = 0x04
_PSH = 0x08
_ACK = 0x10
_URG = 0x20


def _endpoint(ip: pd.Series, port: pd.Series) -> pd.Series:
    """Compact ``ip:port`` string used for canonical bidirectional flow keying."""
    return ip.astype(str) + ":" + port.astype(str)


def assign_flows(df: pd.DataFrame, timeout_s: float = 30.0) -> pd.DataFrame:
    """Tag every packet row with a ``flow_id`` (string) and ``is_fwd`` (bool).

    Bidirectional 5-tuple: the canonical key sorts the two endpoints so packets
    in both directions of the same connection land in the same flow. ``is_fwd``
    is true for packets matching the FIRST packet's direction within the flow.
    A new flow segment is opened whenever the inter-packet gap exceeds
    ``timeout_s`` (mirrors the v1 timeout but without the 20-packet cap).
    """
    if df.empty:
        out = df.copy()
        out["flow_id"] = pd.Series([], dtype="object")
        out["is_fwd"] = pd.Series([], dtype=bool)
        return out

    df = df.copy()
    a = _endpoint(df["src_ip"], df["src_port"])
    b = _endpoint(df["dst_ip"], df["dst_port"])
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    df["_key"] = (
        df["label"].astype(str) + "|" + lo + "|" + hi + "|" + df["proto"].astype(str)
    )
    # forward = the (src,sport) matches the canonical low endpoint
    df["_is_fwd_raw"] = (a == lo).to_numpy()

    df.sort_values(["_key", "ts"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    prev_ts = df.groupby("_key", sort=False)["ts"].shift(1)
    gap = df["ts"] - prev_ts
    new_seg = (prev_ts.isna() | (gap > timeout_s)).astype(np.int64)
    seg_idx = new_seg.groupby(df["_key"], sort=False).cumsum()
    df["flow_id"] = df["_key"] + "#" + seg_idx.astype(str)

    # `is_fwd` should reference the first packet of THIS flow segment, not the
    # canonical-low endpoint, so the direction marker is consistent with what
    # downstream stats expect ("fwd = initiator -> responder").
    first_dir = df.groupby("flow_id", sort=False)["_is_fwd_raw"].transform("first")
    df["is_fwd"] = df["_is_fwd_raw"] == first_dir
    df.drop(columns=["_key", "_is_fwd_raw"], inplace=True)
    return df


def _stat_block(g: pd.api.typing.DataFrameGroupBy, col: str, prefix: str) -> dict:
    """Return ``{prefix}_{mean,std,min,max,sum}`` aggregates for the column."""
    s = g[col]
    return {
        f"{prefix}_mean": s.mean(),
        f"{prefix}_std": s.std().fillna(0.0),
        f"{prefix}_min": s.min(),
        f"{prefix}_max": s.max(),
        f"{prefix}_sum": s.sum(),
    }


def _iat_block(
    df: pd.DataFrame, mask: pd.Series | None, prefix: str
) -> dict[str, pd.Series]:
    """Inter-arrival-time aggregates over the (optionally filtered) df."""
    sub = df if mask is None else df[mask]
    if sub.empty:
        empty = pd.Series(dtype=float)
        return {
            f"{prefix}_mean": empty,
            f"{prefix}_std": empty,
            f"{prefix}_min": empty,
            f"{prefix}_max": empty,
        }
    iat = sub.groupby("flow_id", sort=False)["ts"].diff()
    grouped = iat.groupby(sub["flow_id"], sort=False)
    return {
        f"{prefix}_mean": grouped.mean().fillna(0.0),
        f"{prefix}_std": grouped.std().fillna(0.0),
        f"{prefix}_min": grouped.min().fillna(0.0),
        f"{prefix}_max": grouped.max().fillna(0.0),
    }


def build_flow_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate per-packet rows into per-flow CICFlowMeter-style features.

    The returned frame is indexed by ``flow_id`` and contains ~80 columns covering:
      * Identity: label, proto, src_port, dst_port
      * Counts: tot_pkts, fwd_pkts, bwd_pkts; tot_bytes, fwd_bytes, bwd_bytes
      * Duration & rates: dur, bytes_per_s, pkts_per_s, fwd_pkts_per_s,
        bwd_pkts_per_s, down_up_ratio, mean_pkt_size
      * Packet length (ip_len) stats overall + fwd + bwd
      * Payload length stats overall
      * IAT stats overall + fwd + bwd
      * TCP flag counts + ratios (n_syn, r_syn, n_ack, ...)
      * Time-windowed per-source fan-out (deployment-honest, no label key):
        - 60 s window:  fan_dst_ip, fan_dst_port, fan_dst_subnet, fan_syn,
          fan_pkts, fan_rate_pkts_60s
        - 5 s window:   fan_*_5s variants for bursty patterns
      * Header asymmetry (mean_ip_len_fwd_minus_bwd)
    """
    if df.empty or "flow_id" not in df.columns:
        feats = pd.DataFrame()
        return feats, {"feature_names": [], "n_flows": 0, "proto_counts": {}}

    df = df.copy()
    # Boolean flag bit columns -- only meaningful for TCP rows but harmless on
    # UDP/ICMP rows where flags==0.
    df["syn"] = ((df["flags"].astype(int) & _SYN) > 0) & (df["proto"] == 0)
    df["ack"] = ((df["flags"].astype(int) & _ACK) > 0) & (df["proto"] == 0)
    df["fin"] = ((df["flags"].astype(int) & _FIN) > 0) & (df["proto"] == 0)
    df["rst"] = ((df["flags"].astype(int) & _RST) > 0) & (df["proto"] == 0)
    df["psh"] = ((df["flags"].astype(int) & _PSH) > 0) & (df["proto"] == 0)
    df["urg"] = ((df["flags"].astype(int) & _URG) > 0) & (df["proto"] == 0)

    g = df.groupby("flow_id", sort=False)
    feats = pd.DataFrame(index=list(g.groups.keys()))
    feats.index.name = "flow_id"

    feats["label"] = g["label"].first()
    feats["proto"] = g["proto"].first().astype(int)
    feats["src_port"] = g["src_port"].first().astype(int)
    feats["dst_port"] = g["dst_port"].first().astype(int)

    feats["tot_pkts"] = g.size().astype(np.int64)
    feats["fwd_pkts"] = g["is_fwd"].sum().astype(np.int64)
    feats["bwd_pkts"] = (feats["tot_pkts"] - feats["fwd_pkts"]).astype(np.int64)
    feats["tot_bytes"] = g["ip_len"].sum().astype(np.float64)
    fwd_bytes = (
        df.loc[df["is_fwd"]].groupby("flow_id", sort=False)["ip_len"].sum()
    ).reindex(feats.index).fillna(0.0)
    feats["fwd_bytes"] = fwd_bytes.astype(np.float64)
    feats["bwd_bytes"] = (feats["tot_bytes"] - feats["fwd_bytes"]).astype(np.float64)

    ts_min = g["ts"].min()
    ts_max = g["ts"].max()
    feats["dur"] = (ts_max - ts_min).astype(np.float64)
    safe_dur = feats["dur"].clip(lower=1e-6)
    feats["bytes_per_s"] = (feats["tot_bytes"] / safe_dur).astype(np.float64)
    feats["pkts_per_s"] = (feats["tot_pkts"] / safe_dur).astype(np.float64)
    feats["fwd_pkts_per_s"] = (feats["fwd_pkts"] / safe_dur).astype(np.float64)
    feats["bwd_pkts_per_s"] = (feats["bwd_pkts"] / safe_dur).astype(np.float64)
    feats["down_up_ratio"] = (
        feats["bwd_pkts"] / feats["fwd_pkts"].clip(lower=1)
    ).astype(np.float64)
    feats["mean_pkt_size"] = (
        feats["tot_bytes"] / feats["tot_pkts"].clip(lower=1)
    ).astype(np.float64)

    # Packet-length (ip_len) blocks: overall, fwd-only, bwd-only.
    for prefix, mask in (
        ("plen", None),
        ("plen_fwd", df["is_fwd"]),
        ("plen_bwd", ~df["is_fwd"]),
    ):
        sub = df if mask is None else df[mask]
        gg = sub.groupby("flow_id", sort=False)
        if sub.empty:
            for stat in ("mean", "std", "min", "max", "sum"):
                feats[f"{prefix}_{stat}"] = 0.0
            continue
        block = _stat_block(gg, "ip_len", prefix)
        for k, v in block.items():
            feats[k] = v.reindex(feats.index).fillna(0.0)

    # Payload length stats (overall only).
    block = _stat_block(g, "payload_len", "plen_pay")
    for k, v in block.items():
        feats[k] = v.reindex(feats.index).fillna(0.0)

    # IAT blocks: overall, fwd-only, bwd-only.
    for prefix, mask in (
        ("iat", None),
        ("iat_fwd", df["is_fwd"]),
        ("iat_bwd", ~df["is_fwd"]),
    ):
        block = _iat_block(df, mask, prefix)
        for k, v in block.items():
            feats[k] = v.reindex(feats.index).fillna(0.0)

    # TCP flag counts + ratios.
    for fl in ("syn", "ack", "fin", "rst", "psh", "urg"):
        n_col = f"n_{fl}"
        r_col = f"r_{fl}"
        feats[n_col] = g[fl].sum().astype(np.float64)
        feats[r_col] = feats[n_col] / feats["tot_pkts"].clip(lower=1).astype(
            np.float64
        )

    # Per-source fan-out: how many distinct dst-IPs/dst-ports does this flow's
    # source contact in a rolling time window? Deployment-honest: only uses
    # ``src_ip`` + ``ts`` (no label). Because different PCAP captures have
    # disjoint absolute timestamps, ``floor(ts / W)`` implicitly acts as a
    # capture-id, preventing cross-capture mixing without requiring a label.
    #
    # Two windows expose both micro patterns (5 s: bursty scans / floods) and
    # macro patterns (60 s: distributed recon / slow scans). The unsuffixed
    # ``fan_*`` names preserve backward-compat with the v1 contract and tests.
    df = df.copy()
    df["_subnet24"] = df["dst_ip"].astype(str).str.rsplit(".", n=1).str[0]
    flow_first = df.groupby("flow_id", sort=False).agg(
        _flow_src_ip=("src_ip", "first"),
        _flow_first_ts=("ts", "first"),
    )
    _WINDOWS: tuple[tuple[float, str], ...] = ((60.0, ""), (5.0, "_5s"))
    for w_sec, suffix in _WINDOWS:
        bin_col = f"_bin{suffix or '_60s'}"
        df[bin_col] = np.floor(df["ts"].to_numpy(dtype=np.float64) / w_sec).astype(
            np.int64
        )
        win_agg = (
            df.groupby(["src_ip", bin_col], sort=False)
            .agg(
                **{
                    f"fan_dst_ip{suffix}": ("dst_ip", "nunique"),
                    f"fan_dst_port{suffix}": ("dst_port", "nunique"),
                    f"fan_dst_subnet{suffix}": ("_subnet24", "nunique"),
                    f"fan_syn{suffix}": ("syn", "sum"),
                    f"fan_pkts{suffix}": ("ts", "count"),
                }
            )
            .reset_index()
        )
        flow_first[bin_col] = np.floor(
            flow_first["_flow_first_ts"].to_numpy(dtype=np.float64) / w_sec
        ).astype(np.int64)
        joined = (
            flow_first.reset_index()
            .merge(
                win_agg,
                left_on=["_flow_src_ip", bin_col],
                right_on=["src_ip", bin_col],
                how="left",
            )
            .set_index("flow_id")
        )
        out_cols = (
            f"fan_dst_ip{suffix}",
            f"fan_dst_port{suffix}",
            f"fan_dst_subnet{suffix}",
            f"fan_syn{suffix}",
            f"fan_pkts{suffix}",
        )
        for col in out_cols:
            feats[col] = (
                joined[col].reindex(feats.index).fillna(0.0).astype(np.float64)
            )
        feats[f"fan_rate_pkts{suffix or '_60s'}"] = (
            feats[f"fan_pkts{suffix}"] / float(w_sec)
        ).astype(np.float64)

    # Asymmetry: mean ip_len difference fwd-bwd. Helps distinguish download- vs
    # upload-heavy flows independently of absolute byte totals.
    feats["mean_ip_len_fwd_minus_bwd"] = (
        feats["plen_fwd_mean"] - feats["plen_bwd_mean"]
    )

    # Coerce all numeric columns to float64 to keep the matrix clean for HGBM/HGT.
    numeric_cols = [c for c in feats.columns if c != "label"]
    feats[numeric_cols] = feats[numeric_cols].astype(np.float64)

    meta = {
        "feature_names": [c for c in feats.columns if c != "label"],
        "n_flows": int(len(feats)),
        "proto_counts": df["proto"].value_counts().to_dict(),
    }
    return feats, meta
