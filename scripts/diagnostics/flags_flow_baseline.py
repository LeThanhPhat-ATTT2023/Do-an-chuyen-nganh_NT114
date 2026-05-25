"""PROVE-THE-FIX baseline: re-extract from raw PCAPs WITHOUT the lossy filters.

The current dataset kept only TCP/UDP packets that carry an L4 payload, dropping
every TCP control packet (SYN/ACK/RST/FIN) and all ICMP, and never recorded TCP
flags. Those are exactly the signals that separate scans/floods/recon. This
script re-parses the 13 raw PCAPs keeping ALL packets, records TCP flags + IP
length + direction, builds CICFlowMeter-style bidirectional flow features (incl.
flag counts and per-source fan-out), and runs the same HistGBM ceiling test.

If macro-F1 jumps well past the 0.47 payload-histogram ceiling (esp. on the
Recon/scan cluster), the extraction filter is confirmed as the dominant cause
and the production fix is: re-extract with control packets + flags.

Read-only on raw PCAPs. Per-class packet cap keeps the validation fast.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dpkt

# TCP flag bit masks
_FIN, _SYN, _RST, _PSH, _ACK, _URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
_DLT_EN10MB, _DLT_RAW, _DLT_RAW2 = 1, 12, 101


def parse_pcap(pcap_path: Path, label: str, max_packets: int) -> list[tuple]:
    """Return per-packet tuples for ALL packets (TCP/UDP/ICMP, payload or not)."""
    rows: list[tuple] = []
    try:
        f = open(pcap_path, "rb")
    except OSError:
        return rows
    with f:
        try:
            reader = dpkt.pcap.Reader(f)
            dlt = reader.datalink()
        except Exception:
            return rows
        for i, (ts, buf) in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            try:
                if dlt == _DLT_EN10MB:
                    ip = dpkt.ethernet.Ethernet(buf).data
                elif dlt in (_DLT_RAW, _DLT_RAW2):
                    ip = dpkt.ip.IP(buf)
                else:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                if not isinstance(ip, dpkt.ip.IP):
                    continue
            except Exception:
                continue
            t = ip.data
            sport = dport = -1
            flags = 0
            if isinstance(t, dpkt.tcp.TCP):
                proto = 0  # TCP
                sport, dport, flags = t.sport, t.dport, int(t.flags)
                plen = len(t.data)
            elif isinstance(t, dpkt.udp.UDP):
                proto = 1  # UDP
                sport, dport = t.sport, t.dport
                plen = len(t.data)
            elif isinstance(t, dpkt.icmp.ICMP):
                proto = 2  # ICMP
                plen = len(bytes(t.data))
            else:
                proto = 3  # OTHER
                plen = 0
            try:
                src = socket.inet_ntoa(ip.src)
                dst = socket.inet_ntoa(ip.dst)
            except Exception:
                continue
            rows.append((float(ts), src, dst, sport, dport, proto, flags,
                         int(ip.len), int(plen), label))
    return rows


def build_packets_df(raw_root: Path, max_per_class: int) -> pd.DataFrame:
    all_rows: list[tuple] = []
    for cls_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = cls_dir.name
        for pcap in sorted(cls_dir.glob("*.pcap")):
            print(f"  parsing {label}/{pcap.name} (cap={max_per_class:,}) …", flush=True)
            rows = parse_pcap(pcap, label, max_per_class)
            all_rows.extend(rows)
            print(f"    -> {len(rows):,} packets", flush=True)
    df = pd.DataFrame(all_rows, columns=[
        "ts", "src_ip", "dst_ip", "src_port", "dst_port", "proto",
        "flags", "ip_len", "plen", "label"])
    return df


def assign_flows(df: pd.DataFrame, timeout: float = 30.0) -> pd.DataFrame:
    """Bidirectional 5-tuple flows with idle-timeout segmentation (no packet cap)."""
    # canonical bidirectional key: order the two endpoints
    a = df["src_ip"].astype(str) + ":" + df["src_port"].astype(str)
    b = df["dst_ip"].astype(str) + ":" + df["dst_port"].astype(str)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    df["_key"] = df["label"].astype(str) + "|" + lo + "|" + hi + "|" + df["proto"].astype(str)
    df["_is_fwd"] = (a <= b)  # forward = canonical-low endpoint is the source

    df.sort_values(["_key", "ts"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    prev_ts = df.groupby("_key", sort=False)["ts"].shift(1)
    gap = df["ts"] - prev_ts
    new_seg = (prev_ts.isna() | (gap > timeout)).astype(np.int64)
    # vectorized flow id: cumulative segment count within key + key
    seg_idx = new_seg.groupby(df["_key"], sort=False).cumsum()
    df["flow_id"] = df["_key"] + "#" + seg_idx.astype(str)
    return df


def _stats(g, col):
    s = g[col]
    return s.mean(), s.std().fillna(0.0), s.min(), s.max()


def build_flow_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict]:
    df["iat"] = df.groupby("flow_id", sort=False)["ts"].diff()
    df["syn"] = ((df["flags"] & _SYN) > 0) & (df["proto"] == 0)
    df["ack"] = ((df["flags"] & _ACK) > 0) & (df["proto"] == 0)
    df["fin"] = ((df["flags"] & _FIN) > 0) & (df["proto"] == 0)
    df["rst"] = ((df["flags"] & _RST) > 0) & (df["proto"] == 0)
    df["psh"] = ((df["flags"] & _PSH) > 0) & (df["proto"] == 0)
    df["urg"] = ((df["flags"] & _URG) > 0) & (df["proto"] == 0)

    g = df.groupby("flow_id", sort=False)
    f = pd.DataFrame(index=list(g.groups.keys()))
    f.index.name = "flow_id"

    f["label"] = g["label"].first()
    f["proto"] = g["proto"].first()
    f["src_port"] = g["src_port"].first()
    f["dst_port"] = g["dst_port"].first()
    f["tot_pkts"] = g.size()
    f["fwd_pkts"] = g["_is_fwd"].sum()
    f["bwd_pkts"] = f["tot_pkts"] - f["fwd_pkts"]
    f["tot_bytes"] = g["ip_len"].sum()
    f["dur"] = (g["ts"].max() - g["ts"].min())

    for c in ["ip_len", "plen"]:
        f[f"{c}_mean"], f[f"{c}_std"], f[f"{c}_min"], f[f"{c}_max"] = _stats(g, c)
    f["iat_mean"] = g["iat"].mean().fillna(0.0)
    f["iat_std"] = g["iat"].std().fillna(0.0)
    f["iat_min"] = g["iat"].min().fillna(0.0)
    f["iat_max"] = g["iat"].max().fillna(0.0)

    for fl in ["syn", "ack", "fin", "rst", "psh", "urg"]:
        f[f"n_{fl}"] = g[fl].sum()
        f[f"r_{fl}"] = f[f"n_{fl}"] / f["tot_pkts"].clip(lower=1)

    dur = f["dur"].clip(lower=1e-6)
    f["bytes_per_s"] = f["tot_bytes"] / dur
    f["pkts_per_s"] = f["tot_pkts"] / dur
    f["down_up_ratio"] = f["bwd_pkts"] / f["fwd_pkts"].clip(lower=1)
    f["mean_pkt_size"] = f["tot_bytes"] / f["tot_pkts"].clip(lower=1)

    # per-source fan-out context within the capture (scan signal)
    df_first = df.drop_duplicates("flow_id")[["flow_id", "label", "src_ip"]]
    src_grp = df.groupby(["label", "src_ip"], sort=False)
    fan = pd.DataFrame({
        "fan_dst_ip": src_grp["dst_ip"].nunique(),
        "fan_dst_port": src_grp["dst_port"].nunique(),
        "fan_syn": src_grp["syn"].sum(),
        "fan_pkts": src_grp.size(),
    }).reset_index()
    df_first = df_first.merge(fan, on=["label", "src_ip"], how="left").set_index("flow_id")
    for c in ["fan_dst_ip", "fan_dst_port", "fan_syn", "fan_pkts"]:
        f[c] = df_first[c].reindex(f.index).fillna(0.0)

    label_values = sorted(f["label"].astype(str).unique().tolist())
    label_map = {v: i for i, v in enumerate(label_values)}
    label_names = {i: v for v, i in label_map.items()}
    y = f["label"].astype(str).map(label_map).to_numpy(np.int64)
    X = f.drop(columns=["label"])
    return X, y, label_names


def evaluate(X: np.ndarray, y: np.ndarray, label_names: dict, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_sample_weight

    X = np.nan_to_num(X.astype(np.float32), posinf=0.0, neginf=0.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sw = compute_sample_weight("balanced", ytr)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         l2_regularization=1.0, early_stopping=True,
                                         random_state=seed)
    t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sw)
    pred = clf.predict(Xte)
    macro = float(f1_score(yte, pred, average="macro"))
    acc = float(accuracy_score(yte, pred))
    rep = classification_report(yte, pred, labels=sorted(label_names),
                                target_names=[label_names[i] for i in sorted(label_names)],
                                output_dict=True, zero_division=0)
    per_class = {label_names[i]: round(rep[label_names[i]]["f1-score"], 4) for i in sorted(label_names)}
    cm = confusion_matrix(yte, pred, labels=sorted(label_names))
    cmrow = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    print(f"\n=== flags+fanout flow features  n_feat={X.shape[1]}  fit={time.time()-t0:.1f}s ===")
    print(f"  accuracy = {acc:.4f}   macro_f1 = {macro:.4f}")
    print("  per-class f1 (sorted):")
    for cls, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {cls:28s} {v:.4f}")
    print("  confusion (true -> top-3 predicted):")
    cls_list = sorted(label_names)
    for i, c in enumerate(cls_list):
        order = np.argsort(-cmrow[i])[:3]
        tops = ", ".join(f"{label_names[cls_list[j]]}:{cmrow[i][j]*100:.0f}%" for j in order)
        print(f"    {label_names[c]:28s} -> {tops}")
    return {"accuracy": acc, "macro_f1": macro, "per_class_f1": per_class, "n_features": int(X.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument("--out-json", default="outputs/diagnostics/flags_flow_results.json")
    ap.add_argument("--max-per-class", type=int, default=500_000,
                    help="Cap packets parsed per class (validation speed).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] Parsing raw PCAPs (ALL packets incl. control/ICMP) …", flush=True)
    df = build_packets_df(Path(args.raw_root), args.max_per_class)
    print(f"      total packets parsed = {len(df):,}", flush=True)
    proto_counts = df["proto"].value_counts().to_dict()
    print(f"      proto mix (0=TCP,1=UDP,2=ICMP,3=OTHER): {proto_counts}", flush=True)

    print("[2/4] Assigning bidirectional flows …", flush=True)
    df = assign_flows(df)
    print(f"      flows = {df['flow_id'].nunique():,}", flush=True)

    print("[3/4] Building flow features (flags + fan-out) …", flush=True)
    X, y, label_names = build_flow_features(df)
    print(f"      feature matrix = {X.shape}", flush=True)

    print("[4/4] Training HistGBM …", flush=True)
    res = evaluate(X.to_numpy(np.float32), y, label_names, args.seed)

    print(f"\nPrior ceilings: payload-hist=0.475, rich-flow=0.350, GNN~0.12")
    print(f"Total wall: {time.time()-t0:.1f}s")
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "result": res,
                               "label_names": label_names}, indent=2))
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
