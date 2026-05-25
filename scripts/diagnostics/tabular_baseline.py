"""Tabular ceiling test for the NT114 IDS dataset.

PURPOSE
-------
The HGT pipeline is stuck (overfit probe train_acc~0.40, v8.5 full-run
train_acc~0.15). The overfit probe proves the *representation* fed to the model
cannot separate the 13 classes. This script answers the upstream question
*cheaply and independently of the graph*:

    "Given the SAME flows and SAME labels the GNN sees, what macro-F1 can a
     strong tabular model reach from (a) the current 6 flow features, vs
     (b) rich flow statistics, vs (c) rich flow stats + payload byte content?"

It reconstructs flows with the project's own `build_graph_csv_tables` (so flow
boundaries and labels are IDENTICAL to the graph artifact), engineers three
feature sets, and trains a `HistGradientBoostingClassifier` on each.

Interpretation
--------------
* base6 macro-F1 ~ current GNN (~0.12-0.20)  -> sanity check: features are the cap.
* rich-flow >> base6                          -> behavioural classes (Recon/DDoS)
                                                 are separable; flow features were
                                                 the bottleneck.
* +payload >> rich-flow on web attacks        -> content matters; the semantic
                                                 (text-of-hex) embedding was
                                                 throwing the signal away.
* even +payload stays low / labels look noisy -> investigate labels, not features.

It does NOT touch the graph artifact or training pipeline; it only reads the
interim metadata.csv + payload_256.npy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the project's flow definition so flows/labels match the graph artifact.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from graphslm_ids.offline.preprocessing.graph_csv_builder import build_graph_csv_tables  # noqa: E402


# ── feature engineering ────────────────────────────────────────────────────────

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(b > 0, a / np.where(b == 0, 1, b), 0.0)


def build_flow_features(
    flow_nodes: pd.DataFrame,
    packet_nodes: pd.DataFrame,
    protocol_map: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Return (base6_df, rich_df, feature_groups) indexed by flow_id.

    base6  = exactly the current model's flow_x.
    rich   = base6 + per-flow packet-length / IAT / directionality / scan-spread
             statistics derived from packet-level rows.
    """
    pk = packet_nodes.copy()
    pk["timestamp"] = pd.to_numeric(pk["timestamp"], errors="coerce").fillna(0.0)
    pk["payload_len_raw"] = pd.to_numeric(pk["payload_len_raw"], errors="coerce").fillna(0).astype(np.float64)

    # flow reference src_ip (first packet) → forward direction marker
    flow_ref = flow_nodes.set_index("flow_id")["src_ip"].astype(str)
    pk["_flow_src"] = pk["flow_id"].map(flow_ref).astype(str)
    pk["_is_fwd"] = (pk["src_ip"].astype(str) == pk["_flow_src"])

    # inter-arrival time within each flow (rows already time-ordered per flow)
    pk["_iat"] = pk.groupby("flow_id", sort=False)["timestamp"].diff()

    g = pk.groupby("flow_id", sort=False)
    rich = pd.DataFrame(index=flow_nodes["flow_id"].values)
    rich.index.name = "flow_id"

    # packet-length distribution
    plen = g["payload_len_raw"]
    rich["plen_mean"] = plen.mean()
    rich["plen_std"] = plen.std().fillna(0.0)
    rich["plen_min"] = plen.min()
    rich["plen_max"] = plen.max()
    rich["plen_sum"] = plen.sum()

    # inter-arrival-time distribution (seconds)
    iat = g["_iat"]
    rich["iat_mean"] = iat.mean().fillna(0.0)
    rich["iat_std"] = iat.std().fillna(0.0)
    rich["iat_min"] = iat.min().fillna(0.0)
    rich["iat_max"] = iat.max().fillna(0.0)

    # directionality (fwd = same src as flow's first packet)
    rich["fwd_pkts"] = g["_is_fwd"].sum()
    rich["tot_pkts"] = g.size()
    rich["bwd_pkts"] = rich["tot_pkts"] - rich["fwd_pkts"]
    fwd_bytes = pk.loc[pk["_is_fwd"]].groupby("flow_id", sort=False)["payload_len_raw"].sum()
    rich["fwd_bytes"] = fwd_bytes.reindex(rich.index).fillna(0.0)
    rich["bwd_bytes"] = rich["plen_sum"] - rich["fwd_bytes"]
    rich["down_up_ratio"] = _safe_div(rich["bwd_pkts"].to_numpy(float), rich["fwd_pkts"].to_numpy(float))

    # scan-spread signal: how many distinct dst ports / dst ips / src ports
    rich["n_uniq_dst_port"] = g["dst_port"].nunique()
    rich["n_uniq_dst_ip"] = g["dst_ip"].nunique()
    rich["n_uniq_src_port"] = g["src_port"].nunique()

    # rate features
    fn = flow_nodes.set_index("flow_id")
    dur = pd.to_numeric(fn["duration_seconds"], errors="coerce").fillna(0.0).reindex(rich.index)
    rich["bytes_per_s"] = _safe_div(rich["plen_sum"].to_numpy(float), dur.to_numpy(float))
    rich["pkts_per_s"] = _safe_div(rich["tot_pkts"].to_numpy(float), dur.to_numpy(float))
    rich["mean_pkt_size"] = _safe_div(rich["plen_sum"].to_numpy(float), rich["tot_pkts"].to_numpy(float))

    # ── base6: exactly the current flow_x ──
    base = pd.DataFrame(index=flow_nodes["flow_id"].values)
    base.index.name = "flow_id"
    base["packet_count"] = pd.to_numeric(fn["packet_count"], errors="coerce").fillna(0.0).reindex(base.index).to_numpy()
    base["total_payload_bytes"] = pd.to_numeric(fn["total_payload_bytes"], errors="coerce").fillna(0.0).reindex(base.index).to_numpy()
    base["duration_seconds"] = dur.to_numpy()
    base["src_port"] = pd.to_numeric(fn["src_port"], errors="coerce").fillna(0.0).reindex(base.index).to_numpy()
    base["dst_port"] = pd.to_numeric(fn["dst_port"], errors="coerce").fillna(0.0).reindex(base.index).to_numpy()
    base["protocol_id"] = fn["protocol"].astype(str).map(protocol_map).fillna(-1).reindex(base.index).to_numpy()

    # rich set INCLUDES base6 so it strictly dominates in information.
    rich_full = pd.concat([base, rich], axis=1)

    groups = {
        "base6": list(base.columns),
        "rich": list(rich_full.columns),
    }
    return base, rich_full, groups


def build_payload_histograms(
    flow_ids_sample: np.ndarray,
    packet_nodes: pd.DataFrame,
    payload_path: Path,
    batch_rows: int = 500_000,
) -> tuple[np.ndarray, list[str]]:
    """Per-flow normalized 256-bin byte histogram (+ entropy, printable ratio).

    Aggregates the byte distribution over all packets of each sampled flow by
    streaming payload_256.npy in row batches (memory-bounded).
    """
    n_flows = flow_ids_sample.shape[0]
    flow_to_local = {fid: i for i, fid in enumerate(flow_ids_sample.tolist())}

    pk = packet_nodes[["flow_id", "payload_row_index"]]
    pk = pk[pk["flow_id"].isin(flow_to_local)]
    local_idx = pk["flow_id"].map(flow_to_local).to_numpy(np.int64)
    pay_idx = pk["payload_row_index"].to_numpy(np.int64)

    # sort by payload row for sequential mmap reads
    order = np.argsort(pay_idx, kind="stable")
    pay_idx = pay_idx[order]
    local_idx = local_idx[order]

    payload = np.load(str(payload_path), mmap_mode="r")
    hist = np.zeros((n_flows, 256), dtype=np.float64)

    n = pay_idx.shape[0]
    for start in range(0, n, batch_rows):
        sl = slice(start, min(start + batch_rows, n))
        rows_local = local_idx[sl]
        bytes_batch = np.asarray(payload[pay_idx[sl]], dtype=np.int64)  # (B,256)
        b = bytes_batch.shape[0]
        combined = (rows_local[:, None] * 256 + bytes_batch).ravel()
        binc = np.bincount(combined, minlength=n_flows * 256)
        hist += binc.reshape(n_flows, 256).astype(np.float64)

    totals = hist.sum(axis=1, keepdims=True)
    hist_norm = hist / np.where(totals == 0, 1.0, totals)

    # derived content signals
    eps = 1e-12
    entropy = -(hist_norm * np.log2(hist_norm + eps)).sum(axis=1)
    printable = hist_norm[:, 0x20:0x7f].sum(axis=1)  # fraction printable ASCII
    zero_frac = hist_norm[:, 0]

    feats = np.concatenate(
        [hist_norm, entropy[:, None], printable[:, None], zero_frac[:, None]], axis=1
    ).astype(np.float32)
    names = [f"byte_{i:03d}" for i in range(256)] + ["pl_entropy", "pl_printable", "pl_zerofrac"]
    return feats, names


# ── modeling ────────────────────────────────────────────────────────────────────

def stratified_subsample(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.shape[0] > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def evaluate_feature_set(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    label_names: dict[int, str],
    seed: int,
) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    from sklearn.utils.class_weight import compute_sample_weight

    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    sw = compute_sample_weight("balanced", ytr)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=None, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=seed,
    )
    t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sw)
    fit_s = time.time() - t0

    pred = clf.predict(Xte)
    macro = float(f1_score(yte, pred, average="macro"))
    acc = float(accuracy_score(yte, pred))
    rep = classification_report(
        yte, pred,
        labels=sorted(label_names),
        target_names=[label_names[i] for i in sorted(label_names)],
        output_dict=True, zero_division=0,
    )
    per_class = {label_names[i]: round(rep[label_names[i]]["f1-score"], 4) for i in sorted(label_names)}
    print(f"\n=== [{name}]  n_features={X.shape[1]}  fit={fit_s:.1f}s ===")
    print(f"  accuracy   = {acc:.4f}")
    print(f"  macro_f1   = {macro:.4f}")
    print("  per-class f1:")
    for cls, f1 in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {cls:28s} {f1:.4f}")
    return {"feature_set": name, "n_features": int(X.shape[1]),
            "accuracy": acc, "macro_f1": macro, "per_class_f1": per_class}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-csv", default="data/interim/payload_dataset_14gb/metadata.csv")
    ap.add_argument("--payload-npy", default="data/interim/payload_dataset_14gb/payload_256.npy")
    ap.add_argument("--out-json", default="outputs/diagnostics/tabular_baseline_results.json")
    ap.add_argument("--flow-timeout-seconds", type=float, default=30.0)
    ap.add_argument("--max-packets-per-flow", type=int, default=20)
    ap.add_argument("--max-per-class", type=int, default=20000,
                    help="Stratified cap on flows/class for speed (0 = use all).")
    ap.add_argument("--with-payload", action="store_true", default=True)
    ap.add_argument("--no-payload", dest="with_payload", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t_start = time.time()
    print(f"[1/5] Loading metadata: {args.metadata_csv}", flush=True)
    md = pd.read_csv(
        args.metadata_csv,
        dtype={
            "pcap_file": "category", "label": "category",
            "src_ip": "category", "dst_ip": "category", "protocol": "category",
            "packet_index": np.int32, "src_port": np.int32, "dst_port": np.int32,
            "payload_len_raw": np.int32,
        },
    )
    print(f"      packets={len(md):,}  classes={md['label'].nunique()}", flush=True)

    print("[2/5] Reconstructing flows (project flow definition) …", flush=True)
    tables = build_graph_csv_tables(
        md, flow_timeout_seconds=args.flow_timeout_seconds,
        max_packets_per_flow=args.max_packets_per_flow,
    )
    del md
    flow_nodes = tables.flow_nodes.reset_index(drop=True)
    packet_nodes = tables.packet_nodes.reset_index(drop=True)
    print(f"      flows={len(flow_nodes):,}  packets_in_graph={len(packet_nodes):,}", flush=True)

    # label encoding (sorted-string, identical to graph_artifact_builder._encode_series)
    label_values = sorted({str(v) for v in flow_nodes["label"].tolist()})
    label_map = {v: i for i, v in enumerate(label_values)}
    label_names = {i: v for v, i in label_map.items()}
    y_all = flow_nodes["label"].astype(str).map(label_map).to_numpy(np.int64)

    protocol_values = sorted({str(v) for v in flow_nodes["protocol"].tolist()})
    protocol_map = {v: i for i, v in enumerate(protocol_values)}

    print("[3/5] Engineering flow features …", flush=True)
    base_df, rich_df, _ = build_flow_features(flow_nodes, packet_nodes, protocol_map)

    # subsample flows for speed / memory
    if args.max_per_class > 0:
        sel = stratified_subsample(y_all, args.max_per_class, args.seed)
    else:
        sel = np.arange(y_all.shape[0])
    y = y_all[sel]
    flow_ids_sample = flow_nodes["flow_id"].to_numpy()[sel]
    print(f"      training on {sel.shape[0]:,} flows "
          f"(<= {args.max_per_class}/class)", flush=True)

    base_X = base_df.to_numpy(np.float32)[sel]
    rich_X = rich_df.to_numpy(np.float32)[sel]

    results = []
    results.append(evaluate_feature_set("base6 (current flow_x)", base_X, y, label_names, args.seed))
    results.append(evaluate_feature_set("rich-flow", rich_X, y, label_names, args.seed))

    if args.with_payload:
        print("\n[4/5] Building payload byte histograms …", flush=True)
        pay_X, _ = build_payload_histograms(
            flow_ids_sample, packet_nodes, Path(args.payload_npy),
        )
        rich_pay_X = np.concatenate([rich_X, pay_X], axis=1)
        results.append(evaluate_feature_set("rich-flow + payload", rich_pay_X, y, label_names, args.seed))

    print("\n[5/5] Summary (macro-F1):", flush=True)
    for r in results:
        print(f"  {r['feature_set']:28s} macro_f1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f}")
    print(f"\nGNN reference (v8.5 full run): macro_f1~0.12, train_acc~0.15")
    print(f"Total wall: {time.time()-t_start:.1f}s")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args),
        "n_flows_total": int(y_all.shape[0]),
        "n_flows_trained": int(sel.shape[0]),
        "label_names": label_names,
        "results": results,
    }, indent=2))
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
