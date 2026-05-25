"""Full-scale tabular ceiling on v2 features **including content evidence**.

The vanilla v2 ceiling test trains on the ~73 flow features only and tops out at
macro-F1 ~ 0.79 once the label-keyed fan-out leak is removed. The content-based
attack classes (XSS, SqlInjection, CommandInjection, Uploading_Attack) are
indistinguishable from each other purely by flow shape -- they share dst_port,
byte budget and timing. To recover the 0.97 ceiling **deployment-honestly**, we
add per-flow content evidence that *would actually be available at runtime*:

  1. Payload-signature hits, aggregated per flow. For each MITRE technique that
     any signature rule references (~9 IDs), we add two columns: ``sig_hit_X``
     (binary, any-packet) and ``sig_cnt_X`` (count of packets that triggered).
     This is exactly the same evidence the runtime SLM consumes when emitting
     a grounded report -- here we use it as a feature instead of an edge.

  2. Lightweight per-packet payload aggregates: entropy, printable byte ratio,
     non-null/null byte counts, HTTP request markers, URL length, special-char
     count. Mean / max per flow.

Neither block uses the ground-truth label at training time, so the ceiling
holds at deployment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.v2.cli import (
    _build_payload_matrix_from_pcaps,
)
from graphslm_ids.offline.preprocessing.v2.extractor import extract_packets_dir
from graphslm_ids.offline.preprocessing.v2.flows import (
    assign_flows,
    build_flow_features,
)
from graphslm_ids.offline.preprocessing.v2.signatures import (
    match_flow_signatures,
    match_payload_signatures,
)
from graphslm_ids.offline.preprocessing.v2._signature_rules import (
    ALL_REFERENCED_TECHNIQUES,
)

_HTTP_METHOD_PREFIX = (b"GET ", b"POST ", b"PUT ", b"DELETE ", b"HEAD ",
                       b"OPTIONS ", b"PATCH ")
_SPECIAL_CHARS = set(b"<>'\";&%=(){}[]\\")


def _stratified_subsample(
    labels: np.ndarray, max_per_class: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if max_per_class > 0 and idx.shape[0] > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def _packet_payload_aggregates(
    payload_row: np.ndarray, payload_len: int
) -> np.ndarray:
    """Cheap per-packet payload aggregates: 8 floats.

    Order: [entropy, printable_ratio, zero_ratio, n_special, is_http_req,
    is_http_resp, url_length, n_null_bytes].
    """
    out = np.zeros(8, dtype=np.float32)
    pl = int(min(max(payload_len, 0), payload_row.shape[0]))
    if pl == 0:
        return out
    buf = payload_row[:pl]
    hist = np.bincount(buf, minlength=256).astype(np.float64)
    total = float(pl)
    p = hist / max(total, 1.0)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0
    printable_ratio = float(p[0x20:0x7F].sum())
    zero_ratio = float(p[0])
    n_special = float(sum(1 for b in buf.tolist() if b in _SPECIAL_CHARS))
    raw = buf.tobytes()
    is_req = float(any(raw.startswith(m) for m in _HTTP_METHOD_PREFIX))
    is_resp = float(raw.startswith(b"HTTP/"))
    url_length = 0.0
    if is_req:
        try:
            first_line = raw.split(b"\r\n", 1)[0]
            parts = first_line.split(b" ")
            if len(parts) >= 2:
                url_length = float(len(parts[1]))
        except Exception:
            pass
    n_null = float(raw.count(b"\x00"))
    out[:] = (
        entropy,
        printable_ratio,
        zero_ratio,
        n_special,
        is_req,
        is_resp,
        url_length,
        n_null,
    )
    return out


_PAYLOAD_AGG_NAMES = (
    "pl_entropy_mean",
    "pl_entropy_max",
    "pl_printable_mean",
    "pl_zero_ratio_mean",
    "pl_n_special_mean",
    "pl_n_special_max",
    "pl_is_http_req_frac",
    "pl_is_http_resp_frac",
    "pl_url_length_max",
    "pl_url_length_mean",
    "pl_n_null_sum",
)


def _aggregate_payload_per_flow(
    packets_df: pd.DataFrame, payload_matrix: np.ndarray, flow_ids: np.ndarray
) -> pd.DataFrame:
    """For each packet compute the 8 payload aggregates, then per-flow stats."""
    n = len(packets_df)
    raw_feats = np.zeros((n, 8), dtype=np.float32)
    payload_lens = packets_df["payload_len"].to_numpy(dtype=np.int64)
    for i in range(n):
        pl = int(payload_lens[i])
        if pl <= 0:
            continue
        raw_feats[i] = _packet_payload_aggregates(payload_matrix[i], pl)

    cols = (
        "entropy",
        "printable",
        "zero_ratio",
        "n_special",
        "is_req",
        "is_resp",
        "url_len",
        "n_null",
    )
    df = pd.DataFrame(raw_feats, columns=cols)
    df["flow_id"] = flow_ids
    g = df.groupby("flow_id", sort=False)
    out = pd.DataFrame(index=g.size().index)
    out["pl_entropy_mean"] = g["entropy"].mean()
    out["pl_entropy_max"] = g["entropy"].max()
    out["pl_printable_mean"] = g["printable"].mean()
    out["pl_zero_ratio_mean"] = g["zero_ratio"].mean()
    out["pl_n_special_mean"] = g["n_special"].mean()
    out["pl_n_special_max"] = g["n_special"].max()
    out["pl_is_http_req_frac"] = g["is_req"].mean()
    out["pl_is_http_resp_frac"] = g["is_resp"].mean()
    out["pl_url_length_max"] = g["url_len"].max()
    out["pl_url_length_mean"] = g["url_len"].mean()
    out["pl_n_null_sum"] = g["n_null"].sum()
    return out.astype(np.float64)


def _aggregate_byte_histogram_per_flow(
    packets_df: pd.DataFrame,
    payload_matrix: np.ndarray,
    flow_index: pd.Index,
) -> pd.DataFrame:
    """Mean 256-bin byte histogram per flow (averaged over its packets).

    Strong, deterministic content fingerprint: HTTP traffic differs from
    binary uploads differs from probe payloads differs from encrypted streams.
    Each row sums to ~1 over the 256 bins (only summed for packets with
    payload_len > 0; flows with no payload get a zero vector).
    """
    payload_lens = packets_df["payload_len"].to_numpy(dtype=np.int64)
    flow_ids_per_pkt = packets_df["flow_id"].to_numpy()
    fid_to_row = {fid: i for i, fid in enumerate(flow_index)}

    sums = np.zeros((len(flow_index), 256), dtype=np.float64)
    counts = np.zeros(len(flow_index), dtype=np.int64)
    for i in range(len(packets_df)):
        pl = int(payload_lens[i])
        if pl <= 0:
            continue
        r = fid_to_row.get(flow_ids_per_pkt[i])
        if r is None:
            continue
        # Normalized per-packet histogram, then accumulate; final mean = sum / count.
        buf = payload_matrix[i, :pl]
        hist = np.bincount(buf, minlength=256).astype(np.float64)
        hist /= max(float(pl), 1.0)
        sums[r] += hist
        counts[r] += 1
    nz = counts > 0
    sums[nz] /= counts[nz, None]
    cols = [f"hb_{i:03d}" for i in range(256)]
    return pd.DataFrame(sums, index=flow_index, columns=cols)


def _aggregate_ngram_per_flow(
    packets_df: pd.DataFrame,
    payload_matrix: np.ndarray,
    flow_index: pd.Index,
    n: int = 4,
    n_buckets: int = 256,
) -> pd.DataFrame:
    """Mean hashed n-gram histogram per flow (sequence content fingerprint).

    Each n-byte window is hashed to one of ``n_buckets`` via md5 mod n_buckets.
    For each packet, the per-window counts are normalized by the number of
    windows; flow-level vector is the mean of per-packet vectors over packets
    with non-empty payload. This captures sequence patterns (e.g., ``<script``,
    ``OR 1=1``, ``; rm -rf``) that pure byte-frequency histograms collapse.
    """
    import hashlib

    payload_lens = packets_df["payload_len"].to_numpy(dtype=np.int64)
    flow_ids_per_pkt = packets_df["flow_id"].to_numpy()
    fid_to_row = {fid: i for i, fid in enumerate(flow_index)}

    sums = np.zeros((len(flow_index), n_buckets), dtype=np.float64)
    counts = np.zeros(len(flow_index), dtype=np.int64)
    for i in range(len(packets_df)):
        pl = int(payload_lens[i])
        if pl < n:
            continue
        r = fid_to_row.get(flow_ids_per_pkt[i])
        if r is None:
            continue
        raw = payload_matrix[i, :pl].tobytes()
        n_windows = pl - n + 1
        per_pkt = np.zeros(n_buckets, dtype=np.float64)
        for j in range(n_windows):
            h = hashlib.md5(raw[j : j + n]).digest()
            bucket = int.from_bytes(h[:4], "big") % n_buckets
            per_pkt[bucket] += 1.0
        per_pkt /= float(n_windows)
        sums[r] += per_pkt
        counts[r] += 1
    nz = counts > 0
    sums[nz] /= counts[nz, None]
    cols = [f"hng_{i:03d}" for i in range(n_buckets)]
    return pd.DataFrame(sums, index=flow_index, columns=cols)


def _aggregate_signature_hits_per_flow(
    packets_df: pd.DataFrame,
    payload_matrix: np.ndarray,
    flow_features: pd.DataFrame,
) -> pd.DataFrame:
    """Per-flow signature evidence: per-technique binary hit + count."""
    techniques = sorted(ALL_REFERENCED_TECHNIQUES)
    tid_to_col = {t: i for i, t in enumerate(techniques)}
    n_t = len(techniques)
    n_pkts = len(packets_df)
    payload_lens = packets_df["payload_len"].to_numpy(dtype=np.int64)
    flow_ids_per_pkt = packets_df["flow_id"].to_numpy()

    # Per-packet payload-signature counter, sparse so we only allocate hit rows.
    # We accumulate two dicts keyed by flow_id: {flow_id: counts(np.float32[n_t])}.
    per_flow_counts: dict[str, np.ndarray] = {}
    for i in range(n_pkts):
        pl = int(payload_lens[i])
        if pl <= 0:
            continue
        raw = payload_matrix[i, :pl].tobytes()
        if not raw:
            continue
        hits = match_payload_signatures(raw)
        if not hits:
            continue
        fid = flow_ids_per_pkt[i]
        arr = per_flow_counts.setdefault(fid, np.zeros(n_t, dtype=np.float32))
        for tid, _w in hits:
            j = tid_to_col.get(tid)
            if j is not None:
                arr[j] += 1.0

    # Flow-level signatures fire on the aggregated row.
    flow_hits = np.zeros((len(flow_features), n_t), dtype=np.float32)
    for row_i, (fid, row) in enumerate(flow_features.iterrows()):
        hits = match_flow_signatures(row)
        for tid, w in hits:
            j = tid_to_col.get(tid)
            if j is not None:
                # Treat the flow signature as one additional "hit".
                flow_hits[row_i, j] += float(w)

    # Compose the per-flow count matrix.
    cnt = np.zeros((len(flow_features), n_t), dtype=np.float32)
    fid_to_row = {fid: i for i, fid in enumerate(flow_features.index)}
    for fid, arr in per_flow_counts.items():
        r = fid_to_row.get(fid)
        if r is not None:
            cnt[r] = arr
    cnt += flow_hits  # flow signatures contribute to the same technique column

    cnt_df = pd.DataFrame(
        cnt,
        index=flow_features.index,
        columns=[f"sig_cnt_{t.replace('.', '_')}" for t in techniques],
    )
    hit_df = (cnt_df > 0).astype(np.float64)
    hit_df.columns = [f"sig_hit_{t.replace('.', '_')}" for t in techniques]
    return pd.concat([cnt_df.astype(np.float64), hit_df], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument(
        "--out-json", default="outputs/v2/v2_ceiling_content_results.json"
    )
    ap.add_argument("--max-per-class-packets", type=int, default=500_000)
    ap.add_argument("--max-per-class-flows", type=int, default=30_000)
    ap.add_argument("--payload-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    cap = None if args.max_per_class_packets == 0 else args.max_per_class_packets
    print(f"[ceiling+] parsing pcaps under {args.raw_root} (cap={cap}) ...",
          flush=True)
    packets_df = extract_packets_dir(Path(args.raw_root), max_per_class=cap)
    print(f"[ceiling+] parsed packets = {len(packets_df):,}", flush=True)

    print("[ceiling+] second pass: payload bytes ...", flush=True)
    payload_matrix = _build_payload_matrix_from_pcaps(
        Path(args.raw_root), packets_df, args.payload_length
    )
    print(f"[ceiling+] payload matrix shape = {payload_matrix.shape}", flush=True)

    print("[ceiling+] assembling bidirectional flows ...", flush=True)
    # `assign_flows` sort+reset_index drops the original packet positions, so
    # we stash them in an explicit column FIRST. Then ``_orig_idx`` lets us
    # slice the payload matrix in the tagged order.
    packets_df = packets_df.reset_index(drop=True).copy()
    packets_df["_orig_idx"] = np.arange(len(packets_df), dtype=np.int64)
    tagged = assign_flows(packets_df)
    feats, _meta = build_flow_features(tagged)
    print(
        f"[ceiling+] flows = {len(feats):,}  flow_features = {feats.shape[1] - 1}",
        flush=True,
    )

    payload_matrix_tagged = payload_matrix[tagged["_orig_idx"].to_numpy()]
    tagged_compact = tagged.reset_index(drop=True)
    print("[ceiling+] aggregating per-flow payload features ...", flush=True)
    pl_feats = _aggregate_payload_per_flow(
        tagged_compact, payload_matrix_tagged, tagged_compact["flow_id"].to_numpy()
    )
    pl_feats = pl_feats.reindex(feats.index).fillna(0.0)
    print(f"[ceiling+] payload aggregates  cols = {pl_feats.shape[1]}", flush=True)

    print("[ceiling+] computing per-flow signature evidence ...", flush=True)
    sig_feats = _aggregate_signature_hits_per_flow(
        tagged_compact, payload_matrix_tagged, feats
    )
    print(f"[ceiling+] signature evidence  cols = {sig_feats.shape[1]}", flush=True)

    print("[ceiling+] aggregating per-flow byte histogram ...", flush=True)
    hist_feats = _aggregate_byte_histogram_per_flow(
        tagged_compact, payload_matrix_tagged, feats.index
    )
    print(f"[ceiling+] byte histogram cols = {hist_feats.shape[1]}", flush=True)

    print("[ceiling+] aggregating per-flow hashed 4-gram histogram ...", flush=True)
    ngram_feats = _aggregate_ngram_per_flow(
        tagged_compact, payload_matrix_tagged, feats.index, n=4, n_buckets=256
    )
    print(f"[ceiling+] n-gram cols = {ngram_feats.shape[1]}", flush=True)

    feats_full = pd.concat([feats, pl_feats, sig_feats, hist_feats, ngram_feats], axis=1)
    y_all = feats_full["label"].astype("category")
    label_names = list(y_all.cat.categories)
    y = y_all.cat.codes.to_numpy(dtype=np.int64)
    X = feats_full.drop(columns=["label"]).to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = [c for c in feats_full.columns if c != "label"]
    print(
        f"[ceiling+] total features (flow + payload + signatures) = {X.shape[1]}",
        flush=True,
    )

    sel = _stratified_subsample(y, args.max_per_class_flows, args.seed)
    X_sub, y_sub = X[sel], y[sel]
    print(f"[ceiling+] training HGBM on {sel.shape[0]:,} flows", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    Xtr, Xte, ytr, yte = train_test_split(
        X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=args.seed
    )
    sw = compute_sample_weight("balanced", ytr)
    clf = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.06,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=args.seed,
    )
    fit_t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sw)
    fit_s = time.time() - fit_t0
    pred = clf.predict(Xte)
    macro = float(f1_score(yte, pred, average="macro"))
    acc = float(accuracy_score(yte, pred))
    rep = classification_report(
        yte, pred, labels=list(range(len(label_names))),
        target_names=label_names, output_dict=True, zero_division=0,
    )
    per_class = {n: round(rep[n]["f1-score"], 4) for n in label_names}

    print(f"\n[ceiling+] fit={fit_s:.1f}s  acc={acc:.4f}  macro_f1={macro:.4f}")
    print("[ceiling+] per-class f1 (sorted):")
    for cls, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:28s} {v:.4f}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "n_packets_parsed": int(len(packets_df)),
                "n_flows_total": int(len(feats)),
                "n_flows_trained": int(sel.shape[0]),
                "n_features": int(X.shape[1]),
                "feature_block_sizes": {
                    "flow": int(feats.shape[1] - 1),
                    "payload": int(pl_feats.shape[1]),
                    "signatures": int(sig_feats.shape[1]),
                    "byte_histogram": int(hist_feats.shape[1]),
                    "ngram_4_256": int(ngram_feats.shape[1]),
                },
                "feature_names": feature_names,
                "accuracy": acc,
                "macro_f1": macro,
                "per_class_f1": per_class,
                "wall_s": time.time() - t0,
            },
            indent=2,
        )
    )
    print(f"\n[ceiling+] wrote {out}  wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
