"""Single-command orchestrator: raw pcaps -> v3 Smart-BOTH graph artifact.

End-to-end on local CPU (~2.5 hours wall-clock for the 14 GB subset):

    Stage 1.  Parse pcaps -> per-packet metadata DataFrame  (v2/extractor)
    Stage 2.  Re-walk pcaps -> attach payload bytes column   (mirrors v2/cli)
    Stage 3.  Bidirectional flow assembly + ~80 features     (v2/flows)
    Stage 4.  Random + temporal splits                       (v3/split)
    Stage 5.  PMI Ensemble fit on TRAIN packets              (v3/pmi_learner)
    Stage 6.  Graph build                                    (v3/graph_builder)

Output:
    <out-npz>           — np.savez of every node + edge array
    <out-meta>          — json metadata sidecar
    <pmi-table>         — parquet of (token, technique, family, weight)
    <splits-json>       — random + temporal flow-id splits

Usage:
    D:\\v\\nt114\\Scripts\\python.exe -m graphslm_ids.offline.preprocessing.cli ^
        --raw-root data/raw/14gb ^
        --out-npz outputs/v3/graph.npz ^
        --out-meta outputs/v3/graph.meta.json ^
        --pmi-table outputs/v3/pmi_table.parquet ^
        --pmi-meta outputs/v3/pmi.meta.json ^
        --splits-json outputs/v3/splits.json
"""
from __future__ import annotations

import argparse
import logging
import socket  # noqa: F401  # kept for symmetry with v2 CLI second-pass
import time
from pathlib import Path

import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.extractor import (
    extract_packets_dir,
    _payload_class_worker,
)
from graphslm_ids.offline.preprocessing.flows import (
    assign_flows,
    build_flow_features,
)
from graphslm_ids.offline.preprocessing.graph_builder import (
    build_v3_graph_artifact,
    save_v3_artifact,
)
from graphslm_ids.offline.preprocessing.pmi_learner import fit_and_save_pmi_table
from graphslm_ids.offline.preprocessing.split import make_splits, save_splits

_LOG = logging.getLogger("v3.cli")


def _attach_payload_column(
    raw_root: Path,
    packets_df: pd.DataFrame,
    payload_length: int,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Reuse the v2 second-pass byte extractor and attach per-row payload bytes.

    The v2 helper returns an ``(n, payload_length)`` uint8 matrix aligned to
    ``packets_df`` row order. We split each row into a contiguous bytes object
    sized to the recorded ``payload_len`` so downstream tokenisation does not
    need to strip the zero pad.
    """
    payload_matrix = _build_payload_matrix_from_pcaps(
        raw_root, packets_df, payload_length, n_workers=n_workers
    )
    plens = packets_df["payload_len"].astype(int).to_numpy()
    payloads: list[bytes] = []
    for i, row_bytes in enumerate(payload_matrix):
        # The v2 helper zero-pads to ``payload_length``; trim back to the
        # original payload length so PMI sees real content, not trailing 00s.
        k = int(min(payload_length, max(0, plens[i])))
        payloads.append(row_bytes[:k].tobytes())
    out = packets_df.copy()
    out["payload"] = payloads
    return out


def _augment_feats_with_flow_start_ts(
    feats_df: pd.DataFrame, packets_df: pd.DataFrame
) -> pd.DataFrame:
    """Add ``flow_start_ts`` to ``feats_df`` (required by v3/split.make_splits).

    ``build_flow_features`` derives ``dur`` from ``(ts_max - ts_min)`` but does
    not persist the absolute timestamps. We recompute the per-flow minimum
    timestamp here so the temporal split has the wall-clock anchor it needs.
    """
    if "flow_start_ts" in feats_df.columns:
        return feats_df
    ts_min = (
        packets_df.groupby("flow_id", sort=False)["ts"]
        .min()
        .reindex(feats_df.index)
    )
    out = feats_df.copy()
    out["flow_start_ts"] = ts_min.astype(np.float64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--pmi-table", required=True)
    ap.add_argument("--pmi-meta", required=True)
    ap.add_argument("--splits-json", required=True)

    ap.add_argument("--mitre-techniques-csv", default="data/mitre/mitre_techniques.csv")
    ap.add_argument(
        "--mitre-techniques-npy",
        default="data/mitre/mitre_techniques_embeddings.npy",
    )
    ap.add_argument(
        "--mitre-tactic-csv",
        default="data/mitre/mitre_technique_tactic_edges.csv",
    )
    ap.add_argument(
        "--mitre-stix", default="data/mitre/enterprise-attack.json"
    )
    ap.add_argument(
        "--class-technique-map", default="data/mitre/class_technique_map.csv"
    )
    ap.add_argument(
        "--technique-family", default="data/mitre/technique_family.csv"
    )

    ap.add_argument("--payload-length", type=int, default=256)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = no cap")
    ap.add_argument("--pmi-subsample-per-class", type=int, default=25_000)
    ap.add_argument("--pmi-max-total", type=int, default=200_000)
    ap.add_argument("--tau-edge", type=float, default=0.4)
    ap.add_argument("--temporal-train-frac", type=float, default=0.80)
    ap.add_argument("--temporal-val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")

    # ── Worker-count overrides (default: auto-detect from available RAM/CPUs) ─
    ap.add_argument(
        "--n-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Workers for PCAP-loading stages (Stage 1 + 2).  "
            "Default: auto-scale from available RAM "
            "(e.g. 8 on g6e.2xlarge 64 GB, ~1-2 on 16 GB laptop)."
        ),
    )
    ap.add_argument(
        "--n-compute-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Workers for CPU-compute stages (Stage 6: packet_x + evidence pool).  "
            "Default: os.cpu_count()."
        ),
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    raw_root = Path(args.raw_root)
    cap = None if args.max_per_class == 0 else args.max_per_class
    t_total = time.time()

    # ── Stage 1: extract packet metadata ──────────────────────────────────────
    _LOG.info("[1/6] parsing pcaps from %s (cap=%s) ...", raw_root, cap)
    t0 = time.time()
    packets_df = extract_packets_dir(raw_root, max_per_class=cap, n_workers=args.n_workers)
    _LOG.info(
        "  -> packets=%d classes=%d (%.1fs)",
        len(packets_df),
        packets_df["label"].nunique(),
        time.time() - t0,
    )

    # ── Stage 2: attach payload bytes ─────────────────────────────────────────
    _LOG.info("[2/6] second pass: copying payload bytes ...")
    t0 = time.time()
    packets_df = _attach_payload_column(raw_root, packets_df, args.payload_length, n_workers=args.n_workers)
    _LOG.info("  -> payload column attached (%.1fs)", time.time() - t0)

    # ── Stage 3: bidirectional flows + ~80 features ──────────────────────────
    _LOG.info("[3/6] flow assembly + feature build ...")
    t0 = time.time()
    packets_df = assign_flows(packets_df)
    feats_df, _meta = build_flow_features(packets_df)
    feats_df = _augment_feats_with_flow_start_ts(feats_df, packets_df)
    _LOG.info(
        "  -> flows=%d feature_dim=%d (%.1fs)",
        len(feats_df),
        feats_df.shape[1],
        time.time() - t0,
    )

    # ── Stage 4: random + temporal splits ────────────────────────────────────
    _LOG.info("[4/6] building random + temporal splits ...")
    t0 = time.time()
    splits = make_splits(
        feats_df,
        train_frac=args.temporal_train_frac,
        val_frac=args.temporal_val_frac,
        seed=args.seed,
    )
    save_splits(splits, Path(args.splits_json))
    _LOG.info(
        "  -> random train/val/test = %d/%d/%d; temporal = %d/%d/%d (%.1fs)",
        splits["random"]["train"].size,
        splits["random"]["val"].size,
        splits["random"]["test"].size,
        splits["temporal"]["train"].size,
        splits["temporal"]["val"].size,
        splits["temporal"]["test"].size,
        time.time() - t0,
    )

    # ── Stage 5: PMI ensemble fit on TRAIN packets only ──────────────────────
    _LOG.info("[5/6] PMI ensemble fit on TRAIN (temporal split) ...")
    t0 = time.time()
    # Use the temporal-split train ids — it's the stricter constraint (no
    # future packets leak into the PMI fit, so the ensemble is honest under
    # both evaluation protocols).
    train_flow_ids = set(splits["temporal"]["train"].tolist())
    train_packet_mask = packets_df["flow_id"].astype(str).isin(train_flow_ids)
    train_packets = packets_df.loc[train_packet_mask].reset_index(drop=True)
    _LOG.info("  -> train packets for PMI fit: %d", len(train_packets))
    pmi_meta = fit_and_save_pmi_table(
        train_packets,
        class_technique_map_path=Path(args.class_technique_map),
        technique_family_path=Path(args.technique_family),
        out_pmi_table=Path(args.pmi_table),
        out_meta_json=Path(args.pmi_meta),
        n_per_class=args.pmi_subsample_per_class,
        max_total=args.pmi_max_total,
        seed=args.seed,
    )
    _LOG.info(
        "  -> pmi_table rows=%d unique_tokens=%d unique_techniques=%d (%.1fs)",
        pmi_meta["pmi_table_rows"],
        pmi_meta["pmi_table_unique_tokens"],
        pmi_meta["pmi_table_unique_techniques"],
        time.time() - t0,
    )

    # ── Stage 6: graph assembly ──────────────────────────────────────────────
    _LOG.info("[6/6] assembling v3 graph artifact ...")
    t0 = time.time()
    arts = build_v3_graph_artifact(
        packets_df,
        feats_df,
        splits,
        mitre_techniques_csv=Path(args.mitre_techniques_csv),
        mitre_technique_embeddings_npy=Path(args.mitre_techniques_npy),
        mitre_technique_tactic_csv=Path(args.mitre_tactic_csv),
        mitre_stix_json=Path(args.mitre_stix),
        class_technique_map_csv=Path(args.class_technique_map),
        technique_family_csv=Path(args.technique_family),
        pmi_table_parquet=Path(args.pmi_table),
        payload_length=args.payload_length,
        tau_edge=args.tau_edge,
        n_workers=args.n_compute_workers,
    )
    save_v3_artifact(arts, Path(args.out_npz), Path(args.out_meta))
    _LOG.info("  -> artifact saved (%.1fs)", time.time() - t0)

    md = arts["metadata"]
    _LOG.info(
        "[v3] OK  flows=%d packets=%d hosts=%d techniques=%d  "
        "evidence_edges=%s  wall_total=%.1fs",
        md["num_flows"],
        md["num_packets"],
        md["num_hosts"],
        md["num_techniques"],
        md["evidence_edge_counts"],
        time.time() - t_total,
    )


def _build_payload_matrix_from_pcaps(
    raw_root: Path,
    packets_df,
    payload_length: int,
    n_workers: int | None = None,
) -> np.ndarray:
    """Re-walk the pcaps and copy payload bytes — parallel version.

    Builds a per-label row_index map, then spawns one worker per class
    directory. Each worker writes directly to its non-overlapping rows of a
    shared memmap so no large arrays cross process boundaries.

    ``n_workers`` is auto-detected from available RAM when None (see
    :func:`_resource.auto_pcap_workers`). Pass an explicit value to override
    (e.g. ``--n-workers 8`` on a 64 GB cloud instance).
    """
    import multiprocessing as _mp
    import os
    import tempfile

    n = len(packets_df)
    if "label" not in packets_df.columns:
        return np.zeros((n, payload_length), dtype=np.uint8)

    # ── Build per-label row_index: label -> {pos: [row_indices]} ────────────
    per_label_index: dict[str, dict[int, list[int]]] = {}
    cum_per_label: dict[str, int] = {}
    for i, label in enumerate(packets_df["label"].astype(str).tolist()):
        pos = cum_per_label.get(label, 0)
        per_label_index.setdefault(label, {}).setdefault(pos, []).append(i)
        cum_per_label[label] = pos + 1

    # ── Create shared memmap (workers write, main reads back) ───────────────
    tmp_fd, mmap_path = tempfile.mkstemp(prefix="payload_matrix_", suffix=".mmap")
    import os as _os
    _os.close(tmp_fd)
    out_mmap = np.memmap(mmap_path, dtype=np.uint8, mode="w+", shape=(n, payload_length))
    out_mmap.flush()
    del out_mmap  # close so workers can open in 'r+' mode

    # ── Build task list (one task per class directory) ───────────────────────
    tasks: list[tuple] = []
    for cls_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = cls_dir.name
        slice_ = per_label_index.get(label, {})
        if not slice_:
            continue
        tasks.append((str(cls_dir), label, slice_, payload_length, mmap_path, n))

    # Auto-scale workers based on available RAM; respect explicit override.
    from graphslm_ids.offline.preprocessing._resource import auto_pcap_workers
    n_workers = auto_pcap_workers(len(tasks), override=n_workers)
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for done_label in pool.imap_unordered(_payload_class_worker, tasks):
            pass  # progress visible in worker logs / task manager

    # ── Load completed matrix into RAM (or keep as memmap if too large) ──────
    result = np.array(
        np.memmap(mmap_path, dtype=np.uint8, mode="r", shape=(n, payload_length))
    )
    try:
        _os.unlink(mmap_path)
    except OSError:
        pass  # Windows: file locked until GC; will be cleaned by OS
    return result


def _decode_ip(buf: bytes, dlt: int, dpkt_mod):
    if dlt == 1:  # Ethernet
        return dpkt_mod.ethernet.Ethernet(buf).data
    if dlt in (12, 101):
        return dpkt_mod.ip.IP(buf)
    if dlt == 113:
        return dpkt_mod.ip.IP(buf[16:])
    return dpkt_mod.ethernet.Ethernet(buf).data


def _extract_payload(ip, dpkt_mod) -> bytes:
    t = ip.data
    if isinstance(t, dpkt_mod.tcp.TCP):
        return bytes(t.data)
    if isinstance(t, dpkt_mod.udp.UDP):
        return bytes(t.data)
    if isinstance(t, dpkt_mod.icmp.ICMP):
        return bytes(t.data)
    return b""


if __name__ == "__main__":
    main()


__all__ = ["main"]
