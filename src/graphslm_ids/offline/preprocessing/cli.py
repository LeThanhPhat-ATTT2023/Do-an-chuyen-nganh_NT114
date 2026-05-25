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

from graphslm_ids.offline.preprocessing.extractor import extract_packets_dir
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
    raw_root: Path, packets_df: pd.DataFrame, payload_length: int
) -> pd.DataFrame:
    """Reuse the v2 second-pass byte extractor and attach per-row payload bytes.

    The v2 helper returns an ``(n, payload_length)`` uint8 matrix aligned to
    ``packets_df`` row order. We split each row into a contiguous bytes object
    sized to the recorded ``payload_len`` so downstream tokenisation does not
    need to strip the zero pad.
    """
    payload_matrix = _build_payload_matrix_from_pcaps(
        raw_root, packets_df, payload_length
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
    packets_df = extract_packets_dir(raw_root, max_per_class=cap)
    _LOG.info(
        "  -> packets=%d classes=%d (%.1fs)",
        len(packets_df),
        packets_df["label"].nunique(),
        time.time() - t0,
    )

    # ── Stage 2: attach payload bytes ─────────────────────────────────────────
    _LOG.info("[2/6] second pass: copying payload bytes ...")
    t0 = time.time()
    packets_df = _attach_payload_column(raw_root, packets_df, args.payload_length)
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
    raw_root: Path, packets_df, payload_length: int
) -> np.ndarray:
    """Re-walk the pcaps and copy each kept packet's payload bytes into a matrix.

    The extractor emits per-packet metadata but not the raw bytes (keeping the
    metadata path lightweight). For the graph artifact we need the actual
    bytes so payload features and evidence edges can fire on real content.
    This helper does a SECOND deterministic pass aligned to the metadata DF:
    for each row we already extracted, we re-open the same pcap and copy the
    payload of the packet at the recorded ``(pcap, packet_index)`` position.

    NOTE: rows are grouped by pcap path (inferred from label folder layout)
    and each pcap walked once. Rows that don't survive matching are zero-padded.
    """
    n = len(packets_df)
    out = np.zeros((n, payload_length), dtype=np.uint8)
    row_index: dict[tuple[str, int], list[int]] = {}
    if "label" not in packets_df.columns:
        return out
    cum_per_label: dict[str, int] = {}
    for i, label in enumerate(packets_df["label"].astype(str).tolist()):
        pos = cum_per_label.get(label, 0)
        row_index.setdefault((label, pos), []).append(i)
        cum_per_label[label] = pos + 1

    import dpkt
    import socket  # noqa: F401  # imported for symmetry with extractor

    for cls_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = cls_dir.name
        running_pos = 0
        for pcap in sorted(cls_dir.glob("*.pcap")):
            try:
                f = open(pcap, "rb")
            except OSError:
                continue
            with f:
                try:
                    reader = dpkt.pcap.Reader(f)
                except Exception:
                    continue
                dlt = reader.datalink()
                for _, (_ts, buf) in enumerate(reader):
                    try:
                        ip = _decode_ip(buf, dlt, dpkt)
                    except Exception:
                        ip = None
                    if not isinstance(ip, dpkt.ip.IP):
                        continue
                    rows = row_index.get((label, running_pos))
                    if rows:
                        payload = _extract_payload(ip, dpkt)
                        if payload:
                            vec = np.zeros(payload_length, dtype=np.uint8)
                            k = min(payload_length, len(payload))
                            vec[:k] = np.frombuffer(payload[:k], dtype=np.uint8)
                            for r in rows:
                                out[r] = vec
                    running_pos += 1
    return out


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
