"""Assemble the v3 Smart-BOTH heterogeneous graph artifact.

Inputs:
    * packets_df  (per-packet, includes ``payload`` bytes + ``flow_id``)
    * feats_df    (per-flow CICFlowMeter features from v2)
    * splits_dict (random + temporal split flow IDs from v3/split.py)
    * MITRE CSVs + STIX json
    * pmi_table.parquet from v3/pmi_learner.py

Output: dict of numpy arrays + ``metadata`` dict + ``artifact_version='v3'``.

Schema (matches spec §3 / §5):

    Nodes (5):
        flow, packet, host, technique, tactic
    Edges (13 forward types):
        contain                       flow      -> packet
        link  (next_packet)           packet    -> packet     attr: [delta_t]
        from_host                     flow      -> host       attr: [byte_count_fwd]
        to_host                       flow      -> host       attr: [byte_count_bwd]
        burst_neighbor                flow      -> flow       attr: [share_src, share_dst]
        evidence_injection            packet    -> technique  attr: [weight]
        evidence_command_exec         packet    -> technique  attr: [weight]
        evidence_file_upload          packet    -> technique  attr: [weight]
        evidence_recon                packet    -> technique  attr: [weight]
        evidence_c2_beacon            packet    -> technique  attr: [weight]
        flow_technique                flow      -> technique  attr: [weight]
        has_subtechnique              technique -> technique
        technique_tactic              technique -> tactic     attr: [1.0]

Memory budget (177K flows, ~600K packets after control drop, 691 techniques):

    flow_x  float32     177K * 85   ≈   60 MB
    packet_x float32    600K * 2323 ≈  5.6 GB  <-- dominant
    host_x  float32     8K * 4      <    1 MB
    technique_x         691 * 768   ≈    2 MB
    edge buffers (typed evidence, streamed)  ~ 200 MB total

Build wall-clock target: 60-90 min on 16-core CPU (dominated by Source 2
Aho-Corasick matching over 600K packets and the per-packet packet_x feature
build).
"""
from __future__ import annotations

import logging
import multiprocessing as _mp
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.payload_features import (
    FEATURE_DIM as PAYLOAD_FEATURE_DIM,
    FEATURE_NAMES as PAYLOAD_FEATURE_NAMES,
    compute_packet_payload_features,
)
from graphslm_ids.offline.preprocessing.edge_writers import MemmapEdgeWriter
from graphslm_ids.offline.preprocessing.ensemble import (
    aggregate_evidence,
    build_pmi_lookup_from_table,
    lookup_pmi_per_packet,
)
from graphslm_ids.offline.preprocessing.flow_consensus import flow_consensus_hits
from graphslm_ids.offline.preprocessing.procedure_matcher import ProcedureMatcher

_LOG = logging.getLogger(__name__)

# ─── Parallel worker functions (module-level so they are picklable on Windows) ──

def _packet_x_worker(args: tuple) -> tuple[int, int]:
    """Compute payload features for one chunk and write directly to a memmap slice.

    Args delivered as a single tuple (required by Pool.map):
        (start_idx, payloads_chunk, plens_chunk, payload_length, store_dtype,
         mmap_path_str, n_total)
    Returns (start_idx, chunk_len) for progress logging.
    """
    start_idx, payloads_chunk, plens_chunk, payload_length, store_dtype, mmap_path_str, n_total = args
    dtype = np.dtype(store_dtype)
    # Open the pre-created memmap in read-write mode; each worker writes only
    # its own slice so no locking is needed.
    out = np.memmap(mmap_path_str, dtype=dtype, mode="r+", shape=(n_total, PAYLOAD_FEATURE_DIM))
    buf = np.zeros(payload_length, dtype=np.uint8)
    for i, (raw, plen) in enumerate(zip(payloads_chunk, plens_chunk, strict=True)):
        if not raw:
            continue
        buf[:] = 0
        k = min(payload_length, len(raw))
        buf[:k] = np.frombuffer(raw[:k], dtype=np.uint8)
        row_f32 = compute_packet_payload_features(buf, min(plen, payload_length))
        out[start_idx + i] = row_f32 if dtype == np.dtype("float32") else row_f32.astype(dtype)
    out.flush()
    return start_idx, len(payloads_chunk)


# Per-worker globals for the evidence pool (initializer sets these once per process).
_g_proc_matcher: Any = None
_g_pmi_lookup: Any = None
_g_tech_family: Any = None


def _evidence_worker_init(stix_json_path: str, pmi_lookup: dict, technique_family_map: dict) -> None:
    """Pool initializer: build ProcedureMatcher once per worker process."""
    global _g_proc_matcher, _g_pmi_lookup, _g_tech_family
    _g_proc_matcher = ProcedureMatcher(Path(stix_json_path))
    _g_pmi_lookup = pmi_lookup
    _g_tech_family = technique_family_map


def _evidence_batch_worker(batch: list[tuple]) -> list[tuple[int, list]]:
    """Process a batch of packets and return evidence edges.

    Args:
        batch: list of (pidx, payload_bytes, flow_hits_dict)

    Returns:
        list of (pidx, [(tech, family, weight), ...]) — only non-empty.
    """
    results: list[tuple[int, list]] = []
    for pidx, payload, flow_hits in batch:
        if not payload:
            continue
        pmi_hits = lookup_pmi_per_packet(payload, _g_pmi_lookup)
        proc_hits = _g_proc_matcher.weight_per_technique(payload)
        edges = aggregate_evidence(pmi_hits, proc_hits, flow_hits, _g_tech_family)
        if edges:
            results.append((pidx, edges))
    return results

# The 5 typed evidence edge types in spec-defined order. Keeping the order
# fixed makes the audit script and the trainer config trivially aligned.
_EVIDENCE_FAMILIES: tuple[str, ...] = (
    "injection",
    "command_exec",
    "file_upload",
    "recon",
    "c2_beacon",
)


def _coerce_payload(raw: Any) -> bytes:
    """Tokenizer-friendly bytes coercion (mirrors v3/pmi_learner._coerce_payload)."""
    if raw is None:
        return b""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    if isinstance(raw, np.ndarray):
        return raw.astype(np.uint8, copy=False).tobytes()
    if isinstance(raw, (list, tuple)):
        return bytes(raw)
    return b""


def _build_host_tier(
    feats_df: pd.DataFrame,
    packets_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Construct host nodes + edges flow->host (from/to).

    Host = unique IPv4 string that appears as src_ip or dst_ip across any flow.
    Features (4-d): out_degree, in_degree, n_distinct_dst_ports, n_distinct_dst_hosts.

    Returns:
        ``(host_x, host_to_idx, from_host_eidx, to_host_eidx, fwd_bytes, bwd_bytes)``
        where the byte arrays are aligned with the ``from_host`` / ``to_host``
        edge indices respectively.
    """
    # Use the flow features frame (one row per flow) to enumerate the host
    # universe — packets_df may not have src/dst columns the v3 pipeline
    # uniformly relies on after assign_flows reshuffles rows.
    if "src_ip" in feats_df.columns and "dst_ip" in feats_df.columns:
        src_ips = feats_df["src_ip"].astype(str).to_numpy()
        dst_ips = feats_df["dst_ip"].astype(str).to_numpy()
    else:
        # Fallback: pull from packets_df by grouping on flow_id.
        first_packets = (
            packets_df.groupby("flow_id", sort=False).first().reindex(feats_df.index)
        )
        src_ips = first_packets["src_ip"].astype(str).to_numpy()
        dst_ips = first_packets["dst_ip"].astype(str).to_numpy()
        # Stash the columns on feats_df so downstream burst_neighbor can read them.
        feats_df["src_ip"] = src_ips
        feats_df["dst_ip"] = dst_ips

    unique_hosts = sorted(set(src_ips.tolist()) | set(dst_ips.tolist()))
    host_to_idx = {h: i for i, h in enumerate(unique_hosts)}
    n_hosts = len(unique_hosts)

    # Aggregations: out_degree, in_degree, n_distinct_dst_ports, n_distinct_dst_hosts
    # are computed across flow records (one count per flow).
    out_deg = np.zeros(n_hosts, dtype=np.float32)
    in_deg = np.zeros(n_hosts, dtype=np.float32)
    dst_port_sets: dict[int, set] = {i: set() for i in range(n_hosts)}
    dst_host_sets: dict[int, set] = {i: set() for i in range(n_hosts)}

    dst_ports = feats_df["dst_port"].to_numpy() if "dst_port" in feats_df.columns else None
    for i, (s, d) in enumerate(zip(src_ips, dst_ips, strict=True)):
        si = host_to_idx[s]
        di = host_to_idx[d]
        out_deg[si] += 1.0
        in_deg[di] += 1.0
        if dst_ports is not None:
            dst_port_sets[si].add(int(dst_ports[i]))
        dst_host_sets[si].add(d)

    host_x = np.zeros((n_hosts, 4), dtype=np.float32)
    host_x[:, 0] = out_deg
    host_x[:, 1] = in_deg
    host_x[:, 2] = np.asarray(
        [len(dst_port_sets[i]) for i in range(n_hosts)], dtype=np.float32
    )
    host_x[:, 3] = np.asarray(
        [len(dst_host_sets[i]) for i in range(n_hosts)], dtype=np.float32
    )

    # Flow -> host edge indices (forward order: flow index, host index).
    flow_idx_arr = np.arange(len(feats_df), dtype=np.int64)
    from_host_eidx = np.vstack(
        [
            flow_idx_arr,
            np.asarray([host_to_idx[s] for s in src_ips], dtype=np.int64),
        ]
    )
    to_host_eidx = np.vstack(
        [
            flow_idx_arr,
            np.asarray([host_to_idx[d] for d in dst_ips], dtype=np.int64),
        ]
    )

    fwd_bytes = feats_df.get("fwd_bytes", pd.Series(np.zeros(len(feats_df)))).to_numpy(
        dtype=np.float32
    )
    bwd_bytes = feats_df.get("bwd_bytes", pd.Series(np.zeros(len(feats_df)))).to_numpy(
        dtype=np.float32
    )
    return host_x, host_to_idx, from_host_eidx, to_host_eidx, fwd_bytes, bwd_bytes


def _build_burst_neighbor_edges(
    feats_df: pd.DataFrame,
    radius_sec: float = 1.0,
    max_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """KDTree over flow start timestamps -> share_src/share_dst homophily edges.

    For each flow we query a 1-D KDTree at radius ``radius_sec`` and keep at
    most ``max_neighbors`` candidates. Edge is emitted iff src or dst IP is
    shared. Bidirectional (we emit ``(i, j)`` AND ``(j, i)`` whenever ``i < j``
    and the predicate fires) so downstream HGT does not need to know which
    side of the pair the seed was.

    Attr is ``[share_src_ip, share_dst_ip]`` as int8 packed into float32 (the
    edge writer is float32; the values are 0.0 / 1.0 so int8 packing isn't
    necessary at storage).
    """
    from scipy.spatial import cKDTree

    if "flow_start_ts" in feats_df.columns:
        ts = feats_df["flow_start_ts"].to_numpy(dtype=np.float64)
    elif "ts_min" in feats_df.columns:
        ts = feats_df["ts_min"].to_numpy(dtype=np.float64)
    else:
        # No timestamp column -> can't build temporal homophily. Return empty.
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 2), dtype=np.float32)

    if len(ts) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 2), dtype=np.float32)

    src_ips = feats_df["src_ip"].astype(str).to_numpy()
    dst_ips = feats_df["dst_ip"].astype(str).to_numpy()

    tree = cKDTree(ts.reshape(-1, 1))
    # query_ball_tree(self, ...) returns lists of indices per row; for our
    # workload (~177K flows, radius 1s) the per-row list is normally small.
    neighbor_lists = tree.query_ball_tree(tree, r=radius_sec)

    src_buf: list[int] = []
    dst_buf: list[int] = []
    attr_buf: list[list[float]] = []
    for i, neigh in enumerate(neighbor_lists):
        # Filter to forward pairs (i < j) so we don't double-count, then we
        # emit both (i, j) AND (j, i) for HGT bidirectionality.
        candidates = [j for j in neigh if j > i]
        if not candidates:
            continue
        # Truncate to max_neighbors deterministically (sorted ascending).
        for j in candidates[:max_neighbors]:
            share_src = float(src_ips[i] == src_ips[j])
            share_dst = float(dst_ips[i] == dst_ips[j])
            if share_src == 0.0 and share_dst == 0.0:
                continue
            attr = [share_src, share_dst]
            src_buf.append(int(i))
            dst_buf.append(int(j))
            attr_buf.append(attr)
            src_buf.append(int(j))
            dst_buf.append(int(i))
            attr_buf.append(attr)

    if not src_buf:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 2), dtype=np.float32)
    edge_index = np.vstack(
        [
            np.asarray(src_buf, dtype=np.int64),
            np.asarray(dst_buf, dtype=np.int64),
        ]
    )
    edge_attr = np.asarray(attr_buf, dtype=np.float32)
    return edge_index, edge_attr


def _build_has_subtechnique_edges(
    techniques_df: pd.DataFrame,
    technique_id_to_idx: dict[str, int],
) -> np.ndarray:
    """Parse ``T1190.001 -> T1190`` style hierarchy edges from the techniques CSV.

    Returns a ``(2, N)`` int64 array. Sub-technique IDs are detected via the
    ``.`` separator in the technique_id; the parent ID is the prefix.
    """
    src: list[int] = []
    dst: list[int] = []
    for tid in techniques_df["technique_id"].astype(str).tolist():
        if "." not in tid:
            continue
        parent = tid.split(".", 1)[0]
        if parent not in technique_id_to_idx or tid not in technique_id_to_idx:
            continue
        # Spec: parent --[has_subtechnique]--> sub-technique
        src.append(technique_id_to_idx[parent])
        dst.append(technique_id_to_idx[tid])
    if not src:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack(
        [np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)]
    )


def _build_packet_x(
    payloads: list[bytes],
    payload_lengths: list[int],
    payload_length: int = 256,
    store_dtype: str = "float16",
    mmap_path: Path | None = None,
    n_workers: int | None = None,
) -> np.ndarray:
    """Compute the (n_packets, 2323) packet feature matrix — parallel version.

    Uses ``multiprocessing.Pool`` when ``n_workers > 1`` (default = all CPUs).
    Each worker writes directly to its slice of the memmap file so no large
    arrays are transferred back over IPC — only (start, len) progress tokens.

    If ``mmap_path`` is None a RAM array is used (only safe for small n).
    """
    n = len(payloads)
    dtype = np.dtype(store_dtype)
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # Create / zero the output array BEFORE spawning workers.
    mmap_path_str: str | None = None
    if mmap_path is not None:
        # mode='w+' creates / truncates the file and zeros all entries.
        out: np.ndarray = np.memmap(
            mmap_path, dtype=dtype, mode="w+", shape=(n, PAYLOAD_FEATURE_DIM)
        )
        out.flush()
        del out          # close so workers can open in 'r+' mode
        mmap_path_str = str(mmap_path)
    else:
        # Fallback: single-threaded, in-RAM (small datasets / tests).
        n_workers = 1

    # ── Split into chunks, one per worker ──────────────────────────────────
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    chunks: list[tuple] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((
            start,
            payloads[start:end],
            payload_lengths[start:end],
            payload_length,
            store_dtype,
            mmap_path_str,
            n,
        ))

    if n_workers == 1 or mmap_path_str is None:
        # Single-process path (tests / no mmap).
        buf = np.zeros(payload_length, dtype=np.uint8)
        out_arr = np.zeros((n, PAYLOAD_FEATURE_DIM), dtype=dtype)
        for i, (raw, plen) in enumerate(zip(payloads, payload_lengths, strict=True)):
            if not raw:
                continue
            buf[:] = 0
            k = min(payload_length, len(raw))
            buf[:k] = np.frombuffer(raw[:k], dtype=np.uint8)
            row_f32 = compute_packet_payload_features(buf, min(plen, payload_length))
            out_arr[i] = row_f32 if dtype == np.dtype("float32") else row_f32.astype(dtype)
        return out_arr

    _LOG.info("packet_x: launching %d workers for %d packets", n_workers, n)
    ctx = _mp.get_context("spawn")   # explicit spawn = safe on Windows + Linux
    with ctx.Pool(processes=n_workers) as pool:
        for done_start, done_len in pool.imap_unordered(_packet_x_worker, chunks):
            _LOG.debug("packet_x chunk done: start=%d len=%d", done_start, done_len)

    # Re-open the completed memmap for the caller (read-only is enough but
    # the rest of the pipeline may index into it, so 'r+' keeps behaviour).
    return np.memmap(mmap_path, dtype=dtype, mode="r+", shape=(n, PAYLOAD_FEATURE_DIM))


def _flow_evidence_summary(
    flow_to_packets: dict[int, list[int]],
    packet_to_edges: list[list[tuple[str, str, float]]],
    family_to_idx: dict[str, int],
) -> np.ndarray:
    """5-d per-flow summary of typed-evidence edges.

    Columns: ``[evidence_count, evidence_max_weight, dominant_family_id,
                n_distinct_families, sum_log1p_weight]``.
    """
    n_flows = len(flow_to_packets)
    summary = np.zeros((n_flows, 5), dtype=np.float32)
    for flow_idx, packet_idxs in flow_to_packets.items():
        count = 0
        max_w = 0.0
        family_counts: dict[str, int] = {}
        sum_log = 0.0
        for pidx in packet_idxs:
            for _tech, family, w in packet_to_edges[pidx]:
                count += 1
                if w > max_w:
                    max_w = w
                family_counts[family] = family_counts.get(family, 0) + 1
                sum_log += float(np.log1p(w))
        if count == 0:
            continue
        # Dominant family = family with most edges; tie -> first-encountered.
        dominant = max(family_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        summary[flow_idx, 0] = float(count)
        summary[flow_idx, 1] = float(max_w)
        summary[flow_idx, 2] = float(family_to_idx.get(dominant, -1))
        summary[flow_idx, 3] = float(len(family_counts))
        summary[flow_idx, 4] = float(sum_log)
    return summary


def build_v3_graph_artifact(
    packets_df: pd.DataFrame,
    feats_df: pd.DataFrame,
    splits_dict: dict[str, dict[str, np.ndarray]],
    *,
    mitre_techniques_csv: Path,
    mitre_technique_embeddings_npy: Path,
    mitre_technique_tactic_csv: Path,
    mitre_stix_json: Path,
    class_technique_map_csv: Path,
    technique_family_csv: Path,
    pmi_table_parquet: Path,
    payload_length: int = 256,
    tau_edge: float = 0.4,
    tmp_dir: Path | None = None,
    n_workers: int | None = None,
) -> dict[str, Any]:
    """End-to-end v3 graph build. See module docstring for schema.

    Args:
        packets_df: from v2 extractor + ``assign_flows``. Must contain
            ``flow_id``, ``payload`` (bytes-like), ``payload_len``, ``ts``.
        feats_df: from v2 ``build_flow_features``. Indexed by ``flow_id``.
        splits_dict: from v3/split. Not used to subset the graph (the artifact
            contains ALL flows), but the random/temporal split ID arrays are
            persisted into ``metadata`` so the trainer can mask.
        mitre_*_csv / *_npy / *_json: paths into ``data/mitre/``.
        class_technique_map_csv: maps dataset class -> technique with mapping
            weight; consumed by the PMI projection (already baked into
            ``pmi_table_parquet``) and exported in metadata.
        technique_family_csv: maps technique -> family for typed-edge routing.
        pmi_table_parquet: output of :func:`pmi_learner.fit_and_save_pmi_table`.
        payload_length: byte window for the packet payload features.
        tau_edge: ensemble threshold (see :func:`ensemble.aggregate_evidence`).
        tmp_dir: directory for streaming edge spills. Defaults to a
            ``tempfile.mkdtemp()`` location and is NOT cleaned up by this
            function (caller's job).

    Returns:
        Dict of NumPy arrays + ``metadata``. ``artifact_version = "v3"``.
    """
    if packets_df.empty:
        raise ValueError("Cannot build a v3 artifact from empty packets_df")
    if feats_df.empty:
        raise ValueError("Cannot build a v3 artifact from empty feats_df")

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="v3_edges_"))
    else:
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("v3 build: tmp_dir=%s", tmp_dir)

    t_total = time.time()

    # ─── 1. Flow nodes + label encoding ─────────────────────────────────────
    # Exclude non-numeric columns (label, src_ip/dst_ip strings, flow_start_ts
    # may be kept since it's numeric). We coerce by checking dtype.
    _NON_FEATURE_COLS = {"label", "src_ip", "dst_ip", "src_ip_str", "dst_ip_str"}
    flow_feature_columns = [
        c for c in feats_df.columns
        if c not in _NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(feats_df[c])
    ]
    flow_x_core = feats_df[flow_feature_columns].to_numpy(dtype=np.float32)
    labels = feats_df["label"].astype(str).tolist()
    uniq_labels = sorted(set(labels))
    label_mapping = {l: i for i, l in enumerate(uniq_labels)}
    flow_y = np.asarray([label_mapping[l] for l in labels], dtype=np.int64)
    flow_id_to_idx = {fid: i for i, fid in enumerate(feats_df.index.tolist())}
    n_flows = len(feats_df)
    _LOG.info("flow nodes: n=%d, features=%d", n_flows, flow_x_core.shape[1])

    # ─── 2. Packet tier: drop control packets (payload_len == 0) ────────────
    if "payload_len" not in packets_df.columns:
        raise ValueError("packets_df must contain 'payload_len' column")
    keep_mask = packets_df["payload_len"].to_numpy() > 0
    packets_kept = packets_df.loc[keep_mask].reset_index(drop=True)
    n_packets = len(packets_kept)
    n_raw_packets = len(packets_df)
    _LOG.info(
        "packet nodes: kept=%d / raw=%d (dropped %d control packets)",
        n_packets,
        n_raw_packets,
        n_raw_packets - n_packets,
    )
    # Map packet -> flow index for streaming edge construction.
    packet_flow_ids = packets_kept["flow_id"].astype(str).tolist()
    packet_flow_idx = np.asarray(
        [flow_id_to_idx.get(fid, -1) for fid in packet_flow_ids], dtype=np.int64
    )
    if (packet_flow_idx < 0).any():
        # Packets referencing flows the feature builder dropped: filter them.
        mask = packet_flow_idx >= 0
        packets_kept = packets_kept.loc[mask].reset_index(drop=True)
        packet_flow_idx = packet_flow_idx[mask]
        packet_flow_ids = [packet_flow_ids[i] for i in np.where(mask)[0]]
        n_packets = len(packets_kept)
        _LOG.info("packet nodes: filtered to %d (after orphan-flow drop)", n_packets)

    payloads = [
        _coerce_payload(p) for p in packets_kept.get("payload", pd.Series([])).tolist()
    ]
    payload_lens = packets_kept["payload_len"].astype(int).tolist()

    # ─── 3. packet_x ────────────────────────────────────────────────────────
    # Write to a memmap file so the full (n_packets, 2323) matrix never lives
    # in RAM. Essential when n_packets > 1 M on a 16 GB machine.
    t0 = time.time()
    packet_x_mmap_path = tmp_dir / "packet_x.mmap"
    packet_x = _build_packet_x(
        payloads,
        payload_lens,
        payload_length=payload_length,
        mmap_path=packet_x_mmap_path,
        n_workers=n_workers,
    )
    _LOG.info("packet_x: shape=%s dtype=%s mmap=%s (%.1fs)",
              packet_x.shape, packet_x.dtype, packet_x_mmap_path, time.time() - t0)

    # ─── 4. Host tier ───────────────────────────────────────────────────────
    t0 = time.time()
    feats_df_local = feats_df.copy()  # _build_host_tier may add src/dst columns
    host_x, host_to_idx, from_host_eidx, to_host_eidx, fwd_bytes, bwd_bytes = (
        _build_host_tier(feats_df_local, packets_df)
    )
    from_host_edge_attr = fwd_bytes.reshape(-1, 1)
    to_host_edge_attr = bwd_bytes.reshape(-1, 1)
    _LOG.info("host nodes: n=%d (%.1fs)", host_x.shape[0], time.time() - t0)

    # ─── 5. Contain edges (flow -> packet) ──────────────────────────────────
    contain_edge_index = np.vstack(
        [packet_flow_idx, np.arange(n_packets, dtype=np.int64)]
    )

    # ─── 6. Link edges (packet -> packet, next_packet) ──────────────────────
    if n_packets > 0:
        flow_arr = packet_flow_idx
        ts_arr = packets_kept["ts"].to_numpy(dtype=np.float64)
        # next_packet edges exist between rows i-1 and i iff they share flow.
        same_flow = np.zeros(n_packets, dtype=bool)
        same_flow[1:] = flow_arr[1:] == flow_arr[:-1]
        dst_pkt = np.where(same_flow)[0]
        src_pkt = dst_pkt - 1
        delta_t = (ts_arr[dst_pkt] - ts_arr[src_pkt]).astype(np.float32)
        link_edge_index = np.vstack(
            [src_pkt.astype(np.int64), dst_pkt.astype(np.int64)]
        )
        link_edge_attr = delta_t.reshape(-1, 1)
    else:
        link_edge_index = np.empty((2, 0), dtype=np.int64)
        link_edge_attr = np.empty((0, 1), dtype=np.float32)

    # ─── 7. burst_neighbor (flow -> flow homophily) ─────────────────────────
    t0 = time.time()
    burst_eidx, burst_attr = _build_burst_neighbor_edges(feats_df_local)
    _LOG.info(
        "burst_neighbor edges: n=%d (%.1fs)",
        burst_eidx.shape[1],
        time.time() - t0,
    )

    # ─── 8. MITRE technique tier ────────────────────────────────────────────
    techniques_df = pd.read_csv(mitre_techniques_csv)
    technique_x = np.load(mitre_technique_embeddings_npy)
    if technique_x.shape[0] != len(techniques_df):
        raise ValueError("technique embeddings row count != techniques CSV row count")
    technique_id_to_idx = {
        tid: i for i, tid in enumerate(techniques_df["technique_id"].astype(str).tolist())
    }
    n_techniques = len(techniques_df)

    # ─── 9. Tactic tier + technique_tactic edges (reuse v2 helper) ──────────
    tt_edge_index, tt_edge_attr, tactic_to_idx, n_tactics = _load_mitre_tactics(
        mitre_technique_tactic_csv, mitre_techniques_csv, technique_id_to_idx
    )
    tactic_x = np.arange(n_tactics, dtype=np.int64)[:, None]

    # ─── 10. has_subtechnique (technique -> technique) ──────────────────────
    has_sub_eidx = _build_has_subtechnique_edges(techniques_df, technique_id_to_idx)

    # ─── 11. PMI ensemble: streaming packet -> typed-evidence edges ────────
    t0 = time.time()
    pmi_table = pd.read_parquet(pmi_table_parquet)
    pmi_lookup = build_pmi_lookup_from_table(pmi_table)
    _LOG.info(
        "pmi_lookup: %d tokens, %d (token,tech) entries",
        len(pmi_lookup),
        int(pmi_table.shape[0]),
    )

    # ProcedureMatcher is built inside each worker via _evidence_worker_init.
    flow_consensus_map = flow_consensus_hits(feats_df)
    _LOG.info("flow_consensus: %d flows had signature hits", len(flow_consensus_map))

    technique_family_df = pd.read_csv(technique_family_csv)
    technique_family_map: dict[str, str] = {
        str(r.technique): str(r.family)
        for r in technique_family_df.itertuples(index=False)
    }

    family_to_idx = {fam: i for i, fam in enumerate(_EVIDENCE_FAMILIES)}
    evidence_writers: dict[str, MemmapEdgeWriter] = {
        fam: MemmapEdgeWriter(tmp_dir, ("packet", f"evidence_{fam}", "technique"), attr_dim=1)
        for fam in _EVIDENCE_FAMILIES
    }
    # Cache per-packet edges for the flow-evidence summary AND keep memory in
    # check by storing only the (tech, family, weight) triples, not payloads.
    packet_to_edges: list[list[tuple[str, str, float]]] = [list() for _ in range(n_packets)]

    # Build batches for parallel processing: each batch = list of
    # (pidx, payload_bytes, flow_hits_dict).  Batch size ~2 K keeps IPC
    # round-trips and per-batch pickle overhead balanced.
    from graphslm_ids.offline.preprocessing._resource import auto_compute_workers
    # Evidence pool: pure CPU-compute, not RAM-limited per worker.
    # n_workers override comes from --n-compute-workers; None → os.cpu_count().
    _n_ev_tasks = max(1, n_packets // 500)   # rough batch-count ceiling for cap
    n_workers_ev = auto_compute_workers(_n_ev_tasks, override=n_workers)
    batch_size = max(500, n_packets // (n_workers_ev * 8))
    batches: list[list[tuple]] = []
    for b_start in range(0, n_packets, batch_size):
        b_end = min(b_start + batch_size, n_packets)
        batch: list[tuple] = []
        for pidx in range(b_start, b_end):
            flow_id = packet_flow_ids[pidx]
            batch.append((pidx, payloads[pidx], flow_consensus_map.get(flow_id, {})))
        batches.append(batch)

    _LOG.info(
        "evidence: %d packets → %d batches × ~%d, %d workers",
        n_packets, len(batches), batch_size, n_workers_ev,
    )

    ctx = _mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers_ev,
        initializer=_evidence_worker_init,
        initargs=(str(mitre_stix_json), pmi_lookup, technique_family_map),
    ) as pool:
        for batch_results in pool.imap_unordered(
            _evidence_batch_worker, batches, chunksize=2
        ):
            for pidx, edges in batch_results:
                packet_to_edges[pidx] = edges
                for tech, family, weight in edges:
                    tidx = technique_id_to_idx.get(tech)
                    writer = evidence_writers.get(family)
                    if tidx is None or writer is None:
                        continue
                    writer.append(src=pidx, dst=tidx, attr=float(weight))

    finalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fam, w in evidence_writers.items():
        finalized[fam] = w.finalize()
        w.close()
        _LOG.info("evidence_%s edges: n=%d", fam, finalized[fam][0].shape[1])
    _LOG.info("PMI ensemble pass: total %.1fs", time.time() - t0)

    # ─── 12. flow_technique edges (flow consensus -> direct edges) ─────────
    flow_tech_src: list[int] = []
    flow_tech_dst: list[int] = []
    flow_tech_w: list[float] = []
    for fid, hits in flow_consensus_map.items():
        fidx = flow_id_to_idx.get(fid)
        if fidx is None:
            continue
        for tech, w in hits.items():
            tidx = technique_id_to_idx.get(tech)
            if tidx is None:
                continue
            flow_tech_src.append(fidx)
            flow_tech_dst.append(tidx)
            flow_tech_w.append(float(w))
    if flow_tech_src:
        flow_tech_eidx = np.vstack(
            [
                np.asarray(flow_tech_src, dtype=np.int64),
                np.asarray(flow_tech_dst, dtype=np.int64),
            ]
        )
        flow_tech_attr = np.asarray(flow_tech_w, dtype=np.float32).reshape(-1, 1)
    else:
        flow_tech_eidx = np.empty((2, 0), dtype=np.int64)
        flow_tech_attr = np.empty((0, 1), dtype=np.float32)

    # ─── 13. Flow evidence summary -> flow_x augmentation ──────────────────
    flow_to_packets: dict[int, list[int]] = {i: [] for i in range(n_flows)}
    for pidx, fidx in enumerate(packet_flow_idx.tolist()):
        flow_to_packets[fidx].append(pidx)
    ev_summary = _flow_evidence_summary(flow_to_packets, packet_to_edges, family_to_idx)
    flow_x = np.concatenate([flow_x_core, ev_summary], axis=1)
    flow_feature_names = flow_feature_columns + [
        "ev_count",
        "ev_max_weight",
        "ev_dominant_family_id",
        "ev_n_distinct_families",
        "ev_sum_log1p_weight",
    ]

    # ─── 14. Assemble artifact dict ─────────────────────────────────────────
    arts: dict[str, Any] = {
        "flow_x": flow_x.astype(np.float32),
        "flow_y": flow_y,
        "packet_x": packet_x,
        "host_x": host_x,
        "technique_x": technique_x.astype(np.float32),
        "tactic_x": tactic_x,
        "contain_edge_index": contain_edge_index,
        "link_edge_index": link_edge_index,
        "link_edge_attr": link_edge_attr,
        "from_host_edge_index": from_host_eidx,
        "from_host_edge_attr": from_host_edge_attr,
        "to_host_edge_index": to_host_eidx,
        "to_host_edge_attr": to_host_edge_attr,
        "burst_neighbor_edge_index": burst_eidx,
        "burst_neighbor_edge_attr": burst_attr,
        "evidence_injection_edge_index": finalized["injection"][0],
        "evidence_injection_edge_attr": finalized["injection"][1],
        "evidence_command_exec_edge_index": finalized["command_exec"][0],
        "evidence_command_exec_edge_attr": finalized["command_exec"][1],
        "evidence_file_upload_edge_index": finalized["file_upload"][0],
        "evidence_file_upload_edge_attr": finalized["file_upload"][1],
        "evidence_recon_edge_index": finalized["recon"][0],
        "evidence_recon_edge_attr": finalized["recon"][1],
        "evidence_c2_beacon_edge_index": finalized["c2_beacon"][0],
        "evidence_c2_beacon_edge_attr": finalized["c2_beacon"][1],
        "flow_technique_edge_index": flow_tech_eidx,
        "flow_technique_edge_attr": flow_tech_attr,
        "has_subtechnique_edge_index": has_sub_eidx,
        "technique_tactic_edge_index": tt_edge_index,
        "technique_tactic_edge_attr": tt_edge_attr,
    }

    # Persist splits into metadata (lists of flow_id strings — small, JSON-safe).
    splits_meta: dict[str, dict[str, list[str]]] = {}
    for protocol, by_split in splits_dict.items():
        splits_meta[protocol] = {
            split_name: [str(x) for x in arr.tolist()]
            for split_name, arr in by_split.items()
        }

    # Per-family edge counts in metadata for the audit script.
    evidence_counts = {
        fam: int(finalized[fam][0].shape[1]) for fam in _EVIDENCE_FAMILIES
    }

    # Canonical flow_id ordering used for flow_x rows. Persisted so the trainer
    # can map string flow_ids from splits.json back to integer node indices.
    flow_id_order = [str(fid) for fid in feats_df.index.tolist()]

    arts["metadata"] = {
        "artifact_version": "v3",
        "num_flows": int(n_flows),
        "num_packets": int(n_packets),
        "num_packets_raw": int(n_raw_packets),
        "num_hosts": int(host_x.shape[0]),
        "num_techniques": int(n_techniques),
        "num_tactics": int(n_tactics),
        "payload_length": int(payload_length),
        "tau_edge": float(tau_edge),
        "flow_feature_names": flow_feature_names,
        "flow_id_order": flow_id_order,
        "packet_feature_names": PAYLOAD_FEATURE_NAMES,
        "label_mapping": label_mapping,
        "host_to_idx": host_to_idx,
        "technique_id_to_idx": technique_id_to_idx,
        "tactic_to_idx": tactic_to_idx,
        "family_to_idx": family_to_idx,
        "evidence_edge_counts": evidence_counts,
        "n_contain_edges": int(contain_edge_index.shape[1]),
        "n_link_edges": int(link_edge_index.shape[1]),
        "n_from_host_edges": int(from_host_eidx.shape[1]),
        "n_to_host_edges": int(to_host_eidx.shape[1]),
        "n_burst_neighbor_edges": int(burst_eidx.shape[1]),
        "n_flow_technique_edges": int(flow_tech_eidx.shape[1]),
        "n_has_subtechnique_edges": int(has_sub_eidx.shape[1]),
        "n_technique_tactic_edges": int(tt_edge_index.shape[1]),
        "splits": splits_meta,
        "wall_seconds_total": round(time.time() - t_total, 2),
    }
    return arts


def save_v3_artifact(arts: dict[str, Any], out_npz: Path, out_meta_json: Path) -> None:
    """Persist the v3 artifact: NPZ (arrays) + meta JSON.

    Standalone — does not require ``build_v3_graph_artifact`` to have been
    run in this process. Caller is responsible for cleaning up the
    ``tmp_dir`` passed into the builder (this function does not touch it).
    """
    import json

    out_npz = Path(out_npz)
    out_meta_json = Path(out_meta_json)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)

    arrays = {k: v for k, v in arts.items() if isinstance(v, np.ndarray)}
    np.savez(str(out_npz), **arrays)
    with out_meta_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(arts["metadata"]), f, indent=2)


def _load_mitre_tactics(
    technique_tactic_csv: Path,
    techniques_csv: Path,
    technique_id_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    """Read MITRE technique<->tactic edges -> (edge_index, edge_attr, mapping, n_tactics)."""
    edges_df = pd.read_csv(technique_tactic_csv)
    required = {"technique_id", "tactic_shortname"}
    if not required.issubset(edges_df.columns):
        raise ValueError(
            f"technique_tactic_csv must contain {sorted(required)}, got {list(edges_df.columns)}"
        )
    tactics = sorted({str(t) for t in edges_df["tactic_shortname"].tolist()})
    tactic_to_idx = {t: i for i, t in enumerate(tactics)}

    src: list[int] = []
    dst: list[int] = []
    for _, row in edges_df.iterrows():
        tid = str(row["technique_id"])
        tac = str(row["tactic_shortname"])
        if tid not in technique_id_to_idx:
            continue
        src.append(technique_id_to_idx[tid])
        dst.append(tactic_to_idx[tac])
    if src:
        edge_index = np.vstack(
            [np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)]
        )
        edge_attr = np.ones((edge_index.shape[1], 1), dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 1), dtype=np.float32)
    return edge_index, edge_attr, tactic_to_idx, len(tactics)


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas types so json.dump won't choke."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


__all__ = [
    "build_v3_graph_artifact",
    "save_v3_artifact",
]
