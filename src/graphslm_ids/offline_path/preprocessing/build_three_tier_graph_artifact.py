from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from graphslm_ids.offline_path.preprocessing.graph_artifact_builder import build_graph_artifact
from graphslm_ids.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build three-tier heterogeneous graph artifact with flow-packet structure and "
            "packet/flow to MITRE technique tactical edges from cosine similarity."
        )
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--payload-npy", required=True)
    parser.add_argument("--student-embedding-npy", required=True)
    parser.add_argument("--mitre-techniques-csv", required=True)
    parser.add_argument("--mitre-technique-embeddings-npy", required=True)
    parser.add_argument("--mitre-technique-tactic-edges-csv", required=True)
    parser.add_argument("--output-npz", default="data/processed/graph_artifact_3tier_t082_k5.npz")
    parser.add_argument("--output-meta-json", default=None)
    parser.add_argument("--flow-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-packets-per-flow", type=int, default=20)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--packet-top-k", type=int, default=5)
    parser.add_argument("--flow-top-k", type=int, default=5)
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device for similarity: auto | cpu | cuda | cuda:0",
    )
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=50_000,
        help="Packets per batch for GPU/CPU similarity computation.",
    )
    return parser.parse_args()


# ── device setup ──────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[device] GPU: {name}  ({vram_gb:.1f} GB VRAM)", flush=True)
        else:
            dev = torch.device("cpu")
            n = os.cpu_count() or 1
            torch.set_num_threads(n)
            print(f"[device] CPU  ({n} threads)", flush=True)
        return dev

    dev = torch.device(device_str)
    if dev.type == "cpu":
        n = os.cpu_count() or 1
        torch.set_num_threads(n)
        print(f"[device] CPU  ({n} threads)", flush=True)
    else:
        idx = dev.index if dev.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        vram_gb = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        print(f"[device] GPU {idx}: {name}  ({vram_gb:.1f} GB VRAM)", flush=True)
    return dev


# ── normalization ─────────────────────────────────────────────────────────────

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


# ── batched similarity (GPU → CPU fallback) ───────────────────────────────────

def _flow_sum_device(device: torch.device, num_flows: int, n_tech: int) -> torch.device:
    """Pick the device for the flow_tech_sum accumulator.

    Keeps it on GPU when at least 2× the matrix bytes fit free, so the per-batch
    index_add_ avoids copying the (B, N_tech) sim matrix across the bus. Falls
    back to CPU torch (still much faster than np.add.at because torch CPU
    index_add_ is multi-threaded C++).
    """
    bytes_needed = int(num_flows) * int(n_tech) * 4
    if device.type != "cuda":
        return torch.device("cpu")
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return torch.device("cpu")
    if int(free) > bytes_needed * 2:
        return device
    return torch.device("cpu")


def _compute_sim_edges_and_flow_sums(
    student_emb: np.ndarray,         # (N_payload, D) mmap — full student embedding matrix
    packet_payload_idx: np.ndarray,  # (N_pkt,) indices into student_emb (graph packet order)
    mitre_emb: np.ndarray,           # (N_tech, D) L2-normalised float32
    contain_flow: np.ndarray,        # (E,) flow graph-indices  (contain_edge_index[0])
    contain_pkt: np.ndarray,         # (E,) packet graph-indices (contain_edge_index[1])
    num_flows: int,
    packet_top_k: int,
    threshold: float,
    device: torch.device,
    batch_size: int,
    packet_semantic_out: np.ndarray, # (N_pkt, D) mmap — written in-place (avoids RAM spike)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream packet embeddings from mmap in batches — never materialises the full
    N_pkt × D array in RAM.  Each batch:
      1. Reads student_emb[packet_payload_idx[start:end]] from disk via mmap.
      2. Writes raw floats to packet_semantic_out (disk-backed).
      3. L2-normalises in-place for cosine similarity.
      4. Computes top-k packet→technique edges and accumulates flow sums.

    GPU memory stays bounded to ~batch_size × D × 4 bytes per step.
    Falls back to CPU automatically on CUDA OOM.
    """
    N_pkt  = len(packet_payload_idx)
    N_tech = mitre_emb.shape[0]
    k      = min(packet_top_k, N_tech)

    # Pre-sort contain-edges by packet index for O(log N) batch lookup
    if contain_pkt.size > 0:
        order   = np.argsort(contain_pkt, kind="stable")
        s_pkt   = contain_pkt[order]
        s_flow  = contain_flow[order]
    else:
        s_pkt  = contain_pkt
        s_flow = contain_flow

    mitre_t = torch.from_numpy(np.ascontiguousarray(mitre_emb)).to(device)  # (N_tech, D)

    src_chunks:   list[np.ndarray] = []
    dst_chunks:   list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []

    accum_device = _flow_sum_device(device, num_flows, N_tech)
    print(
        f"[similarity] flow_tech_sum accumulator on {accum_device} "
        f"({num_flows * N_tech * 4 / 1e9:.2f} GB)",
        flush=True,
    )
    flow_tech_sum_t   = torch.zeros((num_flows, N_tech), dtype=torch.float32, device=accum_device)
    flow_tech_count_t = torch.zeros((num_flows,),         dtype=torch.float32, device=accum_device)

    for start in tqdm(range(0, N_pkt, batch_size), desc="similarity", unit="batch", leave=False):
        end     = min(start + batch_size, N_pkt)
        row_idx = packet_payload_idx[start:end]

        # Read only this batch from the mmap — O(batch) RAM, not O(N_pkt)
        batch_raw = np.ascontiguousarray(student_emb[row_idx], dtype=np.float32)

        # Persist raw embedding to the disk-backed output (zero additional RAM)
        packet_semantic_out[start:end] = batch_raw

        # L2-normalise per-batch for cosine similarity
        norms      = np.linalg.norm(batch_raw, axis=1, keepdims=True)
        batch_norm = batch_raw / np.maximum(norms, 1e-12)
        del batch_raw

        batch = torch.from_numpy(batch_norm).to(device)
        del batch_norm

        # matrix multiply with OOM fallback to CPU
        try:
            sim = batch @ mitre_t.T          # (B, N_tech)
        except RuntimeError as exc:
            if device.type == "cuda" and "memory" in str(exc).lower():
                torch.cuda.empty_cache()
                sim = batch.cpu() @ mitre_t.cpu().T
            else:
                raise

        # ── packet→technique top-k edges ──────────────────────────────────
        topk_scores, topk_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
        above   = topk_scores >= threshold
        row_ids = torch.arange(start, end, device=sim.device).unsqueeze(1).expand_as(topk_scores)

        src_chunks.append(row_ids[above].cpu().numpy().astype(np.int64))
        dst_chunks.append(topk_idx[above].cpu().numpy().astype(np.int64))
        score_chunks.append(topk_scores[above].cpu().numpy().astype(np.float32))

        # ── flow-level accumulation (vectorized index_add_ on accum_device) ──
        if s_pkt.size > 0:
            lo = int(np.searchsorted(s_pkt, start, side="left"))
            hi = int(np.searchsorted(s_pkt, end,   side="left"))
            if lo < hi:
                local_pkt_t = torch.from_numpy(
                    np.ascontiguousarray(s_pkt[lo:hi] - start, dtype=np.int64)
                ).to(sim.device, non_blocking=True)
                batch_flow_t = torch.from_numpy(
                    np.ascontiguousarray(s_flow[lo:hi], dtype=np.int64)
                )
                # Slice on sim's device first so the H2D copy (when accum_device
                # ≠ sim.device) shrinks from O(B * N_tech) to O(len(local_pkt) * N_tech).
                sim_subset = sim.index_select(0, local_pkt_t)
                if accum_device != sim.device:
                    sim_subset = sim_subset.to(accum_device, non_blocking=True)
                batch_flow_t = batch_flow_t.to(accum_device, non_blocking=True)
                flow_tech_sum_t.index_add_(0, batch_flow_t, sim_subset)
                ones = torch.ones((batch_flow_t.shape[0],), dtype=torch.float32, device=accum_device)
                flow_tech_count_t.index_add_(0, batch_flow_t, ones)

        # Note: ``torch.cuda.empty_cache()`` used to live here but it forces a
        # device-wide synchronisation, which serialises the otherwise pipelined
        # H2D / matmul / index_add stream. We only call it on the OOM fallback
        # path now — the caching allocator already reuses freed blocks across
        # iterations without help.
        del sim, batch

    del mitre_t

    pkt_src   = np.concatenate(src_chunks)   if src_chunks   else np.empty(0, np.int64)
    pkt_dst   = np.concatenate(dst_chunks)   if dst_chunks   else np.empty(0, np.int64)
    pkt_score = np.concatenate(score_chunks) if score_chunks else np.empty(0, np.float32)

    flow_tech_sum   = flow_tech_sum_t.detach().cpu().numpy()
    flow_tech_count = flow_tech_count_t.detach().cpu().numpy()
    del flow_tech_sum_t, flow_tech_count_t
    if accum_device.type == "cuda":
        torch.cuda.empty_cache()

    return pkt_src, pkt_dst, pkt_score, flow_tech_sum, flow_tech_count


def _select_flow_edges(
    flow_tech_sum:   np.ndarray,  # (N_flows, N_tech)
    flow_tech_count: np.ndarray,  # (N_flows,)
    flow_top_k:      int,
    threshold:       float,
    device:          torch.device,
    chunk_size:      int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute flow→technique top-k edges from accumulated sums.

    Processes rows in chunks to avoid materialising the full (N_flows, N_tech)
    float32 matrix, which can exceed available RAM for large datasets.
    """
    N_flows, N_tech = flow_tech_sum.shape
    k = min(flow_top_k, N_tech)

    src_chunks:   list[np.ndarray] = []
    dst_chunks:   list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []

    for start in range(0, N_flows, chunk_size):
        end = min(start + chunk_size, N_flows)

        count_safe = np.maximum(flow_tech_count[start:end, np.newaxis], 1.0)
        avg_chunk  = (flow_tech_sum[start:end] / count_safe).astype(np.float32)

        avg_t = torch.from_numpy(avg_chunk).to(device)
        topk_scores, topk_idx = torch.topk(avg_t, k=k, dim=1, largest=True, sorted=False)
        above   = topk_scores >= threshold
        row_ids = (
            torch.arange(start, end, device=avg_t.device)
            .unsqueeze(1).expand_as(topk_scores)
        )

        src_chunks.append(row_ids[above].cpu().numpy().astype(np.int64))
        dst_chunks.append(topk_idx[above].cpu().numpy().astype(np.int64))
        score_chunks.append(topk_scores[above].cpu().numpy().astype(np.float32))

        del avg_t, topk_scores, topk_idx, avg_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    src   = np.concatenate(src_chunks)   if src_chunks   else np.empty(0, np.int64)
    dst   = np.concatenate(dst_chunks)   if dst_chunks   else np.empty(0, np.int64)
    score = np.concatenate(score_chunks) if score_chunks else np.empty(0, np.float32)

    return src, dst, score


# ── MITRE tactic arrays ───────────────────────────────────────────────────────

def _build_technique_tactic_arrays(
    techniques_df: pd.DataFrame,
    technique_tactic_edges_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    technique_id_to_idx = {
        str(tid): idx
        for idx, tid in enumerate(techniques_df["technique_id"].astype(str).tolist())
    }
    tactic_shortnames = sorted({str(x) for x in technique_tactic_edges_df["tactic_shortname"].astype(str).tolist()})
    tactic_to_idx = {name: idx for idx, name in enumerate(tactic_shortnames)}

    src_list: list[int] = []
    dst_list: list[int] = []
    for row in technique_tactic_edges_df.itertuples(index=False):
        tid = str(row.technique_id)
        tac = str(row.tactic_shortname)
        if tid not in technique_id_to_idx or tac not in tactic_to_idx:
            continue
        src_list.append(technique_id_to_idx[tid])
        dst_list.append(tactic_to_idx[tac])

    if src_list:
        edge_index = np.vstack([
            np.asarray(src_list, dtype=np.int64),
            np.asarray(dst_list, dtype=np.int64),
        ])
        edge_attr = np.ones((len(src_list), 1), dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr  = np.empty((0, 1),  dtype=np.float32)

    return edge_index, edge_attr, technique_id_to_idx, tactic_to_idx


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = _resolve_device(args.device)

    metadata_csv                   = Path(args.metadata_csv)
    payload_npy                    = Path(args.payload_npy)
    student_embedding_npy          = Path(args.student_embedding_npy)
    mitre_techniques_csv           = Path(args.mitre_techniques_csv)
    mitre_technique_embeddings_npy = Path(args.mitre_technique_embeddings_npy)
    mitre_technique_tactic_edges   = Path(args.mitre_technique_tactic_edges_csv)
    output_npz                     = Path(args.output_npz)
    output_meta_json               = (
        Path(args.output_meta_json) if args.output_meta_json else output_npz.with_suffix(".meta.json")
    )

    # ── load inputs ───────────────────────────────────────────────────────────
    print("[1/6] Loading inputs …", flush=True)
    # Optimised dtypes cut DataFrame RAM by ~50 % vs default int64/object inference
    _cat = "category"
    metadata = pd.read_csv(metadata_csv, dtype={
        "pcap_file":       _cat,
        "label":           _cat,
        "src_ip":          _cat,
        "dst_ip":          _cat,
        "protocol":        _cat,
        "packet_index":    np.int32,
        "src_port":        np.int32,
        "dst_port":        np.int32,
        "payload_len_raw": np.int32,
    })
    payload_matrix    = np.load(payload_npy,        mmap_mode="r")
    student_embedding = np.load(student_embedding_npy, mmap_mode="r")

    if student_embedding.ndim != 2:
        raise ValueError("student embedding must be 2D")
    if student_embedding.shape[0] != payload_matrix.shape[0]:
        raise ValueError("student embedding rows must align with payload rows")

    techniques_df = pd.read_csv(mitre_techniques_csv)
    if "technique_id" not in techniques_df.columns:
        raise ValueError("MITRE techniques CSV must contain technique_id column")

    mitre_embedding = np.asarray(np.load(mitre_technique_embeddings_npy), dtype=np.float32)
    if mitre_embedding.ndim != 2:
        raise ValueError("MITRE technique embedding must be 2D")
    if mitre_embedding.shape[0] != techniques_df.shape[0]:
        raise ValueError("MITRE embedding rows must align with MITRE techniques CSV rows")
    if mitre_embedding.shape[1] != student_embedding.shape[1]:
        raise ValueError("Embedding dimensions must match between student and MITRE techniques")

    # ── build base graph (flow/packet topology) ───────────────────────────────
    print("[2/6] Building flow-packet graph structure …", flush=True)
    base_artifact = build_graph_artifact(
        metadata=metadata,
        payload_matrix=payload_matrix,
        flow_timeout_seconds=args.flow_timeout_seconds,
        max_packets_per_flow=args.max_packets_per_flow,
    )

    # Free the metadata DataFrame immediately — graph structure is built, no longer needed
    import gc
    del metadata
    gc.collect()
    print("[mem] metadata freed", flush=True)

    # ── allocate disk-backed mmap for packet_semantic_x ───────────────────────
    # Keeps N_pkt × D × 4 bytes off RAM; written in-place during the similarity loop.
    packet_payload_row_index = base_artifact.arrays["packet_payload_row_index"]
    N_pkt         = int(packet_payload_row_index.shape[0])
    embedding_dim = int(student_embedding.shape[1])
    packet_semantic_npy = output_npz.with_name(output_npz.stem + "_packet_semantic_x.npy")
    pkt_sem_gb = N_pkt * embedding_dim * 4 / 1e9
    print(f"[3/6] Allocating packet_semantic_x on disk  ({N_pkt:,} × {embedding_dim} = {pkt_sem_gb:.2f} GB) …", flush=True)
    packet_semantic_mmap = np.lib.format.open_memmap(
        str(packet_semantic_npy), mode="w+", dtype=np.float32, shape=(N_pkt, embedding_dim)
    )

    # ── L2-normalise MITRE embeddings (tiny — stays in RAM) ───────────────────
    mitre_emb_norm = _normalize_rows(mitre_embedding)

    # ── batched GPU/CPU cosine similarity ─────────────────────────────────────
    contain_edge_index = base_artifact.arrays["contain_edge_index"]
    num_flows          = int(base_artifact.metadata["num_flows"])
    contain_flow       = contain_edge_index[0] if contain_edge_index.shape[1] > 0 else np.empty(0, np.int64)
    contain_pkt        = contain_edge_index[1] if contain_edge_index.shape[1] > 0 else np.empty(0, np.int64)

    print(
        f"[4/6] Cosine similarity  "
        f"packets={N_pkt:,}  techniques={mitre_emb_norm.shape[0]}  "
        f"device={device}  batch={args.sim_batch_size:,}",
        flush=True,
    )
    pkt_src, pkt_dst, pkt_score, flow_tech_sum, flow_tech_count = _compute_sim_edges_and_flow_sums(
        student_emb=student_embedding,
        packet_payload_idx=packet_payload_row_index,
        mitre_emb=mitre_emb_norm,
        contain_flow=contain_flow,
        contain_pkt=contain_pkt,
        num_flows=num_flows,
        packet_top_k=args.packet_top_k,
        threshold=args.similarity_threshold,
        device=device,
        batch_size=args.sim_batch_size,
        packet_semantic_out=packet_semantic_mmap,
    )
    packet_semantic_mmap.flush()
    del packet_semantic_mmap  # close mmap; file stays on disk
    gc.collect()

    if pkt_src.size > 0:
        packet_technique_edge_index = np.vstack([pkt_src, pkt_dst])
        packet_technique_edge_attr  = pkt_score[:, np.newaxis]
    else:
        packet_technique_edge_index = np.empty((2, 0), dtype=np.int64)
        packet_technique_edge_attr  = np.empty((0, 1),  dtype=np.float32)

    # ── flow→technique edges ──────────────────────────────────────────────────
    print("[5/6] Building flow→technique edges …", flush=True)
    flow_src, flow_dst, flow_score = _select_flow_edges(
        flow_tech_sum=flow_tech_sum,
        flow_tech_count=flow_tech_count,
        flow_top_k=args.flow_top_k,
        threshold=args.similarity_threshold,
        device=device,
    )

    if flow_src.size > 0:
        flow_technique_edge_index = np.vstack([flow_src, flow_dst])
        flow_technique_edge_attr  = flow_score[:, np.newaxis]
    else:
        flow_technique_edge_index = np.empty((2, 0), dtype=np.int64)
        flow_technique_edge_attr  = np.empty((0, 1),  dtype=np.float32)

    # ── technique→tactic edges ────────────────────────────────────────────────
    technique_tactic_edges_df = pd.read_csv(mitre_technique_tactic_edges)
    required_edge_cols = {"technique_id", "tactic_shortname"}
    if not required_edge_cols.issubset(technique_tactic_edges_df.columns):
        raise ValueError(f"MITRE technique-tactic edges CSV must contain {sorted(required_edge_cols)}")

    (
        technique_tactic_edge_index,
        technique_tactic_edge_attr,
        technique_id_to_idx,
        tactic_shortname_to_idx,
    ) = _build_technique_tactic_arrays(techniques_df, technique_tactic_edges_df)

    # ── assemble and save ─────────────────────────────────────────────────────
    # packet_semantic_x is already on disk as packet_semantic_npy (mmap file written above).
    # Excluding it from the compressed NPZ avoids decompressing/recompressing ~59 GB in RAM.
    print("[6/6] Saving NPZ …", flush=True)
    arrays = dict(base_artifact.arrays)
    arrays.update(
        {
            "technique_x":                   mitre_embedding,
            "packet_technique_edge_index":   packet_technique_edge_index,
            "packet_technique_edge_attr":    packet_technique_edge_attr,
            "flow_technique_edge_index":     flow_technique_edge_index,
            "flow_technique_edge_attr":      flow_technique_edge_attr,
            "technique_tactic_edge_index":   technique_tactic_edge_index,
            "technique_tactic_edge_attr":    technique_tactic_edge_attr,
        }
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed savez: ``packet_semantic_x`` is the only big array, and it's
    # already excluded (lives as the sidecar memmap built earlier). Everything
    # left here is int64 edge indices + small static features — they compress
    # poorly (high-entropy ids) and savez_compressed used to spend most of its
    # runtime on the deflate pass for no real disk-size win.
    np.savez(output_npz, **arrays)

    meta = {
        "created_at_utc":                   datetime.now(timezone.utc).isoformat(),
        "metadata_csv":                     str(metadata_csv),
        "payload_npy":                      str(payload_npy),
        "student_embedding_npy":            str(student_embedding_npy),
        "mitre_techniques_csv":             str(mitre_techniques_csv),
        "mitre_technique_embeddings_npy":   str(mitre_technique_embeddings_npy),
        "mitre_technique_tactic_edges_csv": str(mitre_technique_tactic_edges),
        "graph_output_npz":                 str(output_npz),
        "packet_semantic_x_npy":           str(packet_semantic_npy),
        "device":                           str(device),
        "sim_batch_size":                   int(args.sim_batch_size),
        "flow_timeout_seconds":             float(args.flow_timeout_seconds),
        "max_packets_per_flow":             int(args.max_packets_per_flow),
        "similarity_threshold":             float(args.similarity_threshold),
        "packet_top_k":                     int(args.packet_top_k),
        "flow_top_k":                       int(args.flow_top_k),
        "num_flows":                        int(base_artifact.metadata["num_flows"]),
        "num_packets":                      int(base_artifact.metadata["num_packets"]),
        "num_techniques":                   int(mitre_embedding.shape[0]),
        "num_tactics":                      int(len(tactic_shortname_to_idx)),
        "num_contain_edges":                int(base_artifact.metadata["num_contain_edges"]),
        "num_link_edges":                   int(base_artifact.metadata["num_link_edges"]),
        "num_packet_technique_edges":       int(packet_technique_edge_index.shape[1]),
        "num_flow_technique_edges":         int(flow_technique_edge_index.shape[1]),
        "num_technique_tactic_edges":       int(technique_tactic_edge_index.shape[1]),
        "payload_length":                   int(base_artifact.metadata["payload_length"]),
        "embedding_dim":                    int(student_embedding.shape[1]),
        "label_mapping":                    base_artifact.metadata.get("label_mapping", {}),
        "protocol_mapping":                 base_artifact.metadata.get("protocol_mapping", {}),
        "flow_feature_names":               base_artifact.metadata.get("flow_feature_names", []),
        "technique_id_to_idx":              technique_id_to_idx,
        "tactic_shortname_to_idx":          tactic_shortname_to_idx,
    }
    write_json(output_meta_json, meta)

    print(f"[OK] Three-tier graph: {output_npz}", flush=True)
    print(f"[OK] Metadata:         {output_meta_json}", flush=True)
    print(
        f"[OK] Counts → flows={meta['num_flows']:,}  packets={meta['num_packets']:,}  "
        f"techniques={meta['num_techniques']}  tactics={meta['num_tactics']}  "
        f"pkt→tech edges={meta['num_packet_technique_edges']:,}  "
        f"flow→tech edges={meta['num_flow_technique_edges']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
