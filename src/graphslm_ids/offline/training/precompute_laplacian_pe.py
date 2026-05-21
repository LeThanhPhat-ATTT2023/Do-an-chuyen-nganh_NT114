"""One-time precompute of Laplacian PE for the NT114 flow graph.

Usage:
  python -m graphslm_ids.offline.training.precompute_laplacian_pe \
    --graph-store-root /home/ubuntu/dataset/graph_store_v1 \
    --output-path /home/ubuntu/dataset/graph_store_v1/laplacian_pe_flow.npy \
    --k 16

For NT114 the flow-flow graph is derived implicitly via shared packets:
two flows are "neighbors" if they share at least one packet. We build
this by aggregating ``flow→contains→packet`` edges. The resulting
adjacency is sparse (~tens of millions of edges) and the top-16
eigenvectors can be computed in ~5-15 minutes via Lanczos.

This script is intentionally separate from the training entry-point so
the PE computation is one-time and reusable across training runs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from graphslm_ids.offline.training.laplacian_pe import (
    compute_laplacian_pe,
    save_laplacian_pe,
)
from graphslm_ids.offline.training.on_disk_graph_store import OnDiskHeteroGraphStore


def build_flow_flow_edges_via_packets(store: OnDiskHeteroGraphStore) -> np.ndarray:
    """Construct symmetric flow-flow edges by co-packet membership.

    Two flows are connected if they share at least one packet node. For
    very large graphs this can produce many edges per shared packet
    (quadratic in the packet's flow count). We deduplicate via the
    structured-void view trick to keep memory bounded.

    Returns:
        ``(2, E)`` int64 array of flow-flow edges (no self-loops, deduplicated).
    """
    edge_key = ("flow", "contains", "packet")
    csr_indptr, csr_indices, _ = store.csr_for_edge(edge_key)
    # csr_indptr: per-flow offsets into csr_indices (which lists packet IDs).

    num_packets = store.manifest.get("num_packets") or int(csr_indices.max() + 1)
    # Inverse mapping: packet → list of flows.
    flow_for_packet: list[list[int]] = [[] for _ in range(num_packets)]
    num_flows = len(csr_indptr) - 1
    for flow_id in range(num_flows):
        start, end = csr_indptr[flow_id], csr_indptr[flow_id + 1]
        for pkt in csr_indices[start:end]:
            flow_for_packet[int(pkt)].append(flow_id)

    # For each packet, emit pairwise flow edges. Cap per-packet to avoid
    # explosion when a packet is referenced by very many flows.
    MAX_FLOWS_PER_PACKET = 32
    edges_src: list[int] = []
    edges_dst: list[int] = []
    for flows in flow_for_packet:
        if len(flows) < 2:
            continue
        if len(flows) > MAX_FLOWS_PER_PACKET:
            flows = flows[:MAX_FLOWS_PER_PACKET]
        for i in range(len(flows)):
            for j in range(i + 1, len(flows)):
                edges_src.append(flows[i])
                edges_dst.append(flows[j])

    if not edges_src:
        return np.zeros((2, 0), dtype=np.int64)

    src = np.array(edges_src, dtype=np.int64)
    dst = np.array(edges_dst, dtype=np.int64)
    edges = np.stack([src, dst], axis=1)
    # Deduplicate using structured-void view.
    edges_v = np.ascontiguousarray(edges).view(
        np.dtype([("a", np.int64), ("b", np.int64)])
    )
    _, unique_idx = np.unique(edges_v, return_index=True)
    edges_unique = edges[np.sort(unique_idx)]
    return edges_unique.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-store-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    store = OnDiskHeteroGraphStore(Path(args.graph_store_root))
    num_flows = int(store.manifest["num_flows"])
    print(f"[pe] graph store: num_flows={num_flows}", flush=True)

    print("[pe] building flow-flow edges via shared packets ...", flush=True)
    t0 = time.time()
    edge_index = build_flow_flow_edges_via_packets(store)
    print(
        f"[pe] flow-flow edges: shape={edge_index.shape} "
        f"(elapsed {time.time() - t0:.1f}s)",
        flush=True,
    )

    print(f"[pe] computing top-{args.k} Laplacian eigenvectors ...", flush=True)
    t0 = time.time()
    pe = compute_laplacian_pe(
        edge_index=edge_index,
        num_nodes=num_flows,
        k=args.k,
        seed=args.seed,
    )
    print(
        f"[pe] PE shape={pe.shape} dtype={pe.dtype} "
        f"(elapsed {time.time() - t0:.1f}s)",
        flush=True,
    )
    print(f"[pe] saving to {args.output_path}", flush=True)
    save_laplacian_pe(pe, args.output_path)
    print(f"[pe] done. file size = {Path(args.output_path).stat().st_size / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
