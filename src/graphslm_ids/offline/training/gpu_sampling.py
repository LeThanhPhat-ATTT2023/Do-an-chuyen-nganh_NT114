"""Torch (GPU-capable) neighbor sampling for Phase C.

Mirrors the numpy CSR sampler in on_disk_graph_store.py but operates on torch
tensors so the whole K-hop expansion can run on the compute device. The numpy
versions remain the parity oracle (see tests).
"""
from __future__ import annotations

import torch


def gather_csr_neighbors_torch(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    attr: torch.Tensor,
    src_ids: torch.Tensor,
    fanout: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched CSR neighbor gather. Returns (local_indptr, nbrs, weights).

    fanout=None keeps every neighbour in CSR-ascending order (exact parity with
    the numpy oracle). With fanout, keeps min(degree, fanout) per row via a
    uniform-key top-k, then restores CSR-ascending order inside each row.
    """
    device = indptr.device
    src = src_ids.reshape(-1).to(torch.int64)
    n_src = int(src.numel())
    empty = (
        torch.zeros(n_src + 1, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
    )
    if n_src == 0 or (fanout is not None and int(fanout) == 0):
        return empty

    starts = indptr[src]
    ends = indptr[src + 1]
    degrees = (ends - starts).to(torch.int64)
    total_in = int(degrees.sum())
    if total_in == 0:
        return empty

    counts = degrees if fanout is None else torch.clamp(degrees, max=int(fanout))

    row_of_edge = torch.repeat_interleave(
        torch.arange(n_src, device=device, dtype=torch.int64), degrees
    )
    cum_in = torch.zeros(n_src + 1, dtype=torch.int64, device=device)
    torch.cumsum(degrees, dim=0, out=cum_in[1:])
    offset_in_row = torch.arange(total_in, device=device, dtype=torch.int64) - cum_in[row_of_edge]
    global_edge = torch.repeat_interleave(starts, degrees) + offset_in_row

    if int(counts.sum()) == total_in:
        selected = global_edge
    else:
        keys = torch.rand(total_in, device=device, generator=generator)
        # Sort by (row asc, key asc); keep first counts[r] per row.
        order = torch.argsort(row_of_edge.to(torch.float64) * (keys.max() + 1.0) + keys, stable=True)
        sorted_row = row_of_edge[order]
        pos_in_seg = torch.arange(total_in, device=device, dtype=torch.int64) - cum_in[sorted_row]
        keep = pos_in_seg < counts[sorted_row]
        selected_in_order = order[keep]
        # Restore CSR-ascending order within each row.
        sel_rows = row_of_edge[selected_in_order]
        sel_glob = global_edge[selected_in_order]
        max_glob = int(global_edge.max().item()) + 1
        perm = torch.argsort(sel_rows * max_glob + sel_glob, stable=True)
        selected = selected_in_order[perm]

    local_indptr = torch.zeros(n_src + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=local_indptr[1:])
    nbrs = indices[selected].to(torch.int64)
    weights = attr[selected].to(torch.float32)
    return local_indptr, nbrs, weights
