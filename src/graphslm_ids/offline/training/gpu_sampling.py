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


class TorchLocalMap:
    """Torch port of neighbor_sampling._LocalMap.

    Assigns dense local ids to global ids in first-occurrence order, with a
    sorted view for O(log n) bulk lookup. Mirrors the numpy version's ordering
    exactly so subgraphs are identical across backends.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._chunks: list[torch.Tensor] = []
        self._count = 0
        self._sorted = torch.empty(0, dtype=torch.int64, device=device)
        self._local_of_sorted = torch.empty(0, dtype=torch.int64, device=device)

    @property
    def count(self) -> int:
        return self._count

    def add(self, candidates: torch.Tensor) -> torch.Tensor:
        arr = candidates.reshape(-1).to(torch.int64)
        if arr.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        uniq_sorted, inverse = torch.unique(arr, sorted=True, return_inverse=True)
        order = torch.arange(arr.numel(), device=self.device, dtype=torch.int64)
        # first_idx[u] = min(order[i] for i where arr[i] == uniq_sorted[u])
        first_idx = torch.full(
            (uniq_sorted.numel(),), arr.numel(), dtype=torch.int64, device=self.device
        )
        first_idx = first_idx.scatter_reduce(0, inverse, order, reduce="amin", include_self=True)

        if self._sorted.numel():
            pos = torch.searchsorted(self._sorted, uniq_sorted)
            pos_clamped = pos.clamp(max=self._sorted.numel() - 1)
            present = (pos < self._sorted.numel()) & (self._sorted[pos_clamped] == uniq_sorted)
            new_mask = ~present
        else:
            new_mask = torch.ones(uniq_sorted.numel(), dtype=torch.bool, device=self.device)

        if not bool(new_mask.any()):
            return torch.empty(0, dtype=torch.int64, device=self.device)

        new_sorted = uniq_sorted[new_mask]
        new_first = first_idx[new_mask]
        ins_order = torch.argsort(new_first, stable=True)
        new_in_insert = new_sorted[ins_order]
        n_new = int(new_in_insert.numel())
        new_locals_insert = torch.arange(
            self._count, self._count + n_new, dtype=torch.int64, device=self.device
        )
        self._count += n_new
        self._chunks.append(new_in_insert)

        new_locals_for_sorted = torch.empty(n_new, dtype=torch.int64, device=self.device)
        new_locals_for_sorted[ins_order] = new_locals_insert

        if self._sorted.numel():
            merged_vals = torch.cat([self._sorted, new_sorted])
            merged_locs = torch.cat([self._local_of_sorted, new_locals_for_sorted])
            sort_idx = torch.argsort(merged_vals, stable=True)
            self._sorted = merged_vals[sort_idx]
            self._local_of_sorted = merged_locs[sort_idx]
        else:
            self._sorted = new_sorted
            self._local_of_sorted = new_locals_for_sorted
        return new_in_insert

    def lookup(self, query: torch.Tensor) -> torch.Tensor:
        arr = query.reshape(-1).to(torch.int64)
        out = torch.full(arr.shape, -1, dtype=torch.int64, device=self.device)
        if arr.numel() == 0 or self._sorted.numel() == 0:
            return out
        pos = torch.searchsorted(self._sorted, arr)
        in_bounds = pos < self._sorted.numel()
        pos_clamped = pos.clamp(max=self._sorted.numel() - 1)
        match = in_bounds & (self._sorted[pos_clamped] == arr)
        out[match] = self._local_of_sorted[pos[match]]
        return out

    def globals_array(self) -> torch.Tensor:
        if self._count == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if len(self._chunks) == 1:
            return self._chunks[0]
        out = torch.cat(self._chunks)
        self._chunks = [out]
        return out
