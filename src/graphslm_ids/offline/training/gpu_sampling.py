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


import numpy as np
from graphslm_ids.offline.training.neighbor_sampling import MiniBatchSubgraph
from graphslm_ids.offline.training.on_disk_graph_store import edge_key_to_name


def _unique_preserve_order_torch(values, device):
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    _, first_idx = np.unique(arr, return_index=True)
    ordered = arr[np.sort(first_idx)]
    return torch.from_numpy(ordered).to(device)


class GpuNeighborBackend:
    """Holds CSR topology + small node features on the compute device."""

    def __init__(self, in_memory_backend, device: torch.device) -> None:
        self.device = device
        self.edge_types = list(in_memory_backend.edge_types)
        self.feature_dims = dict(in_memory_backend.feature_dims)
        self._num_flows = in_memory_backend.num_flows
        self._num_techniques = in_memory_backend.num_techniques
        self._num_tactics = in_memory_backend.num_tactics
        art = in_memory_backend.artifact
        self._flow_x = torch.from_numpy(
            np.asarray(art.node_features["flow"], dtype=np.float32)
        ).to(device)
        self._flow_y = torch.from_numpy(
            np.asarray(art.flow_y, dtype=np.int64)
        ).to(device)
        self._csr: dict = {}
        for ek, (indptr, indices, attr) in in_memory_backend._edge_csr.items():
            self._csr[ek] = (
                torch.from_numpy(indptr).to(device),
                torch.from_numpy(indices).to(device),
                torch.from_numpy(attr).to(device),
            )

    @property
    def num_flows(self):
        return self._num_flows

    @property
    def num_techniques(self):
        return self._num_techniques

    @property
    def num_tactics(self):
        return self._num_tactics

    def get_flow_features(self, flow_ids):
        return self._flow_x[flow_ids]

    def get_flow_labels(self, flow_ids):
        return self._flow_y[flow_ids]

    def out_neighbors(self, edge_type, src_ids, fanout=None, generator=None):
        if edge_type not in self._csr:
            n = int(src_ids.numel())
            return (
                torch.zeros(n + 1, dtype=torch.int64, device=self.device),
                torch.empty(0, dtype=torch.int64, device=self.device),
                torch.empty(0, dtype=torch.float32, device=self.device),
            )
        indptr, indices, attr = self._csr[edge_type]
        return gather_csr_neighbors_torch(indptr, indices, attr, src_ids, fanout, generator)


class TorchHeteroNeighborSampler:
    """Torch K-hop sampler. Mirrors HeteroNeighborSampler.sample() but on
    device. Always defers packet/technique features (store gathers them).
    """

    def __init__(
        self,
        *,
        backend: GpuNeighborBackend,
        hops: int,
        fanouts=None,
        reverse_fanouts=None,
        always_include_all_tactics: bool = True,
        always_include_all_techniques: bool = True,
        standardize_flow_features: bool = False,
        seed: int = 42,
    ) -> None:
        self.backend = backend
        self.hops = int(hops)
        self.fanouts = {str(k): int(v) for k, v in (fanouts or {}).items()}
        self.reverse_fanouts = {str(k): int(v) for k, v in (reverse_fanouts or {}).items()}
        self.always_include_all_tactics = bool(always_include_all_tactics)
        self.always_include_all_techniques = bool(always_include_all_techniques)
        self.standardize_flow_features = bool(standardize_flow_features)
        self.generator = torch.Generator(device=backend.device).manual_seed(seed)

    def _fanout(self, edge_type):
        name = edge_key_to_name(edge_type)
        rel = edge_type[1]
        if rel.startswith("rev_") and name in self.reverse_fanouts:
            return self.reverse_fanouts[name]
        if name in self.fanouts:
            return self.fanouts[name]
        if rel in self.fanouts:
            return self.fanouts[rel]
        return None

    def sample(self, seed_flow_ids):
        dev = self.backend.device
        seeds = _unique_preserve_order_torch(seed_flow_ids, dev)
        maps = {t: TorchLocalMap(dev) for t in ("flow", "packet", "technique", "tactic")}
        maps["flow"].add(seeds)
        edge_chunks = {ek: [] for ek in self.backend.edge_types}
        frontier = {"flow": seeds.clone()}

        for _ in range(self.hops):
            nxt = {}
            for ek in self.backend.edge_types:
                src_t, _, dst_t = ek
                src_arr = frontier.get(src_t)
                if src_arr is None or src_arr.numel() == 0:
                    continue
                fan = self._fanout(ek)
                if fan == 0:
                    continue
                li, dst_ids, w = self.backend.out_neighbors(ek, src_arr, fan, self.generator)
                if dst_ids.numel() == 0:
                    continue
                counts = torch.diff(li)
                src_rep = torch.repeat_interleave(src_arr, counts)
                edge_chunks[ek].append((src_rep, dst_ids, w))
                maps[src_t].add(src_arr)
                newly = maps[dst_t].add(dst_ids)
                if newly.numel():
                    nxt.setdefault(dst_t, []).append(newly)
            if not nxt:
                frontier = {}
                break
            frontier = {
                t: (c[0] if len(c) == 1 else torch.cat(c)) for t, c in nxt.items()
            }

        if self.always_include_all_techniques:
            maps["technique"].add(torch.arange(self.backend.num_techniques, device=dev, dtype=torch.int64))
        if self.always_include_all_tactics:
            maps["tactic"].add(torch.arange(self.backend.num_tactics, device=dev, dtype=torch.int64))

        edge_index = {}
        edge_attr = {}
        for ek in self.backend.edge_types:
            chunks = edge_chunks.get(ek, [])
            if not chunks:
                edge_index[ek] = torch.empty((2, 0), dtype=torch.int64, device=dev)
                edge_attr[ek] = torch.empty(0, dtype=torch.float32, device=dev)
                continue
            src_t, _, dst_t = ek
            sg = torch.cat([c[0] for c in chunks])
            dg = torch.cat([c[1] for c in chunks])
            wg = torch.cat([c[2] for c in chunks])
            sl = maps[src_t].lookup(sg)
            dl = maps[dst_t].lookup(dg)
            valid = (sl >= 0) & (dl >= 0)
            if bool(valid.any()):
                edge_index[ek] = torch.stack([sl[valid], dl[valid]])
                edge_attr[ek] = wg[valid]
            else:
                edge_index[ek] = torch.empty((2, 0), dtype=torch.int64, device=dev)
                edge_attr[ek] = torch.empty(0, dtype=torch.float32, device=dev)

        flow_ids = maps["flow"].globals_array()
        packet_ids = maps["packet"].globals_array()
        technique_ids = maps["technique"].globals_array()
        tactic_ids = maps["tactic"].globals_array()

        seed_local = maps["flow"].lookup(seeds)
        seed_mask = torch.zeros(flow_ids.numel(), dtype=torch.bool, device=dev)
        valid_seed = seed_local >= 0
        if bool(valid_seed.any()):
            seed_mask[seed_local[valid_seed]] = True

        node_features = {
            "flow": self.backend.get_flow_features(flow_ids),
            "packet": torch.empty((0, self.backend.feature_dims["packet"]), device=dev),
            "technique": torch.empty((0, self.backend.feature_dims["technique"]), device=dev),
            "tactic": tactic_ids.reshape(-1, 1).to(torch.int64),
        }
        return MiniBatchSubgraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            seed_mask=seed_mask,
            seed_labels=self.backend.get_flow_labels(seeds),
            seed_flow_ids=seeds,
            local_to_global={
                "flow": flow_ids,
                "packet": packet_ids,
                "technique": technique_ids,
                "tactic": tactic_ids,
            },
        )
