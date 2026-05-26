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
        # Sort by (row asc, key asc) via two stable sorts. Avoids the
        # float64 mantissa trick (row*(max_key+1)+key), which silently breaks
        # once n_src*(max_key+1) exceeds 2^52.
        inner = torch.argsort(keys, stable=True)
        order = inner[torch.argsort(row_of_edge[inner], stable=True)]
        sorted_row = row_of_edge[order]
        pos_in_seg = torch.arange(total_in, device=device, dtype=torch.int64) - cum_in[sorted_row]
        keep = pos_in_seg < counts[sorted_row]
        selected_in_order = order[keep]
        # Restore CSR-ascending order within each row -- same two-stable-sort
        # idiom keyed on (row asc, global_edge asc).
        sel_rows = row_of_edge[selected_in_order]
        sel_glob = global_edge[selected_in_order]
        inner = torch.argsort(sel_glob, stable=True)
        perm = inner[torch.argsort(sel_rows[inner], stable=True)]
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
        self._num_hosts = getattr(in_memory_backend, "num_hosts", 0)
        art = in_memory_backend.artifact
        self._flow_x = torch.from_numpy(
            np.asarray(art.node_features["flow"], dtype=np.float32)
        ).to(device)
        self._flow_y = torch.from_numpy(
            np.asarray(art.flow_y, dtype=np.int64)
        ).to(device)
        self._host_x: torch.Tensor | None = None
        if "host" in self.feature_dims and "host" in art.node_features:
            self._host_x = torch.from_numpy(
                np.asarray(art.node_features["host"], dtype=np.float32)
            ).to(device)
        # Technique features are tiny (~hundreds of rows) — keep them on device
        # and gather per-batch so node_features["technique"] is never empty.
        # (Packet stays deferred to packet_store because it's large.)
        self._technique_x: torch.Tensor | None = None
        if "technique" in art.node_features:
            self._technique_x = torch.from_numpy(
                np.asarray(art.node_features["technique"], dtype=np.float32)
            ).to(device)
        # Store CSR indptr/indices as int32 on device — halves topology footprint
        # vs int64. Node counts (~600k) fit easily under int32 max (~2.1B), and
        # gather_csr_neighbors_torch casts intermediate arithmetic back to int64
        # explicitly, so output dtype is unchanged and there is no quality impact
        # on downstream sampling/model. attr stays float32 (already 4 bytes).
        # Guard against overflow: error loudly if any single edge type exceeds
        # int32 range so future scale-ups don't silently corrupt indices.
        INT32_MAX = (1 << 31) - 1
        self._csr: dict = {}
        topology_bytes = 0
        for ek, (indptr, indices, attr) in in_memory_backend._edge_csr.items():
            n_edges = int(indices.shape[0])
            n_src_plus_one = int(indptr.shape[0])
            max_node_id = int(indices.max()) if n_edges else 0
            max_offset = int(indptr.max()) if n_src_plus_one else 0
            if max(max_node_id, max_offset) > INT32_MAX:
                # Fall back to int64 for this edge type rather than corrupt
                # indices; preserves the optimization for the common case.
                indptr_t = torch.from_numpy(indptr.astype(np.int64, copy=False)).to(device)
                indices_t = torch.from_numpy(indices.astype(np.int64, copy=False)).to(device)
            else:
                indptr_t = torch.from_numpy(indptr.astype(np.int32, copy=False)).to(device)
                indices_t = torch.from_numpy(indices.astype(np.int32, copy=False)).to(device)
            attr_t = torch.from_numpy(np.asarray(attr, dtype=np.float32)).to(device)
            self._csr[ek] = (indptr_t, indices_t, attr_t)
            topology_bytes += (
                indptr_t.element_size() * indptr_t.numel()
                + indices_t.element_size() * indices_t.numel()
                + attr_t.element_size() * attr_t.numel()
            )
        self._topology_bytes = topology_bytes

    @property
    def num_flows(self):
        return self._num_flows

    @property
    def num_techniques(self):
        return self._num_techniques

    @property
    def num_tactics(self):
        return self._num_tactics

    @property
    def num_hosts(self):
        return self._num_hosts

    def get_flow_features(self, flow_ids):
        return self._flow_x[flow_ids]

    def get_host_features(self, host_ids):
        if self._host_x is None:
            return torch.empty(
                (0, int(self.feature_dims.get("host", 0))),
                dtype=torch.float32,
                device=self.device,
            )
        return self._host_x[host_ids]

    def get_technique_features(self, technique_ids):
        if self._technique_x is None:
            return torch.empty(
                (0, int(self.feature_dims.get("technique", 0))),
                dtype=torch.float32,
                device=self.device,
            )
        return self._technique_x[technique_ids]

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
        flow_feature_stats: dict | None = None,
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
        # Parity with numpy HeteroNeighborSampler: when enabled AND stats are
        # provided, cache mean/std on device as 1xD row vectors so sample() can
        # standardize flow_x in-line (broadcast subtract / divide).
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        if self.standardize_flow_features and flow_feature_stats is not None:
            mean = flow_feature_stats.get("mean")
            std = flow_feature_stats.get("std")
            if mean is not None and std is not None:
                self._mean = torch.from_numpy(
                    np.asarray(mean, dtype=np.float32)
                ).reshape(1, -1).to(backend.device)
                self._std = torch.from_numpy(
                    np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
                ).reshape(1, -1).to(backend.device)

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
        maps = {t: TorchLocalMap(dev) for t in ("flow", "packet", "technique", "tactic", "host")}
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
        has_host = "host" in self.backend.feature_dims
        host_ids = (
            maps["host"].globals_array()
            if has_host
            else torch.empty(0, dtype=torch.int64, device=dev)
        )

        seed_local = maps["flow"].lookup(seeds)
        seed_mask = torch.zeros(flow_ids.numel(), dtype=torch.bool, device=dev)
        valid_seed = seed_local >= 0
        if bool(valid_seed.any()):
            seed_mask[seed_local[valid_seed]] = True

        flow_x = self.backend.get_flow_features(flow_ids)
        if (
            self._mean is not None
            and self._std is not None
            and flow_x.shape[1] == self._mean.shape[1]
        ):
            flow_x = (flow_x - self._mean) / self._std
        node_features = {
            "flow": flow_x,
            # packet stays deferred — to_torch_batch fills via packet_store (big tensor).
            "packet": torch.empty((0, self.backend.feature_dims["packet"]), device=dev),
            # technique is small — gather on device so encode() can project it.
            "technique": (
                self.backend.get_technique_features(technique_ids)
                if technique_ids.numel()
                else torch.empty(
                    (0, int(self.backend.feature_dims["technique"])),
                    dtype=torch.float32,
                    device=dev,
                )
            ),
            "tactic": tactic_ids.reshape(-1, 1).to(torch.int64),
        }
        if has_host:
            node_features["host"] = (
                self.backend.get_host_features(host_ids)
                if host_ids.numel()
                else torch.empty(
                    (0, int(self.backend.feature_dims["host"])),
                    dtype=torch.float32,
                    device=dev,
                )
            )
        local_to_global = {
            "flow": flow_ids,
            "packet": packet_ids,
            "technique": technique_ids,
            "tactic": tactic_ids,
        }
        if has_host:
            local_to_global["host"] = host_ids
        return MiniBatchSubgraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            seed_mask=seed_mask,
            seed_labels=self.backend.get_flow_labels(seeds),
            seed_flow_ids=seeds,
            local_to_global=local_to_global,
        )
