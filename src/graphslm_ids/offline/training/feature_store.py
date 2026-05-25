"""Auto-scaling tiered feature store for packet_x.

Tiers: GPU hot cache (static frequency placement) -> CPU RAM warm -> disk
memmap cold. Only packet_x is tiered; all other node features are small and
stay fully resident on the compute device. See
docs/superpowers/specs/2026-05-25-tiered-feature-store-design.md.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

_LOG = logging.getLogger(__name__)


class PacketSource(Protocol):
    """Read-only row source for packet_x. Returns numpy rows in native dtype."""

    num_rows: int
    dim: int

    def gather(self, ids: np.ndarray) -> np.ndarray:
        ...


class ArrayPacketSource:
    """packet_x held fully in RAM as a numpy array."""

    def __init__(self, array: np.ndarray) -> None:
        if array.ndim != 2:
            raise ValueError("packet_x array must be 2-D (n_packets, dim)")
        self._array = array
        self.num_rows = int(array.shape[0])
        self.dim = int(array.shape[1])

    def gather(self, ids: np.ndarray) -> np.ndarray:
        return self._array[np.asarray(ids, dtype=np.int64)]


class MemmapPacketSource:
    """packet_x backed by a flat row-major .bin file via np.memmap (disk cold tier).

    OS page cache turns RAM into an automatic LRU over recently-read rows, so
    this single class covers both "fits in RAM" (kernel caches everything) and
    "larger than RAM" (kernel evicts cold pages).
    """

    def __init__(
        self, bin_path: Path, num_rows: int, dim: int, dtype: str = "float16"
    ) -> None:
        self._mm = np.memmap(
            str(bin_path), dtype=np.dtype(dtype), mode="r", shape=(num_rows, dim)
        )
        self.num_rows = int(num_rows)
        self.dim = int(dim)

    def gather(self, ids: np.ndarray) -> np.ndarray:
        # Copy out of the memmap so downstream torch.from_numpy owns the buffer.
        return np.asarray(self._mm[np.asarray(ids, dtype=np.int64)])


def compute_cache_capacity(
    *,
    device: torch.device,
    num_rows: int,
    row_bytes: int,
    free_bytes_override: int | None = None,
    model_reserve_bytes: int = 2 * 1024**3,
    cache_fraction: float = 0.6,
) -> int:
    """Number of packet rows that fit in the GPU hot cache.

    CPU device with no override -> 0 (no GPU cache tier). When
    ``free_bytes_override`` is given, it is used regardless of device type
    (test hook). Result is clamped to ``[0, num_rows]``.
    """
    if free_bytes_override is not None:
        free = int(free_bytes_override)
    elif device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
    else:
        return 0
    usable = max(0, free - int(model_reserve_bytes)) * float(cache_fraction)
    k = int(usable // int(row_bytes))
    return max(0, min(k, int(num_rows)))


class TieredFeatureStore:
    """Gathers packet rows from a GPU hot cache, falling back to the source.

    The hottest ``capacity`` rows (by ``freq_order``) live on ``device`` as a
    contiguous cache tensor; ``id_to_slot[g]`` gives the cache slot of global id
    ``g`` or -1 if not cached. ``gather`` returns rows on ``device`` in
    ``cache_dtype`` (bfloat16 on cuda, float32 on cpu by default).
    """

    def __init__(
        self,
        *,
        source: PacketSource,
        device: torch.device,
        freq_order: np.ndarray,
        capacity: int,
        cache_dtype: str | None = None,
    ) -> None:
        self.source = source
        self.device = device
        self.dim = int(source.dim)
        self.num_rows = int(source.num_rows)
        if cache_dtype is None:
            cache_dtype = "bfloat16" if device.type == "cuda" else "float32"
        self._torch_dtype = getattr(torch, cache_dtype)
        self.capacity = int(max(0, min(capacity, self.num_rows)))

        self._id_to_slot = np.full(self.num_rows, -1, dtype=np.int64)
        if self.capacity > 0:
            cached_ids = np.asarray(freq_order, dtype=np.int64)[: self.capacity]
            self._cached_ids = cached_ids
            self._id_to_slot[cached_ids] = np.arange(self.capacity, dtype=np.int64)
            rows = np.asarray(source.gather(cached_ids), dtype=np.float32)
            self._cache = torch.from_numpy(rows).to(
                device=device, dtype=self._torch_dtype
            )
            _LOG.info(
                "TieredFeatureStore: cached %d/%d rows on %s (%.2f GB)",
                self.capacity,
                self.num_rows,
                device,
                self._cache.element_size() * self._cache.nelement() / 1024**3,
            )
        else:
            self._cached_ids = np.empty(0, dtype=np.int64)
            self._cache = torch.empty((0, self.dim), device=device, dtype=self._torch_dtype)

    def gather(self, packet_ids: np.ndarray) -> torch.Tensor:
        ids = np.asarray(packet_ids, dtype=np.int64).reshape(-1)
        n = ids.shape[0]
        out = torch.empty((n, self.dim), device=self.device, dtype=self._torch_dtype)
        if n == 0:
            return out

        slots = self._id_to_slot[ids]
        hit = slots >= 0
        if hit.any():
            hit_idx = torch.from_numpy(np.nonzero(hit)[0]).to(self.device)
            hit_slots = torch.from_numpy(slots[hit]).to(self.device)
            out.index_copy_(0, hit_idx, self._cache.index_select(0, hit_slots))
        miss = ~hit
        if miss.any():
            miss_pos = np.nonzero(miss)[0]
            miss_rows = np.asarray(self.source.gather(ids[miss]), dtype=np.float32)
            miss_t = torch.from_numpy(miss_rows).to(
                device=self.device, dtype=self._torch_dtype
            )
            out.index_copy_(0, torch.from_numpy(miss_pos).to(self.device), miss_t)
        return out

    def state_dict(self) -> dict:
        return {"freq_order": self._cached_ids, "capacity": self.capacity}


def compute_access_frequency(
    *,
    sampler,
    seed_flow_ids: np.ndarray,
    num_packets: int,
    batch_size: int,
    n_warmup_batches: int,
    seed: int = 42,
) -> np.ndarray:
    """Count how often each packet node is sampled over warmup batches.

    Deterministic given ``seed``: seeds are shuffled with a fixed RNG and
    consumed in fixed-size batches. Returns an int64 array of length
    ``num_packets`` (use ``argsort(desc)`` to get freq_order for the cache).
    """
    seeds = np.asarray(seed_flow_ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(seeds.shape[0])
    seeds = seeds[order]
    freq = np.zeros(int(num_packets), dtype=np.int64)
    n = min(n_warmup_batches, max(1, seeds.shape[0] // max(1, batch_size)))
    for b in range(n):
        chunk = seeds[b * batch_size : (b + 1) * batch_size]
        if chunk.size == 0:
            break
        batch = sampler.sample(chunk.tolist())
        pkt = np.asarray(batch.local_to_global["packet"], dtype=np.int64)
        if pkt.size:
            np.add.at(freq, pkt, 1)
    return freq
