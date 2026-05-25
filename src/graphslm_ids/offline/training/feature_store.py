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
