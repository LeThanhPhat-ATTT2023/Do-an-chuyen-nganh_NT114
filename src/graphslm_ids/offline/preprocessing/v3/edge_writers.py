"""Streaming spill-to-disk edge buffer for the v3 graph builder.

The v3 build emits 5 typed-evidence edge types over 600K packets; in the
worst case each packet contributes 5 candidate edges and the in-memory list
of (src, dst, weight) triples for one edge type peaks around 12M entries
(~150 MB as a Python list of tuples, ~3x that as the underlying objects).
With 5 edge types in flight simultaneously, the naive "build then concatenate"
approach blows past the 16 GB local RAM budget.

:class:`MemmapEdgeWriter` solves this by:

1. Buffering appended edges in two parallel Python lists (src+dst, attr).
2. When the buffer reaches ``initial_cap`` rows, writing the chunk to two
   ``.bin`` files (one for ``int64`` edge_index, one for ``float32``
   edge_attr) under ``tmp_dir/<edge_type>/`` and clearing the buffer.
3. ``finalize()`` concatenates all on-disk chunks plus the residual in-memory
   buffer into the final ``(2, N)`` and ``(N, attr_dim)`` arrays the graph
   builder needs.

Why .bin files and not np.memmap directly? Two reasons:

* We don't know ``N`` upfront — would have to over-allocate or grow the
  memmap, both of which are painful on Windows where filesystem-level resize
  is not atomic.
* Chunked .bin lets the finalize step decide whether to materialise the
  result in RAM (small edges) or hand back a memmap (large edges). For v3
  it's always small enough to materialise; the API leaves room to switch
  later without breaking callers.

Memory budget: the writer itself holds at most ``initial_cap * (2*8 + 4*attr_dim)``
bytes in RAM (default 1M rows * 20 bytes/row ≈ 20 MB per writer).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_LOG = logging.getLogger(__name__)


def _edge_type_dirname(edge_type: tuple[str, str, str]) -> str:
    """Filesystem-safe directory name for a ``(src, rel, dst)`` triple."""
    return "__".join(edge_type)


class MemmapEdgeWriter:
    """Append-only edge buffer that spills to disk when memory pressure hits.

    Usage:
        >>> w = MemmapEdgeWriter(tmp_dir, ("packet", "evidence_injection", "technique"))
        >>> w.append(src=12, dst=4, attr=0.7)
        >>> w.extend(src_array, dst_array, attr_array)
        >>> edge_index, edge_attr = w.finalize()
        >>> w.close()

    Not thread-safe. Each writer owns its own spill directory.
    """

    def __init__(
        self,
        tmp_dir: Path,
        edge_type: tuple[str, str, str],
        attr_dim: int = 1,
        initial_cap: int = 1_000_000,
    ) -> None:
        """Configure the writer and create its spill directory.

        Args:
            tmp_dir: root directory for spill files. A subdirectory named
                after ``edge_type`` is created underneath.
            edge_type: ``(src_node_type, relation, dst_node_type)``. Used
                only for naming the spill directory and logging.
            attr_dim: per-edge attribute dimensionality. Pass 1 for scalar
                weights, >1 for multi-channel attrs.
            initial_cap: number of rows the in-memory buffer holds before
                spilling to disk.
        """
        if attr_dim < 1:
            raise ValueError("attr_dim must be >= 1")
        if initial_cap < 1:
            raise ValueError("initial_cap must be >= 1")
        self._attr_dim = int(attr_dim)
        self._cap = int(initial_cap)
        self._edge_type = edge_type
        self._spill_dir = Path(tmp_dir) / _edge_type_dirname(edge_type)
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        # Two parallel buffers so flush can dump them as contiguous arrays
        # without zip/unzip overhead. ``_attr_buf`` is a list of rows
        # (each row a list of length attr_dim) — boxed but linear-time to
        # convert to np.ndarray at flush time.
        self._src_buf: list[int] = []
        self._dst_buf: list[int] = []
        self._attr_buf: list[list[float]] = []
        self._chunk_idx = 0
        self._closed = False

    def _flush(self) -> None:
        """Write the in-memory buffer to disk and clear it."""
        if not self._src_buf:
            return
        n = len(self._src_buf)
        edge_index = np.empty((2, n), dtype=np.int64)
        edge_index[0, :] = self._src_buf
        edge_index[1, :] = self._dst_buf
        edge_attr = np.asarray(self._attr_buf, dtype=np.float32).reshape(
            n, self._attr_dim
        )

        idx_path = self._spill_dir / f"chunk_{self._chunk_idx:05d}_idx.bin"
        attr_path = self._spill_dir / f"chunk_{self._chunk_idx:05d}_attr.bin"
        edge_index.tofile(idx_path)
        edge_attr.tofile(attr_path)

        self._src_buf.clear()
        self._dst_buf.clear()
        self._attr_buf.clear()
        self._chunk_idx += 1

    def _normalise_attr(self, attr: float | list[float] | None) -> list[float]:
        """Coerce an attr value into a length-``attr_dim`` list of floats."""
        if attr is None:
            return [0.0] * self._attr_dim
        if isinstance(attr, (int, float)):
            if self._attr_dim != 1:
                raise ValueError(
                    f"attr_dim={self._attr_dim} but got scalar attr; "
                    "pass a list of the correct length"
                )
            return [float(attr)]
        attr_list = [float(x) for x in attr]
        if len(attr_list) != self._attr_dim:
            raise ValueError(
                f"attr length {len(attr_list)} != attr_dim {self._attr_dim}"
            )
        return attr_list

    def append(
        self,
        src: int,
        dst: int,
        attr: float | list[float] | None = None,
    ) -> None:
        """Add one edge. Spills the in-memory buffer once it hits ``initial_cap``."""
        if self._closed:
            raise RuntimeError("MemmapEdgeWriter is closed")
        self._src_buf.append(int(src))
        self._dst_buf.append(int(dst))
        self._attr_buf.append(self._normalise_attr(attr))
        if len(self._src_buf) >= self._cap:
            self._flush()

    def extend(
        self,
        src_array: np.ndarray,
        dst_array: np.ndarray,
        attr_array: np.ndarray | None = None,
    ) -> None:
        """Bulk-append from NumPy arrays. Spills as soon as the cap is reached.

        ``attr_array`` shape may be ``(N,)`` (only valid when ``attr_dim==1``)
        or ``(N, attr_dim)``. ``None`` defaults every new row's attr to zeros.
        """
        if self._closed:
            raise RuntimeError("MemmapEdgeWriter is closed")
        src_array = np.asarray(src_array, dtype=np.int64).ravel()
        dst_array = np.asarray(dst_array, dtype=np.int64).ravel()
        if src_array.shape != dst_array.shape:
            raise ValueError("src_array and dst_array must have the same shape")
        n = src_array.size
        if attr_array is None:
            attr2d = np.zeros((n, self._attr_dim), dtype=np.float32)
        else:
            attr2d = np.asarray(attr_array, dtype=np.float32)
            if attr2d.ndim == 1:
                if self._attr_dim != 1:
                    raise ValueError(
                        f"attr_array is 1-D but attr_dim={self._attr_dim}; "
                        "pass a 2-D array of the correct shape"
                    )
                attr2d = attr2d.reshape(-1, 1)
            if attr2d.shape != (n, self._attr_dim):
                raise ValueError(
                    f"attr_array shape {attr2d.shape} incompatible with "
                    f"(n={n}, attr_dim={self._attr_dim})"
                )

        # Convert to Python lists at the buffer boundary — keeps append() and
        # extend() consistent and avoids carrying mixed types through flush.
        self._src_buf.extend(src_array.tolist())
        self._dst_buf.extend(dst_array.tolist())
        self._attr_buf.extend(attr2d.tolist())
        if len(self._src_buf) >= self._cap:
            self._flush()

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialise the full edge_index + edge_attr from all chunks.

        Returns:
            ``(edge_index, edge_attr)`` of shapes ``(2, N)`` and
            ``(N, attr_dim)``. Empty edge case yields zero-width arrays of
            the correct dtypes.
        """
        if self._closed:
            raise RuntimeError("MemmapEdgeWriter is closed")
        # Flush whatever remains in memory so finalize only deals with files
        # — simpler than special-casing the residual buffer.
        self._flush()

        idx_chunks: list[np.ndarray] = []
        attr_chunks: list[np.ndarray] = []
        for chunk_idx in range(self._chunk_idx):
            idx_path = self._spill_dir / f"chunk_{chunk_idx:05d}_idx.bin"
            attr_path = self._spill_dir / f"chunk_{chunk_idx:05d}_attr.bin"
            if not idx_path.exists():
                continue
            # int64 file holds 2*n entries laid out as row-major (2, n).
            raw = np.fromfile(idx_path, dtype=np.int64)
            if raw.size == 0:
                continue
            n = raw.size // 2
            idx_chunks.append(raw.reshape(2, n))
            attr_raw = np.fromfile(attr_path, dtype=np.float32)
            attr_chunks.append(attr_raw.reshape(n, self._attr_dim))

        if not idx_chunks:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, self._attr_dim), dtype=np.float32)
            return edge_index, edge_attr

        edge_index = np.concatenate(idx_chunks, axis=1)
        edge_attr = np.concatenate(attr_chunks, axis=0)
        _LOG.info(
            "MemmapEdgeWriter[%s]: finalized %d edges over %d chunks",
            _edge_type_dirname(self._edge_type),
            edge_index.shape[1],
            len(idx_chunks),
        )
        return edge_index, edge_attr

    def close(self) -> None:
        """Release buffers. Spill directory is NOT removed (caller's job)."""
        self._src_buf.clear()
        self._dst_buf.clear()
        self._attr_buf.clear()
        self._closed = True
