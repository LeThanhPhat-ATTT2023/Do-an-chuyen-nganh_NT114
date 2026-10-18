"""Dense helpers on (N, D) client-delta matrices.

N (clients per round) is small; D (HGT parameters, ~6.5M for the full model) is
large, so coordinate-wise reductions run over column chunks to bound memory.

Convention: torch for the (N, D) work, numpy float64 for per-client scalars.
Inputs must be finite; the server rejects non-finite deltas before aggregation.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

CHUNK = 1 << 20


def _check_chunk(chunk: int) -> None:
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")


def pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    out = torch.zeros((n, n), dtype=torch.float64, device=x.device)
    for i in range(n):
        # Direct differences on purpose (no |a|^2+|b|^2-2ab cancellation): do not
        # "simplify" to cdist/vector_norm. In-place square halves peak memory.
        out[i] = (x - x[i]).pow_(2).sum(dim=1).to(torch.float64)
    return out


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    """float64 (N, N) cosine matrix; zero-norm rows give cosine 0 (including a 0 diagonal entry)."""
    gram = (x @ x.T).to(torch.float64)
    norms = gram.diagonal().clamp_min(1e-24).sqrt()
    return gram / (norms[:, None] * norms[None, :])


def row_norms(x: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """float64 (N,) L2 norms from chunked sums of squares.

    float32 ``Tensor.norm()`` over millions of columns has ~2e-4 relative error;
    accumulating per-chunk sums in float64 avoids that without a float64 copy of x.
    """
    _check_chunk(chunk)
    acc = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    for a in range(0, x.shape[1], chunk):
        acc += x[:, a:a + chunk].pow(2).sum(dim=1).to(torch.float64)
    return acc.sqrt()


def cosine_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """float64 cosine of each row of x with ref; a zero-norm row (or ref) gives 0, not NaN."""
    ref = ref.to(x.device, x.dtype)
    # Elementwise product + chunked sum, accumulated in float64 (the row_norms
    # pattern): a float32 sgemv's accumulation depends on the BLAS thread count.
    dots = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    for a in range(0, x.shape[1], CHUNK):
        dots += (x[:, a:a + CHUNK] * ref[a:a + CHUNK]).sum(dim=1).to(torch.float64)
    denom = row_norms(x) * row_norms(ref.unsqueeze(0))[0]
    return dots / denom.clamp_min(1e-24)


def _sorted_chunks(x: torch.Tensor, chunk: int) -> Iterator[tuple[int, torch.Tensor]]:
    for start in range(0, x.shape[1], chunk):
        yield start, x[:, start:start + chunk].sort(dim=0).values


def coordinate_median(x: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """Coordinate-wise median; even N averages the two middle values (unlike torch.median)."""
    _check_chunk(chunk)
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot take the median of zero rows")
    out = torch.empty(x.shape[1], dtype=x.dtype, device=x.device)
    for start, s in _sorted_chunks(x, chunk):
        end = start + s.shape[1]
        out[start:end] = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return out


def coordinate_trimmed_mean(x: torch.Tensor, k: int, chunk: int = CHUNK) -> torch.Tensor:
    _check_chunk(chunk)
    n = x.shape[0]
    if k < 0 or 2 * k >= n:
        raise ValueError(f"cannot trim {k} per side from {n} rows")
    out = torch.empty(x.shape[1], dtype=x.dtype, device=x.device)
    for start, s in _sorted_chunks(x, chunk):
        out[start:start + s.shape[1]] = s[k:n - k].mean(dim=0)
    return out


def weighted_trimmed_mean(x: torch.Tensor, w: torch.Tensor, beta: float, chunk: int = CHUNK) -> torch.Tensor:
    """Coordinate-wise trimmed mean that trims weight mass ``beta`` from each end.

    Per coordinate the values are sorted and carry their weights (normalized to sum 1) with
    cumulative mass ``c_i``; element i keeps ``max(0, min(c_i, 1 - beta) - max(c_{i-1}, beta))``
    of its mass and the result is ``sum(kept * value) / (1 - 2 beta)``. Equal weights and
    ``beta = k / N`` give ``coordinate_trimmed_mean(x, k)``; ``beta = 0`` gives the weighted mean.
    """
    _check_chunk(chunk)
    beta = float(beta)
    if not 0.0 <= beta < 0.5:
        raise ValueError(f"beta must be in [0, 0.5), got {beta}")
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot take a weighted trimmed mean of zero rows")
    w = torch.as_tensor(w).to(x.device, torch.float64).reshape(-1)
    if w.shape[0] != n:
        raise ValueError(f"got {w.shape[0]} weights for {n} rows")
    if bool((w < 0).any()) or not float(w.sum()) > 0:
        raise ValueError("weights must be non-negative with a positive sum")
    w = w / w.sum()
    out = torch.empty(x.shape[1], dtype=x.dtype, device=x.device)
    for start in range(0, x.shape[1], chunk):
        vals, order = x[:, start:start + chunk].sort(dim=0)
        cum = w[order].cumsum_(dim=0)  # float64 mass; in-place ops bound the per-chunk memory
        del order
        kept = cum.clamp(max=1.0 - beta)
        lower = cum.clamp_(min=beta)  # max(c_{i-1}, beta) once shifted down by one row
        kept[1:] -= lower[:-1]
        kept[0] -= beta  # c_{-1} = 0
        kept.clamp_(min=0.0)
        out[start:start + vals.shape[1]] = (kept.mul_(vals).sum(dim=0) / (1.0 - 2.0 * beta)).to(x.dtype)
    return out


def top_right_singular_vector(x: torch.Tensor) -> torch.Tensor:
    _u, _s, vh = torch.linalg.svd(x.to(torch.float32), full_matrices=False)
    return vh[0].to(x.dtype)


def robust_center_scale(
    values: np.ndarray | list[float], axis: int = 0, floor: float = 1e-6
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Median and floored MAD scale along ``axis`` (reduced axis removed; scalars for 1-D)."""
    v = np.asarray(values, dtype=np.float64)
    center = np.median(v, axis=axis, keepdims=True)
    mad = np.median(np.abs(v - center), axis=axis, keepdims=True)
    scale = np.maximum(1.4826 * mad, floor)
    return np.squeeze(center, axis=axis)[()], np.squeeze(scale, axis=axis)[()]


def robust_z(values: np.ndarray | list[float], floor: float = 1e-6) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    center, scale = robust_center_scale(v, axis=0, floor=floor)
    return (v - center) / scale


def _offdiag_rows(mat: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mat, dtype=np.float64)
    mask = ~np.eye(m.shape[0], dtype=bool)
    return [m[i][mask[i]] for i in range(m.shape[0])]


def median_excluding_self(mat: np.ndarray) -> np.ndarray:
    """Per-row median of off-diagonal entries; an empty row yields 1.0 (meant for cosine matrices)."""
    return np.array([float(np.median(r)) if r.size else 1.0 for r in _offdiag_rows(mat)])


def mean_excluding_self(mat: np.ndarray) -> np.ndarray:
    """Per-row mean of off-diagonal entries; an empty row yields 1.0 (meant for cosine matrices)."""
    return np.array([float(r.mean()) if r.size else 1.0 for r in _offdiag_rows(mat)])
