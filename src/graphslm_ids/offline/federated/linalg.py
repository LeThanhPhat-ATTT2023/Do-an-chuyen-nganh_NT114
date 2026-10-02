"""Dense helpers on (N, D) client-delta matrices.

N (clients per round) is small; D (HGT parameters, ~6.5M for the full model) is
large, so coordinate-wise reductions run over column chunks to bound memory.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

CHUNK = 1 << 20


def pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    out = torch.zeros((n, n), dtype=torch.float64, device=x.device)
    for i in range(n):
        out[i] = (x - x[i]).pow(2).sum(dim=1).to(torch.float64)
    return out


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    gram = (x @ x.T).to(torch.float64)
    norms = gram.diagonal().clamp_min(1e-24).sqrt()
    return gram / (norms[:, None] * norms[None, :])


def cosine_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    ref = ref.to(x.device, x.dtype)
    dots = (x @ ref).to(torch.float64)
    denom = x.norm(dim=1).to(torch.float64) * float(ref.norm())
    return dots / denom.clamp_min(1e-24)


def _sorted_chunks(x: torch.Tensor, chunk: int) -> Iterator[tuple[int, torch.Tensor]]:
    for start in range(0, x.shape[1], chunk):
        yield start, x[:, start:start + chunk].sort(dim=0).values


def coordinate_median(x: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    n = x.shape[0]
    out = torch.empty(x.shape[1], dtype=x.dtype, device=x.device)
    for start, s in _sorted_chunks(x, chunk):
        end = start + s.shape[1]
        out[start:end] = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return out


def coordinate_trimmed_mean(x: torch.Tensor, k: int, chunk: int = CHUNK) -> torch.Tensor:
    n = x.shape[0]
    if k < 0 or 2 * k >= n:
        raise ValueError(f"cannot trim {k} per side from {n} rows")
    out = torch.empty(x.shape[1], dtype=x.dtype, device=x.device)
    for start, s in _sorted_chunks(x, chunk):
        out[start:start + s.shape[1]] = s[k:n - k].mean(dim=0)
    return out


def top_right_singular_vector(x: torch.Tensor) -> torch.Tensor:
    _u, _s, vh = torch.linalg.svd(x.to(torch.float32), full_matrices=False)
    return vh[0]


def robust_center_scale(
    values: np.ndarray | list[float], axis: int = 0, floor: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    center = np.median(v, axis=axis)
    scale = 1.4826 * np.median(np.abs(v - center), axis=axis)
    return center, np.maximum(scale, floor)


def robust_z(values: np.ndarray | list[float], floor: float = 1e-6) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    center, scale = robust_center_scale(v, axis=0, floor=floor)
    return (v - center) / scale


def _offdiag_rows(mat: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mat, dtype=np.float64)
    mask = ~np.eye(m.shape[0], dtype=bool)
    return [m[i][mask[i]] for i in range(m.shape[0])]


def median_excluding_self(mat: np.ndarray) -> np.ndarray:
    return np.array([float(np.median(r)) if r.size else 1.0 for r in _offdiag_rows(mat)])


def mean_excluding_self(mat: np.ndarray) -> np.ndarray:
    return np.array([float(r.mean()) if r.size else 1.0 for r in _offdiag_rows(mat)])
