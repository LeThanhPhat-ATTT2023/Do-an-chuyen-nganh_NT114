"""Aggregation rules over (N, D) client deltas.

An aggregator maps the stacked deltas and their base weights to one aggregate delta (an
``AggResult``). Rules that filter clients say who they dropped, either with ``selected``
(boolean mask) or with explicit per-client ``decisions`` plus ``trust`` and ``weights``.
Aggregators never see a model, graph or trainer: only tensors and a small ``ctx`` dict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from graphslm_ids.offline.federated.linalg import (
    coordinate_median,
    coordinate_trimmed_mean,
    pairwise_sq_dists,
    top_right_singular_vector,
)

ACCEPT = "accept"
DOWNWEIGHT = "downweight"
REJECT = "reject"
INVALID = "invalid"


@dataclass
class AggResult:
    """Aggregate delta plus optional per-client verdicts (aligned with the input rows)."""

    delta: torch.Tensor
    selected: torch.Tensor | None = None
    decisions: list[str] | None = None
    trust: np.ndarray | None = None
    weights: np.ndarray | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def weighted_mean(deltas: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(deltas.device, deltas.dtype)
    total = float(w.sum())
    if not total > 0:
        raise ValueError(f"aggregation weights must sum to a positive value, got {total}")
    return (w / total) @ deltas


class Aggregator:
    """Subclasses must not mutate ``deltas`` in place: the server keeps ``last_inputs``
    aliasing the same tensor.

    ``ctx`` keys: ``client_ids`` (list[int]), ``round_idx`` (int),
    ``layout`` (ParamLayout | None) and ``prev_aggregate`` (Tensor[D] | None)."""

    name = "base"

    def aggregate(self, deltas: torch.Tensor, weights: torch.Tensor, ctx: dict | None = None) -> AggResult:
        raise NotImplementedError


class FedAvg(Aggregator):
    name = "fedavg"

    def aggregate(self, deltas, weights, ctx=None):
        return AggResult(weighted_mean(deltas, weights))


def _resolve_f(f: int | None, f_ratio: float, n: int) -> int:
    return int(f) if f is not None else int(math.floor(float(f_ratio) * n))


def largest_meanshift_cluster(feats: np.ndarray, quantile: float = 0.5) -> tuple[np.ndarray, bool]:
    """Boolean mask of the largest MeanShift cluster; (all True, True) on fallback."""
    from sklearn.cluster import MeanShift, estimate_bandwidth

    n = int(feats.shape[0])
    everyone = np.ones(n, dtype=bool)
    if n < 3:
        return everyone, True
    bandwidth = float(estimate_bandwidth(feats, quantile=quantile, n_samples=min(50, n)))
    if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
        return everyone, True
    try:
        labels = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False).fit(feats).labels_
    except ValueError:
        return everyone, True
    valid = labels >= 0
    if not valid.any():
        return everyone, True
    top = int(np.bincount(labels[valid]).argmax())
    return labels == top, False


def sign_features(deltas: torch.Tensor, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """SignGuard features: fractions of positive / zero / negative coordinates on one
    contiguous random block (official code), each divided by its max over clients."""
    d = deltas.shape[1]
    block = max(1, int(round(ratio * d)))
    start = int(rng.integers(0, d - block + 1))
    sub = deltas[:, start:start + block]
    feats = torch.stack(
        [(sub > 0).double().mean(1), (sub == 0).double().mean(1), (sub < 0).double().mean(1)], dim=1
    ).cpu().numpy()
    peak = feats.max(axis=0)
    peak[peak <= 0] = 1.0
    return feats / peak


class Median(Aggregator):
    name = "median"

    def aggregate(self, deltas, weights, ctx=None):
        return AggResult(coordinate_median(deltas))


class TrimmedMean(Aggregator):
    name = "trimmed_mean"

    def __init__(self, trim_ratio: float = 0.2, trim_k: int | None = None) -> None:
        self.trim_ratio = float(trim_ratio)
        self.trim_k = None if trim_k is None else int(trim_k)

    def aggregate(self, deltas, weights, ctx=None):
        n = deltas.shape[0]
        k = self.trim_k if self.trim_k is not None else int(math.floor(self.trim_ratio * n))
        if k < 0 or 2 * k >= n:
            raise ValueError(f"trimmed_mean trims {k} per side but only {n} deltas were given")
        return AggResult(coordinate_trimmed_mean(deltas, k), info={"trim_k": k})


class MultiKrum(Aggregator):
    name = "multi_krum"

    def __init__(self, f: int | None = None, f_ratio: float = 0.2, m: int | None = None) -> None:
        self.f = None if f is None else int(f)
        self.f_ratio = float(f_ratio)
        self.m = None if m is None else int(m)

    def aggregate(self, deltas, weights, ctx=None):
        weights = weights.to(deltas.device, torch.float32)
        n = deltas.shape[0]
        f = _resolve_f(self.f, self.f_ratio, n)
        if f < 0 or n <= 2 * f + 2:
            raise ValueError(f"multi_krum needs n > 2f + 2 (n={n}, f={f})")
        m = max(1, min(n - f if self.m is None else self.m, n))
        d2 = pairwise_sq_dists(deltas).cpu().numpy()
        # Krum score: sum of squared distances to the n - f - 2 nearest other clients.
        k = n - f - 2
        scores = np.array([np.sort(np.delete(d2[i], i))[:k].sum() for i in range(n)])
        selected = np.argsort(scores, kind="stable")[:m]
        mask = torch.zeros(n, dtype=torch.bool, device=deltas.device)
        mask[selected] = True
        return AggResult(
            weighted_mean(deltas[mask], weights[mask]),
            selected=mask,
            scores={"krum": scores},
            info={"f": f, "m": m},
        )


class SignGuard(Aggregator):
    """Xu et al. ICDCS'22 (+ official code): norm filter, sign-statistics MeanShift,
    median-norm clipped mean over the intersection (plain variant)."""

    name = "signguard"

    def __init__(
        self,
        lower: float = 0.1,
        upper: float = 3.0,
        coord_ratio: float = 0.1,
        quantile: float = 0.5,
        seed: int = 0,
    ) -> None:
        if not 0.0 < coord_ratio <= 1.0:
            raise ValueError(f"coord_ratio must be in (0, 1], got {coord_ratio}")
        self.lower, self.upper = float(lower), float(upper)
        self.coord_ratio = float(coord_ratio)
        self.quantile = float(quantile)
        self.rng = np.random.default_rng(seed)

    def aggregate(self, deltas, weights, ctx=None):
        weights = weights.to(deltas.device, torch.float32)
        n = deltas.shape[0]
        norms = deltas.norm(dim=1)
        med = max(float(norms.median()), 1e-12)
        ratio = norms / med
        s1 = ((ratio >= self.lower) & (ratio <= self.upper)).cpu().numpy()
        s2, fallback = largest_meanshift_cluster(sign_features(deltas, self.coord_ratio, self.rng), self.quantile)
        sel = s1 & s2
        if not sel.any():
            sel, fallback = s1, True
        if not sel.any():
            sel = np.ones(n, dtype=bool)
        mask = torch.as_tensor(sel, device=deltas.device)
        clip = torch.clamp(med / norms.clamp_min(1e-12), max=1.0)
        delta = weighted_mean(deltas[mask] * clip[mask].unsqueeze(1), weights[mask])
        return AggResult(
            delta,
            selected=mask,
            scores={"norm_ratio": ratio.cpu().numpy()},
            info={"fallback": bool(fallback), "median_norm": med},
        )


class DnC(Aggregator):
    """Divide-and-Conquer (Shejwalkar & Houmansadr, NDSS'21, Alg. 2)."""

    name = "dnc"

    def __init__(
        self,
        niters: int = 1,
        c: float = 1.0,
        b: int = 10000,
        f: int | None = None,
        f_ratio: float = 0.2,
        seed: int = 0,
    ) -> None:
        if niters < 1 or b < 1:
            raise ValueError(f"dnc needs niters >= 1 and b >= 1, got niters={niters}, b={b}")
        self.niters, self.c, self.b = int(niters), float(c), int(b)
        self.f = None if f is None else int(f)
        self.f_ratio = float(f_ratio)
        self.rng = np.random.default_rng(seed)

    def aggregate(self, deltas, weights, ctx=None):
        weights = weights.to(deltas.device, torch.float32)
        n, d = deltas.shape
        f = _resolve_f(self.f, self.f_ratio, n)
        if f < 0:
            raise ValueError(f"dnc needs f >= 0, got {f}")
        keep_n = max(1, n - int(math.floor(self.c * f)))
        good = np.ones(n, dtype=bool)
        for _ in range(self.niters):
            dims = np.sort(self.rng.choice(d, size=min(self.b, d), replace=False))
            x = deltas[:, torch.as_tensor(dims, device=deltas.device)]
            x = x - x.mean(dim=0, keepdim=True)
            s = (x @ top_right_singular_vector(x)).pow(2).cpu().numpy()
            keep = np.zeros(n, dtype=bool)
            keep[np.argsort(s, kind="stable")[:keep_n]] = True
            good &= keep
        if not good.any():
            good[:] = True
        mask = torch.as_tensor(good, device=deltas.device)
        return AggResult(weighted_mean(deltas[mask], weights[mask]), selected=mask, info={"f": f, "keep": keep_n})


_AGGREGATORS: dict[str, type[Aggregator]] = {
    "fedavg": FedAvg,
    "median": Median,
    "trimmed_mean": TrimmedMean,
    "multi_krum": MultiKrum,
    "signguard": SignGuard,
    "dnc": DnC,
}


def build_aggregator(cfg: dict, *, assumed_malicious_ratio: float = 0.2, seed: int = 0) -> Aggregator:
    """Build the rule named by ``cfg["name"]`` with ``cfg["params"]`` as constructor kwargs.

    ``assumed_malicious_ratio`` and ``seed`` are the defense-wide defaults that rules needing
    them fill in when ``params`` leaves them out: the ratio feeds Trimmed Mean, Multi-Krum and
    DnC; the seed feeds SignGuard and DnC.
    """
    if "name" not in cfg:
        raise ValueError(f"aggregator config needs a 'name'; choose from {sorted(_AGGREGATORS)}")
    name = str(cfg["name"]).lower()
    if name not in _AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; choose from {sorted(_AGGREGATORS)}")
    params = dict(cfg.get("params") or {})
    if name == "trimmed_mean":
        params.setdefault("trim_ratio", float(assumed_malicious_ratio))
    if name in {"multi_krum", "dnc"}:
        params.setdefault("f_ratio", float(assumed_malicious_ratio))
    if name in {"signguard", "dnc"}:
        params.setdefault("seed", int(seed))
    try:
        return _AGGREGATORS[name](**params)
    except TypeError as exc:
        raise ValueError(f"bad params for aggregator {name!r}: {exc}") from exc
