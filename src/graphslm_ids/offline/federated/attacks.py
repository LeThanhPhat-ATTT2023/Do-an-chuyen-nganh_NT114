"""Model-poisoning attacks on flattened deltas (experiment-only Byzantine hook).

``knowledge="omniscient"``: the reference set is the round's benign deltas (worst
case). ``"partial"``: only the attackers' own honest deltas. ALIE / Min-Max / Min-Sum /
ScaleSign make every attacker send one shared vector (collusion by construction).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from scipy.stats import norm as _normal

from graphslm_ids.offline.federated.linalg import pairwise_sq_dists

KNOWLEDGE = ("omniscient", "partial")


@dataclass
class AttackContext:
    honest: torch.Tensor
    benign: torch.Tensor
    num_participants: int
    knowledge: str = "omniscient"

    def reference(self) -> torch.Tensor:
        if self.knowledge not in KNOWLEDGE:
            raise ValueError(f"unknown attack knowledge {self.knowledge!r}; expected one of {KNOWLEDGE}")
        if self.knowledge == "omniscient" and self.benign.shape[0] > 0:
            return self.benign
        if self.honest.shape[0] > 0:
            return self.honest
        if self.benign.shape[0] > 0:
            return self.benign
        raise ValueError("attack has neither benign nor honest reference deltas")


def search_gamma(
    oracle: Callable[[float], bool], gamma_init: float = 10.0, tol: float = 1e-5, max_iter: int = 200
) -> float:
    """Largest gamma with oracle(gamma) True — NDSS'21 Alg. 1 with the official
    code's step schedule (first move is +-gamma_init/2)."""
    gamma = float(gamma_init)
    step = float(gamma_init)
    succ = 0.0
    for _ in range(max_iter):
        if abs(succ - gamma) <= tol:
            break
        if oracle(gamma):
            succ = gamma
            gamma += step / 2.0
        else:
            gamma -= step / 2.0
        step /= 2.0
    return succ


def alie_z(num_participants: int, num_malicious: int) -> float:
    """Baruch et al. 2019, Alg. 3: s = floor(n/2 + 1) - m; z = Phi^-1((n - s) / n)."""
    n = int(num_participants)
    s = math.floor(n / 2 + 1) - int(num_malicious)
    q = min(max((n - s) / n, 1e-6), 1.0 - 1e-6)
    return float(_normal.ppf(q))


def perturbation_direction(ref: torch.Tensor, kind: str) -> torch.Tensor:
    mu = ref.mean(dim=0)
    if kind == "std":
        return -ref.std(dim=0, unbiased=False)
    if kind == "unit":
        return -mu / mu.norm().clamp_min(1e-12)
    if kind == "sign":
        return -torch.sign(mu)
    raise ValueError(f"unknown perturbation {kind!r}; expected std, unit or sign")


def sign_counts(x: torch.Tensor) -> tuple[int, int, int]:
    return int((x > 0).sum()), int((x == 0).sum()), int((x < 0).sum())


def krum_sign_counts(honest: torch.Tensor) -> tuple[int, int, int]:
    """ScaleSign Eq. 10: the attacker sign-count triple closest (Krum-style) to the rest."""
    counts = torch.tensor([sign_counts(row) for row in honest], dtype=torch.float64)
    if counts.shape[0] == 1:
        return tuple(int(v) for v in counts[0].tolist())
    d = torch.cdist(counts, counts).pow(2)
    j = int(d.sum(dim=1).argmin())
    return tuple(int(v) for v in counts[j].tolist())


def match_sign_counts(g: torch.Tensor, target_pos: int, target_neg: int) -> torch.Tensor:
    """ScaleSign Alg. 4 sign repair: zero the smallest surplus entries, or promote the
    entries nearest zero to the boundary value of the missing sign."""
    g = g.clone()
    pos_idx = torch.nonzero(g > 0).flatten()
    p_now = int(pos_idx.numel())
    if target_pos < p_now:
        sel = torch.topk(g[pos_idx], p_now - target_pos, largest=False).indices
        g[pos_idx[sel]] = 0.0
    elif target_pos > p_now:
        tau = float(g[pos_idx].min()) if p_now else 1e-12
        nonpos = torch.nonzero(g <= 0).flatten()
        k = min(target_pos - p_now, int(nonpos.numel()))
        sel = torch.topk(g[nonpos], k, largest=True).indices
        g[nonpos[sel]] = tau
    neg_idx = torch.nonzero(g < 0).flatten()
    n_now = int(neg_idx.numel())
    if target_neg < n_now:
        sel = torch.topk(g[neg_idx], n_now - target_neg, largest=True).indices
        g[neg_idx[sel]] = 0.0
    elif target_neg > n_now:
        tau = float(g[neg_idx].max()) if n_now else -1e-12
        zero_idx = torch.nonzero(g == 0).flatten()
        k = min(target_neg - n_now, int(zero_idx.numel()))
        g[zero_idx[:k]] = tau
    return g


class Attack:
    name = "base"
    needs_honest = False

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        raise NotImplementedError


class NoAttack(Attack):
    name = "none"
    needs_honest = True

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        return ctx.honest.clone()


class SignFlip(Attack):
    name = "sign_flip"
    needs_honest = True

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        if ctx.honest.shape[0] != num_malicious:
            raise ValueError("sign_flip needs one honest delta per attacker")
        return -self.scale * ctx.honest


class ALIE(Attack):
    name = "alie"

    def __init__(self, z: float | None = None) -> None:
        self.z = None if z is None else float(z)

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        ref = ctx.reference()
        z = self.z if self.z is not None else alie_z(ctx.num_participants, num_malicious)
        vec = ref.mean(dim=0) + z * ref.std(dim=0, unbiased=False)
        return vec.unsqueeze(0).repeat(num_malicious, 1)


class _GammaAttack(Attack):
    def __init__(self, perturbation: str = "std", gamma_init: float = 10.0, tol: float = 1e-5) -> None:
        if perturbation not in {"std", "unit", "sign"}:
            raise ValueError(f"unknown perturbation {perturbation!r}; expected std, unit or sign")
        self.perturbation = perturbation
        self.gamma_init = float(gamma_init)
        self.tol = float(tol)
        self.last_gamma = 0.0

    def _oracle(
        self, a: torch.Tensor, b: torch.Tensor, cc: float, dists: torch.Tensor
    ) -> Callable[[float], bool]:
        raise NotImplementedError

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        ref = ctx.reference()
        mu = ref.mean(dim=0)
        p = perturbation_direction(ref, self.perturbation)
        diff = mu.unsqueeze(0) - ref
        # ||mu + g p - r_i||^2 = a_i + 2 g b_i + g^2 cc  -> each oracle call is O(N).
        a = diff.pow(2).sum(dim=1).to(torch.float64)
        b = (diff @ p).to(torch.float64)
        cc = float(p.pow(2).sum())
        oracle = self._oracle(a, b, cc, pairwise_sq_dists(ref))
        self.last_gamma = search_gamma(oracle, self.gamma_init, self.tol)
        return (mu + self.last_gamma * p).unsqueeze(0).repeat(num_malicious, 1)


class MinMax(_GammaAttack):
    """Shejwalkar & Houmansadr NDSS'21: max_i ||m - g_i|| <= max_ij ||g_i - g_j||."""

    name = "min_max"

    def _oracle(self, a, b, cc, dists):
        bound = float(dists.max())
        return lambda g: float((a + 2.0 * g * b + g * g * cc).max()) <= bound


class MinSum(_GammaAttack):
    """NDSS'21 Eq. 7 (max over i on the right-hand side, as in the paper)."""

    name = "min_sum"

    def _oracle(self, a, b, cc, dists):
        bound = float(dists.sum(dim=1).max())
        return lambda g: float((a + 2.0 * g * b + g * g * cc).sum()) <= bound


class ScaleSign(Attack):
    """Yang et al., TIFS 2025, §V. Gamma search uses the NDSS oracle instead of the
    printed Alg. 2 (which re-checks a stale g* and never moves on failure)."""

    name = "scale_sign"
    needs_honest = True

    def __init__(self, gamma_init: float = 10.0, tol: float = 1e-3, constraint: str = "min_max") -> None:
        if constraint not in {"min_max", "min_sum"}:
            raise ValueError(f"unknown scale_sign constraint {constraint!r}")
        self.gamma_init = float(gamma_init)
        self.tol = float(tol)
        self.constraint = constraint
        self.last_gamma = 1.0

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        honest = ctx.honest
        if honest.shape[0] == 0:
            raise ValueError("scale_sign needs the attackers' honest deltas")
        gbar = honest.mean(dim=0)
        zero = torch.zeros_like(gbar)
        pos = torch.where(gbar > 0, gbar, zero)
        neg = torch.where(gbar < 0, gbar, zero)
        ref = ctx.reference()
        # ||g pos + neg/g - r_i||^2 with disjoint supports of pos and neg.
        aa = float(pos.pow(2).sum())
        bb = float(neg.pow(2).sum())
        rr = ref.pow(2).sum(dim=1).to(torch.float64)
        ar = (ref @ pos).to(torch.float64)
        br = (ref @ neg).to(torch.float64)
        dists = pairwise_sq_dists(ref)

        def sq(g: float) -> torch.Tensor:
            return g * g * aa + bb / (g * g) + rr - 2.0 * g * ar - (2.0 / g) * br

        # gamma in [1, gamma_init]: the NDSS step schedule alone can walk up to ~2x
        # gamma_init when every gamma is feasible, so the interval is enforced here.
        if self.constraint == "min_max":
            bound = float(dists.max())
            ok = lambda g: 1.0 <= g <= self.gamma_init and float(sq(g).max()) <= bound  # noqa: E731
        else:
            bound = float(dists.sum(dim=1).max())
            ok = lambda g: 1.0 <= g <= self.gamma_init and float(sq(g).sum()) <= bound  # noqa: E731
        self.last_gamma = max(search_gamma(ok, self.gamma_init, self.tol), 1.0)
        crafted = self.last_gamma * pos + neg / self.last_gamma
        target_pos, _target_zero, target_neg = krum_sign_counts(honest)
        crafted = match_sign_counts(crafted, target_pos, target_neg)
        return crafted.unsqueeze(0).repeat(num_malicious, 1)


ATTACKS: dict[str, type[Attack]] = {
    "none": NoAttack,
    "sign_flip": SignFlip,
    "alie": ALIE,
    "min_max": MinMax,
    "min_sum": MinSum,
    "scale_sign": ScaleSign,
}


def build_attack(cfg: dict) -> Attack:
    name = str(cfg.get("name", "none")).lower()
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}; choose from {sorted(ATTACKS)}")
    params = dict(cfg.get("params") or {})
    try:
        return ATTACKS[name](**params)
    except TypeError as exc:
        raise ValueError(f"bad params for attack {name!r}: {exc}") from exc
