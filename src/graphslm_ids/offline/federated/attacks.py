"""Model-poisoning attacks on flattened deltas (experiment-only Byzantine hook).

``knowledge="omniscient"``: the reference set is the round's benign deltas (worst
case). ``"partial"``: only the attackers' own honest deltas. ALIE / Min-Max / Min-Sum /
ScaleSign make every attacker send one shared vector (collusion by construction).

Conventions:

* ``sigma`` is the *population* std (``unbiased=False``), as in the spec formulas. The
  reference implementations typically use the sample std (``torch.std`` default), which
  is larger by sqrt(n / (n - 1)); here sigma is smaller by sqrt((n - 1) / n), so ALIE's
  shift ``z * sigma`` shrinks by the same factor (n = 10: 0.949, n = 50: 0.990).
* Every attack records diagnostics in ``attack.info`` (reset on each ``craft``);
  ``last_gamma`` is the gamma of the last gamma-searching craft (``None`` otherwise).
* Distances and dot products over the D ~ 6.4M coordinates are accumulated in float64
  over column chunks (the ``linalg.row_norms`` pattern): a float32 sgemv has ~1e-4
  relative error at that size and depends on the BLAS thread count.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from scipy.stats import norm as _normal

from graphslm_ids.offline.federated.linalg import CHUNK, pairwise_sq_dists, row_norms

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
        if self.knowledge == "partial":
            # Never fall back to benign deltas: that would leak knowledge the attacker lacks.
            if self.honest.shape[0] == 0:
                raise ValueError("partial knowledge needs the attackers' own honest deltas")
            return self.honest
        if self.benign.shape[0] > 0:
            return self.benign
        if self.honest.shape[0] > 0:
            return self.honest
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


def _finite(label: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return v


def _above(label: str, value: float, floor: float = 0.0) -> float:
    v = _finite(label, value)
    if v <= floor:
        raise ValueError(f"{label} must be > {floor:g}, got {value!r}")
    return v


def _sq_norm(v: torch.Tensor) -> float:
    return float(row_norms(v.unsqueeze(0))[0]) ** 2


def _centered_stats(
    ref: torch.Tensor, center: torch.Tensor, vectors: list[torch.Tensor]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """float64 per-row ``||e||^2`` and ``e . v`` (for each v) with ``e = center - ref``.

    One pass over column chunks, accumulated in float64: no (N, D) copy of ``e`` and no
    float32 sgemv. Centering on the crafted vector's anchor keeps the closed-form
    distances free of the ``|r|^2`` cancellation an origin-centred expansion suffers.
    """
    n = ref.shape[0]
    sq = torch.zeros(n, dtype=torch.float64, device=ref.device)
    dots = [torch.zeros(n, dtype=torch.float64, device=ref.device) for _ in vectors]
    for a in range(0, ref.shape[1], CHUNK):
        e = center[a:a + CHUNK] - ref[:, a:a + CHUNK]
        sq += e.pow(2).sum(dim=1).to(torch.float64)
        for acc, v in zip(dots, vectors):
            acc += (e * v[a:a + CHUNK]).sum(dim=1).to(torch.float64)
    return sq, dots


def perturbation_direction(ref: torch.Tensor, kind: str) -> torch.Tensor:
    """Perturbation ``p`` of the gamma attacks (``mu + gamma * p``).

    ``std``: ``-sigma``. ``unit`` / ``sign``: ``-mu`` / ``-sign(mu)`` rescaled to
    ``||sigma||`` so gamma is unitless and comparable across kinds and data scales (the
    set ``{mu + gamma p}`` is unchanged; the fixed gamma_init / tolerance would otherwise
    saturate or lose precision when ``||mu||`` is far from 1).
    """
    if kind not in {"std", "unit", "sign"}:
        raise ValueError(f"unknown perturbation {kind!r}; expected std, unit or sign")
    sigma = ref.std(dim=0, unbiased=False)
    if kind == "std":
        return -sigma
    mu = ref.mean(dim=0)
    direction = -mu if kind == "unit" else -torch.sign(mu)
    norm = float(row_norms(direction.unsqueeze(0))[0])
    sigma_norm = float(row_norms(sigma.unsqueeze(0))[0])
    return direction * (sigma_norm / max(norm, 1e-12))


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
    """``craft`` returns the (num_malicious, D) malicious deltas and refreshes ``info``.

    ``info`` is a diagnostics dict reset on every call; gamma-searching attacks fill
    ``gamma`` / ``saturated`` (gamma reached the ceiling of its search range, i.e. the
    constraint was slack) and keep ``last_gamma`` in sync.
    """

    name = "base"
    needs_honest = False

    def __init__(self) -> None:
        self.info: dict = {}
        self.last_gamma: float | None = None

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        self.info = {}
        return self._craft(ctx, num_malicious)

    def _craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        raise NotImplementedError


class NoAttack(Attack):
    name = "none"
    needs_honest = True

    def _craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        return ctx.honest.clone()


class SignFlip(Attack):
    name = "sign_flip"
    needs_honest = True

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = _above("sign_flip scale", scale)

    def _craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        if ctx.honest.shape[0] != num_malicious:
            raise ValueError("sign_flip needs one honest delta per attacker")
        return -self.scale * ctx.honest


class ALIE(Attack):
    name = "alie"

    def __init__(self, z: float | None = None) -> None:
        super().__init__()
        self.z = None if z is None else _finite("alie z", z)

    def _craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        ref = ctx.reference()
        z = self.z if self.z is not None else alie_z(ctx.num_participants, num_malicious)
        self.info["z"] = z
        vec = ref.mean(dim=0) + z * ref.std(dim=0, unbiased=False)
        return vec.unsqueeze(0).repeat(num_malicious, 1)


class _GammaAttack(Attack):
    def __init__(self, perturbation: str = "std", gamma_init: float = 10.0, tol: float = 1e-5) -> None:
        super().__init__()
        if perturbation not in {"std", "unit", "sign"}:
            raise ValueError(f"unknown perturbation {perturbation!r}; expected std, unit or sign")
        self.perturbation = perturbation
        self.gamma_init = _above(f"{self.name} gamma_init", gamma_init)
        self.tol = _above(f"{self.name} tol", tol)
        self.last_gamma = 0.0

    def _oracle(
        self, a: torch.Tensor, b: torch.Tensor, cc: float, dists: torch.Tensor
    ) -> Callable[[float], bool]:
        raise NotImplementedError

    def _craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        ref = ctx.reference()
        mu = ref.mean(dim=0)
        p = perturbation_direction(ref, self.perturbation)
        # ||mu + g p - r_i||^2 = a_i + 2 g b_i + g^2 cc  -> each oracle call is O(N).
        a, (b,) = _centered_stats(ref, mu, [p])
        cc = _sq_norm(p)
        oracle = self._oracle(a, b, cc, pairwise_sq_dists(ref))
        self.last_gamma = search_gamma(oracle, self.gamma_init, self.tol)
        # The step schedule reaches at most ~2 * gamma_init; landing there = slack constraint.
        saturated = self.last_gamma >= 2.0 * self.gamma_init - 4.0 * self.tol
        self.info = {"gamma": self.last_gamma, "saturated": bool(saturated)}
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
    printed Alg. 2 (which re-checks a stale g* and never moves on failure).

    gamma lies in [1, gamma_init] and is searched as ``t = gamma - 1`` in
    ``[0, gamma_init - 1]``: searching gamma directly and rejecting ``g < 1`` makes the
    downward probes (gamma_init / 2, / 4, / 8, ...) jump below 1 and return a no-op
    whenever gamma* < ~gamma_init / 8 (typical for correlated clients).
    """

    name = "scale_sign"
    needs_honest = True

    def __init__(self, gamma_init: float = 10.0, tol: float = 1e-3, constraint: str = "min_max") -> None:
        super().__init__()
        if constraint not in {"min_max", "min_sum"}:
            raise ValueError(f"unknown scale_sign constraint {constraint!r}")
        self.gamma_init = _above("scale_sign gamma_init", gamma_init, floor=1.0)
        self.tol = _above("scale_sign tol", tol)
        self.constraint = constraint
        self.last_gamma = 1.0

    def _craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        honest = ctx.honest
        if honest.shape[0] == 0:
            raise ValueError("scale_sign needs the attackers' honest deltas")
        gbar = honest.mean(dim=0)
        zero = torch.zeros_like(gbar)
        pos = torch.where(gbar > 0, gbar, zero)
        neg = torch.where(gbar < 0, gbar, zero)
        ref = ctx.reference()
        # crafted = g pos + neg / g and gbar = pos + neg, so with e = gbar - r_i
        #   crafted - r_i = e + u pos + w neg,  u = g - 1,  w = 1/g - 1 = -u / g,
        # and pos, neg have disjoint supports (pos . neg = 0):
        #   ||crafted - r_i||^2 = |e|^2 + 2u e.pos + 2w e.neg + u^2 |pos|^2 + w^2 |neg|^2.
        # Centred on gbar (not the origin) so the sums do not cancel catastrophically.
        ee, (ep, en) = _centered_stats(ref, gbar, [pos, neg])
        aa, bb = _sq_norm(pos), _sq_norm(neg)
        dists = pairwise_sq_dists(ref)

        def sq(g: float) -> torch.Tensor:
            u = g - 1.0
            w = -u / g
            return ee + 2.0 * u * ep + 2.0 * w * en + u * u * aa + w * w * bb

        if self.constraint == "min_max":
            bound = float(dists.max())
            feasible = lambda g: float(sq(g).max()) <= bound  # noqa: E731
        else:
            bound = float(dists.sum(dim=1).max())
            feasible = lambda g: float(sq(g).sum()) <= bound  # noqa: E731
        t_max = self.gamma_init - 1.0
        t = search_gamma(lambda t: t <= t_max and feasible(1.0 + t), t_max, self.tol)
        self.last_gamma = 1.0 + t
        crafted = self.last_gamma * pos + neg / self.last_gamma
        target_pos, _target_zero, target_neg = krum_sign_counts(honest)
        crafted = match_sign_counts(crafted, target_pos, target_neg)
        self.info = {
            "gamma": self.last_gamma,
            "saturated": bool(t >= t_max - self.tol),
            "target_pos": target_pos,
            "target_neg": target_neg,
        }
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
    """Build the attack from the ``attack`` config section (``name``, ``params``; the
    ``knowledge`` key, when present, is validated here though it lives on the context)."""
    name = str(cfg.get("name", "none")).lower()
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}; choose from {sorted(ATTACKS)}")
    knowledge = cfg.get("knowledge", "omniscient")
    if knowledge not in KNOWLEDGE:
        raise ValueError(f"unknown attack knowledge {knowledge!r}; expected one of {KNOWLEDGE}")
    params = dict(cfg.get("params") or {})
    try:
        return ATTACKS[name](**params)
    except TypeError as exc:
        raise ValueError(f"bad params for attack {name!r}: {exc}") from exc
