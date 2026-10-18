"""RAGuard: the proposed defense (spec §3.7).

Relation-aware screening: every HGT parameter group (``ParamLayout`` group) is compared only
among the clients for which it is *active* (non-zero delta), so a benign client that never sees
a relation (exactly-zero delta there, see ``params.py``) is not judged on that group. Per group
and active client, three signals are robust-z scored inside the group:

* magnitude: ``log(||d_g|| / median ||d_g||)``;
* direction: cosine to ``ref`` (the previous aggregate restricted to the group, or the
  coordinate median of the active rows when there is none);
* sign: share of ``ref``'s non-zero coordinates whose sign the client matches.

Each signal has a 1-sigma dead zone, ``hi(z) = clip((z - 1) / (z_max - 1), 0, 1)``, and they are
combined by noisy-OR into a group suspicion; a client's suspicion ``U`` is the parameter-count
weighted mean (or the max) over the groups it was evaluated in. Round history turns ``U`` into
trust (EMA plus abrupt-change detection), trust into accept / downweight / reject, and the
surviving clients' median-norm clipped deltas are aggregated per group with a weighted trimmed
mean (or a coordinate median); a group nobody active keeps is aggregated to zero.

Recorded decisions: the signals are computed on the RAW deltas (so a x50 update is visible),
the aggregation uses the CLIPPED deltas. ``abrupt_min_jump`` and the 1.4826 MAD factor of the
abrupt-change rule are our choices (the spec states the rule in sigma units without a floor).
The sign-agreement share is discrete (step ``1/n``, n = coordinates with a non-zero reference
sign), so its robust-z scale is ``max(1.4826 * MAD, scale_floor, 1/n)``: with MAD = 0 a single
disagreeing coordinate would otherwise read as dozens of sigmas. ``z_max = 4`` (spec: 3) is
likewise our choice, made on no-attack data to keep benign false rejections low (spec §3.7).

For models whose layout has no relation groups (MLP / CNN: only ``other`` / ``classifier``)
the screening degenerates to (almost) whole-vector screening; that is intended.

This module only sees tensors and a small ``ctx`` dict: no model, graph or trainer code.
"""

from __future__ import annotations

import numpy as np
import torch

from graphslm_ids.offline.federated.aggregators import (
    ACCEPT,
    DOWNWEIGHT,
    REJECT,
    AggResult,
    Aggregator,
    build_aggregator,
)
from graphslm_ids.offline.federated.linalg import (
    CHUNK,
    coordinate_median,
    cosine_to,
    robust_center_scale,
    row_norms,
    weighted_trimmed_mean,
)
from graphslm_ids.offline.federated.params import ParamLayout

MAD_SIGMA = 1.4826  # MAD -> sigma for a normal distribution
ABRUPT_TRUST_CAP = 0.5  # trust of a client flagged for an abrupt change is capped here


def robust_z_sigma(values: np.ndarray, floor: float) -> np.ndarray:
    """``(v - median) / max(1.4826 * MAD, floor)`` over a 1-D array (float64)."""
    v = np.asarray(values, dtype=np.float64)
    center, scale = robust_center_scale(v, axis=0, floor=float(floor))
    return (v - center) / scale


def sign_agreement(x: torch.Tensor, ref: torch.Tensor) -> np.ndarray:
    """Per row of x: share of the coordinates with ``sign(ref) != 0`` where ``sign(x) == sign(ref)``.

    A ``ref`` without non-zero coordinates gives 0 for every row.
    """
    ref = ref.to(x.device, x.dtype)
    agree = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    count = 0
    for a in range(0, x.shape[1], CHUNK):
        s_ref = torch.sign(ref[a:a + CHUNK])
        nonzero = s_ref != 0
        count += int(nonzero.sum())
        agree += ((torch.sign(x[:, a:a + CHUNK]) == s_ref) & nonzero).sum(dim=1).to(torch.float64)
    return (agree / max(count, 1)).cpu().numpy()


def _clip_scales(deltas: torch.Tensor, factor: float) -> tuple[torch.Tensor, float]:
    """Per-row factors ``min(1, factor * median_norm / norm)`` (1 for a zero row) and the median norm."""
    norms = row_norms(deltas)
    median_norm = float(norms.median())
    scale = torch.ones_like(norms)
    nonzero = norms > 0
    scale[nonzero] = (float(factor) * median_norm / norms[nonzero]).clamp(max=1.0)
    return scale.to(deltas.dtype), median_norm


def clip_to_median(deltas: torch.Tensor, factor: float) -> torch.Tensor:
    """Each row scaled by ``min(1, factor * median_norm / norm)``; ``median = float(norms.median())``."""
    scale, _ = _clip_scales(deltas, factor)
    return deltas * scale.unsqueeze(1)


def client_suspicion(
    u: np.ndarray, evaluated: np.ndarray, group_sizes: np.ndarray, reduce: str
) -> np.ndarray:
    """Per-client suspicion from per-group suspicions ``u`` [N, G].

    ``mean``: parameter-count (``group_sizes``) weighted mean over the evaluated groups;
    ``max``: max over the evaluated groups. A client with no evaluated group gets 0.
    """
    u = np.asarray(u, dtype=np.float64)
    ev = np.asarray(evaluated, dtype=bool)
    sizes = np.asarray(group_sizes, dtype=np.float64)
    if reduce == "mean":
        mass = ev * sizes[None, :]
        total = mass.sum(axis=1)
        return np.divide((u * mass).sum(axis=1), total, out=np.zeros(u.shape[0]), where=total > 0)
    if reduce == "max":
        return np.where(ev, u, 0.0).max(axis=1, initial=0.0)
    raise ValueError(f"unknown client_reduce {reduce!r}; expected mean or max")


def decide(
    trust: np.ndarray, *, downweight: bool = True, reject_below: float = 0.5, accept_above: float = 0.8
) -> tuple[list[str], np.ndarray]:
    """``trust < reject_below`` -> reject (weight 0); below ``accept_above`` -> downweight (weight =
    trust); else accept (weight 1). With ``downweight=False`` everything not rejected is accepted
    with weight 1. A non-finite trust is rejected."""
    t = np.asarray(trust, dtype=np.float64).reshape(-1)
    decisions: list[str] = []
    weights = np.zeros_like(t)
    for i, value in enumerate(t):
        if not np.isfinite(value) or value < reject_below:
            decisions.append(REJECT)
        elif downweight and value < accept_above:
            decisions.append(DOWNWEIGHT)
            weights[i] = value
        else:
            decisions.append(ACCEPT)
            weights[i] = 1.0
    return decisions, weights


def _hi(z: np.ndarray, z_max: float) -> np.ndarray:
    return np.clip((z - 1.0) / (z_max - 1.0), 0.0, 1.0)


def sign_suspicion(agreement: np.ndarray, n_ref: int, *, scale_floor: float, z_max: float) -> np.ndarray:
    """``hi(-z)`` of the sign-agreement shares, whose robust-z scale is floored at the signal's
    step ``1 / n_ref`` as well (``n_ref``: coordinates with a non-zero reference sign), i.e.
    ``max(1.4826 * MAD, scale_floor, 1 / n_ref)``: missing one coordinate stays in the dead zone."""
    floor = max(float(scale_floor), 1.0 / max(int(n_ref), 1))
    return _hi(-robust_z_sigma(agreement, floor), z_max)


class RAGuard(Aggregator):
    """Relation-aware, history-aware screening + per-group weighted trimmed mean (spec §3.7)."""

    name = "raguard"

    def __init__(
        self,
        beta: float = 0.2,
        clip_factor: float = 1.0,
        z_max: float = 4.0,
        min_active: int = 3,
        scale_floor: float = 1e-3,
        ema_current: float = 0.8,
        warmup_rounds: int = 2,
        abrupt_k: float = 3.0,
        abrupt_min_jump: float = 0.15,
        reject_below: float = 0.5,
        accept_above: float = 0.8,
        relation_aware: bool = True,
        temporal: bool = True,
        downweight: bool = True,
        aggregator: str = "trimmed_mean",
        client_reduce: str = "mean",
    ) -> None:
        self.beta = float(beta)
        self.clip_factor = float(clip_factor)
        self.z_max = float(z_max)
        self.min_active = int(min_active)
        self.scale_floor = float(scale_floor)
        self.ema_current = float(ema_current)
        self.warmup_rounds = int(warmup_rounds)
        self.abrupt_k = float(abrupt_k)
        self.abrupt_min_jump = float(abrupt_min_jump)
        self.reject_below = float(reject_below)
        self.accept_above = float(accept_above)
        for flag, value in (("relation_aware", relation_aware), ("temporal", temporal), ("downweight", downweight)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{flag} must be a boolean, got {value!r}")
        self.relation_aware = bool(relation_aware)
        self.temporal = bool(temporal)
        self.downweight = bool(downweight)
        self.aggregator = str(aggregator)
        self.client_reduce = str(client_reduce)
        if not 0.0 <= self.beta < 0.5:
            raise ValueError(f"beta must be in [0, 0.5), got {self.beta}")
        if not self.clip_factor > 0:
            raise ValueError(f"clip_factor must be positive, got {self.clip_factor}")
        if not self.z_max > 1.0:
            raise ValueError(f"z_max must be > 1, got {self.z_max}")
        if self.min_active < 1:
            raise ValueError(f"min_active must be >= 1, got {self.min_active}")
        if not self.scale_floor > 0:
            raise ValueError(f"scale_floor must be positive, got {self.scale_floor}")
        if not 0.0 <= self.ema_current <= 1.0:
            raise ValueError(f"ema_current must be in [0, 1], got {self.ema_current}")
        if self.warmup_rounds < 0:
            raise ValueError(f"warmup_rounds must be >= 0, got {self.warmup_rounds}")
        if self.abrupt_k < 0 or self.abrupt_min_jump < 0:
            raise ValueError(f"abrupt_k and abrupt_min_jump must be >= 0, got {self.abrupt_k}, {self.abrupt_min_jump}")
        if not 0.0 <= self.reject_below <= self.accept_above:
            raise ValueError(
                f"need 0 <= reject_below <= accept_above, got {self.reject_below}, {self.accept_above}"
            )
        if self.aggregator not in {"trimmed_mean", "median"}:
            raise ValueError(f"unknown RAGuard aggregator {self.aggregator!r}; expected trimmed_mean or median")
        if self.client_reduce not in {"mean", "max"}:
            raise ValueError(f"unknown client_reduce {self.client_reduce!r}; expected mean or max")
        # Per-client history, keyed by client id.
        self._trust_prev: dict[int, float] = {}
        self._u_prev: dict[int, float] = {}
        self._jumps: dict[int, list[float]] = {}
        self._calls = 0

    # -- history ---------------------------------------------------------------------------

    def update_trust(
        self, client_ids: list[int], suspicion: np.ndarray, round_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trust and abrupt-change flags for this round's clients; updates the per-client history.

        Warm-up rounds (``round_idx <= warmup_rounds``) or ``temporal=False``: ``trust = 1 - U``.
        Later: ``trust = ema_current * (1 - U) + (1 - ema_current) * trust_prev`` (``1 - U`` on
        first sighting); with >= 2 recorded jumps ``|U_t - U_{t-1}|`` and a new jump above
        ``max(median + abrupt_k * 1.4826 * MAD, abrupt_min_jump)`` of them, the client is flagged
        and its trust capped at 0.5. The new jump is always appended afterwards.
        """
        u = np.asarray(suspicion, dtype=np.float64).reshape(-1)
        if len(client_ids) != u.shape[0]:
            raise ValueError(f"got {len(client_ids)} client ids for {u.shape[0]} suspicion values")
        use_history = self.temporal and int(round_idx) > self.warmup_rounds
        trust = np.empty_like(u)
        abrupt = np.zeros(u.shape[0], dtype=bool)
        for i, cid in enumerate(int(c) for c in client_ids):
            current = float(u[i])
            value = 1.0 - current
            u_prev = self._u_prev.get(cid)
            jumps = self._jumps.setdefault(cid, [])
            if use_history:
                trust_prev = self._trust_prev.get(cid)
                if trust_prev is not None:
                    value = self.ema_current * value + (1.0 - self.ema_current) * trust_prev
                if u_prev is not None and len(jumps) >= 2:
                    hist = np.asarray(jumps, dtype=np.float64)
                    center = float(np.median(hist))
                    mad = MAD_SIGMA * float(np.median(np.abs(hist - center)))
                    if abs(current - u_prev) > max(center + self.abrupt_k * mad, self.abrupt_min_jump):
                        abrupt[i] = True
                        value = min(value, ABRUPT_TRUST_CAP)
            if u_prev is not None:
                jumps.append(abs(current - u_prev))
            self._trust_prev[cid] = value
            self._u_prev[cid] = current
            trust[i] = value
        return trust, abrupt

    # -- aggregation -----------------------------------------------------------------------

    def _groups(
        self, layout: ParamLayout | None, d: int, device: torch.device
    ) -> list[tuple[str, slice | torch.Tensor, int]]:
        """(name, columns, size) per screened group; one ``all`` group without relation awareness."""
        if not self.relation_aware or layout is None:
            return [("all", slice(None), d)]
        if int(layout.numel) != d:
            raise ValueError(f"layout has {layout.numel} coordinates, deltas have {d}")
        groups = []
        for g in layout.group_names:
            cols = layout.group_coords([g]).to(device)
            groups.append((g, cols, int(cols.numel())))
        return groups

    def _group_suspicion(self, sub: torch.Tensor, norms: np.ndarray, ref: torch.Tensor) -> np.ndarray:
        """Noisy-OR of the magnitude / direction / sign signals over the active rows ``sub``."""
        z_mag = robust_z_sigma(np.log(norms / np.median(norms)), self.scale_floor)
        z_cos = robust_z_sigma(cosine_to(sub, ref).cpu().numpy(), self.scale_floor)
        n_ref = int(torch.count_nonzero(ref))  # coordinates with a non-zero reference sign
        s_sgn = sign_suspicion(sign_agreement(sub, ref), n_ref, scale_floor=self.scale_floor, z_max=self.z_max)
        keep = (1.0 - _hi(np.abs(z_mag), self.z_max)) * (1.0 - _hi(-z_cos, self.z_max)) * (1.0 - s_sgn)
        return 1.0 - keep

    def aggregate(self, deltas: torch.Tensor, weights: torch.Tensor, ctx: dict | None = None) -> AggResult:
        ctx = ctx or {}
        n, d = deltas.shape
        device = deltas.device
        self._calls += 1
        ids = ctx.get("client_ids")
        ids = list(range(n)) if ids is None else [int(c) for c in ids]
        if len(ids) != n:
            raise ValueError(f"ctx has {len(ids)} client ids for {n} deltas")
        round_idx = ctx.get("round_idx")
        round_idx = self._calls if round_idx is None else int(round_idx)
        prev = ctx.get("prev_aggregate")
        if prev is not None:
            prev = prev.detach().to(device, deltas.dtype).reshape(-1)
            if prev.shape[0] != d:
                raise ValueError(f"prev_aggregate has {prev.shape[0]} coordinates, deltas have {d}")
        groups = self._groups(ctx.get("layout"), d, device)

        # 1) Per-group signals on the RAW deltas, among the clients active in the group.
        num_groups = len(groups)
        u = np.zeros((n, num_groups))
        evaluated = np.zeros((n, num_groups), dtype=bool)
        active = np.zeros((n, num_groups), dtype=bool)
        sizes = np.array([size for _, _, size in groups], dtype=np.float64)
        num_evaluated = 0
        for gi, (_, cols, _) in enumerate(groups):
            sub = deltas[:, cols]
            norms = row_norms(sub).cpu().numpy()
            act = norms > 0
            active[:, gi] = act
            if int(act.sum()) < self.min_active:
                continue
            sub_act = sub[torch.as_tensor(act, device=device)]
            ref = None
            if prev is not None:
                prev_g = prev[cols]
                if float(row_norms(prev_g.unsqueeze(0))[0]) > 0:
                    ref = prev_g
            if ref is None:
                ref = coordinate_median(sub_act)
            u[act, gi] = self._group_suspicion(sub_act, norms[act], ref)
            evaluated[act, gi] = True
            num_evaluated += 1
        suspicion = client_suspicion(u, evaluated, sizes, self.client_reduce)

        # 2) History -> trust -> decisions.
        trust, abrupt = self.update_trust(ids, suspicion, round_idx)
        decisions, decision_w = decide(
            trust, downweight=self.downweight, reject_below=self.reject_below, accept_above=self.accept_above
        )
        w = weights.detach().to("cpu", torch.float64).reshape(-1).numpy() * decision_w
        keep = np.array([dec != REJECT for dec in decisions], dtype=bool)

        # 3) Per-group aggregation of the CLIPPED deltas of the kept, active clients.
        scale, median_norm = _clip_scales(deltas, self.clip_factor)
        w_t = torch.as_tensor(w, dtype=torch.float64, device=device)
        out = torch.zeros(d, dtype=deltas.dtype, device=device)
        for gi, (_, cols, _) in enumerate(groups):
            rows = keep & active[:, gi]
            if self.aggregator == "trimmed_mean":
                rows &= w > 0  # zero-weight rows carry no mass; dropping them avoids a 0/0
            if not rows.any():
                continue  # nobody kept trained this group: it stays 0
            idx = torch.as_tensor(np.flatnonzero(rows), device=device)
            # Tensor (advanced) indexing always copies, so clipping in place never touches ``deltas``.
            part = deltas[idx] if isinstance(cols, slice) else deltas[idx.unsqueeze(1), cols.unsqueeze(0)]
            part.mul_(scale[idx].unsqueeze(1))
            if self.aggregator == "trimmed_mean":
                out[cols] = weighted_trimmed_mean(part, w_t[idx], self.beta)
            else:
                out[cols] = coordinate_median(part)

        return AggResult(
            out,
            selected=torch.as_tensor(keep, device=device),
            decisions=decisions,
            trust=trust,
            weights=w,
            scores={"suspicion": suspicion, "abrupt": abrupt},
            info={
                "beta": self.beta,
                "num_groups": num_groups,
                "num_groups_evaluated": num_evaluated,
                "median_norm": median_norm,
                "num_clipped": int((scale < 1).sum()),
                "num_abrupt": int(abrupt.sum()),
            },
        )


def build_defense(cfg: dict, *, assumed_malicious_ratio: float = 0.2, seed: int = 0) -> Aggregator:
    """``raguard`` -> ``RAGuard(**params)`` with ``beta`` defaulting to ``assumed_malicious_ratio``;
    any other name -> ``build_aggregator`` (same arguments). Bad names / params raise ValueError."""
    if str(cfg.get("name", "")).lower() != RAGuard.name:
        return build_aggregator(cfg, assumed_malicious_ratio=assumed_malicious_ratio, seed=seed)
    params = dict(cfg.get("params") or {})
    params.setdefault("beta", float(assumed_malicious_ratio))
    try:
        return RAGuard(**params)
    except TypeError as exc:
        raise ValueError(f"bad params for defense {RAGuard.name!r}: {exc}") from exc
