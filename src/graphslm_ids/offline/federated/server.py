"""FL coordinator: integrity checks -> aggregation -> global update.

The server only ever receives ``ClientMessage`` (round id, delta, sample count). It never
learns which clients are malicious; the simulation grades its per-client decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from graphslm_ids.offline.federated.aggregators import ACCEPT, INVALID, REJECT, AggResult, Aggregator
from graphslm_ids.offline.federated.params import ParamLayout
from graphslm_ids.offline.federated.records import to_jsonable


@dataclass
class ClientMessage:
    client_id: int
    round_idx: int
    delta: torch.Tensor
    num_samples: int


@dataclass
class ClientDecision:
    client_id: int
    decision: str
    trust: float
    weight: float
    norm: float | None
    reason: str = ""


@dataclass
class RoundReport:
    round_idx: int
    decisions: list[ClientDecision]
    applied: bool
    aggregate_norm: float
    seconds: float
    info: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)

    def decision_map(self) -> dict[int, str]:
        return {d.client_id: d.decision for d in self.decisions}


def _per_client(values: np.ndarray | None, n: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"aggregator returned {arr.shape[0]} {name} values for {n} clients")
    return arr


class FLServer:
    def __init__(
        self,
        *,
        global_vec: torch.Tensor,
        aggregator: Aggregator,
        layout: ParamLayout | None = None,
        weighting: str = "samples",
        server_lr: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if weighting not in {"samples", "uniform"}:
            raise ValueError(f"unknown weighting {weighting!r}; expected samples or uniform")
        self.device = torch.device(device)
        self.global_vec = global_vec.detach().to(self.device, torch.float32).clone()
        self.aggregator = aggregator
        self.layout = layout
        self.weighting = weighting
        self.server_lr = float(server_lr)
        self.round_idx = 1
        self.prev_aggregate: torch.Tensor | None = None
        # (client ids, deltas [N, D], num_samples [N]) of the valid messages of the last step.
        self.last_inputs: tuple[list[int], torch.Tensor, torch.Tensor] | None = None

    @property
    def dim(self) -> int:
        return int(self.global_vec.numel())

    def validate(self, msg: ClientMessage) -> str | None:
        """Return why ``msg`` is unusable this round, or None."""
        if int(msg.round_idx) != self.round_idx:
            return f"stale_round:{msg.round_idx}"
        if msg.delta.ndim != 1 or int(msg.delta.numel()) != self.dim:
            return f"shape:{tuple(msg.delta.shape)}"
        # Checked after the float32 cast the server aggregates in: a float64 1e300 overflows to inf.
        if not bool(torch.isfinite(msg.delta.detach().to(torch.float32)).all()):
            return "non_finite"
        return None

    def _resolve(
        self, result: AggResult, base: torch.Tensor, norms: torch.Tensor, ids: list[int]
    ) -> list[ClientDecision]:
        """Per-client verdicts: explicit decisions, else the selected mask, else accept all."""
        n = len(ids)
        trust = _per_client(result.trust, n, "trust")
        weights = _per_client(result.weights, n, "weight")
        reasons = [""] * n
        if result.decisions is not None:
            if len(result.decisions) != n:
                raise ValueError(f"aggregator returned {len(result.decisions)} decisions for {n} clients")
            decisions = list(result.decisions)
        elif result.selected is not None:
            chosen = torch.as_tensor(result.selected).reshape(-1).cpu().tolist()
            if len(chosen) != n:
                raise ValueError(f"aggregator returned a selection of {len(chosen)} entries for {n} clients")
            decisions = [ACCEPT if c else REJECT for c in chosen]
            reasons = ["" if c else "not_selected" for c in chosen]
            if trust is None:
                trust = np.array([1.0 if c else 0.0 for c in chosen])
        else:
            decisions = [ACCEPT] * n
        if trust is None:
            trust = np.array([0.0 if d == REJECT else 1.0 for d in decisions])
        if weights is None:
            weights = np.array([0.0 if d == REJECT else float(base[i]) for i, d in enumerate(decisions)])
        return [
            ClientDecision(cid, decisions[i], float(trust[i]), float(weights[i]), float(norms[i]), reasons[i])
            for i, cid in enumerate(ids)
        ]

    def step(self, messages: list[ClientMessage]) -> RoundReport:
        start = time.perf_counter()
        r = self.round_idx
        self.last_inputs = None
        slots: list[ClientDecision | None] = [None] * len(messages)
        valid_pos: list[int] = []
        for pos, msg in enumerate(messages):
            reason = self.validate(msg)
            if reason:
                slots[pos] = ClientDecision(int(msg.client_id), INVALID, 0.0, 0.0, None, reason)
            else:
                valid_pos.append(pos)
        if not valid_pos:
            self.round_idx += 1
            return RoundReport(r, [d for d in slots if d is not None], False, 0.0, time.perf_counter() - start)

        valid = [messages[pos] for pos in valid_pos]
        ids = [int(m.client_id) for m in valid]
        deltas = torch.stack([m.delta.detach().to(self.device, torch.float32) for m in valid])
        samples = torch.tensor([float(m.num_samples) for m in valid], device=self.device)
        self.last_inputs = (ids, deltas, samples)
        base = samples if self.weighting == "samples" else torch.ones_like(samples)
        ctx = {"client_ids": ids, "round_idx": r, "layout": self.layout, "prev_aggregate": self.prev_aggregate}
        result = self.aggregator.aggregate(deltas, base, ctx)

        resolved = self._resolve(result, base, deltas.norm(dim=1), ids)
        for pos, decision in zip(valid_pos, resolved):
            slots[pos] = decision
        applied = any(d.decision != REJECT for d in resolved)
        agg_norm = 0.0
        if applied:
            delta = result.delta.detach().to(self.device, torch.float32)
            if tuple(delta.shape) != (self.dim,) or not bool(torch.isfinite(delta).all()):
                raise ValueError(
                    f"aggregator {self.aggregator.name!r} returned an invalid aggregate "
                    f"(shape {tuple(delta.shape)}, expected ({self.dim},); all finite required)"
                )
            self.global_vec += self.server_lr * delta
            self.prev_aggregate = delta.clone()
            agg_norm = float(delta.norm())
        self.round_idx += 1
        return RoundReport(
            r,
            [d for d in slots if d is not None],
            applied,
            agg_norm,
            time.perf_counter() - start,
            to_jsonable(result.info),
            to_jsonable(result.scores),
        )
