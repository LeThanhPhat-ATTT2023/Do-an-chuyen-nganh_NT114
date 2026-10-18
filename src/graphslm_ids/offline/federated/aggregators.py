"""Aggregation rules over (N, D) client deltas.

An aggregator maps the stacked deltas and their base weights to one aggregate delta (an
``AggResult``). Rules that filter clients say who they dropped, either with ``selected``
(boolean mask) or with explicit per-client ``decisions`` plus ``trust`` and ``weights``.
Aggregators never see a model, graph or trainer: only tensors and a small ``ctx`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

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
    """``ctx`` keys: ``client_ids`` (list[int]), ``round_idx`` (int),
    ``layout`` (ParamLayout | None) and ``prev_aggregate`` (Tensor[D] | None)."""

    name = "base"

    def aggregate(self, deltas: torch.Tensor, weights: torch.Tensor, ctx: dict | None = None) -> AggResult:
        raise NotImplementedError


class FedAvg(Aggregator):
    name = "fedavg"

    def aggregate(self, deltas, weights, ctx=None):
        return AggResult(weighted_mean(deltas, weights))


_AGGREGATORS: dict[str, type[Aggregator]] = {
    "fedavg": FedAvg,
}


def build_aggregator(cfg: dict, *, assumed_malicious_ratio: float = 0.2, seed: int = 0) -> Aggregator:
    """Build the rule named by ``cfg["name"]`` with ``cfg["params"]`` as constructor kwargs.

    ``assumed_malicious_ratio`` and ``seed`` are the defense-wide defaults that rules needing
    them fill in when ``params`` leaves them out; the rules registered so far take neither.
    """
    if "name" not in cfg:
        raise ValueError(f"aggregator config needs a 'name'; choose from {sorted(_AGGREGATORS)}")
    name = str(cfg["name"]).lower()
    if name not in _AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; choose from {sorted(_AGGREGATORS)}")
    params = dict(cfg.get("params") or {})
    try:
        return _AGGREGATORS[name](**params)
    except TypeError as exc:
        raise ValueError(f"bad params for aggregator {name!r}: {exc}") from exc
