"""Split TRAIN flows into simulated FL clients (IID or Dirichlet label skew)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ClientPartition:
    client_flow_ids: list[np.ndarray]
    scheme: str
    alpha: float | None
    seed: int

    @property
    def num_clients(self) -> int:
        return len(self.client_flow_ids)

    def sizes(self) -> list[int]:
        return [int(ids.shape[0]) for ids in self.client_flow_ids]

    def all_flow_ids(self) -> np.ndarray:
        if not self.client_flow_ids:
            return np.empty((0,), dtype=np.int64)
        return np.sort(np.concatenate(self.client_flow_ids))

    def class_histogram(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int64)
        return np.stack(
            [np.bincount(labels[ids], minlength=num_classes) for ids in self.client_flow_ids]
        ).astype(np.int64)

    def to_json(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "alpha": self.alpha,
            "seed": self.seed,
            "client_flow_ids": [ids.astype(np.int64).tolist() for ids in self.client_flow_ids],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ClientPartition":
        return cls(
            client_flow_ids=[np.asarray(ids, dtype=np.int64) for ids in payload["client_flow_ids"]],
            scheme=str(payload["scheme"]),
            alpha=None if payload.get("alpha") is None else float(payload["alpha"]),
            seed=int(payload["seed"]),
        )


def _check_unique(ids: np.ndarray) -> None:
    if np.unique(ids).size != ids.size:
        raise ValueError("train_ids contains duplicate flow ids; each flow must appear once")


def partition_iid(train_ids: np.ndarray, num_clients: int, seed: int) -> ClientPartition:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    if num_clients > train_ids.size:
        raise ValueError(
            f"num_clients ({num_clients}) exceeds the number of train flows ({train_ids.size})"
        )
    _check_unique(train_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_ids)
    return ClientPartition([np.sort(c) for c in np.array_split(perm, num_clients)], "iid", None, int(seed))


def _attempt_rng(seed: int, attempt: int) -> np.random.Generator:
    # [seed, attempt] keeps retry streams of different seeds independent
    # (seed + attempt would make seed s / attempt 1 identical to seed s+1 / attempt 0).
    # Attempt 0 is bit-identical to default_rng(seed), the original single-shot stream.
    return np.random.default_rng([seed, attempt])


def _subsample_rng(seed: int) -> np.random.Generator:
    # Own stream: a SeedSequence with spawn_key (1,) differs from default_rng(seed) and from
    # every retry stream default_rng([seed, attempt]), so subsample draws never replay the
    # IID permutation or a Dirichlet attempt (``[seed, 1]`` would equal Dirichlet attempt 1).
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))


def _dirichlet_attempt(
    train_ids: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """One Dirichlet label-skew split of ``train_ids`` (``y`` are their labels) into clients."""
    buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for c in classes:
        ids_c = train_ids[y == c].copy()
        rng.shuffle(ids_c)
        props = rng.dirichlet(np.full(num_clients, float(alpha)))
        cuts = (np.cumsum(props)[:-1] * ids_c.shape[0]).astype(np.int64)
        for k, part in enumerate(np.split(ids_c, cuts)):
            if part.size:
                buckets[k].append(part)
    return [np.sort(np.concatenate(b)) if b else np.empty((0,), dtype=np.int64) for b in buckets]


def partition_dirichlet(
    train_ids: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_flows_per_client: int = 1,
    max_retries: int = 100,
) -> ClientPartition:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0")
    if min_flows_per_client < 1:
        raise ValueError(f"min_flows_per_client must be >= 1, got {min_flows_per_client}")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    if num_clients * min_flows_per_client > train_ids.size:
        raise ValueError(
            f"num_clients ({num_clients}) * min_flows_per_client ({min_flows_per_client}) exceeds "
            f"the number of train flows ({train_ids.size}); lower num_clients or min_flows_per_client"
        )
    _check_unique(train_ids)
    y = np.asarray(labels, dtype=np.int64)[train_ids]
    classes = np.unique(y)
    for attempt in range(max_retries):
        clients = _dirichlet_attempt(train_ids, y, classes, num_clients, alpha, _attempt_rng(seed, attempt))
        if min(c.shape[0] for c in clients) >= min_flows_per_client:
            return ClientPartition(clients, "dirichlet", float(alpha), int(seed))
    raise ValueError(
        f"Dirichlet partition (alpha={alpha}) could not give every client >= "
        f"{min_flows_per_client} flows after {max_retries} attempts; increase alpha "
        "or lower min_flows_per_client"
    )


def _stratified_quota(counts: np.ndarray, k: int) -> np.ndarray:
    """Per-class quotas summing to exactly ``k`` (requires ``k <= counts.sum()``).

    Largest-remainder allocation in exact integer arithmetic: each class starts at
    ``floor(count * k / n)`` with a floor of 1, capped at its count; leftover slots go one by
    one to the classes with the largest fractional remainder that still have spare flows
    (ties toward lower class ids). If ``k`` is below the number of classes the floor-of-1
    quotas are trimmed from the largest quota first (ties toward lower class ids), so some
    classes are dropped. If the floor-of-1 bumps overshoot ``k``, classes above 1 give back
    one at a time, the one with the smallest fractional remainder first; the bumped classes
    keep their single flow.
    """
    n = int(counts.sum())
    exact = counts * k  # the proportional quota, scaled by n
    never = np.iinfo(np.int64).min // 2  # sentinel that sorts last and cannot overflow when negated
    quota = np.minimum(np.maximum(1, exact // n), counts)
    while quota.sum() > k:
        if k < counts.size:
            quota[int(np.argmax(quota))] -= 1
        else:
            surplus = np.where(quota > 1, quota * n - exact, never)
            quota[int(np.argmax(surplus))] -= 1
    while quota.sum() < k:
        remainder = np.where(quota < counts, exact - quota * n, never)
        order = np.lexsort((np.arange(counts.size), -remainder))
        for i in order[: k - int(quota.sum())]:
            if quota[i] < counts[i]:
                quota[i] += 1
    return quota


def _stratified_take(ids: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick exactly ``min(k, len(ids))`` of ``ids``, stratified by label.

    Quotas follow ``_stratified_quota`` (proportional, largest-remainder, at least one per
    class whenever ``k`` is at least the number of classes). Returns sorted int64 ids.
    """
    if k >= ids.shape[0]:
        return np.sort(ids).astype(np.int64)
    y = labels[ids]
    classes, counts = np.unique(y, return_counts=True)
    quota = _stratified_quota(counts.astype(np.int64), k)
    picked = [
        rng.choice(ids[y == c], size=int(q), replace=False)
        for c, q in zip(classes, quota)
        if q > 0
    ]
    return np.sort(np.concatenate(picked)).astype(np.int64)


def subsample_partition(
    partition: ClientPartition, labels: np.ndarray, max_flows_per_client: int, seed: int
) -> ClientPartition:
    cap = int(max_flows_per_client)
    if cap < 1:
        raise ValueError(f"max_flows_per_client must be >= 1, got {max_flows_per_client}")
    rng = _subsample_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    out = [
        ids if ids.shape[0] <= cap else _stratified_take(ids, labels, cap, rng)
        for ids in partition.client_flow_ids
    ]
    return ClientPartition(out, partition.scheme, partition.alpha, partition.seed)


def make_partition(train_ids: np.ndarray, labels: np.ndarray, cfg: dict[str, Any]) -> ClientPartition:
    scheme = str(cfg.get("scheme", "iid")).lower()
    num_clients = int(cfg.get("num_clients", 10))
    seed = int(cfg.get("seed", 42))
    cap = cfg.get("max_flows_per_client")
    if cap is not None:
        cap = int(cap)
        if cap < 1:
            raise ValueError(f"max_flows_per_client must be >= 1, got {cap}")
    if scheme == "iid":
        part = partition_iid(train_ids, num_clients, seed)
    elif scheme == "dirichlet":
        min_flows = int(cfg.get("min_flows_per_client", 1))
        if cap is not None and cap < min_flows:
            raise ValueError(
                f"max_flows_per_client ({cap}) is below min_flows_per_client ({min_flows}); "
                "subsampling would leave clients under the minimum"
            )
        part = partition_dirichlet(
            train_ids,
            labels,
            num_clients,
            float(cfg.get("alpha", 0.5)),
            seed,
            min_flows,
        )
    else:
        raise ValueError(f"unknown partition scheme {scheme!r}; expected 'iid' or 'dirichlet'")
    if cap is not None:
        part = subsample_partition(part, labels, cap, seed)
    return part
