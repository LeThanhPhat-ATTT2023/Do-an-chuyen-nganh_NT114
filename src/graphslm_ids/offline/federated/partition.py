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


def partition_iid(train_ids: np.ndarray, num_clients: int, seed: int) -> ClientPartition:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(train_ids, dtype=np.int64))
    return ClientPartition([np.sort(c) for c in np.array_split(perm, num_clients)], "iid", None, int(seed))


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
    train_ids = np.asarray(train_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[train_ids]
    classes = np.unique(y)
    for attempt in range(max_retries):
        rng = np.random.default_rng(seed + attempt)
        buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for c in classes:
            ids_c = train_ids[y == c].copy()
            rng.shuffle(ids_c)
            props = rng.dirichlet(np.full(num_clients, float(alpha)))
            cuts = (np.cumsum(props)[:-1] * ids_c.shape[0]).astype(np.int64)
            for k, part in enumerate(np.split(ids_c, cuts)):
                if part.size:
                    buckets[k].append(part)
        clients = [
            np.sort(np.concatenate(b)) if b else np.empty((0,), dtype=np.int64) for b in buckets
        ]
        if min(c.shape[0] for c in clients) >= min_flows_per_client:
            return ClientPartition(clients, "dirichlet", float(alpha), int(seed))
    raise ValueError(
        f"Dirichlet partition (alpha={alpha}) could not give every client >= "
        f"{min_flows_per_client} flows after {max_retries} attempts; increase alpha "
        "or lower min_flows_per_client"
    )


def _stratified_take(ids: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    y = labels[ids]
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.floor(counts * (k / ids.shape[0])).astype(np.int64))
    quota = np.minimum(quota, counts)
    while quota.sum() > k:
        quota[int(np.argmax(quota))] -= 1
    picked = [
        rng.choice(ids[y == c], size=int(q), replace=False)
        for c, q in zip(classes, quota)
        if q > 0
    ]
    return np.sort(np.concatenate(picked)).astype(np.int64)


def subsample_partition(
    partition: ClientPartition, labels: np.ndarray, max_flows_per_client: int, seed: int
) -> ClientPartition:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    cap = int(max_flows_per_client)
    out = [
        ids if ids.shape[0] <= cap else _stratified_take(ids, labels, cap, rng)
        for ids in partition.client_flow_ids
    ]
    return ClientPartition(out, partition.scheme, partition.alpha, partition.seed)


def make_partition(train_ids: np.ndarray, labels: np.ndarray, cfg: dict[str, Any]) -> ClientPartition:
    scheme = str(cfg.get("scheme", "iid")).lower()
    num_clients = int(cfg.get("num_clients", 10))
    seed = int(cfg.get("seed", 42))
    if scheme == "iid":
        part = partition_iid(train_ids, num_clients, seed)
    elif scheme == "dirichlet":
        part = partition_dirichlet(
            train_ids,
            labels,
            num_clients,
            float(cfg.get("alpha", 0.5)),
            seed,
            int(cfg.get("min_flows_per_client", 1)),
        )
    else:
        raise ValueError(f"unknown partition scheme {scheme!r}; expected 'iid' or 'dirichlet'")
    cap = cfg.get("max_flows_per_client")
    if cap:
        part = subsample_partition(part, labels, int(cap), seed)
    return part
