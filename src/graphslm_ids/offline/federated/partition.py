"""Split TRAIN flows into simulated FL clients (IID, Dirichlet label skew, explicit class shares)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

VALID_SCHEMES = ("iid", "dirichlet", "explicit")


@dataclass
class ClientPartition:
    client_flow_ids: list[np.ndarray]
    scheme: str
    alpha: float | None
    seed: int
    counts: np.ndarray | None = None  # (K, C) int64 client x class table; set by the explicit scheme
    params: dict[str, Any] = field(default_factory=dict)  # scheme parameters worth recording

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
            "counts": None if self.counts is None else np.asarray(self.counts, dtype=np.int64).tolist(),
            "params": dict(self.params),
            "client_flow_ids": [ids.astype(np.int64).tolist() for ids in self.client_flow_ids],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ClientPartition":
        return cls(
            client_flow_ids=[np.asarray(ids, dtype=np.int64) for ids in payload["client_flow_ids"]],
            scheme=str(payload["scheme"]),
            alpha=None if payload.get("alpha") is None else float(payload["alpha"]),
            seed=int(payload["seed"]),
            counts=None if payload.get("counts") is None else np.asarray(payload["counts"], dtype=np.int64),
            params=dict(payload.get("params") or {}),
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


# -- explicit class-share table ---------------------------------------------------------------


def group_members(groups: dict[str, Any], num_clients: int) -> dict[str, list[int]]:
    """Validated 0-based client indices of every gateway group.

    ``groups`` maps a group name to ``{"clients": [first, last], "family": ...}`` with 1-based
    inclusive client ranges. The ranges must lie in ``1..num_clients``, be disjoint and together
    cover every client.
    """
    if not isinstance(groups, dict) or not groups:
        raise ValueError("partition.groups must be a non-empty mapping of group name -> {clients, family}")
    owner: dict[int, str] = {}
    members: dict[str, list[int]] = {}
    for name, spec in groups.items():
        rng = spec.get("clients") if isinstance(spec, dict) else None
        if (
            not isinstance(rng, (list, tuple))
            or len(rng) != 2
            or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in rng)
        ):
            raise ValueError(f"group {name!r}: 'clients' must be a [first, last] pair of 1-based client numbers")
        first, last = int(rng[0]), int(rng[1])
        if not 1 <= first <= last <= num_clients:
            raise ValueError(f"group {name!r}: clients [{first}, {last}] must satisfy 1 <= first <= last <= {num_clients}")
        for k in range(first - 1, last):
            if k in owner:
                raise ValueError(f"groups {owner[k]!r} and {name!r} both contain client {k + 1}")
            owner[k] = str(name)
        members[str(name)] = list(range(first - 1, last))
    missing = [k + 1 for k in range(num_clients) if k not in owner]
    if missing:
        raise ValueError(f"groups must cover all {num_clients} clients; no group contains client(s) {missing}")
    return members


def _explicit_plan(
    num_clients: int, groups: dict[str, Any], families: dict[str, list[str]], dominance: Any
) -> tuple[float, dict[str, list[int]], dict[str, str]]:
    """Validate the explicit-scheme settings; returns (dominance, group -> clients, class -> group)."""
    try:
        p = float(dominance)
    except (TypeError, ValueError):
        raise ValueError(f"partition.dominance must be a number in [0, 1], got {dominance!r}") from None
    if not 0.0 <= p <= 1.0:  # also rejects NaN
        raise ValueError(f"partition.dominance must lie in [0, 1], got {dominance!r}")
    members = group_members(groups, int(num_clients))
    if not isinstance(families, dict):
        raise ValueError("partition.families must be a mapping of family name -> class names")
    group_of_family: dict[str, str] = {}
    for name, spec in groups.items():
        family = spec.get("family")
        if family is None:
            continue
        if family not in families:
            raise ValueError(f"group {name!r}: family {family!r} is not in partition.families {sorted(families)}")
        if family in group_of_family:
            raise ValueError(f"family {family!r} is used by groups {group_of_family[family]!r} and {name!r}; at most one")
        group_of_family[family] = str(name)
    class_family: dict[str, str] = {}
    group_of_class: dict[str, str] = {}
    for family, class_names in families.items():
        for cls in class_names:
            if class_family.setdefault(str(cls), str(family)) != family:
                raise ValueError(f"class {cls!r} is listed in both families {class_family[str(cls)]!r} and {family!r}")
            if family in group_of_family:
                group_of_class[str(cls)] = group_of_family[family]
    return p, members, group_of_class


def explicit_fractions(
    label_names: dict[int, str],
    num_clients: int,
    groups: dict[str, Any],
    families: dict[str, list[str]],
    dominance: float,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    """``(K, C)`` share of each class that every client receives; every column sums to 1.

    A class of family ``F`` whose group has ``n_F`` clients goes ``p / n_F + (1 - p) / K`` to each
    client inside the group and ``(1 - p) / K`` to every other client. A class in no family, or
    in a family no group uses, is split ``1 / K`` evenly. ``C`` is ``max(label_names) + 1``
    unless ``num_classes`` is larger.
    """
    p, members, group_of_class = _explicit_plan(num_clients, groups, families, dominance)
    k_total = int(num_clients)
    n_classes = max([int(num_classes or 0)] + [int(c) + 1 for c in label_names])
    frac = np.full((k_total, n_classes), 1.0 / k_total, dtype=np.float64)
    for c in range(n_classes):
        group = group_of_class.get(str(label_names.get(c)))
        if group is None:
            continue
        inside = members[group]
        frac[:, c] = (1.0 - p) / k_total
        frac[inside, c] += p / len(inside)
    if not np.allclose(frac.sum(axis=0), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("explicit class shares do not sum to 1 for every class")
    return frac


def explicit_counts(fractions: np.ndarray, class_counts: np.ndarray) -> np.ndarray:
    """``(K, C)`` integer table ``n[k, c]`` whose column sums are exactly ``class_counts``.

    ``n = floor(f * N_c)`` (the ``1e-9`` absorbs float error, e.g. ``0.17 * 400 = 67.999...``);
    the ``N_c - sum(n)`` leftovers go one each to the clients with the largest fractional part,
    ties to the smaller client index.
    """
    f = np.asarray(fractions, dtype=np.float64)
    totals = np.asarray(class_counts, dtype=np.int64)
    if f.ndim != 2 or totals.shape != (f.shape[1],):
        raise ValueError(f"fractions {f.shape} and class_counts {totals.shape} do not line up")
    scaled = f * totals
    base = np.floor(scaled + 1e-9).astype(np.int64)
    left = totals - base.sum(axis=0)
    if (left < 0).any():
        raise ValueError("fractions sum to more than 1 for some class")
    # Rounded so two shares that are equal on paper stay tied despite float noise.
    remainder = np.round(scaled - base, 9)
    out = base.copy()
    for c in range(f.shape[1]):
        winners = np.lexsort((np.arange(f.shape[0]), -remainder[:, c]))[: int(left[c])]
        out[winners, c] += 1
    return out


def capped_counts(fractions: np.ndarray, class_counts: np.ndarray, cap: int | None) -> tuple[np.ndarray, float]:
    """The share table shrunk uniformly so that no client holds more than ``cap`` flows.

    Returns ``(n', s)``. Without a cap, or when ``n = explicit_counts(f, N)`` already has every
    client total ``<= cap``, that is ``(n, 1.0)``. Otherwise ``n' = explicit_counts(f, floor(s * N_c
    + 1e-9))``: the same fractions and tie rule on smaller class totals, so the client x class shares
    stay those of ``f``. ``s`` starts at ``cap / max_k sum_c n[k, c]`` and, while the largest client
    total of ``n'`` still exceeds ``cap``, is multiplied by ``cap / that total``. Deterministic.
    """
    f = np.asarray(fractions, dtype=np.float64)
    totals = np.asarray(class_counts, dtype=np.int64)
    counts = explicit_counts(f, totals)
    top = int(counts.sum(axis=1).max()) if counts.size else 0
    if cap is None or top <= cap:
        return counts, 1.0
    scale = cap / top
    while True:
        counts = explicit_counts(f, np.floor(scale * totals + 1e-9).astype(np.int64))
        top = int(counts.sum(axis=1).max())
        if top <= cap:
            return counts, scale
        scale *= cap / top


def partition_explicit(
    train_ids: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str],
    cfg: dict[str, Any],
    order: np.ndarray | None = None,
) -> ClientPartition:
    """Deterministic client x class split: the share table, then consecutive time segments.

    The train flows of class ``c`` are sorted by ``(order[id], id)`` (by ``id`` without ``order``)
    and cut into consecutive runs of ``n[0, c], n[1, c], ...`` flows; client ``k`` takes run
    ``k``. With ``max_flows_per_client`` the WHOLE table shrinks uniformly (``capped_counts``,
    scale recorded as ``params["cap_scale"]``) so the shares of ``f`` survive, and client ``k``
    keeps the first ``n'[k, c]`` flows of its full-data run of class ``c``. Nothing is random;
    ``seed`` is only recorded.
    """
    num_clients = int(cfg.get("num_clients", 10))
    min_flows = int(cfg.get("min_flows_per_client", 1))
    cap = cfg.get("max_flows_per_client")
    if cap is not None:
        cap = int(cap)
        if cap < 1:
            raise ValueError(f"max_flows_per_client must be >= 1, got {cap}")
    if min_flows < 1:
        raise ValueError(f"min_flows_per_client must be >= 1, got {min_flows}")
    groups, families = cfg.get("groups"), cfg.get("families")
    if groups is None or families is None:
        raise ValueError("partition scheme 'explicit' needs partition.groups and partition.families")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    _check_unique(train_ids)
    ranks = None
    if order is not None:
        ranks = np.asarray(order, dtype=np.float64)
        if train_ids.size and int(train_ids.max()) >= ranks.shape[0]:
            raise ValueError(f"order has {ranks.shape[0]} entries but train_ids reach flow id {int(train_ids.max())}")
    y = labels[train_ids]
    num_classes = max([int(c) + 1 for c in label_names] + [int(y.max()) + 1 if y.size else 0])
    dominance = float(cfg.get("dominance", 0.8))
    fractions = explicit_fractions(label_names, num_clients, groups, families, dominance, num_classes=num_classes)
    class_counts = np.bincount(y, minlength=num_classes)
    full = explicit_counts(fractions, class_counts)  # fixes the run boundaries
    kept, cap_scale = capped_counts(fractions, class_counts, cap)  # how much of each run survives the cap
    grown = np.argwhere(kept > full)
    if grown.size:
        # Largest remainder is not monotone in the class total (Alabama paradox), so a shrunk
        # table can, for tiny classes, give a client a flow its full-data run does not have.
        k, c = (int(v) for v in grown[0])
        raise AssertionError(
            f"capped explicit table gives client {k + 1} {int(kept[k, c])} flows of class {c} but its "
            f"full-data run has {int(full[k, c])}; choose another max_flows_per_client"
        )
    small = np.flatnonzero(kept.sum(axis=1) < min_flows)
    if small.size:
        raise ValueError(
            f"explicit partition leaves client(s) {(small + 1).tolist()} below min_flows_per_client "
            f"({min_flows}); lower min_flows_per_client or raise max_flows_per_client"
        )
    parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        ids_c = train_ids[y == c]
        ids_c = np.sort(ids_c) if ranks is None else ids_c[np.lexsort((ids_c, ranks[ids_c]))]
        bounds = np.concatenate([[0], np.cumsum(full[:, c])])
        for k in range(num_clients):
            parts[k].append(ids_c[bounds[k] : bounds[k] + kept[k, c]])
    clients = [np.sort(np.concatenate(p)).astype(np.int64) for p in parts]
    params = {
        "dominance": dominance,
        "groups": {
            str(g): {"clients": [int(s["clients"][0]), int(s["clients"][1])], "family": s.get("family")}
            for g, s in groups.items()
        },
        "families": {str(f): [str(n) for n in names] for f, names in families.items()},
        "ordered_by": "flow_id" if ranks is None else "flow_start_ts",
        "max_flows_per_client": cap,
        "cap_scale": float(cap_scale),
    }
    return ClientPartition(clients, "explicit", None, int(cfg.get("seed", 42)), counts=kept, params=params)


def check_partition_config(cfg: dict[str, Any]) -> str:
    """Reject a bad ``partition`` section without touching any data; returns the scheme."""
    scheme = str(cfg.get("scheme", "iid")).lower()
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"unknown partition scheme {scheme!r}; expected one of {VALID_SCHEMES}")
    if scheme == "explicit":
        groups, families = cfg.get("groups"), cfg.get("families")
        if groups is None or families is None:
            raise ValueError("partition scheme 'explicit' needs partition.groups and partition.families")
        _explicit_plan(int(cfg.get("num_clients", 10)), groups, families, cfg.get("dominance", 0.8))
    return scheme


def make_partition(
    train_ids: np.ndarray,
    labels: np.ndarray,
    cfg: dict[str, Any],
    *,
    label_names: dict[int, str] | None = None,
    order: np.ndarray | None = None,
) -> ClientPartition:
    """Build the client split named by ``cfg["scheme"]``.

    ``label_names`` (class id -> name) is required by ``explicit``; ``order`` (flow id -> start
    time) is optional there and ignored by the other schemes.
    """
    scheme = str(cfg.get("scheme", "iid")).lower()
    num_clients = int(cfg.get("num_clients", 10))
    seed = int(cfg.get("seed", 42))
    if scheme == "explicit":
        if label_names is None:
            raise ValueError("partition scheme 'explicit' needs label_names (class id -> name)")
        return partition_explicit(train_ids, labels, label_names, cfg, order)
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
        raise ValueError(f"unknown partition scheme {scheme!r}; expected one of {VALID_SCHEMES}")
    if cap is not None:
        part = subsample_partition(part, labels, cap, seed)
    return part
