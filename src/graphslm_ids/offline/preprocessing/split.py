"""Train/val/test split protocols for the v3 pipeline.

Produces TWO independently-sampled splits over the same flow set so the same
HGT can be trained twice and the random-vs-temporal F1 gap can be reported as
the v3 contribution:

* ``random`` — stratified by attack class, shuffled by a seeded RNG. Mirrors
  the protocol used by prior work (E-GraphSAGE, TFE-GNN). Inflates F1 because
  test flows are drawn from the same PCAP burst as training flows.
* ``temporal`` — per-class chronological cut on ``flow_start_ts``. First 80%
  in time -> train, next 10% -> val, last 10% -> test. Deployment-realistic:
  no test flow can have a training-set successor.

Classes with fewer than 3 flows are pinned to ``train`` for both protocols —
splitting them creates empty val/test buckets that downstream stratified
metrics handle worse than a dropped class.

Memory budget: ~500 MB. We do not materialise feature arrays here; only the
``flow_id`` index, the ``label`` column, and the ``flow_start_ts`` column are
referenced. For 200K flows that's ~10 MB of working data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _stratified_random(
    ids: np.ndarray,
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Per-class shuffle then slice into (train, val, test) by fraction.

    Implementation note: we shuffle the per-class index *array*, not the row
    order of any DataFrame, so the caller's frame is never mutated.
    """
    train_lists: list[np.ndarray] = []
    val_lists: list[np.ndarray] = []
    test_lists: list[np.ndarray] = []
    for cls in np.unique(labels):
        mask = labels == cls
        cls_ids = ids[mask].copy()
        if cls_ids.size < 3:
            # Too few flows to split — pin all to train (see module docstring).
            train_lists.append(cls_ids)
            continue
        rng.shuffle(cls_ids)
        n = cls_ids.size
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Guarantee val and test each receive at least one flow when n >= 3,
        # so downstream stratified metrics don't get a NaN bucket.
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))
        train_lists.append(cls_ids[:n_train])
        val_lists.append(cls_ids[n_train : n_train + n_val])
        test_lists.append(cls_ids[n_train + n_val :])
    return {
        "train": np.concatenate(train_lists) if train_lists else np.array([], dtype=ids.dtype),
        "val": np.concatenate(val_lists) if val_lists else np.array([], dtype=ids.dtype),
        "test": np.concatenate(test_lists) if test_lists else np.array([], dtype=ids.dtype),
    }


def _stratified_temporal(
    ids: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> dict[str, np.ndarray]:
    """Per-class chronological cut into (train, val, test) by fraction.

    Sort key is ``flow_start_ts`` ascending. Ties broken by original index for
    determinism. NaN timestamps are pushed to the END of the class (treated as
    "newest" so they land in test rather than corrupting train).
    """
    train_lists: list[np.ndarray] = []
    val_lists: list[np.ndarray] = []
    test_lists: list[np.ndarray] = []
    for cls in np.unique(labels):
        mask = labels == cls
        cls_ids = ids[mask]
        cls_ts = timestamps[mask]
        if cls_ids.size < 3:
            train_lists.append(cls_ids)
            continue
        # np.argsort on a NaN-bearing float array places NaN last by default
        # in numpy >= 1.4 — which is exactly the behaviour we want.
        order = np.argsort(cls_ts, kind="stable")
        cls_ids_sorted = cls_ids[order]
        n = cls_ids_sorted.size
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))
        train_lists.append(cls_ids_sorted[:n_train])
        val_lists.append(cls_ids_sorted[n_train : n_train + n_val])
        test_lists.append(cls_ids_sorted[n_train + n_val :])
    return {
        "train": np.concatenate(train_lists) if train_lists else np.array([], dtype=ids.dtype),
        "val": np.concatenate(val_lists) if val_lists else np.array([], dtype=ids.dtype),
        "test": np.concatenate(test_lists) if test_lists else np.array([], dtype=ids.dtype),
    }


def make_splits(
    flows_df: pd.DataFrame,
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Build the v3 random + temporal splits.

    Args:
        flows_df: DataFrame indexed by ``flow_id``. Must have columns
            ``label`` (str) and ``flow_start_ts`` (float, unix seconds).
        train_frac: per-class fraction allocated to ``train``.
        val_frac: per-class fraction allocated to ``val``. ``test_frac`` is
            implicit = ``1 - train_frac - val_frac``.
        seed: RNG seed for the random split. Temporal split is seed-free.

    Returns:
        ``{"random": {"train", "val", "test"}, "temporal": {"train", "val", "test"}}``
        where every leaf is a 1-D NumPy array of ``flow_id`` strings.

    Raises:
        ValueError: if required columns are missing or fractions don't sum
            to a valid interior of the simplex.
    """
    if "label" not in flows_df.columns:
        raise ValueError("flows_df must contain a 'label' column")
    if "flow_start_ts" not in flows_df.columns:
        raise ValueError("flows_df must contain a 'flow_start_ts' column")
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0):
        raise ValueError("train_frac and val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 to leave room for test")

    ids = np.asarray(flows_df.index.values)
    labels = flows_df["label"].to_numpy()
    timestamps = flows_df["flow_start_ts"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    return {
        "random": _stratified_random(ids, labels, train_frac, val_frac, rng),
        "temporal": _stratified_temporal(
            ids, labels, timestamps, train_frac, val_frac
        ),
    }


def save_splits(splits: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    """Persist splits as JSON (lists, not arrays, so the file is portable).

    The JSON layout mirrors the in-memory dict so :func:`load_splits` is a
    pure inverse.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict[str, list[Any]]] = {}
    for protocol, by_split in splits.items():
        payload[protocol] = {
            split_name: [str(x) for x in arr.tolist()]
            for split_name, arr in by_split.items()
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_splits(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Inverse of :func:`save_splits`. Returns dict of NumPy arrays of str."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, np.ndarray]] = {}
    for protocol, by_split in payload.items():
        out[protocol] = {
            split_name: np.asarray(ids, dtype=object)
            for split_name, ids in by_split.items()
        }
    return out
