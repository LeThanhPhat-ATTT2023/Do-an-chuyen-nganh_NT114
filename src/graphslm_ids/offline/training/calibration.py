"""Pure, dependency-light calibration utilities for the fair-compare track.

All functions operate on numpy arrays of RAW logits (N, C) + integer labels (N,)
so they can be unit-tested without a model, and reused by the Tier-0 driver.
Decision rule everywhere: argmax(logits + bias). Selection always on VAL.
"""
from __future__ import annotations

import numpy as np


def macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Unweighted mean per-class F1 (same definition as the trainer/sklearn).

    Classes absent from both preds and labels (denominator 0) count as F1=0,
    matching sklearn's ``zero_division=0`` default.
    """
    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (labels == c)))
        fp = int(np.sum((preds == c) & (labels != c)))
        fn = int(np.sum((preds != c) & (labels == c)))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s))


def apply_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Argmax over logits shifted by a per-class additive bias."""
    return (np.asarray(logits, dtype=np.float64) + np.asarray(bias, dtype=np.float64)).argmax(axis=1)


def tune_per_class_bias(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    grid: np.ndarray | None = None,
    rounds: int = 3,
) -> np.ndarray:
    """Greedy coordinate ascent: for each class, pick the additive bias on `grid`
    that maximizes VAL macro-F1, holding the others fixed; repeat `rounds` times.
    Returns the bias vector (C,). Selection is ON VAL ONLY.

    Cost is O(rounds * C * len(grid)) macro_f1 evaluations, each O(N * C). Intended
    for an offline val split of ~tens of thousands of flows (seconds on CPU); for a
    much larger val set, subsample rows before calling.
    """
    if grid is None:
        grid = np.linspace(-4.0, 4.0, 33)
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("grid must be a non-empty 1-D array")
    val_logits = np.asarray(val_logits, dtype=np.float64)
    val_labels = np.asarray(val_labels).ravel()
    bias = np.zeros(num_classes, dtype=np.float64)
    best = macro_f1(apply_bias(val_logits, bias), val_labels, num_classes)
    for _ in range(rounds):
        improved = False
        for c in range(num_classes):
            cur = bias[c]
            best_b = cur
            for b in grid:
                bias[c] = b
                f1 = macro_f1(apply_bias(val_logits, bias), val_labels, num_classes)
                if f1 > best:
                    best, best_b, improved = f1, b, True
            bias[c] = best_b
        if not improved:
            break
    return bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax (subtract row max before exp)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def combine_tta(logits_list: list[np.ndarray]) -> np.ndarray:
    """Average softmax probabilities over K stochastic-sampling passes, then map
    back to an effective logit = log(mean_prob) so per-class bias still composes
    additively. Input: list of (N, C) RAW logit arrays for the SAME rows/order."""
    if not logits_list:
        raise ValueError("logits_list is empty")
    probs = np.mean([_softmax(l) for l in logits_list], axis=0)
    return np.log(np.clip(probs, 1e-12, None))
