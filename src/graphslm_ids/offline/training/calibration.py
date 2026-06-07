"""Pure, dependency-light calibration utilities for the fair-compare track.

All functions operate on numpy arrays of RAW logits (N, C) + integer labels (N,)
so they can be unit-tested without a model, and reused by the Tier-0 driver.
Decision rule everywhere: argmax(logits + bias). Selection always on VAL.
"""
from __future__ import annotations

import numpy as np


def macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Unweighted mean per-class F1 (same definition as the trainer/sklearn)."""
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
