"""Pure, deterministic reporting helpers for the Smart-BOTH evaluation.

These functions are intentionally free of any model / graph / torch dependency
so they can be unit-tested in isolation (see ``tests/test_eval_reporting.py``)
and reused from ``v3_eval_both_splits.py`` to enrich the structured JSON output
with the defensibility artifacts the v5 spec asks for:

  * confusion matrix (rows = true, cols = pred),
  * per-class support,
  * bootstrap 95% confidence intervals for macro-F1 and each per-class F1.

F1 definition
-------------
Matched **exactly** to the production trainer
(``train_hgt_flow_classifier.metrics_from_predictions``) so the bootstrap point
estimate equals the headline number reported during training:

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * p * r / max(p + r, 1e-12)
    macro_f1  = mean of per-class f1 over classes with support > 0

Determinism
-----------
The bootstrap resamples test indices with a seeded ``numpy`` RNG
(``np.random.default_rng(seed)``); the same ``seed`` always yields the same CI.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "confusion_matrix",
    "per_class_support",
    "per_class_f1_from_labels",
    "bootstrap_f1_ci",
    "apply_bias",
    "calibrate_bias_for_macro_f1",
]


def _as_int_1d(arr: Any) -> np.ndarray:
    """Coerce an input to a contiguous 1-D int64 array (no copy when possible)."""
    out = np.asarray(arr).reshape(-1)
    if out.dtype != np.int64:
        out = out.astype(np.int64)
    return out


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    num_classes: int,
) -> np.ndarray:
    """Return the ``(num_classes, num_classes)`` confusion matrix.

    Rows index the **true** class, columns the **predicted** class, so
    ``cm[i, j]`` is the number of samples whose true label is ``i`` and whose
    predicted label is ``j``. ``cm.sum(axis=1)`` is per-class support and
    ``cm.sum()`` is the number of samples.

    Implemented with a single ``np.bincount`` over flattened ``true*C + pred``
    indices — fully deterministic and O(N).
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    t = _as_int_1d(y_true)
    p = _as_int_1d(y_pred)
    if t.shape != p.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length; "
            f"got {t.shape[0]} and {p.shape[0]}"
        )
    if t.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    if t.min() < 0 or p.min() < 0 or t.max() >= num_classes or p.max() >= num_classes:
        raise ValueError(
            f"labels must lie in [0, {num_classes}); "
            f"got true in [{t.min()}, {t.max()}], pred in [{p.min()}, {p.max()}]"
        )
    flat = t * num_classes + p
    cm = np.bincount(flat, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).astype(np.int64)


def per_class_support(y_true: Any, num_classes: int) -> dict[int, int]:
    """Return ``{class_id: count}`` for every class in ``range(num_classes)``.

    Classes absent from ``y_true`` are reported with a count of ``0`` so the
    JSON output always lists the full label space. Keys and values are plain
    python ``int`` (JSON-serializable).
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    t = _as_int_1d(y_true)
    if t.size and (t.min() < 0 or t.max() >= num_classes):
        raise ValueError(
            f"labels must lie in [0, {num_classes}); got [{t.min()}, {t.max()}]"
        )
    counts = np.bincount(t, minlength=num_classes)
    return {int(cid): int(counts[cid]) for cid in range(num_classes)}


def per_class_f1_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-class F1 + support, matching the trainer's definition.

    Returns ``(f1, support)`` as float64 / int64 arrays of length
    ``num_classes``. ``f1[c]`` is well-defined for every class (it is ``0.0``
    when a class has no true and no predicted samples); ``support[c]`` lets the
    caller decide which classes enter the macro average.

    Inputs are assumed pre-validated (this is the bootstrap hot loop).
    """
    cls = np.arange(num_classes)[:, None]            # (C, 1)
    pred_eq = (y_pred[None, :] == cls)               # (C, N)
    lab_eq = (y_true[None, :] == cls)                # (C, N)
    tp = np.logical_and(pred_eq, lab_eq).sum(axis=1).astype(np.float64)
    fp = np.logical_and(pred_eq, ~lab_eq).sum(axis=1).astype(np.float64)
    fn = np.logical_and(~pred_eq, lab_eq).sum(axis=1).astype(np.float64)
    support = lab_eq.sum(axis=1).astype(np.int64)

    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / np.maximum(tp + fn, 1.0)
    f1 = 2.0 * prec * rec / np.maximum(prec + rec, 1e-12)
    return f1, support


def _macro_f1(f1: np.ndarray, support: np.ndarray) -> float:
    mask = support > 0
    if not mask.any():
        return 0.0
    return float(f1[mask].mean())


def apply_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return ``argmax(logits + bias, axis=1)`` as an int64 array.

    ``bias`` (length ``C``) is the per-class additive shift on the *logit*
    scale — the multiclass generalisation of a per-class decision threshold.
    ``bias = 0`` recovers the raw model argmax.
    """
    lg = np.asarray(logits, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64).reshape(1, -1)
    if lg.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C); got shape {lg.shape}")
    if b.shape[1] != lg.shape[1]:
        raise ValueError(
            f"bias length {b.shape[1]} != num classes {lg.shape[1]}"
        )
    return (lg + b).argmax(axis=1).astype(np.int64)


def calibrate_bias_for_macro_f1(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    *,
    n_rounds: int = 12,
    grid_half_width: float = 6.0,
    grid_points: int = 25,
    shrink: float = 0.6,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tune a per-class additive logit bias that maximises macro-F1 on a holdout.

    This is post-hoc decision-rule calibration: it learns a vector ``b`` (length
    ``num_classes``) such that ``argmax(logits + b)`` has higher macro-F1 than
    the raw ``argmax(logits)`` on the supplied (validation) split. It is the
    multiclass generalisation of per-class threshold tuning to maximise F1
    (Lipton et al. 2014) and is closely related to logit adjustment
    (Menon et al. ICLR 2021), but it is *learned per class* rather than tied to
    a single scalar τ on the log-prior — so it can penalise exactly the one
    over-predicted "sink" class without disturbing the rest.

    **Honest-evaluation contract:** call this on the VALIDATION split only, then
    apply the returned ``b`` (via :func:`apply_bias`) to the TEST split. The bias
    never sees test labels, so there is no test-set peeking.

    Algorithm — coordinate ascent. Start from ``b = 0``. For ``n_rounds`` passes,
    sweep the classes in a fixed (seeded) order; for each class try a symmetric
    grid of candidate biases (holding the others fixed) and keep the value that
    most improves holdout macro-F1, breaking ties toward smaller |bias|. The grid
    half-width shrinks by ``shrink`` each round (coarse→fine). Deterministic given
    ``seed``. The zero vector is always in the candidate set, so the result can
    never score below the raw baseline on the tuning data.

    Returns ``(bias, info)`` where ``info`` carries ``macro_f1_raw``,
    ``macro_f1_calibrated``, ``rounds_run`` and the final ``grid_half_width``.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    lg = np.asarray(logits, dtype=np.float64)
    y = _as_int_1d(labels)
    if lg.ndim != 2 or lg.shape[1] != num_classes:
        raise ValueError(
            f"logits must be (N, {num_classes}); got shape {lg.shape}"
        )
    if lg.shape[0] != y.shape[0]:
        raise ValueError(
            f"logits and labels must have the same length; "
            f"got {lg.shape[0]} and {y.shape[0]}"
        )

    def _macro(bias: np.ndarray) -> float:
        pred = (lg + bias.reshape(1, -1)).argmax(axis=1).astype(np.int64)
        f1, sup = per_class_f1_from_labels(y, pred, num_classes)
        return _macro_f1(f1, sup)

    bias = np.zeros(num_classes, dtype=np.float64)
    raw_macro = _macro(bias)
    best_macro = raw_macro

    rng = np.random.default_rng(seed)
    class_order = np.arange(num_classes)

    half = float(grid_half_width)
    rounds_run = 0
    for _ in range(n_rounds):
        rounds_run += 1
        rng.shuffle(class_order)
        improved = False
        grid = np.linspace(-half, half, grid_points)
        for c in class_order:
            current = bias[c]
            best_val = current
            best_local = best_macro
            # Always include the current value (delta 0) so we never regress.
            for delta in grid:
                cand = current + float(delta)
                bias[c] = cand
                m = _macro(bias)
                # Strict improvement, or equal macro with a smaller |bias|
                # (prefer the least-aggressive correction — better generalisation).
                if m > best_local + 1e-12 or (
                    abs(m - best_local) <= 1e-12 and abs(cand) < abs(best_val)
                ):
                    best_local = m
                    best_val = cand
            bias[c] = best_val
            if best_local > best_macro + 1e-12:
                improved = True
            best_macro = best_local
        half *= shrink
        if not improved:
            break

    info: dict[str, Any] = {
        "macro_f1_raw": float(raw_macro),
        "macro_f1_calibrated": float(best_macro),
        "rounds_run": int(rounds_run),
        "final_grid_half_width": float(half),
        "n_rounds": int(n_rounds),
        "grid_points": int(grid_points),
        "seed": int(seed),
    }
    return bias, info


def bootstrap_f1_ci(
    y_true: Any,
    y_pred: Any,
    num_classes: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Seeded percentile-bootstrap CIs for macro-F1 and every per-class F1.

    For ``n_boot`` iterations, resample the test indices **with replacement**
    (same size N) using a seeded RNG, recompute F1 on the resample, and take the
    ``[alpha/2, 1-alpha/2]`` percentiles as the CI. The point estimate is the F1
    on the *full* (un-resampled) data, so it exactly matches the trainer.

    Returns::

        {
          "n_boot": int, "seed": int, "alpha": float, "n_test": int,
          "macro_f1": {"point": float, "lo": float, "hi": float},
          "per_class": {class_id: {"point","lo","hi","support"}, ...},
        }

    All bounds are clamped to ``[0, 1]`` (F1 is already in that range; this only
    guards against floating-point drift). Per-class entries are reported for
    every class with ``support > 0`` on the full data.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")

    t = _as_int_1d(y_true)
    p = _as_int_1d(y_pred)
    if t.shape != p.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length; "
            f"got {t.shape[0]} and {p.shape[0]}"
        )
    n = t.shape[0]

    # Point estimates on the full data (canonical, seed-independent).
    point_f1, point_support = per_class_f1_from_labels(t, p, num_classes)
    point_macro = _macro_f1(point_f1, point_support)

    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    def _empty_result() -> dict[str, Any]:
        return {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "alpha": float(alpha),
            "n_test": int(n),
            "macro_f1": {
                "point": float(point_macro),
                "lo": float(point_macro),
                "hi": float(point_macro),
            },
            "per_class": {},
        }

    if n == 0:
        return _empty_result()

    rng = np.random.default_rng(seed)
    macro_samples = np.empty(n_boot, dtype=np.float64)
    per_class_samples = np.empty((n_boot, num_classes), dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)            # resample with replacement
        bt = t[idx]
        bp = p[idx]
        f1_b, sup_b = per_class_f1_from_labels(bt, bp, num_classes)
        per_class_samples[b] = f1_b
        macro_samples[b] = _macro_f1(f1_b, sup_b)

    def _ci(samples: np.ndarray, point: float) -> dict[str, float]:
        lo = float(np.clip(np.percentile(samples, lo_pct), 0.0, 1.0))
        hi = float(np.clip(np.percentile(samples, hi_pct), 0.0, 1.0))
        pt = float(np.clip(point, 0.0, 1.0))
        # Guarantee the CI brackets the point estimate even if the percentile
        # band falls entirely to one side of it (small n_boot / degenerate).
        lo = min(lo, pt)
        hi = max(hi, pt)
        return {"point": pt, "lo": lo, "hi": hi}

    result: dict[str, Any] = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "n_test": int(n),
        "macro_f1": _ci(macro_samples, point_macro),
        "per_class": {},
    }
    for cid in range(num_classes):
        if point_support[cid] <= 0:
            continue
        entry = _ci(per_class_samples[:, cid], float(point_f1[cid]))
        entry["support"] = int(point_support[cid])
        result["per_class"][int(cid)] = entry
    return result
