"""Unit tests for the PURE reporting helpers in ``scripts/eval/eval_reporting.py``.

These tests deliberately avoid loading any model, graph, or checkpoint — they
exercise only the deterministic numpy reporting functions so they run fast on
local CPU and stay reproducible (the bootstrap is seeded).

The helper module lives under ``scripts/`` (not an installed package), so we
add it to ``sys.path`` explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/eval is not an installed package; make it importable.
_EVAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import eval_reporting as er  # noqa: E402


# --------------------------------------------------------------------------- #
# confusion_matrix
# --------------------------------------------------------------------------- #
def test_confusion_matrix_hand_checked() -> None:
    # 3 classes. Rows = true, cols = pred.
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0])
    cm = er.confusion_matrix(y_true, y_pred, num_classes=3)

    expected = np.array(
        [
            [1, 1, 0],  # true 0: one ->0, one ->1
            [0, 2, 0],  # true 1: both ->1
            [1, 0, 2],  # true 2: one ->0, two ->2
        ],
        dtype=np.int64,
    )
    assert cm.shape == (3, 3)
    assert cm.dtype == np.int64
    np.testing.assert_array_equal(cm, expected)
    # Row sums == per-class support; total == number of samples.
    np.testing.assert_array_equal(cm.sum(axis=1), np.array([2, 2, 3]))
    assert int(cm.sum()) == y_true.shape[0]


def test_confusion_matrix_perfect_is_diagonal() -> None:
    y = np.array([0, 1, 2, 3, 2, 1, 0])
    cm = er.confusion_matrix(y, y, num_classes=4)
    # All mass on the diagonal.
    assert int(cm.sum()) == int(np.trace(cm))
    np.testing.assert_array_equal(np.diag(cm), np.array([2, 2, 2, 1]))


def test_confusion_matrix_handles_absent_class() -> None:
    # num_classes larger than any observed label -> trailing empty row/col.
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    cm = er.confusion_matrix(y_true, y_pred, num_classes=4)
    assert cm.shape == (4, 4)
    assert int(cm[2].sum()) == 0
    assert int(cm[3].sum()) == 0


# --------------------------------------------------------------------------- #
# per_class_support
# --------------------------------------------------------------------------- #
def test_per_class_support_counts() -> None:
    y_true = np.array([0, 0, 0, 2, 2, 3])
    sup = er.per_class_support(y_true, num_classes=4)
    assert sup == {0: 3, 1: 0, 2: 2, 3: 1}
    # Keys are plain python ints (JSON-serializable), values are plain ints.
    for k, v in sup.items():
        assert isinstance(k, int)
        assert isinstance(v, int)


def test_per_class_support_sums_to_total() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    sup = er.per_class_support(y_true, num_classes=5)
    assert sum(sup.values()) == 200


# --------------------------------------------------------------------------- #
# bootstrap_f1_ci
# --------------------------------------------------------------------------- #
def _make_data(seed: int = 7, n: int = 300, num_classes: int = 4):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, num_classes, size=n)
    # Predictions correct ~70% of the time, otherwise a random other class.
    y_pred = y_true.copy()
    flip = rng.random(n) < 0.3
    y_pred[flip] = rng.integers(0, num_classes, size=int(flip.sum()))
    return y_true, y_pred, num_classes


def test_bootstrap_is_deterministic_same_seed() -> None:
    y_true, y_pred, nc = _make_data()
    a = er.bootstrap_f1_ci(y_true, y_pred, nc, n_boot=200, seed=42)
    b = er.bootstrap_f1_ci(y_true, y_pred, nc, n_boot=200, seed=42)
    assert a == b


def test_bootstrap_different_seed_changes_ci() -> None:
    y_true, y_pred, nc = _make_data()
    a = er.bootstrap_f1_ci(y_true, y_pred, nc, n_boot=200, seed=1)
    b = er.bootstrap_f1_ci(y_true, y_pred, nc, n_boot=200, seed=2)
    # The point estimate is seed-independent; the CI bounds should differ.
    assert a["macro_f1"]["point"] == pytest.approx(b["macro_f1"]["point"])
    assert (a["macro_f1"]["lo"], a["macro_f1"]["hi"]) != (
        b["macro_f1"]["lo"],
        b["macro_f1"]["hi"],
    )


def test_bootstrap_ci_brackets_point_and_in_unit_interval() -> None:
    y_true, y_pred, nc = _make_data()
    out = er.bootstrap_f1_ci(y_true, y_pred, nc, n_boot=500, seed=42)

    assert "macro_f1" in out
    # Every per-class id 0..nc-1 with support>0 must be reported.
    for cid in range(nc):
        assert cid in out["per_class"]

    def _check(entry: dict) -> None:
        lo, point, hi = entry["lo"], entry["point"], entry["hi"]
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert 0.0 <= point <= 1.0
        assert lo <= point <= hi

    _check(out["macro_f1"])
    for entry in out["per_class"].values():
        _check(entry)


def test_bootstrap_point_matches_trainer_f1_definition() -> None:
    # Cross-check the point estimate against the project's canonical F1 math:
    # precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),
    # f1=2pr/max(p+r,1e-12); macro = mean over classes with support>0.
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0])
    nc = 3

    def _ref_f1(cid: int) -> float:
        tp = int(((y_pred == cid) & (y_true == cid)).sum())
        fp = int(((y_pred == cid) & (y_true != cid)).sum())
        fn = int(((y_pred != cid) & (y_true == cid)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return 2.0 * prec * rec / max(prec + rec, 1e-12)

    ref_per_class = {cid: _ref_f1(cid) for cid in range(nc)}
    ref_macro = float(np.mean([ref_per_class[c] for c in range(nc)]))

    out = er.bootstrap_f1_ci(y_true, y_pred, nc, n_boot=50, seed=42)
    assert out["macro_f1"]["point"] == pytest.approx(ref_macro)
    for cid in range(nc):
        assert out["per_class"][cid]["point"] == pytest.approx(ref_per_class[cid])


def test_bootstrap_perfect_predictions_ci_is_one() -> None:
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    out = er.bootstrap_f1_ci(y, y, num_classes=3, n_boot=100, seed=42)
    # Every resample is still perfect -> point and both bounds are 1.0.
    assert out["macro_f1"]["point"] == pytest.approx(1.0)
    assert out["macro_f1"]["lo"] == pytest.approx(1.0)
    assert out["macro_f1"]["hi"] == pytest.approx(1.0)


def test_bootstrap_alpha_widens_interval() -> None:
    y_true, y_pred, nc = _make_data()
    tight = er.bootstrap_f1_ci(y_true, y_pred, nc, n_boot=500, seed=42, alpha=0.5)
    wide = er.bootstrap_f1_ci(y_true, y_pred, nc, n_boot=500, seed=42, alpha=0.05)
    tight_w = tight["macro_f1"]["hi"] - tight["macro_f1"]["lo"]
    wide_w = wide["macro_f1"]["hi"] - wide["macro_f1"]["lo"]
    # A larger alpha (50%) is a narrower interval than a 5% alpha (95% CI).
    assert tight_w <= wide_w


# --------------------------------------------------------------------------- #
# per-class additive-bias calibration (max macro-F1, tuned on a holdout)
#
# This is the post-hoc decision-rule calibration: find a per-class bias vector
# b (length C) that, added to the logits before argmax, maximises macro-F1 on
# the validation split. It is the multiclass generalisation of "tune the
# decision threshold per class" (Lipton et al. 2014; Menon et al. ICLR 2021)
# and is the correct lever for a precision-sink class that is over-predicted.
# --------------------------------------------------------------------------- #
def _logits_from_preds(y_pred: np.ndarray, num_classes: int, conf: float = 4.0) -> np.ndarray:
    """Build deterministic logits whose argmax equals ``y_pred``.

    The chosen class gets logit ``conf``; all others get ``0``. This lets a test
    construct a known raw-prediction pattern and then check that a bias shifts
    the argmax in the intended direction.
    """
    n = y_pred.shape[0]
    logits = np.zeros((n, num_classes), dtype=np.float64)
    logits[np.arange(n), y_pred] = conf
    return logits


def test_apply_bias_shifts_argmax() -> None:
    # Two samples, raw argmax = class 0 for both (logit 4 vs 0,0).
    logits = _logits_from_preds(np.array([0, 0]), num_classes=3)
    # A big positive bias on class 1 overtakes the 4.0 margin -> both predict 1.
    bias = np.array([0.0, 5.0, 0.0])
    pred = er.apply_bias(logits, bias)
    np.testing.assert_array_equal(pred, np.array([1, 1]))


def test_apply_bias_zero_is_raw_argmax() -> None:
    logits = _logits_from_preds(np.array([2, 0, 1, 2]), num_classes=3)
    pred = er.apply_bias(logits, np.zeros(3))
    np.testing.assert_array_equal(pred, np.array([2, 0, 1, 2]))


def test_calibrate_perfect_predictions_keeps_macro_f1_at_one() -> None:
    # When raw argmax is already perfect, calibration cannot do worse than 1.0.
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    logits = _logits_from_preds(y, num_classes=3)
    bias, info = er.calibrate_bias_for_macro_f1(logits, y, num_classes=3)
    pred = er.apply_bias(logits, bias)
    f1, sup = er.per_class_f1_from_labels(y, pred, 3)
    assert er._macro_f1(f1, sup) == pytest.approx(1.0)
    # The tuned macro-F1 recorded in info matches the holdout it optimised on.
    assert info["macro_f1_calibrated"] == pytest.approx(1.0)


def test_calibration_fixes_over_predicted_sink_class() -> None:
    """A class that is over-predicted (eats others) should be penalised by the
    bias, raising macro-F1 above the raw-argmax baseline.

    Construction: class 2 is a 'sink'. Its own 100 samples are correct, but it
    also wins (by a small margin) on 60 samples that truly belong to classes 0
    and 1. Raw argmax therefore has perfect recall on 2 but poor precision, and
    poor recall on 0/1 -> low macro-F1. A negative bias on class 2 should hand
    those borderline samples back to 0/1 and raise macro-F1.
    """
    rng = np.random.default_rng(0)
    nc = 3
    parts_logits = []
    parts_labels = []

    # True 0 / True 1: 100 each, correctly and confidently classified.
    for c in (0, 1):
        lg = np.full((100, nc), -2.0)
        lg[:, c] = 3.0
        parts_logits.append(lg)
        parts_labels.append(np.full(100, c))

    # True 0 / True 1 borderline: class 2 wins by a SMALL margin (the sink).
    for c in (0, 1):
        lg = np.full((30, nc), -2.0)
        lg[:, c] = 1.0
        lg[:, 2] = 1.2  # 2 narrowly beats the true class
        parts_logits.append(lg)
        parts_labels.append(np.full(30, c))

    # True 2: 100 samples, confidently correct.
    lg = np.full((100, nc), -2.0)
    lg[:, 2] = 3.0
    parts_logits.append(lg)
    parts_labels.append(np.full(100, 2))

    logits = np.concatenate(parts_logits, axis=0)
    labels = np.concatenate(parts_labels, axis=0)

    raw_pred = er.apply_bias(logits, np.zeros(nc))
    raw_f1, raw_sup = er.per_class_f1_from_labels(labels, raw_pred, nc)
    raw_macro = er._macro_f1(raw_f1, raw_sup)

    bias, info = er.calibrate_bias_for_macro_f1(logits, labels, num_classes=nc, seed=0)
    cal_pred = er.apply_bias(logits, bias)
    cal_f1, cal_sup = er.per_class_f1_from_labels(labels, cal_pred, nc)
    cal_macro = er._macro_f1(cal_f1, cal_sup)

    # Calibration must not hurt, and on this constructed sink it must help.
    assert cal_macro >= raw_macro
    assert cal_macro > raw_macro + 0.05
    # The sink class (2) should be penalised relative to the recovered classes.
    assert bias[2] < bias[0]
    assert bias[2] < bias[1]
    # info reports both numbers it measured on the holdout.
    assert info["macro_f1_raw"] == pytest.approx(raw_macro)
    assert info["macro_f1_calibrated"] == pytest.approx(cal_macro)


def test_calibration_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(300, 4))
    labels = rng.integers(0, 4, size=300)
    b1, _ = er.calibrate_bias_for_macro_f1(logits, labels, num_classes=4, seed=42)
    b2, _ = er.calibrate_bias_for_macro_f1(logits, labels, num_classes=4, seed=42)
    np.testing.assert_array_equal(b1, b2)


def test_calibration_never_decreases_holdout_macro_f1() -> None:
    # On arbitrary noisy data, the tuned bias must be at least as good as b=0
    # on the SAME data it was tuned on (it always considers the zero vector).
    rng = np.random.default_rng(123)
    logits = rng.normal(size=(500, 5))
    labels = rng.integers(0, 5, size=500)
    bias, info = er.calibrate_bias_for_macro_f1(logits, labels, num_classes=5, seed=1)
    assert info["macro_f1_calibrated"] >= info["macro_f1_raw"] - 1e-9
