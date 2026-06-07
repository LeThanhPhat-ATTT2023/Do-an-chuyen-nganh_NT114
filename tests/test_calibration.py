import numpy as np
from graphslm_ids.offline.training.calibration import macro_f1, apply_bias


def test_macro_f1_perfect_is_one():
    labels = np.array([0, 1, 2, 0, 1, 2])
    preds = labels.copy()
    assert macro_f1(preds, labels, num_classes=3) == 1.0


def test_macro_f1_matches_manual():
    labels = np.array([0, 1, 2])
    preds = np.array([0, 0, 2])
    f1 = macro_f1(preds, labels, num_classes=3)
    assert abs(f1 - (2 / 3 + 0.0 + 1.0) / 3) < 1e-9


def test_apply_bias_shifts_argmax():
    logits = np.array([[1.0, 0.9], [0.5, 0.6]])
    assert apply_bias(logits, np.zeros(2)).tolist() == [0, 1]
    assert apply_bias(logits, np.array([0.0, 1.0])).tolist() == [1, 1]
