"""Unit tests for the dataset-agnostic monitor-score selector (v8.6).

`val_balanced` exists so checkpoint selection optimizes accuracy AND macro-F1
jointly — and like every other monitor it reads only the val_metrics dict, so
it adapts automatically to any class count / distribution.
"""
from __future__ import annotations

import math

import pytest


def _score(monitor, **metrics):
    from graphslm_ids.offline.training.train_hgt_flow_classifier import _compute_monitor_score
    return _compute_monitor_score(monitor, metrics)


def test_macro_f1_monitor_returns_macro_f1():
    assert _score("val_macro_f1", accuracy=0.9, macro_f1=0.4, loss=1.0) == pytest.approx(0.4)


def test_accuracy_monitor_returns_accuracy():
    assert _score("val_accuracy", accuracy=0.9, macro_f1=0.4, loss=1.0) == pytest.approx(0.9)


def test_balanced_monitor_is_mean_of_acc_and_macro_f1():
    assert _score("val_balanced", accuracy=0.9, macro_f1=0.4, loss=1.0) == pytest.approx(0.65)


def test_balanced_monitor_rewards_both_over_lopsided():
    """A model good at both should outscore one that maxes accuracy but ignores
    minority classes (high acc, low macro_f1)."""
    both = _score("val_balanced", accuracy=0.80, macro_f1=0.60, loss=1.0)   # 0.70
    lopsided = _score("val_balanced", accuracy=0.95, macro_f1=0.10, loss=1.0)  # 0.525
    assert both > lopsided


def test_loss_monitor_is_negative_loss():
    assert _score("val_loss", accuracy=0.0, macro_f1=0.0, loss=1.5) == pytest.approx(-1.5)


def test_loss_monitor_handles_none_loss():
    assert _score("val_loss", accuracy=0.0, macro_f1=0.0, loss=None) == -float("inf")


def test_balanced_monitor_propagates_nan():
    """Epoch-1 skipped val → NaN metrics → NaN score (caller's isnan guard skips it)."""
    s = _score("val_balanced", accuracy=float("nan"), macro_f1=float("nan"), loss=None)
    assert math.isnan(s)


def test_unknown_monitor_raises():
    with pytest.raises(ValueError):
        _score("val_top1", accuracy=0.9, macro_f1=0.4, loss=1.0)
