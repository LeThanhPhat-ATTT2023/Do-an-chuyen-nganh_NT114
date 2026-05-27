"""Unit tests for the per-class F1 log formatter.

During a training plateau, macro-F1 is typically capped by a few tail classes.
`_format_per_class_f1` surfaces them as a compact one-line, worst-first summary
so the operator can see *which* classes are failing without digging into the
history JSON.
"""
from __future__ import annotations


def _fmt(per_class, **kw):
    from graphslm_ids.offline.training.train_hgt_flow_classifier import _format_per_class_f1
    return _format_per_class_f1(per_class, **kw)


def test_orders_worst_f1_first():
    per_class = {
        "A": {"f1": 0.90, "support": 100, "precision": 0.9, "recall": 0.9},
        "B": {"f1": 0.10, "support": 50, "precision": 0.1, "recall": 0.1},
        "C": {"f1": 0.50, "support": 30, "precision": 0.5, "recall": 0.5},
    }
    out = _fmt(per_class)
    assert out.index("B=") < out.index("C=") < out.index("A=")


def test_includes_f1_and_support():
    per_class = {"B": {"f1": 0.10, "support": 50, "precision": 0.1, "recall": 0.1}}
    out = _fmt(per_class)
    assert "B=0.100" in out
    assert "n=50" in out


def test_excludes_zero_support_classes():
    per_class = {
        "A": {"f1": 0.0, "support": 0, "precision": 0.0, "recall": 0.0},
        "B": {"f1": 0.5, "support": 10, "precision": 0.5, "recall": 0.5},
    }
    out = _fmt(per_class)
    assert "A=" not in out
    assert "B=" in out


def test_top_k_limits_to_worst_k():
    per_class = {
        "A": {"f1": 0.9, "support": 1, "precision": 0.0, "recall": 0.0},
        "B": {"f1": 0.1, "support": 1, "precision": 0.0, "recall": 0.0},
        "C": {"f1": 0.5, "support": 1, "precision": 0.0, "recall": 0.0},
    }
    out = _fmt(per_class, top_k=2)
    assert "B=" in out and "C=" in out
    assert "A=" not in out


def test_empty_per_class_returns_empty_string():
    assert _fmt({}) == ""
