"""Tests for HGAA filter network components.

Reference paper: Zhao et al., HGAA (Symmetry 2025, 17, 1623), §3.2.

Multiclass adaptation: instead of paper's "is this an anomaly" filter,
we validate that augmented samples remain in their intended class via
per-class centroid similarity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


# ---------- Task 4: ClassCentroidBank ----------


def test_centroid_bank_zero_init():
    """A fresh bank has all centroids at zero (no observations yet)."""
    from graphslm_ids.offline.training.hgaa_filter_network import ClassCentroidBank

    bank = ClassCentroidBank(num_classes=5, embedding_dim=8)
    assert bank.centroids.shape == (5, 8)
    assert torch.allclose(bank.centroids, torch.zeros(5, 8))


def test_centroid_bank_first_update_initializes_centroids():
    """First update must initialize centroids to the per-class mean (not EMA)."""
    from graphslm_ids.offline.training.hgaa_filter_network import ClassCentroidBank

    bank = ClassCentroidBank(num_classes=3, embedding_dim=4, ema_decay=0.9)
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1])
    bank.update(embeddings, labels)

    # Class 0 mean: (1+3)/2 = 2 in dim 0
    assert torch.allclose(bank.centroids[0], torch.tensor([2.0, 0.0, 0.0, 0.0]))
    # Class 1 mean: 2 in dim 1
    assert torch.allclose(bank.centroids[1], torch.tensor([0.0, 2.0, 0.0, 0.0]))
    # Class 2 untouched
    assert torch.allclose(bank.centroids[2], torch.zeros(4))


def test_centroid_bank_subsequent_update_uses_ema():
    """After init, updates should be EMA-blended with previous centroid."""
    from graphslm_ids.offline.training.hgaa_filter_network import ClassCentroidBank

    bank = ClassCentroidBank(num_classes=2, embedding_dim=2, ema_decay=0.8)
    # First update: centroid for class 0 = [1, 0]
    bank.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    # Second update with [3, 0]: EMA = 0.8 * 1 + 0.2 * 3 = 1.4
    bank.update(torch.tensor([[3.0, 0.0]]), torch.tensor([0]))
    assert torch.allclose(bank.centroids[0], torch.tensor([1.4, 0.0]), atol=1e-6)


def test_centroid_bank_cosine_similarity_returns_unit_for_aligned_vectors():
    """Cosine sim of a vector with its own centroid should be 1.0."""
    from graphslm_ids.offline.training.hgaa_filter_network import ClassCentroidBank

    bank = ClassCentroidBank(num_classes=2, embedding_dim=4)
    bank.update(torch.tensor([[2.0, 0.0, 0.0, 0.0]]), torch.tensor([0]))
    sim = bank.cosine_similarity(torch.tensor([5.0, 0.0, 0.0, 0.0]), class_id=0)
    assert abs(sim - 1.0) < 1e-5


def test_centroid_bank_cosine_similarity_zero_for_orthogonal():
    """Orthogonal vector should give cosine sim 0."""
    from graphslm_ids.offline.training.hgaa_filter_network import ClassCentroidBank

    bank = ClassCentroidBank(num_classes=2, embedding_dim=4)
    bank.update(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([0]))
    sim = bank.cosine_similarity(torch.tensor([0.0, 1.0, 0.0, 0.0]), class_id=0)
    assert abs(sim) < 1e-5


def test_centroid_bank_state_dict_roundtrip():
    """save/load via state_dict preserves centroids exactly."""
    from graphslm_ids.offline.training.hgaa_filter_network import ClassCentroidBank

    bank1 = ClassCentroidBank(num_classes=3, embedding_dim=4, ema_decay=0.5)
    bank1.update(torch.randn(10, 4), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
    state = bank1.state_dict()

    bank2 = ClassCentroidBank(num_classes=3, embedding_dim=4, ema_decay=0.5)
    bank2.load_state_dict(state)
    assert torch.allclose(bank1.centroids, bank2.centroids)


# ---------- Task 5: Resilient checkpoint loader ----------


def _make_dummy_hgt_checkpoint(tmp_path: Path, num_classes: int, hidden_dim: int = 32) -> Path:
    """Create a fake HGT-like checkpoint file with classifier head of given size."""
    state = {
        "model_state_dict": {
            "input_projection.flow.0.weight": torch.randn(hidden_dim, 8),
            "input_projection.flow.0.bias": torch.randn(hidden_dim),
            "layers.0.query.flow.weight": torch.randn(hidden_dim, hidden_dim),
            "classifier.1.weight": torch.randn(hidden_dim, hidden_dim),
            "classifier.4.weight": torch.randn(num_classes, hidden_dim),
            "classifier.4.bias": torch.randn(num_classes),
        }
    }
    path = tmp_path / "fake_hgt.pt"
    torch.save(state, path)
    return path


class _DummyHGT(torch.nn.Module):
    """Minimal model with the same param NAMES as the real HGT classifier."""

    def __init__(self, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.input_projection = torch.nn.ModuleDict(
            {"flow": torch.nn.Sequential(torch.nn.Linear(8, hidden_dim))}
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {"query": torch.nn.ModuleDict({"flow": torch.nn.Linear(hidden_dim, hidden_dim, bias=False)})}
                )
            ]
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, num_classes),
        )


def test_load_resilient_succeeds_when_num_classes_matches(tmp_path):
    """Same num_classes: classifier head loads cleanly with no missing keys."""
    from graphslm_ids.offline.training.hgaa_filter_network import (
        load_hgt_state_dict_resilient,
    )

    ckpt = _make_dummy_hgt_checkpoint(tmp_path, num_classes=12)
    model = _DummyHGT(num_classes=12)
    missing, unexpected = load_hgt_state_dict_resilient(model, ckpt, num_classes_new=12)

    # Classifier weights should be loaded (no missing classifier keys)
    assert not any("classifier.4" in k for k in missing)


def test_load_resilient_drops_classifier_head_on_num_classes_mismatch(tmp_path):
    """num_classes changed: classifier.4 should be missing (re-init), not crash."""
    from graphslm_ids.offline.training.hgaa_filter_network import (
        load_hgt_state_dict_resilient,
    )

    # Checkpoint has 12 classes
    ckpt = _make_dummy_hgt_checkpoint(tmp_path, num_classes=12)
    # New model has 13 classes (NT114 case!)
    model = _DummyHGT(num_classes=13)
    missing, unexpected = load_hgt_state_dict_resilient(model, ckpt, num_classes_new=13)

    # The mismatched head should be in `missing` (we dropped it from ckpt
    # state, so model's new head is left at random init).
    assert any("classifier.4" in k for k in missing), (
        f"expected classifier.4 in missing, got missing={missing}"
    )
    # But non-classifier params must NOT be in missing
    non_clf_missing = [k for k in missing if "classifier" not in k]
    assert not non_clf_missing, f"non-classifier params missing: {non_clf_missing}"


def test_load_resilient_handles_raw_state_dict_not_wrapped(tmp_path):
    """Checkpoint without 'model_state_dict' wrapper should still load."""
    from graphslm_ids.offline.training.hgaa_filter_network import (
        load_hgt_state_dict_resilient,
    )

    raw_state = {
        "classifier.4.weight": torch.randn(12, 32),
        "classifier.4.bias": torch.randn(12),
    }
    path = tmp_path / "raw.pt"
    torch.save(raw_state, path)

    model = _DummyHGT(num_classes=12)
    missing, unexpected = load_hgt_state_dict_resilient(model, path, num_classes_new=12)
    # Doesn't crash; some params will be in missing (we didn't save them all)
    assert isinstance(missing, list)
    assert isinstance(unexpected, list)
