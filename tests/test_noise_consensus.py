"""Tests for the noise-robust training core (EG-HGT noise-consensus, 2026-06).

These are the spec for ``graphslm_ids.offline.training.noise_consensus``: four
pure, deterministic helpers that let the model SELF-DETECT instance-dependent
label noise during a single training run, with no hand-crafted signatures and no
test-set peeking. See docs/superpowers/specs/2026-06-09-neighbor-consensus-
noise-robust-hgt-design.md.

The four units (all CPU, fast, no model/graph load):
  1. neighbor_consensus  — soft graph-neighbor label agreement (Signal 2).
  2. evidence_support    — MITRE-evidence grounding of a flow's attack label
                           (Signal 1, the Tầng-3 contribution).
  3. combine_clean_confidence — geometric blend of the two signals.
  4. curriculum_weight   — warmup + clamp -> per-sample loss weight.
  5. EMAConsensusBuffer  — persistent per-flow EMA across epochs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from graphslm_ids.offline.training.noise_consensus import (
    EMAConsensusBuffer,
    combine_clean_confidence,
    curriculum_weight,
    evidence_support,
    neighbor_consensus,
)


# --------------------------------------------------------------------------- #
# Signal 2: neighbor_consensus
# --------------------------------------------------------------------------- #
def test_consensus_high_when_neighbors_predict_my_label() -> None:
    # 3 flows. Seed flow 0 has label 1. Its two neighbors (1, 2) both predict
    # class 1 with high prob -> consensus for flow 0 should be high (~0.9).
    pred = torch.tensor(
        [
            [0.05, 0.90, 0.05],  # flow 0 (seed)
            [0.05, 0.90, 0.05],  # neighbor
            [0.10, 0.85, 0.05],  # neighbor
        ]
    )
    # undirected neighbor edges 0-1 and 0-2 (as a 2xE index over local node ids)
    edge_index = torch.tensor([[0, 0], [1, 2]])
    seed_idx = torch.tensor([0])
    seed_labels = torch.tensor([1])
    c = neighbor_consensus(pred, edge_index, seed_idx, seed_labels)
    assert c.shape == (1,)
    assert c.item() == pytest.approx((0.90 + 0.85) / 2, abs=1e-6)


def test_consensus_low_when_neighbors_predict_other_label() -> None:
    # Seed flow 0 labeled 2 (an "attack"), but its neighbors are background the
    # model predicts as class 0 -> they do NOT support label 2 -> low consensus.
    pred = torch.tensor(
        [
            [0.10, 0.10, 0.80],  # flow 0 (seed) — model itself unsure-ish
            [0.95, 0.03, 0.02],  # neighbor -> class 0
            [0.92, 0.05, 0.03],  # neighbor -> class 0
        ]
    )
    edge_index = torch.tensor([[0, 0], [1, 2]])
    c = neighbor_consensus(pred, edge_index, torch.tensor([0]), torch.tensor([2]))
    assert c.item() == pytest.approx((0.02 + 0.03) / 2, abs=1e-6)
    assert c.item() < 0.1


def test_consensus_isolated_flow_defaults_to_one() -> None:
    # A seed with no neighbors cannot be judged -> default trust = 1.0 (we never
    # penalise a flow just for being unconnected).
    pred = torch.tensor([[0.2, 0.8]])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    c = neighbor_consensus(pred, edge_index, torch.tensor([0]), torch.tensor([1]))
    assert c.item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Signal 1: evidence_support (the Tầng-3 / MITRE contribution)
# --------------------------------------------------------------------------- #
def test_evidence_support_attack_with_matching_evidence_is_one() -> None:
    # Flow labeled as attack-class 3, and it HAS a matching evidence-edge weight
    # (>0) for that class's family -> fully supported (1.0).
    # evidence_by_flow[i, family] = summed matching evidence weight.
    labels = torch.tensor([3])
    # one family column; flow 0 has weight 2.5 in the family that maps to class 3
    evidence = torch.tensor([[2.5]])
    # class->family map: class 3 -> family col 0; non-attack classes -> -1 (benign)
    class_to_family = torch.tensor([-1, -1, -1, 0])
    s = evidence_support(labels, evidence, class_to_family)
    assert s.item() == pytest.approx(1.0)


def test_evidence_support_attack_without_evidence_is_low() -> None:
    # Flow labeled attack-class 3 but ZERO matching evidence -> prime noise
    # candidate -> support 0.0.
    labels = torch.tensor([3])
    evidence = torch.tensor([[0.0]])
    class_to_family = torch.tensor([-1, -1, -1, 0])
    s = evidence_support(labels, evidence, class_to_family)
    assert s.item() == pytest.approx(0.0)


def test_evidence_support_non_attack_class_is_one() -> None:
    # Benign / volumetric classes are not subject to web label pollution; they
    # are trusted by construction (class_to_family == -1 -> support 1.0).
    labels = torch.tensor([0, 1])
    evidence = torch.tensor([[0.0], [0.0]])
    class_to_family = torch.tensor([-1, -1, -1, 0])
    s = evidence_support(labels, evidence, class_to_family)
    assert torch.allclose(s, torch.ones(2))


# --------------------------------------------------------------------------- #
# combine_clean_confidence
# --------------------------------------------------------------------------- #
def test_combine_is_geometric_blend() -> None:
    ev = torch.tensor([1.0, 0.0, 0.81])
    co = torch.tensor([0.5, 0.5, 0.49])
    lam = 0.5
    out = combine_clean_confidence(ev, co, lam)
    expected = (ev ** lam) * (co ** (1 - lam))
    assert torch.allclose(out, expected, atol=1e-6)
    # a zero in EITHER signal drives the blend toward zero (flow 1: ev=0)
    assert out[1].item() == pytest.approx(0.0)


def test_combine_lambda_one_is_evidence_only() -> None:
    ev = torch.tensor([0.3, 0.9])
    co = torch.tensor([0.99, 0.01])
    out = combine_clean_confidence(ev, co, lam=1.0)
    assert torch.allclose(out, ev, atol=1e-6)


# --------------------------------------------------------------------------- #
# curriculum_weight
# --------------------------------------------------------------------------- #
def test_curriculum_warmup_returns_all_ones() -> None:
    clean = torch.tensor([0.1, 0.5, 0.9])
    w = curriculum_weight(clean, epoch=2, warmup_epochs=5, w_min=0.2)
    assert torch.allclose(w, torch.ones(3))


def test_curriculum_after_warmup_clamps_to_wmin() -> None:
    clean = torch.tensor([0.0, 0.5, 1.0])
    w = curriculum_weight(clean, epoch=10, warmup_epochs=5, w_min=0.2)
    # 0.0 -> clamped up to w_min 0.2; 0.5 stays; 1.0 stays.
    assert w[0].item() == pytest.approx(0.2)
    assert w[1].item() == pytest.approx(0.5)
    assert w[2].item() == pytest.approx(1.0)


def test_curriculum_weight_in_unit_range() -> None:
    clean = torch.rand(100)
    w = curriculum_weight(clean, epoch=10, warmup_epochs=5, w_min=0.3)
    assert (w >= 0.3 - 1e-6).all() and (w <= 1.0 + 1e-6).all()


# --------------------------------------------------------------------------- #
# EMAConsensusBuffer
# --------------------------------------------------------------------------- #
def test_ema_buffer_first_update_is_identity() -> None:
    buf = EMAConsensusBuffer(num_flows=4, decay=0.9, init=1.0)
    vals = torch.tensor([0.2, 0.4])
    idx = torch.tensor([1, 3])
    out = buf.update(idx, vals)
    # first time a flow is seen, EMA returns the raw value (no stale init bias)
    assert out[0].item() == pytest.approx(0.2)
    assert out[1].item() == pytest.approx(0.4)
    assert buf.get(torch.tensor([1])).item() == pytest.approx(0.2)


def test_ema_buffer_second_update_blends() -> None:
    buf = EMAConsensusBuffer(num_flows=2, decay=0.8, init=1.0)
    buf.update(torch.tensor([0]), torch.tensor([1.0]))
    out = buf.update(torch.tensor([0]), torch.tensor([0.0]))
    # EMA: 0.8*1.0 + 0.2*0.0 = 0.8
    assert out.item() == pytest.approx(0.8, abs=1e-6)


def test_ema_buffer_untouched_flows_keep_init() -> None:
    buf = EMAConsensusBuffer(num_flows=3, decay=0.9, init=1.0)
    buf.update(torch.tensor([0]), torch.tensor([0.1]))
    assert buf.get(torch.tensor([2])).item() == pytest.approx(1.0)
