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


# --------------------------------------------------------------------------- #
# DYNAMIC mechanism (ICGNN-style): the model self-detects noise each epoch.
#   - evidence_prediction_contradiction: 1 - cos(q_i, e_i)  (dynamic via q_i)
#   - em_clean_confidence: 2-component fit per epoch -> beta_i (clean posterior)
#   - soft_relabel_target: beta*onehot(label) + (1-beta)*q  (ICGNN Eq.6)
# --------------------------------------------------------------------------- #
from graphslm_ids.offline.training.noise_consensus import (  # noqa: E402
    em_clean_confidence,
    evidence_prediction_contradiction,
    soft_relabel_target,
)


def test_epc_zero_when_prediction_matches_evidence() -> None:
    # model predicts family 0, evidence is purely family 0 -> agree=1 -> EPC=0.
    q = torch.tensor([[1.0, 0.0, 0.0]])
    e = torch.tensor([[2.0, 0.0, 0.0]])   # un-normalised; cosine ignores scale
    epc = evidence_prediction_contradiction(q, e)
    assert epc.shape == (1,)
    assert epc.item() == pytest.approx(0.0, abs=1e-6)


def test_epc_high_when_prediction_contradicts_evidence() -> None:
    # model predicts family 0, evidence says family 1 -> orthogonal -> EPC ~ 1.
    q = torch.tensor([[1.0, 0.0]])
    e = torch.tensor([[0.0, 1.0]])
    epc = evidence_prediction_contradiction(q, e)
    assert epc.item() == pytest.approx(1.0, abs=1e-6)


def test_epc_no_evidence_returns_neutral() -> None:
    # a flow with zero evidence (background) cannot be judged by EPC -> neutral 0.5
    # (we don't want to punish or reward purely from missing evidence; the EM step
    # and label decide). Implementation choice documented in the spec.
    q = torch.tensor([[0.7, 0.3]])
    e = torch.tensor([[0.0, 0.0]])
    epc = evidence_prediction_contradiction(q, e)
    assert epc.item() == pytest.approx(0.5, abs=1e-6)


def test_em_clean_confidence_separates_two_clusters() -> None:
    # Build an EPC distribution: a clean cluster (low EPC ~0.1) and a noisy cluster
    # (high EPC ~0.9). beta should be ~1 for low-EPC and ~0 for high-EPC samples.
    epc = torch.tensor([0.05, 0.1, 0.12, 0.08,   # clean
                        0.88, 0.92, 0.9, 0.95])   # noisy
    beta = em_clean_confidence(epc, n_iter=50)
    assert beta.shape == epc.shape
    assert (beta[:4] > 0.5).all()    # clean cluster -> high clean-confidence
    assert (beta[4:] < 0.5).all()    # noisy cluster -> low clean-confidence


def test_em_clean_confidence_preserves_input_device() -> None:
    # Regression guard: all EM init tensors must live on the input's device, so the
    # routine works on cuda (the bug that crashed the first smoke-train at epoch 5).
    epc = torch.rand(20)
    beta = em_clean_confidence(epc, n_iter=10)
    assert beta.device == epc.device


def test_em_clean_confidence_in_unit_range() -> None:
    torch.manual_seed(0)
    epc = torch.rand(200)
    beta = em_clean_confidence(epc, n_iter=30)
    assert (beta >= 0.0).all() and (beta <= 1.0).all()


def test_soft_relabel_blends_label_and_prediction() -> None:
    # beta=1 -> pure given label; beta=0 -> pure model prediction; 0.5 -> average.
    labels = torch.tensor([0, 1])
    q = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
    num_classes = 2
    # beta = 1: target == one-hot(label)
    t1 = soft_relabel_target(labels, q, torch.tensor([1.0, 1.0]), num_classes)
    assert torch.allclose(t1, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), atol=1e-6)
    # beta = 0: target == model prediction q
    t0 = soft_relabel_target(labels, q, torch.tensor([0.0, 0.0]), num_classes)
    assert torch.allclose(t0, q, atol=1e-6)
    # beta = 0.5 on sample 0: average of one-hot(0) and q[0]
    th = soft_relabel_target(labels[:1], q[:1], torch.tensor([0.5]), num_classes)
    expected = 0.5 * torch.tensor([[1.0, 0.0]]) + 0.5 * q[:1]
    assert torch.allclose(th, expected, atol=1e-6)


def test_soft_relabel_rows_sum_to_one() -> None:
    labels = torch.tensor([0, 2, 1])
    q = torch.softmax(torch.randn(3, 3), dim=1)
    beta = torch.tensor([0.3, 0.7, 0.0])
    t = soft_relabel_target(labels, q, beta, num_classes=3)
    assert torch.allclose(t.sum(dim=1), torch.ones(3), atol=1e-6)


def test_controller_soft_targets_relabels_noisy_attack_flow() -> None:
    """The controller ties EPC -> EM -> soft-relabel for a batch.

    Construct 6 seed flows: 3 genuine attack flows of class 1 (family col 0) whose
    family prediction AGREES with evidence, and 3 'noisy' flows also labeled class 1
    but whose family prediction is benign-ish and evidence is family 0 -> contradiction.
    The soft target of the noisy flows should move OFF the given label toward the model
    prediction, while the clean flows stay near their label.
    """
    from graphslm_ids.offline.training.noise_consensus import NoiseRobustController

    num_classes, num_families = 3, 2
    # class 1 -> family col 0 ; classes 0,2 -> non-attack (-1)
    class_to_family = torch.tensor([-1, 0, -1])
    # per-flow evidence table (global) for 6 flows: all claim family 0 evidence present
    evidence_by_flow = torch.zeros((6, num_families))
    evidence_by_flow[:, 0] = 1.0
    ctrl = NoiseRobustController(
        evidence_by_flow=evidence_by_flow,
        class_to_family=class_to_family,
        num_classes=num_classes,
        num_families=num_families,
        warmup_epochs=0,
        ema_decay=0.0,   # no smoothing -> pure per-epoch for a deterministic test
    )
    seed_global_ids = torch.arange(6)
    seed_labels = torch.tensor([1, 1, 1, 1, 1, 1])
    # family logits: flows 0-2 predict family 0 strongly (agree); 3-5 predict family 1
    family_logits = torch.tensor(
        [[5.0, -5.0]] * 3 + [[-5.0, 5.0]] * 3
    )
    # class logits: clean flows predict class 1; noisy flows predict class 0
    class_logits = torch.tensor(
        [[-5.0, 5.0, -5.0]] * 3 + [[5.0, -5.0, -5.0]] * 3
    )
    targets = ctrl.soft_targets(
        class_logits, family_logits, seed_global_ids, seed_labels, epoch=5
    )
    assert targets.shape == (6, 3)
    assert torch.allclose(targets.sum(dim=1), torch.ones(6), atol=1e-5)
    # clean flows (0-2): target mass on the given label (class 1) stays high
    assert (targets[:3, 1] > 0.8).all()
    # noisy flows (3-5): target mass moved OFF class 1 toward model's class-0 prediction
    assert (targets[3:, 1] < 0.6).all()
    assert (targets[3:, 0] > targets[3:, 1]).all()


def test_build_class_to_family_maps_via_technique() -> None:
    from graphslm_ids.offline.training.noise_consensus import build_class_to_family

    # class -> technique (with weights) ; technique -> family ; family name -> col
    class_to_tech = {
        "CommandInjection": [("T1059", 1.0)],
        "XSS": [("T1190", 0.7), ("T1189", 0.3)],
        "Benign": [],
    }
    tech_to_family = {"T1059": "command_exec", "T1190": "injection", "T1189": "injection"}
    family_to_col = {"injection": 0, "command_exec": 1, "file_upload": 2,
                     "recon": 3, "c2_beacon": 4}
    label_mapping = {"Benign": 0, "CommandInjection": 1, "XSS": 2}
    num_classes = 3
    c2f = build_class_to_family(
        class_to_tech, tech_to_family, family_to_col, label_mapping, num_classes
    )
    assert c2f.shape == (3,)
    assert c2f[0].item() == -1            # Benign -> non-attack
    assert c2f[1].item() == 1            # CommandInjection -> command_exec col 1
    assert c2f[2].item() == 0            # XSS -> injection (dominant T1190) col 0


# --------------------------------------------------------------------------- #
# trainer integration: _compute_train_loss accepts a per-sample weight
# (backward-compatible — sample_weight=None reproduces the unweighted loss).
# --------------------------------------------------------------------------- #
from graphslm_ids.offline.training.train_hgt_flow_classifier import (  # noqa: E402
    _compute_train_loss,
)


@pytest.mark.parametrize("loss_type", ["ce", "focal"])
def test_sample_weight_none_matches_unweighted(loss_type: str) -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    base = _compute_train_loss(logits, labels, None, loss_type=loss_type)
    # all-ones weight must equal the unweighted (mean) loss
    ones = _compute_train_loss(
        logits, labels, None, loss_type=loss_type,
        sample_weight=torch.ones(8),
    )
    assert ones.item() == pytest.approx(base.item(), abs=1e-6)


def test_static_evidence_weight_downweights_unsupported_attack_flows() -> None:
    """End-to-end of Signal 1 as a static per-flow weight vector.

    3 flows: flow0 = attack class 3 WITH matching evidence (keep weight 1),
    flow1 = attack class 3 WITHOUT evidence (noise -> down-weight to w_min),
    flow2 = benign class 0 (always 1).
    """
    from graphslm_ids.offline.training.noise_consensus import static_evidence_weight

    contains = torch.tensor([[0, 1, 2], [0, 1, 2]])      # flow i contains packet i
    evidence_per_family = {0: (torch.tensor([0]), torch.tensor([2.0]))}  # pkt0 -> fam0
    labels = torch.tensor([3, 3, 0])
    class_to_family = torch.tensor([-1, -1, -1, 0])      # class 3 -> family 0; others none
    w = static_evidence_weight(
        contains, evidence_per_family, labels, class_to_family,
        num_families=1, w_min=0.2,
    )
    assert w.shape == (3,)
    assert w[0].item() == pytest.approx(1.0)   # supported attack
    assert w[1].item() == pytest.approx(0.2)   # unsupported attack -> down-weighted
    assert w[2].item() == pytest.approx(1.0)   # benign


def test_build_evidence_by_flow_aggregates_packets_to_flow() -> None:
    from graphslm_ids.offline.training.noise_consensus import build_evidence_by_flow

    # 3 flows, 4 packets. contains: flow0->{pkt0,pkt1}, flow1->{pkt2}, flow2->{pkt3}
    contains = torch.tensor([[0, 0, 1, 2],
                             [0, 1, 2, 3]])
    num_flows = 3
    num_families = 2  # family 0 = command_exec, family 1 = injection
    # evidence edges per family: (packet_id, weight). family 0 has an edge on pkt1
    # (so flow0 gets command_exec); family 1 has an edge on pkt3 (flow2 gets injection).
    evidence_per_family = {
        0: (torch.tensor([1]), torch.tensor([2.5])),   # pkt1 -> command_exec w=2.5
        1: (torch.tensor([3]), torch.tensor([0.8])),   # pkt3 -> injection w=0.8
    }
    ev = build_evidence_by_flow(contains, evidence_per_family, num_flows, num_families)
    assert ev.shape == (3, 2)
    # flow0 has a command_exec packet -> col0 > 0; no injection -> col1 == 0
    assert ev[0, 0].item() == pytest.approx(2.5)
    assert ev[0, 1].item() == pytest.approx(0.0)
    # flow1 has neither
    assert ev[1].sum().item() == pytest.approx(0.0)
    # flow2 has an injection packet
    assert ev[2, 1].item() == pytest.approx(0.8)
    assert ev[2, 0].item() == pytest.approx(0.0)


@pytest.mark.parametrize("loss_type", ["ce", "focal"])
def test_sample_weight_zero_drops_sample(loss_type: str) -> None:
    torch.manual_seed(1)
    logits = torch.randn(6, 4)
    labels = torch.randint(0, 4, (6,))
    # Weighting the last sample to 0 must equal the mean over the first 5
    # (weighted loss normalises by the SUM of weights, not the count).
    w = torch.ones(6)
    w[5] = 0.0
    weighted = _compute_train_loss(
        logits, labels, None, loss_type=loss_type, sample_weight=w,
    )
    sub = _compute_train_loss(
        logits[:5], labels[:5], None, loss_type=loss_type,
    )
    assert weighted.item() == pytest.approx(sub.item(), abs=1e-5)
