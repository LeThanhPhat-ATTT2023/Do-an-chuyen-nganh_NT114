"""EACS — Evidence-Anchored Candidate-Set self-relabeling (unit, CPU, pure)."""
import numpy as np
import pytest
import torch

from graphslm_ids.offline.training.noise_consensus import (
    EACSController,
    soft_target_focal_loss,
)

NUM_CLASSES = 4
BENIGN = 1  # class ids: 0=AttackA, 1=Benign, 2=AttackB, 3=Other


def _make_controller(**kw):
    # 6 flows: 0,1 suspects (AttackA label, no evidence); 2 anchor (AttackA + evidence);
    # 3 benign; 4 other-class; 5 suspect (AttackB label, no evidence).
    suspect = torch.tensor([True, True, False, False, False, True])
    anchor = torch.tensor([False, False, True, False, False, False])
    defaults = dict(
        suspect_mask=suspect,
        anchor_mask=anchor,
        benign_class_id=BENIGN,
        num_classes=NUM_CLASSES,
        warmup_epochs=2,
        ema_decay=0.0,  # no smoothing -> assertions are exact
        lambda_disambig=1.0,  # no consensus term unless a test passes one
    )
    defaults.update(kw)
    return EACSController(**defaults)


def _logits_for(p_y: float, p_benign: float, y: int) -> torch.Tensor:
    """Single-row logits whose softmax puts ~p_y on y and ~p_benign on BENIGN."""
    rest = (1.0 - p_y - p_benign) / (NUM_CLASSES - 2)
    probs = torch.full((1, NUM_CLASSES), rest)
    probs[0, y] = p_y
    probs[0, BENIGN] = p_benign
    return probs.clamp_min(1e-9).log()


def test_warmup_returns_pure_onehot():
    ctrl = _make_controller(warmup_epochs=3)
    logits = _logits_for(0.1, 0.8, y=0)
    tgt = ctrl.soft_targets(logits, torch.tensor([0]), torch.tensor([0]), epoch=3)
    assert torch.allclose(tgt, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_non_suspect_rows_stay_onehot_after_warmup():
    ctrl = _make_controller()
    # flow 2 = anchor, flow 3 = benign, flow 4 = other class
    logits = torch.cat([_logits_for(0.05, 0.9, y=0)] * 3, dim=0)
    gids = torch.tensor([2, 3, 4])
    labels = torch.tensor([0, 1, 3])
    tgt = ctrl.soft_targets(logits, gids, labels, epoch=5)
    expect = torch.zeros(3, NUM_CLASSES)
    expect[torch.arange(3), labels] = 1.0
    assert torch.allclose(tgt, expect)


def test_suspect_beta_is_two_way_disambiguation():
    ctrl = _make_controller()
    # p[y]=0.2, p[Benign]=0.6 -> beta = 0.2/0.8 = 0.25
    logits = _logits_for(0.2, 0.6, y=0)
    tgt = ctrl.soft_targets(logits, torch.tensor([0]), torch.tensor([0]), epoch=5)
    beta = 0.2 / (0.2 + 0.6)
    expect = torch.zeros(1, NUM_CLASSES)
    expect[0, 0] = beta
    expect[0, BENIGN] = 1.0 - beta
    assert torch.allclose(tgt, expect, atol=1e-5)
    # rows sum to 1
    assert torch.allclose(tgt.sum(dim=1), torch.ones(1))


def test_confident_in_label_keeps_label():
    ctrl = _make_controller()
    logits = _logits_for(0.9, 0.02, y=0)
    tgt = ctrl.soft_targets(logits, torch.tensor([0]), torch.tensor([0]), epoch=5)
    assert tgt[0, 0].item() > 0.95


def test_consensus_modulates_beta_geometrically():
    ctrl = _make_controller(lambda_disambig=0.5)
    logits = _logits_for(0.5, 0.5, y=0)  # beta_raw = 0.5
    cons = torch.tensor([0.125])
    tgt = ctrl.soft_targets(
        logits, torch.tensor([0]), torch.tensor([0]), epoch=5, consensus=cons
    )
    # beta = 0.5^0.5 * 0.125^0.5 = sqrt(0.0625) = 0.25
    assert abs(tgt[0, 0].item() - 0.25) < 1e-4


def test_ema_smooths_beta_across_calls():
    ctrl = _make_controller(ema_decay=0.5)
    g = torch.tensor([0])
    y = torch.tensor([0])
    ctrl.soft_targets(_logits_for(0.8, 0.1, y=0), g, y, epoch=5)  # beta1=0.8/0.9
    tgt2 = ctrl.soft_targets(_logits_for(0.1, 0.8, y=0), g, y, epoch=5)
    b1 = 0.8 / 0.9
    b2_raw = 0.1 / 0.9
    expect = 0.5 * b1 + 0.5 * b2_raw
    assert abs(tgt2[0, 0].item() - expect) < 1e-4


def test_epoch_stats_counts_suspects_and_relabels():
    ctrl = _make_controller()
    logits = torch.cat([_logits_for(0.1, 0.8, y=0), _logits_for(0.9, 0.02, y=2)])
    ctrl.soft_targets(logits, torch.tensor([0, 5]), torch.tensor([0, 2]), epoch=5)
    stats = ctrl.epoch_stats(reset=True)
    assert stats["suspects_seen"] == 2
    assert stats["relabeled"] == 1  # only the beta<0.5 one
    assert 0.0 < stats["mean_beta"] < 1.0
    # reset worked
    assert ctrl.epoch_stats(reset=False)["suspects_seen"] == 0


def test_mask_construction_via_builder_logic():
    # mirrors build_eacs_controller's mask derivation (pure tensor logic)
    flow_labels = torch.tensor([0, 0, 0, 1, 3, 2])
    suspect_ids = torch.tensor([0, 2])
    class_to_family = torch.tensor([0, -1, 1, -1])  # AttackA->fam0, AttackB->fam1
    evidence = torch.zeros(6, 2)
    evidence[2, 0] = 0.9  # flow 2 has matching family evidence
    in_sus = torch.isin(flow_labels, suspect_ids)
    fam = class_to_family[flow_labels]
    has_ev = torch.zeros(6, dtype=torch.bool)
    ok = fam >= 0
    has_ev[ok] = evidence[torch.arange(6)[ok], fam[ok]] > 0
    assert (in_sus & ~has_ev).tolist() == [True, True, False, False, False, True]
    assert (in_sus & has_ev).tolist() == [False, False, True, False, False, False]


# ---------- soft_target_focal_loss ----------

def test_soft_loss_equals_hard_focal_when_onehot():
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        _compute_train_loss,
    )
    torch.manual_seed(0)
    logits = torch.randn(16, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (16,))
    weight = torch.rand(NUM_CLASSES) + 0.5
    onehot = torch.nn.functional.one_hot(labels, NUM_CLASSES).float()
    for ls in (0.0, 0.05):
        hard = _compute_train_loss(
            logits, labels, weight,
            loss_type="focal", label_smoothing=ls, focal_gamma=1.5,
        )
        soft = soft_target_focal_loss(
            logits, onehot, weight,
            loss_type="focal", label_smoothing=ls, focal_gamma=1.5,
        )
        assert torch.allclose(hard, soft, atol=1e-6), (ls, hard, soft)


def test_soft_loss_moves_toward_benign_target():
    logits = _logits_for(0.1, 0.8, y=0).repeat(4, 1)
    onehot = torch.zeros(4, NUM_CLASSES)
    onehot[:, 0] = 1.0
    soft = onehot.clone()
    soft[:, 0] = 0.2
    soft[:, BENIGN] = 0.8
    l_hard = soft_target_focal_loss(logits, onehot, None, loss_type="focal", focal_gamma=1.5)
    l_soft = soft_target_focal_loss(logits, soft, None, loss_type="focal", focal_gamma=1.5)
    # prediction already favors Benign -> soft target must give LOWER loss
    assert l_soft.item() < l_hard.item()
