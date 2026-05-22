"""Unit tests for SAM (Sharpness-Aware Minimization) optimizer wrapper."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _make_model_and_sam(rho=0.05, lr=0.1, base=torch.optim.SGD):
    model = torch.nn.Linear(4, 2, bias=False)
    from graphslm_ids.offline.training.sam_optimizer import SAM
    opt = SAM(model.parameters(), base_optimizer=base, rho=rho, lr=lr)
    return model, opt


def _one_cycle(model, opt, x, y):
    """Helper: run one full SAM cycle (first + second step). Returns losses."""
    loss1 = F.cross_entropy(model(x), y)
    loss1.backward()
    opt.first_step(zero_grad=True)
    loss2 = F.cross_entropy(model(x), y)
    loss2.backward()
    opt.second_step(zero_grad=True)
    return loss1.item(), loss2.item()


def test_first_step_perturbs_weights():
    """first_step() must change model weights away from original."""
    model, opt = _make_model_and_sam()
    original = model.weight.data.clone()
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 0, 1])
    F.cross_entropy(model(x), y).backward()
    opt.first_step(zero_grad=True)
    assert not torch.allclose(model.weight.data, original), "weights unchanged after first_step"


def test_second_step_restores_and_updates():
    """After second_step, weights must differ from both original and perturbed."""
    torch.manual_seed(0)
    model, opt = _make_model_and_sam()
    x, y = torch.randn(8, 4), torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])

    original = model.weight.data.clone()
    F.cross_entropy(model(x), y).backward()
    opt.first_step(zero_grad=True)
    perturbed = model.weight.data.clone()

    F.cross_entropy(model(x), y).backward()
    opt.second_step(zero_grad=True)
    final = model.weight.data.clone()

    assert not torch.allclose(final, perturbed), "weights not updated from perturbed"
    assert not torch.allclose(final, original),  "weights same as pre-perturbation"


def test_restore_weights_reverts_without_step():
    """restore_weights() reverts perturbation, does NOT call base_optimizer.step()."""
    torch.manual_seed(1)
    model, opt = _make_model_and_sam()
    original = model.weight.data.clone()
    x, y = torch.randn(4, 4), torch.tensor([0, 1, 0, 1])

    F.cross_entropy(model(x), y).backward()
    opt.first_step(zero_grad=True)
    assert not torch.allclose(model.weight.data, original)

    opt.restore_weights()
    torch.testing.assert_close(model.weight.data, original)


def test_zero_grad_after_second_step():
    """second_step(zero_grad=True) → all p.grad is None."""
    model, opt = _make_model_and_sam()
    x, y = torch.randn(4, 4), torch.tensor([0, 1, 0, 1])
    _one_cycle(model, opt, x, y)
    assert all(p.grad is None for p in model.parameters())


def test_step_raises():
    """Direct step() must raise NotImplementedError."""
    _, opt = _make_model_and_sam()
    with pytest.raises(NotImplementedError):
        opt.step()


def test_param_groups_shared_with_base():
    """SAM.param_groups is the identical object as base_optimizer.param_groups."""
    _, opt = _make_model_and_sam(base=torch.optim.AdamW, lr=0.001)
    assert opt.param_groups is opt.base_optimizer.param_groups


def test_full_cycle_decreases_loss():
    """One SAM cycle with AdamW should decrease loss (overwhelmingly likely)."""
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 3, bias=False)
    from graphslm_ids.offline.training.sam_optimizer import SAM
    opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=0.01)
    x = torch.randn(32, 10)
    y = torch.randint(0, 3, (32,))

    loss_before = F.cross_entropy(model(x), y).item()
    _one_cycle(model, opt, x, y)
    loss_after = F.cross_entropy(model(x), y).item()
    assert loss_after < loss_before, f"loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"


def test_rho_zero_is_plain_optimizer():
    """rho=0 → e_w=0 → SAM degenerates to the base optimizer update."""
    torch.manual_seed(7)
    model_sam = torch.nn.Linear(4, 2, bias=False)
    model_base = torch.nn.Linear(4, 2, bias=False)
    model_base.weight.data.copy_(model_sam.weight.data)

    from graphslm_ids.offline.training.sam_optimizer import SAM
    opt_sam = SAM(model_sam.parameters(), torch.optim.SGD, rho=0.0, lr=0.1, momentum=0.0)
    opt_base = torch.optim.SGD(model_base.parameters(), lr=0.1, momentum=0.0)

    x, y = torch.randn(4, 4), torch.tensor([0, 1, 0, 1])

    loss_s = F.cross_entropy(model_sam(x), y); loss_s.backward()
    opt_sam.first_step(zero_grad=True)
    loss_s2 = F.cross_entropy(model_sam(x), y); loss_s2.backward()
    opt_sam.second_step(zero_grad=True)

    loss_b = F.cross_entropy(model_base(x), y); loss_b.backward()
    opt_base.step(); opt_base.zero_grad()

    torch.testing.assert_close(model_sam.weight.data, model_base.weight.data, atol=1e-5, rtol=1e-5)


def test_defaults_expose_base_optimizer_keys():
    """SAM.defaults must include the base optimizer's keys (betas for AdamW).

    OneCycleLR(cycle_momentum=True) inspects optimizer.defaults for 'betas' or
    'momentum'; without merging base defaults it raises ValueError at __init__.
    """
    _, opt = _make_model_and_sam(base=torch.optim.AdamW, lr=0.001)
    assert "betas" in opt.defaults, "SAM.defaults missing 'betas' from AdamW base"
    assert "rho" in opt.defaults, "SAM.defaults lost its own 'rho'"


def test_onecycle_lr_constructs_over_sam_with_cycle_momentum():
    """OneCycleLR with default cycle_momentum=True must construct over SAM(AdamW).

    Regression for the smoke-test crash:
      ValueError: optimizer must support momentum or beta1 with
      `cycle_momentum` option enabled
    """
    model = torch.nn.Linear(8, 3, bias=False)
    from graphslm_ids.offline.training.sam_optimizer import SAM
    opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=0.001)
    # Must NOT raise — cycle_momentum defaults to True.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=0.001, steps_per_epoch=4, epochs=2,
    )
    # One full SAM cycle + scheduler step keeps lr finite and positive.
    x, y = torch.randn(8, 8), torch.randint(0, 3, (8,))
    F.cross_entropy(model(x), y).backward()
    opt.first_step(zero_grad=True)
    F.cross_entropy(model(x), y).backward()
    opt.second_step(zero_grad=True)
    sched.step()
    assert opt.param_groups[0]["lr"] > 0.0
