"""SAM — Sharpness-Aware Minimization optimizer wrapper.

Foret et al. ICLR 2021 (https://arxiv.org/abs/2010.01412).

Usage pattern in training loop:
    loss.backward()
    optimizer.first_step(zero_grad=True)   # perturb weights
    loss2 = forward(same_batch)
    loss2.backward()
    clip_grad_norm_(model.parameters(), max_norm)
    optimizer.second_step(zero_grad=True)  # restore + real update
    scheduler.step()

Constraints:
  - Incompatible with grad_accum_steps > 1 (perturbation mixes batches).
  - Incompatible with DDP (gradient sync on both passes not handled here).
  - scaler.step() must NOT be called; bypass with the explicit two-step API.
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """SAM wrapper around any base optimizer.

    Args:
        params: model parameters.
        base_optimizer: class of base optimizer (e.g. torch.optim.AdamW).
        rho: perturbation radius (paper default 0.05). rho=0 degenerates to base.
        adaptive: use ASAM scaling (|w| * grad) instead of plain grad.
        **kwargs: forwarded to base_optimizer constructor (lr, weight_decay, ...).
    """

    def __init__(
        self,
        params,
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        # Expose the base optimizer's defaults (lr, betas/momentum, eps, ...) so
        # LR schedulers that introspect `optimizer.defaults` work transparently.
        # In particular OneCycleLR(cycle_momentum=True) requires 'betas' (AdamW)
        # or 'momentum' (SGD) to be present, else it raises at construction.
        self.defaults.update(self.base_optimizer.defaults)
        for group in self.param_groups:
            group.setdefault("rho", rho)
            group.setdefault("adaptive", adaptive)
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Perturb weights in the sharpness direction by rho."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale_p = scale.to(p)
                e_w = (p.abs() if group["adaptive"] else 1.0) * p.grad * scale_p
                self.state[p]["old_p"] = p.data.clone()
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore original weights, then apply real gradient update."""
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data = self.state[p].pop("old_p")
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore_weights(self) -> None:
        """Restore weights without calling base_optimizer.step().

        Use when the second forward produces a non-finite loss so we can
        skip the weight update but still undo the perturbation.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data = self.state[p].pop("old_p")
        self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM requires explicit first_step() / second_step() calls. "
            "Do not call step() directly."
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((p.abs() if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)
