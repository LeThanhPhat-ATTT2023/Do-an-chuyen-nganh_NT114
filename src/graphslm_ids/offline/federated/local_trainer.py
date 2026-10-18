"""Client-side local training for one FL round, architecture-agnostic.

Everything model-specific (data, batches, loss, EACS/GCL) lives behind a
:class:`~graphslm_ids.offline.federated.models.ModelAdapter`; this module owns the shared
recipe: load the global weights, AdamW (fresh, or restored from CPU when
``optimizer_state: persist``), ``local_epochs`` passes capped at ``max_local_steps``,
non-finite losses skipped, gradient clipping, and ``delta = trained - global``.

Gradients are cleared with ``zero_grad(set_to_none=True)``, so a parameter no batch touched
keeps grad ``None``, AdamW skips it and its delta stays exactly zero (``params.py``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from graphslm_ids.offline.federated.config import DEFAULT_FL_CONFIG
from graphslm_ids.offline.federated.models import LOSS_TYPES, ModelAdapter
from graphslm_ids.offline.federated.params import ParamLayout

OPTIMIZER_STATES = ("reset", "persist")


def local_class_weights(
    labels: np.ndarray, num_classes: int, method: str = "cb", beta: float = 0.985, cap: float = 4.0
) -> torch.Tensor:
    """Class weights from one client's labels. Absent classes get 1.0 and stay out of
    the mean-1 normalisation (the centralized helper lets them skew it)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    present = counts > 0
    w = np.ones(num_classes, dtype=np.float64)
    if method == "cb":
        w[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    elif method == "inverse":
        w[present] = counts[present].sum() / (present.sum() * counts[present])
    else:
        raise ValueError(f"unknown class_weight_method {method!r}; expected cb or inverse")
    if present.any():
        w[present] /= w[present].mean()
    return torch.from_numpy(np.minimum(w, float(cap)).astype(np.float32))


def round_lr(client_cfg: dict[str, Any], round_idx: int, total_rounds: int) -> float:
    lr = float(client_cfg.get("lr", 1e-3))
    schedule = str(client_cfg.get("lr_schedule", "cosine_rounds")).lower()
    if schedule == "constant":
        return lr
    if schedule == "cosine_rounds":
        lr_min = float(client_cfg.get("lr_min", 0.0))
        progress = (int(round_idx) - 1) / max(int(total_rounds), 1)
        return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr_schedule {schedule!r}; expected cosine_rounds or constant")


@dataclass
class ClientState:
    client_id: int
    flow_ids: np.ndarray
    flow_labels: np.ndarray
    data: Any
    class_weights: torch.Tensor | None = None
    eacs: Any = None
    optimizer_state: dict | None = None
    rounds_trained: int = 0


@dataclass
class LocalUpdate:
    client_id: int
    delta: torch.Tensor
    num_samples: int
    steps: int
    seconds: float
    metrics: dict[str, Any]


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu").clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


class LocalTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        layout: ParamLayout,
        adapter: ModelAdapter,
        device: str | torch.device,
        client_cfg: dict[str, Any],
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.layout = layout
        self.adapter = adapter
        self.cfg = {**DEFAULT_FL_CONFIG["client"], **client_cfg}
        self.optimizer_state = str(self.cfg["optimizer_state"]).lower()
        if self.optimizer_state not in OPTIMIZER_STATES:
            raise ValueError(
                f"unknown client.optimizer_state {self.optimizer_state!r}; expected one of {OPTIMIZER_STATES}"
            )
        loss_type = str(self.cfg["loss_type"]).lower()
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"unknown client.loss_type {loss_type!r}; expected one of {LOSS_TYPES}")
        max_steps = self.cfg["max_local_steps"]
        self.max_local_steps = int(max_steps) if max_steps else None
        self.local_epochs = int(self.cfg["local_epochs"])
        self.grad_clip_norm = float(self.cfg["grad_clip_norm"] or 0.0)

    def _optimizer(
        self, params: list[torch.nn.Parameter], state: ClientState, lr: float
    ) -> torch.optim.AdamW:
        optimizer = torch.optim.AdamW(
            params,
            lr=float(lr),
            weight_decay=float(self.cfg["weight_decay"]),
            eps=float(self.cfg["adamw_eps"]),
            betas=tuple(float(b) for b in self.cfg["adamw_betas"]),
        )
        if self.optimizer_state == "persist" and state.optimizer_state is not None:
            optimizer.load_state_dict(state.optimizer_state)
            for group in optimizer.param_groups:
                group["lr"] = float(lr)
        return optimizer

    def train(
        self, state: ClientState, global_vec: torch.Tensor, round_idx: int, lr: float, seed: int
    ) -> LocalUpdate:
        start = time.perf_counter()
        torch.manual_seed(int(seed))
        self.model.load_state_dict(self.layout.unflatten(global_vec))
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._optimizer(params, state, lr)
        weights = None if state.class_weights is None else state.class_weights.to(self.device)
        steps = examples = correct = skipped = 0
        loss_sum = 0.0
        done = False
        for epoch in range(self.local_epochs):
            batches = self.adapter.train_batches(
                state.data, state.flow_ids, state.flow_labels, seed=int(seed) + epoch
            )
            for batch in batches:
                out = self.adapter.forward_loss(
                    self.model, batch, class_weights=weights, eacs=state.eacs, round_idx=round_idx
                )
                if out is None or not bool(torch.isfinite(out.loss)):
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
                optimizer.step()
                n = int(out.labels.numel())
                steps += 1
                examples += n
                loss_sum += float(out.loss.detach()) * n
                correct += int((out.logits.argmax(dim=1) == out.labels).sum())
                if self.max_local_steps is not None and steps >= self.max_local_steps:
                    done = True
                    break
            if done:
                break
        optimizer.zero_grad(set_to_none=True)
        if self.optimizer_state == "persist":
            state.optimizer_state = _to_cpu(optimizer.state_dict())
        state.rounds_trained += 1
        trained = self.layout.flatten(self.model.state_dict())
        delta = trained - global_vec.detach().to("cpu", torch.float32)
        return LocalUpdate(
            client_id=state.client_id,
            delta=delta,
            num_samples=int(state.flow_ids.shape[0]),
            steps=steps,
            seconds=time.perf_counter() - start,
            metrics={
                "loss": loss_sum / max(examples, 1),
                "accuracy": correct / max(examples, 1),
                "examples": examples,
                "skipped_batches": skipped,
            },
        )
