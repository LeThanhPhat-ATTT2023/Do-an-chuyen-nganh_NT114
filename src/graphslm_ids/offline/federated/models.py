"""Architecture-agnostic model interface for the FL local trainer and evaluator.

A :class:`ModelAdapter` owns everything model-specific: building the network, turning a
client's flow ids into local training data, batching, the per-batch loss and full-graph /
full-table prediction. :class:`~graphslm_ids.offline.federated.local_trainer.LocalTrainer`
and the simulation only talk to this interface, so HGT and the flat baselines share them.

Adapters do not cache client data: ``client_data`` always builds; caching is the
simulation's job (``ExperimentContext.client_cache``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from graphslm_ids.offline.training.noise_consensus import EACSController

MODEL_NAMES = ("hgt",)
LOSS_TYPES = ("ce", "focal", "cb_focal")


@dataclass
class StepOutput:
    """One training batch: differentiable ``loss``, detached seed ``logits`` (n, C), ``labels`` (n,)."""

    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


class ModelAdapter:
    name: str = ""

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def client_data(self, client_id: int, flow_ids: np.ndarray, seed: int) -> Any:
        raise NotImplementedError

    def client_eacs(self) -> EACSController | None:
        """A fresh per-client ``EACSController`` (``None`` when EACS is off or unsupported)."""
        return None

    def train_batches(self, data: Any, flow_ids: np.ndarray, labels: np.ndarray, *, seed: int) -> Iterable:
        raise NotImplementedError

    def forward_loss(
        self,
        model: nn.Module,
        batch: Any,
        *,
        class_weights: torch.Tensor | None,
        eacs: EACSController | None,
        round_idx: int,
    ) -> StepOutput | None:
        """``None`` means the batch has nothing to train on and is skipped."""
        raise NotImplementedError

    def predict(self, model: nn.Module, flow_ids: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """Global flow ids and float32 CPU logits ``(n, C)``, in the adapter's own order."""
        raise NotImplementedError

    def checkpoint_meta(self) -> dict[str, Any]:
        return {"model_name": self.name}


def _model_name(value: Any) -> str:
    name = str(value or "hgt").lower()
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model.name {value!r}; expected one of {MODEL_NAMES}")
    return name


def build_adapter(
    cfg: dict[str, Any], ctx: Any, flow_stats: dict[str, list[float]], device: str | torch.device
) -> ModelAdapter:
    name = _model_name(cfg["model"].get("name", "hgt"))
    if name == "hgt":
        from graphslm_ids.offline.federated.hgt_adapter import HGTAdapter

        return HGTAdapter(cfg, ctx, flow_stats, device)
    raise AssertionError(name)  # pragma: no cover - MODEL_NAMES and this switch must agree


def rebuild_model(ckpt: dict[str, Any]) -> nn.Module:
    """Rebuild an untrained model from a checkpoint's ``checkpoint_meta`` keys, ``num_classes``
    and ``config["model"]``; the caller loads ``model_state_dict`` into it."""
    name = _model_name(ckpt.get("model_name", "hgt"))
    if name == "hgt":
        from graphslm_ids.offline.federated.hgt_adapter import build_hgt_model

        return build_hgt_model(
            node_input_dims=ckpt["node_input_dims"],
            edge_types=ckpt["edge_types"],
            num_classes=int(ckpt["num_classes"]),
            num_tactics=int(ckpt["num_tactics"]),
            model_cfg=ckpt["config"]["model"],
        )
    raise AssertionError(name)  # pragma: no cover
