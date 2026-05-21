"""HGAA filter network — multiclass adaptation.

Reference paper: Zhao et al., HGAA (Symmetry 2025, 17, 1623), §3.2.

Paper architecture:
  - Teacher N_Teacher (frozen) and Student N_Student (random init) share
    structure; produce (h_i, h_g) node + graph embeddings.
  - Anomaly = high KD(h, ĥ) distance.

Our multiclass adaptation:
  - Teacher = warm-started from v7-p1 checkpoint (Path B, see spec §3.2.2).
  - Student = clone of teacher with reduced depth (2 layers vs 4), then
    fine-tuned briefly to diverge.
  - Replace "is anomaly?" with "is augmented sample still in class c?" via
    a :class:`ClassCentroidBank` that tracks per-class embedding centroids.
  - Accept rule: (structural_distance < τ_struct) AND
    (cosine_sim_to_class_centroid > τ_class).

This module exposes:
  - :class:`ClassCentroidBank` — EMA-updated per-class embedding bank.
  - :func:`load_hgt_state_dict_resilient` — checkpoint loader that handles
    classifier-head dimension mismatches (per spec §1.5 DC-3).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn


_log = logging.getLogger(__name__)


# =========================================================================
# ClassCentroidBank
# =========================================================================


class ClassCentroidBank:
    """EMA-tracked per-class centroid embeddings for filter-time class-membership scoring.

    On the first update for a class, the centroid is initialized to the
    mean of observed embeddings (no EMA blend). Subsequent updates use:

        centroid_c ← ema_decay * centroid_c + (1 - ema_decay) * mean(new_embs_for_c)

    Args:
        num_classes: Total number of classes (data-driven per DC-1).
        embedding_dim: Dimension of teacher's graph-level embedding.
        ema_decay: Smoothing factor in [0, 1). Larger = slower update.
    """

    def __init__(self, num_classes: int, embedding_dim: int, ema_decay: float = 0.99) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.ema_decay = float(ema_decay)
        self.centroids = torch.zeros(num_classes, embedding_dim)
        self._initialized = torch.zeros(num_classes, dtype=torch.bool)

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """EMA-update centroids from a batch of (embedding, label) pairs.

        Args:
            embeddings: ``(B, D)`` tensor on any device.
            labels: ``(B,)`` long tensor with class IDs in ``[0, num_classes)``.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D (B, D), got shape {tuple(embeddings.shape)}")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding_dim mismatch: bank={self.embedding_dim}, got {embeddings.shape[1]}"
            )
        # Bring both onto the bank's device
        embs = embeddings.detach().to(self.centroids.device, dtype=self.centroids.dtype)
        lbls = labels.detach().to(self.centroids.device)

        present_classes = torch.unique(lbls)
        for c in present_classes.tolist():
            c = int(c)
            if not 0 <= c < self.num_classes:
                continue
            mask = lbls == c
            batch_mean = embs[mask].mean(dim=0)
            if not bool(self._initialized[c]):
                self.centroids[c] = batch_mean
                self._initialized[c] = True
            else:
                self.centroids[c] = (
                    self.ema_decay * self.centroids[c]
                    + (1.0 - self.ema_decay) * batch_mean
                )

    @torch.no_grad()
    def cosine_similarity(self, embedding: torch.Tensor, class_id: int) -> float:
        """Compute cosine similarity between a single embedding and class centroid."""
        c = int(class_id)
        if not 0 <= c < self.num_classes:
            raise IndexError(f"class_id {c} out of range [0, {self.num_classes})")
        emb = embedding.detach().to(self.centroids.device, dtype=self.centroids.dtype).reshape(-1)
        centroid = self.centroids[c]
        emb_norm = emb.norm()
        cen_norm = centroid.norm()
        if emb_norm < 1e-12 or cen_norm < 1e-12:
            return 0.0
        return float(torch.dot(emb, centroid) / (emb_norm * cen_norm))

    def state_dict(self) -> dict[str, Any]:
        return {
            "centroids": self.centroids.clone(),
            "initialized": self._initialized.clone(),
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "ema_decay": self.ema_decay,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["num_classes"] != self.num_classes:
            raise ValueError(
                f"num_classes mismatch: bank={self.num_classes}, state={state['num_classes']}"
            )
        if state["embedding_dim"] != self.embedding_dim:
            raise ValueError(
                f"embedding_dim mismatch: bank={self.embedding_dim}, state={state['embedding_dim']}"
            )
        self.centroids = state["centroids"].clone()
        self._initialized = state["initialized"].clone()
        self.ema_decay = float(state["ema_decay"])


# =========================================================================
# Resilient checkpoint loader (DC-3)
# =========================================================================


def load_hgt_state_dict_resilient(
    model: nn.Module,
    ckpt_path: str | Path,
    num_classes_new: int,
) -> tuple[list[str], list[str]]:
    """Load an HGT checkpoint, dropping classifier-head weights if the
    output dimension changed.

    This makes warm-start dataset-portable (spec §1.5 DC-3): the HGT
    backbone (input projections, attention layers, FFN) transfers cleanly
    even when the downstream dataset has a different number of classes;
    only the final classifier head is re-initialized.

    Args:
        model: An instantiated HGT (or compatible) module.
        ckpt_path: Path to a ``.pt`` file saved by the trainer. Either
            top-level state dict OR ``{"model_state_dict": ...}`` wrapping.
        num_classes_new: ``num_classes`` of the current ``model``. If this
            differs from the checkpoint's classifier head, the head keys
            are dropped (model keeps its random-init head).

    Returns:
        ``(missing, unexpected)`` lists from ``load_state_dict(strict=False)``.
    """
    ckpt_path = Path(ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

    # Identify classifier-head keys (final Linear: classifier.<N>.weight/bias).
    # We detect by checking each "classifier.*.weight" entry's row count
    # against num_classes_new; only the final layer has shape (num_classes, *).
    drop_keys: list[str] = []
    for key, tensor in list(state.items()):
        if not key.startswith("classifier."):
            continue
        if not key.endswith(".weight") and not key.endswith(".bias"):
            continue
        # Heuristic: the head linear's weight has 2D shape with rows == old num_classes.
        # Only flag mismatches; inner classifier layers (LayerNorm, hidden Linear)
        # have rows == hidden_dim, which equals model's hidden_dim and matches.
        shape = tensor.shape
        if shape and shape[0] != num_classes_new:
            # Pair weight & bias (or vice versa) for this layer.
            stem = key.rsplit(".", 1)[0]
            for sib in (f"{stem}.weight", f"{stem}.bias"):
                if sib in state and sib not in drop_keys:
                    drop_keys.append(sib)

    if drop_keys:
        _log.warning(
            "Classifier head mismatch (ckpt num_classes != %d). Dropping keys: %s",
            num_classes_new,
            drop_keys,
        )
        for k in drop_keys:
            state.pop(k, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)
