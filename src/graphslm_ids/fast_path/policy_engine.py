from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PolicyDecision:
    label: str
    label_idx: int
    confidence: float
    top_classes: list[dict[str, Any]]
    should_alert: bool
    alert_threshold: float
    trigger_reason: str


class PolicyEngine:
    def __init__(
        self,
        label_to_index: dict[str, int],
        alert_threshold: float = 0.70,
        alert_labels: tuple[str, ...] = ("suspicious", "malicious"),
    ) -> None:
        self.label_to_index = {str(label): int(idx) for label, idx in label_to_index.items()}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self.alert_threshold = float(alert_threshold)
        self.alert_labels = {str(label).lower() for label in alert_labels}

    def decide(self, hgt_output: Any) -> PolicyDecision:
        logits = _as_numpy(hgt_output.logits).reshape(-1)
        probs = _softmax(logits)
        order = np.argsort(-probs)
        top_idx = int(order[0])
        label = self.index_to_label.get(top_idx, str(top_idx))
        confidence = float(probs[top_idx])
        top_classes = [
            {
                "label": self.index_to_label.get(int(idx), str(int(idx))),
                "prob": float(probs[int(idx)]),
                "label_idx": int(idx),
            }
            for idx in order[: min(5, len(order))]
        ]
        label_alert = label.lower() in self.alert_labels
        threshold_alert = confidence >= self.alert_threshold
        should_alert = bool(label_alert and threshold_alert)
        if should_alert:
            reason = f"{label} confidence {confidence:.4f} >= threshold {self.alert_threshold:.4f}"
        elif not label_alert:
            reason = f"{label} is not configured as an alert label"
        else:
            reason = f"{label} confidence {confidence:.4f} < threshold {self.alert_threshold:.4f}"
        return PolicyDecision(
            label=label,
            label_idx=top_idx,
            confidence=confidence,
            top_classes=top_classes,
            should_alert=should_alert,
            alert_threshold=self.alert_threshold,
            trigger_reason=reason,
        )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / max(float(exp.sum()), 1e-12)
