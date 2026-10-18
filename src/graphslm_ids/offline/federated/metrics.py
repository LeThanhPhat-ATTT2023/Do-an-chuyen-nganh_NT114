"""IDS and poisoning metrics for FL runs.

Predictions come from the eval loop as ``(flow_ids, logits)``; the metrics here are built
from confusion matrices against the (possibly noisy) labels and, when given, the clean
answer key. Macro-F1 averages classes with support > 0, exactly like the centralized
trainer, so numbers are comparable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from graphslm_ids.offline.federated.aggregators import DOWNWEIGHT, INVALID, REJECT

FLAGGED = {REJECT, INVALID}
COMPACT_KEYS = (
    "count",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "loss",
    "benign_false_alarm_rate",
    "attack_detection_rate",
    "alarm_precision",
    "family_recall",
)


def _confusion(y: torch.Tensor, pred: torch.Tensor, num_classes: int) -> np.ndarray:
    idx = y.long().reshape(-1) * num_classes + pred.long().reshape(-1)
    return torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes).numpy()


def _safe_div(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def metrics_from_confusion(
    cm: np.ndarray,
    label_names: dict[int, str],
    families: dict[str, list[str]] | None = None,
    benign: str = "Benign",
) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    c = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(c), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(c), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros(c), where=denom > 0)
    present = support > 0
    total = float(cm.sum())
    out: dict[str, Any] = {
        "count": int(total),
        "accuracy": float(tp.sum() / total) if total else None,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / support.sum()) if support.sum() else 0.0,
        "per_class": {
            label_names.get(i, str(i)): {
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in range(c)
        },
    }
    name_to_id = {name: idx for idx, name in label_names.items()}
    b = name_to_id.get(benign)
    if b is not None and b < c:
        attack = [i for i in range(c) if i != b]
        out["benign_false_alarm_rate"] = _safe_div(support[b] - cm[b, b], support[b])
        missed = _safe_div(cm[attack, b].sum(), support[attack].sum())
        out["attack_detection_rate"] = None if missed is None else 1.0 - missed
        false_alarm_share = _safe_div(cm[b, attack].sum(), cm[:, attack].sum())
        out["alarm_precision"] = None if false_alarm_share is None else 1.0 - false_alarm_share
    if families:
        family_recall: dict[str, float | None] = {}
        for fam, members in families.items():
            ids = [name_to_id[m] for m in members if m in name_to_id and name_to_id[m] < c]
            family_recall[fam] = _safe_div(tp[ids].sum(), support[ids].sum()) if ids else None
        out["family_recall"] = family_recall
    return out


def _score(
    logits: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    label_names: dict[int, str],
    families: dict[str, list[str]] | None,
) -> dict[str, Any]:
    num_classes = int(logits.shape[1])
    loss = float(F.cross_entropy(logits, y)) if y.numel() else None
    cm = _confusion(y, pred, num_classes)
    out = metrics_from_confusion(cm, label_names, families)
    out["loss"] = loss
    out["confusion"] = cm.tolist()
    return out


def evaluate_predictions(
    flow_ids: np.ndarray,
    logits: torch.Tensor,
    labels: np.ndarray,
    label_names: dict[int, str],
    *,
    clean_labels: np.ndarray | None = None,
    families: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Score ``logits`` (one row per flow in ``flow_ids``, any order) against ``labels[flow_ids]``.

    ``labels`` and ``clean_labels`` are indexed by global flow id. The result holds the keys of
    ``metrics_from_confusion`` plus ``loss`` (mean cross-entropy) and ``confusion``; with
    ``clean_labels`` the same scores against the clean labels sit under ``"clean"``.
    """
    flow_ids = np.asarray(flow_ids, dtype=np.int64).reshape(-1)
    logits = logits.detach().float().cpu()
    if logits.ndim != 2 or logits.shape[0] != flow_ids.shape[0]:
        raise ValueError(f"logits {tuple(logits.shape)} do not match {flow_ids.shape[0]} flow ids")
    pred = logits.argmax(dim=1)
    y = torch.as_tensor(np.asarray(labels)[flow_ids], dtype=torch.long)
    out = _score(logits, pred, y, label_names, families)
    if clean_labels is not None:
        y_clean = torch.as_tensor(np.asarray(clean_labels)[flow_ids], dtype=torch.long)
        out["clean"] = _score(logits, pred, y_clean, label_names, families)
    return out


def detection_counts(decisions: dict[int, str], malicious: set[int]) -> dict[str, int]:
    counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "downweighted_benign": 0,
        "downweighted_malicious": 0,
        "invalid": 0,
    }
    for cid, decision in decisions.items():
        bad = int(cid) in malicious
        flagged = decision in FLAGGED
        if decision == INVALID:
            counts["invalid"] += 1
        if decision == DOWNWEIGHT:
            counts["downweighted_malicious" if bad else "downweighted_benign"] += 1
        if bad:
            counts["tp" if flagged else "fn"] += 1
        else:
            counts["fp" if flagged else "tn"] += 1
    return counts


def merge_counts(total: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    return {k: int(total.get(k, 0)) + int(new.get(k, 0)) for k in sorted(set(total) | set(new))}


def detection_rates(counts: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn, tn = (int(counts.get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
    tpr = tp / (tp + fn) if tp + fn else None
    return {
        "tpr": tpr,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "malicious_acceptance_rate": None if tpr is None else 1.0 - tpr,
    }


def detection_summary(counts: dict[str, int], screened: bool) -> dict[str, Any]:
    """Run-level detection: the counts and rates accumulated over the screened rounds.

    ``screened`` says whether any round was screened (the aggregator gave per-client verdicts).
    FedAvg / Median / Trimmed Mean accept every client by construction, so their rates are None
    (not applicable) rather than a misleading TPR of 0.
    """
    zero = detection_counts({}, set())
    return {**merge_counts(zero, counts), **detection_rates(counts if screened else {}), "screened": bool(screened)}


def convergence_round(history: list[tuple[int, float | None]], frac: float = 0.95) -> int | None:
    values = [(int(r), float(s)) for r, s in history if s is not None]
    if not values:
        return None
    best = max(s for _r, s in values)
    target = frac * best if best > 0 else best
    for r, s in values:
        if s >= target:
            return r
    return None


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    out = {k: metrics[k] for k in COMPACT_KEYS if k in metrics}
    if metrics.get("clean"):
        out["clean"] = {k: metrics["clean"][k] for k in COMPACT_KEYS if k in metrics["clean"]}
    return out
