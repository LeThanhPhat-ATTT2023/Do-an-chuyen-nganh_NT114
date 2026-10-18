import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.metrics import (
    compact_metrics,
    convergence_round,
    detection_counts,
    detection_rates,
    evaluate_predictions,
    merge_counts,
    metrics_from_confusion,
)

NAMES = {0: "Benign", 1: "XSS", 2: "DDoS-ICMP_Flood"}


def _f1(p, r):
    return 2 * p * r / (p + r)


def test_metrics_from_confusion_rates_and_families():
    cm = np.array([[8, 2, 0], [1, 3, 0], [0, 0, 5]])
    m = metrics_from_confusion(cm, NAMES, families={"web": ["XSS"], "ddos": ["DDoS-ICMP_Flood"], "none": ["Nope"]})
    assert m["count"] == 19
    assert m["benign_false_alarm_rate"] == pytest.approx(0.2)
    assert m["attack_detection_rate"] == pytest.approx(1 - 1 / 9)
    assert m["alarm_precision"] == pytest.approx(1 - 2 / 10)
    assert m["family_recall"]["web"] == pytest.approx(0.75)
    assert m["family_recall"]["none"] is None
    expected = np.mean([_f1(8 / 9, 0.8), _f1(3 / 5, 0.75), 1.0])
    assert m["macro_f1"] == pytest.approx(expected)
    assert m["per_class"]["XSS"]["support"] == 4


def test_macro_f1_ignores_classes_without_support():
    cm = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 0]])
    assert metrics_from_confusion(cm, NAMES)["macro_f1"] == pytest.approx(1.0)


def test_detection_counts_and_rates():
    counts = detection_counts({0: "reject", 1: "accept", 2: "downweight", 3: "invalid", 4: "accept"}, {0, 4})
    assert counts["tp"] == 1 and counts["fn"] == 1 and counts["fp"] == 1 and counts["tn"] == 2
    assert counts["downweighted_benign"] == 1 and counts["invalid"] == 1
    total = merge_counts({"tp": 1}, counts)
    rates = detection_rates(total)
    assert rates["tpr"] == pytest.approx(2 / 3)
    assert rates["fpr"] == pytest.approx(1 / 3)
    assert rates["malicious_acceptance_rate"] == pytest.approx(1 / 3)
    assert detection_rates({})["tpr"] is None


def test_convergence_round_and_compact():
    assert convergence_round([(1, 0.1), (2, 0.5), (3, 0.97), (4, 1.0)]) == 3
    assert convergence_round([(1, None)]) is None
    full = {"macro_f1": 0.5, "per_class": {}, "confusion": [[1]], "clean": {"macro_f1": 0.4, "per_class": {}}}
    assert compact_metrics(full) == {"macro_f1": 0.5, "clean": {"macro_f1": 0.4}}
    assert compact_metrics(None) == {}


def test_evaluate_predictions_noisy_and_clean():
    labels = np.array([0, 1, 2, 1])
    clean = np.array([0, 0, 2, 1])
    gids = np.array([3, 1, 0, 2])  # adapters may return flows in any order
    logits = torch.tensor([[0, 5, 0], [0, 5, 0], [5, 0, 0], [0, 0, 5]], dtype=torch.float32)
    m = evaluate_predictions(gids, logits, labels, NAMES, clean_labels=clean)
    assert m["count"] == 4 and m["accuracy"] == pytest.approx(1.0)
    assert m["clean"]["accuracy"] == pytest.approx(0.75)
    assert m["loss"] > 0 and m["confusion"][1][1] == 2


def test_evaluate_predictions_validates_shapes_and_handles_empty():
    with pytest.raises(ValueError):
        evaluate_predictions(np.array([0, 1]), torch.zeros(3, 3), np.zeros(4, dtype=int), NAMES)
    empty = evaluate_predictions(np.array([], dtype=int), torch.zeros(0, 3), np.zeros(4, dtype=int), NAMES)
    assert empty["count"] == 0 and empty["accuracy"] is None and empty["loss"] is None
