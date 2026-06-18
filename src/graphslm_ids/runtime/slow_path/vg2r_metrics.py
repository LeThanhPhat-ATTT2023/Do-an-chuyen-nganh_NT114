"""Pure dual-rubric math for VG²R. No I/O, no model — testable in isolation.

Axis A (explanation fidelity to HGT): GraphFramEx Fid+/Fid−/sparsity +
characterization. Axis B (report faithfulness): coverage, plausibility, and a
composite F* (harmonic mean). The predict-probability inputs are supplied by
the caller (which re-runs HGT on masked subgraphs), keeping this module pure.
"""
from __future__ import annotations


def fidelity_plus(prob_full: float, prob_without_cited: float) -> float:
    """Necessity: drop in HGT prob when the cited evidence is removed. Higher better."""
    return round(float(prob_full) - float(prob_without_cited), 6)


def fidelity_minus(prob_full: float, prob_only_cited: float) -> float:
    """Sufficiency: change in HGT prob when ONLY cited evidence is kept. Lower better."""
    return round(float(prob_full) - float(prob_only_cited), 6)


def sparsity(num_cited: int, num_total: int) -> float:
    if num_total <= 0:
        return 0.0
    return num_cited / num_total


def characterization(fid_plus: float, fid_minus: float, w_plus: float = 0.5, w_minus: float = 0.5) -> float:
    """Weighted harmonic mean of fid+ and (1 - fid-) (GraphFramEx). In [0, 1]."""
    a = max(min(float(fid_plus), 1.0), 0.0)
    b = max(min(1.0 - float(fid_minus), 1.0), 0.0)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return (w_plus + w_minus) / (w_plus / a + w_minus / b)


def coverage(cited: set[str], salient: set[str]) -> float:
    """Recall of HGT's top-k salient nodes among the report's citations."""
    if not salient:
        return 1.0
    return len(cited & salient) / len(salient)


def plausibility(cited_techniques: list[str], predicted_label: str, class_to_technique: dict[str, list[str]]) -> float:
    """Fraction of cited techniques that belong to the predicted class's MITRE map."""
    allowed = set(class_to_technique.get(predicted_label, []))
    if not cited_techniques:
        return 0.0
    hits = sum(1 for t in cited_techniques if t in allowed)
    return hits / len(cited_techniques)


def composite_f_star(cgr: float, hallucination_rate: float, numeric_accuracy: float,
                     factual_consistency: float, characterization: float) -> float:
    """Harmonic mean of {CGR, 1-HR, NumAcc, FCS, Characterization}. In [0, 1]."""
    parts = [float(cgr), 1.0 - float(hallucination_rate), float(numeric_accuracy),
             float(factual_consistency), float(characterization)]
    parts = [max(min(p, 1.0), 0.0) for p in parts]
    if any(p <= 0.0 for p in parts):
        return 0.0
    return len(parts) / sum(1.0 / p for p in parts)
