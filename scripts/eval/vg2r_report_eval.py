"""Grade VG²R reports on the dual rubric. Reads a directory of per-alert JSON
records (each carrying the rubric inputs), aggregates per-axis means + the
composite F*, and reports bootstrap 95% CI. Runs offline (no Ollama/HGT needed
at grading time); the heavy generation/HGT-masking step writes the records.

Usage:
  D:\\v\\nt114\\Scripts\\python.exe scripts/eval/vg2r_report_eval.py \\
    --records-dir outputs/v3_ob_eacs_v2/vg2r_records --out outputs/v3_ob_eacs_v2/vg2r_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphslm_ids.runtime.slow_path.vg2r_metrics import characterization, composite_f_star


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def bootstrap_ci(values: list[float], seed: int = 42, n: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def aggregate_records(records: list[dict]) -> dict:
    def col(key: str) -> list[float]:
        return [float(r[key]) for r in records if key in r]

    char_vals = [characterization(float(r["fid_plus"]), float(r["fid_minus"]))
                 for r in records if "fid_plus" in r and "fid_minus" in r]
    f_stars = [
        composite_f_star(float(r["cgr"]), float(r["hallucination_rate"]), float(r["numeric_accuracy"]),
                         float(r["factual_consistency"]),
                         characterization(float(r["fid_plus"]), float(r["fid_minus"])))
        for r in records
    ]
    lo, hi = bootstrap_ci(f_stars)
    return {
        "n": len(records),
        "axis_a": {
            "fid_plus_mean": round(_mean(col("fid_plus")), 6),
            "fid_minus_mean": round(_mean(col("fid_minus")), 6),
            "sparsity_mean": round(_mean(col("sparsity")), 6),
            "characterization_mean": round(_mean(char_vals), 6),
        },
        "axis_b": {
            "cgr_mean": round(_mean(col("cgr")), 6),
            "hallucination_rate_mean": round(_mean(col("hallucination_rate")), 6),
            "numeric_accuracy_mean": round(_mean(col("numeric_accuracy")), 6),
            "factual_consistency_mean": round(_mean(col("factual_consistency")), 6),
            "coverage_mean": round(_mean(col("coverage")), 6),
            "plausibility_mean": round(_mean(col("plausibility")), 6),
        },
        "composite_f_star": round(_mean(f_stars), 6),
        "f_star_ci95": [round(lo, 6), round(hi, 6)],
    }


def _load_records(records_dir: Path) -> list[dict]:
    records = []
    for path in sorted(records_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="VG²R dual-rubric eval")
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = _load_records(Path(args.records_dir))
    summary = aggregate_records(records)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
