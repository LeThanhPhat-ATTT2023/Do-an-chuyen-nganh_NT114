from scripts.eval.vg2r_report_eval import aggregate_records, bootstrap_ci


def test_aggregate_records_computes_axes():
    records = [
        {"cgr": 1.0, "hallucination_rate": 0.0, "numeric_accuracy": 1.0,
         "factual_consistency": 1.0, "fid_plus": 0.6, "fid_minus": 0.05,
         "sparsity": 0.2, "coverage": 1.0, "plausibility": 1.0},
        {"cgr": 1.0, "hallucination_rate": 0.0, "numeric_accuracy": 1.0,
         "factual_consistency": 0.9, "fid_plus": 0.5, "fid_minus": 0.10,
         "sparsity": 0.3, "coverage": 0.5, "plausibility": 1.0},
    ]
    summary = aggregate_records(records)
    assert 0.0 <= summary["composite_f_star"] <= 1.0
    assert summary["axis_a"]["fid_plus_mean"] == 0.55
    assert "f_star_ci95" in summary


def test_bootstrap_ci_is_ordered():
    lo, hi = bootstrap_ci([0.8, 0.9, 0.85, 0.95, 0.7], seed=42, n=200)
    assert lo <= hi
