import math
from graphslm_ids.runtime.slow_path.vg2r_metrics import (
    characterization, composite_f_star, coverage, fidelity_minus, fidelity_plus,
    plausibility, sparsity,
)


def test_fidelity_plus_and_minus():
    assert fidelity_plus(prob_full=0.88, prob_without_cited=0.30) == 0.58
    assert fidelity_minus(prob_full=0.88, prob_only_cited=0.85) == 0.03


def test_sparsity():
    assert sparsity(num_cited=2, num_total=10) == 0.2
    assert sparsity(num_cited=0, num_total=0) == 0.0


def test_characterization_high_when_necessary_and_sufficient():
    # high fid+ and low fid- -> high characterization
    high = characterization(fid_plus=0.9, fid_minus=0.05)
    low = characterization(fid_plus=0.1, fid_minus=0.8)
    assert high > low
    assert 0.0 <= high <= 1.0


def test_coverage_recall_of_salient_nodes():
    assert coverage(cited={"E_PKT_001"}, salient={"E_PKT_001", "E_PKT_002"}) == 0.5
    assert coverage(cited=set(), salient=set()) == 1.0


def test_plausibility_matches_class_map():
    cmap = {"SqlInjection": ["T1190"]}
    assert plausibility(cited_techniques=["T1190"], predicted_label="SqlInjection", class_to_technique=cmap) == 1.0
    assert plausibility(cited_techniques=["T1059"], predicted_label="SqlInjection", class_to_technique=cmap) == 0.0


def test_composite_is_harmonic_mean():
    val = composite_f_star(cgr=1.0, hallucination_rate=0.0, numeric_accuracy=1.0,
                           factual_consistency=1.0, characterization=1.0)
    assert math.isclose(val, 1.0)
    worse = composite_f_star(cgr=1.0, hallucination_rate=0.5, numeric_accuracy=1.0,
                            factual_consistency=1.0, characterization=1.0)
    assert worse < 1.0
