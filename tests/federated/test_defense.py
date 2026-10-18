import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.aggregators import Median
from graphslm_ids.offline.federated.defense import RAGuard, build_defense, client_suspicion, clip_to_median, decide
from graphslm_ids.offline.federated.linalg import coordinate_median, weighted_trimmed_mean
from graphslm_ids.offline.federated.params import ParamLayout

LAYOUT = ParamLayout.from_state_dict(
    {
        "layers.0.relation_key.flow__r1__packet.weight": torch.zeros(40),
        "layers.0.relation_key.packet__r2__technique.weight": torch.zeros(40),
        "classifier.weight": torch.zeros(20),
    }
)
R2 = LAYOUT.group_coords(["rel:packet__r2__technique"])


def _round(seed=0, n=20, noise=0.1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(100, generator=g) + noise * torch.randn(n, 100, generator=g)


def _run(guard, x, round_idx=1, prev=None, layout=LAYOUT):
    ctx = {"client_ids": list(range(x.shape[0])), "round_idx": round_idx, "layout": layout, "prev_aggregate": prev}
    return guard.aggregate(x, torch.ones(x.shape[0]), ctx)


def test_decide_thresholds_and_downweight_flag():
    decisions, w = decide(np.array([0.3, 0.6, 0.9]))
    assert decisions == ["reject", "downweight", "accept"] and np.allclose(w, [0.0, 0.6, 1.0])
    decisions, w = decide(np.array([0.3, 0.6, 0.9]), downweight=False)
    assert decisions == ["reject", "accept", "accept"] and np.allclose(w, [0.0, 1.0, 1.0])


def test_client_suspicion_mean_and_max():
    u = np.array([[0.0, 1.0], [0.5, 0.5], [0.9, 0.9]])
    ev = np.array([[True, True], [True, False], [False, False]])
    sizes = np.array([3.0, 1.0])
    assert np.allclose(client_suspicion(u, ev, sizes, "mean"), [0.25, 0.5, 0.0])
    assert np.allclose(client_suspicion(u, ev, sizes, "max"), [1.0, 0.5, 0.0])


def test_rejects_a_scaled_outlier_without_rejecting_benign():
    x = _round()
    x[0] = 50.0 * x[0]
    res = _run(RAGuard(), x)
    assert res.decisions[0] == "reject" and "reject" not in res.decisions[1:]
    assert not bool(res.selected[0]) and res.trust.shape == (20,)


def test_rejects_a_sign_flip():
    x = _round()
    x[0] = -x[0]
    assert _run(RAGuard(), x).decisions[0] == "reject"


def test_zero_relation_group_is_not_penalized_unless_relation_awareness_is_off():
    x = _round()
    x[:6, R2] = 0.0  # non-IID: six benign clients never see relation r2
    assert "reject" not in _run(RAGuard(), x).decisions
    assert "reject" in _run(RAGuard(relation_aware=False), x).decisions[:6]


def test_warmup_uses_cross_sectional_trust_only():
    guard = RAGuard()
    t1, _ = guard.update_trust([0], np.array([0.4]), 1)
    t2, _ = guard.update_trust([0], np.array([0.0]), 2)
    t3, _ = guard.update_trust([0], np.array([0.0]), 3)
    assert t1[0] == pytest.approx(0.6) and t2[0] == pytest.approx(1.0) and t3[0] == pytest.approx(1.0)


def test_abrupt_change_caps_trust_after_warmup():
    guard = RAGuard()
    for r, u in enumerate([0.05, 0.06, 0.05, 0.06], start=1):
        guard.update_trust([0], np.array([u]), r)
    trust, abrupt = guard.update_trust([0], np.array([0.30]), 5)
    assert bool(abrupt[0]) and trust[0] == pytest.approx(0.5)
    plain = RAGuard(temporal=False)
    for r, u in enumerate([0.05, 0.06, 0.05, 0.06, 0.30], start=1):
        trust, abrupt = plain.update_trust([0], np.array([u]), r)
    assert not bool(abrupt[0]) and trust[0] == pytest.approx(0.70)


def test_aggregate_is_weighted_trimmed_mean_or_median_of_clipped_rows():
    x = _round()  # every client is active in every group, so per-group = whole-vector aggregation
    clipped = clip_to_median(x, 1.0)
    res = _run(RAGuard(), x)
    keep = torch.tensor([d != "reject" for d in res.decisions])
    assert bool(keep.all())
    expected = weighted_trimmed_mean(clipped[keep], torch.as_tensor(res.weights, dtype=torch.float32)[keep], 0.2)
    assert torch.allclose(res.delta, expected, atol=1e-6)
    med = _run(RAGuard(aggregator="median"), x)
    assert torch.allclose(med.delta, coordinate_median(clipped), atol=1e-6)


def test_group_nobody_trains_aggregates_to_zero_and_layout_none_is_one_group():
    x = _round()
    x[:, R2] = 0.0
    assert int(torch.count_nonzero(_run(RAGuard(), x).delta[R2])) == 0
    assert _run(RAGuard(), x, layout=None).delta.shape == (100,)


def test_build_defense():
    guard = build_defense({"name": "raguard", "params": {"temporal": False}}, assumed_malicious_ratio=0.3)
    assert isinstance(guard, RAGuard) and guard.beta == 0.3 and guard.temporal is False
    assert isinstance(build_defense({"name": "median"}), Median)
    with pytest.raises(ValueError):
        build_defense({"name": "raguard", "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        RAGuard(beta=0.5)
    with pytest.raises(ValueError):
        RAGuard(aggregator="mean")


def test_sign_signal_scale_is_floored_at_its_resolution():
    from graphslm_ids.offline.federated.defense import robust_z_sigma, sign_suspicion

    agreement = np.ones(20)  # MAD = 0: only the floors set the scale
    agreement[3] = 1 - 1 / 40  # one coordinate off out of n = 40: inside the 1-sigma dead zone
    agreement[7] = 1 - 10 / 40
    assert robust_z_sigma(agreement, max(1e-3, 1 / 40))[3] >= -1 - 1e-9
    s = sign_suspicion(agreement, 40, scale_floor=1e-3, z_max=4.0)
    assert s[3] == pytest.approx(0.0, abs=1e-9) and s[7] > 0
    assert np.allclose(np.delete(s, [3, 7]), 0.0)
    # Without the resolution floor (huge n) the same one-coordinate miss is fully suspicious.
    assert sign_suspicion(agreement, 10**6, scale_floor=1e-3, z_max=4.0)[3] == pytest.approx(1.0)


def test_raguard_info_records_the_effective_beta():
    x = _round()
    assert _run(build_defense({"name": "raguard"}, assumed_malicious_ratio=0.2), x).info["beta"] == 0.2
    assert _run(RAGuard(beta=0.1), x).info["beta"] == 0.1


def test_default_z_max_is_four():
    assert RAGuard().z_max == 4.0
