import pytest
import torch

from graphslm_ids.offline.federated.aggregators import (
    DnC,
    FedAvg,
    Median,
    MultiKrum,
    SignGuard,
    TrimmedMean,
    build_aggregator,
    weighted_mean,
)


def test_fedavg_matches_weighted_mean():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(FedAvg().aggregate(x, torch.tensor([1.0, 3.0])).delta, torch.tensor([2.5, 3.5]))


def test_weighted_mean_rejects_zero_total():
    with pytest.raises(ValueError):
        weighted_mean(torch.ones(2, 3), torch.zeros(2))


def test_build_aggregator_fedavg_and_errors():
    assert isinstance(build_aggregator({"name": "fedavg"}), FedAvg)
    with pytest.raises(ValueError):
        build_aggregator({"name": "nope"})
    with pytest.raises(ValueError):
        build_aggregator({"name": "fedavg", "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        build_aggregator({"params": {}})
    assert isinstance(build_aggregator({"name": "FedAvg", "params": None}), FedAvg)


def _benign(n=10, d=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=g)
    return base + 0.1 * torch.randn(n, d, generator=g)


def test_median_odd_and_even():
    assert torch.allclose(Median().aggregate(torch.tensor([[1.0], [5.0], [3.0]]), torch.ones(3)).delta, torch.tensor([3.0]))
    assert torch.allclose(Median().aggregate(torch.tensor([[1.0], [2.0], [3.0], [10.0]]), torch.ones(4)).delta, torch.tensor([2.5]))


def test_trimmed_mean_drops_extremes_and_rejects_over_trim():
    x = torch.tensor([[0.0], [1.0], [2.0], [100.0]])
    assert torch.allclose(TrimmedMean(trim_k=1).aggregate(x, torch.ones(4)).delta, torch.tensor([1.5]))
    with pytest.raises(ValueError):
        TrimmedMean(trim_k=2).aggregate(x, torch.ones(4))


_ROBUST = [
    Median(),
    TrimmedMean(trim_ratio=0.2),
    MultiKrum(f_ratio=0.2),
    SignGuard(seed=0),
    DnC(f_ratio=0.2, b=50, seed=0),
]


@pytest.mark.parametrize("agg", _ROBUST, ids=lambda a: a.name)
def test_robust_rules_resist_a_large_outlier(agg):
    x = _benign()
    x[0] = -50.0 * x[1]
    clean = x[1:].mean(0)
    fedavg = FedAvg().aggregate(x, torch.ones(10)).delta
    assert (fedavg - clean).norm() > clean.norm()
    out = agg.aggregate(x, torch.ones(10)).delta
    assert (out - clean).norm() < 0.5 * clean.norm()


@pytest.mark.parametrize("agg", _ROBUST[2:], ids=lambda a: a.name)
def test_filtering_rules_report_the_outlier_as_not_selected(agg):
    x = _benign()
    x[0] = -50.0 * x[1]
    result = agg.aggregate(x, torch.ones(10))
    assert result.selected is not None and not bool(result.selected[0])


def test_filtering_rules_drop_a_sign_flipped_cluster():
    g = torch.Generator().manual_seed(1)
    base = torch.rand(200, generator=g) + 0.5  # all-positive mean: sign statistics separate the flip
    x = base + 0.1 * torch.randn(10, 200, generator=g)
    x[:3] = -x[:3]
    for agg in (MultiKrum(f_ratio=0.3), SignGuard(seed=0), DnC(f_ratio=0.3, b=50, seed=0)):
        sel = agg.aggregate(x, torch.ones(10)).selected
        assert not bool(sel[:3].any()) and bool(sel[3:].all()), agg.name


def test_multi_krum_requires_enough_clients():
    with pytest.raises(ValueError):
        MultiKrum(f=2).aggregate(torch.randn(5, 4), torch.ones(5))


def test_aggregators_do_not_mutate_inputs():
    for agg in _ROBUST:
        x = _benign()
        x[0] = -50.0 * x[1]
        before = x.clone()
        agg.aggregate(x, torch.ones(10))
        assert torch.equal(x, before), agg.name


def test_build_aggregator_defaults_and_errors():
    agg = build_aggregator({"name": "trimmed_mean"}, assumed_malicious_ratio=0.3)
    assert isinstance(agg, TrimmedMean) and agg.trim_ratio == 0.3
    assert build_aggregator({"name": "multi_krum"}, assumed_malicious_ratio=0.1).f_ratio == 0.1
    assert build_aggregator({"name": "dnc"}, assumed_malicious_ratio=0.25).f_ratio == 0.25
    for name in ("median", "signguard", "dnc"):
        assert build_aggregator({"name": name}, seed=3).name == name
    with pytest.raises(ValueError):
        build_aggregator({"name": "median", "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        build_aggregator({"name": "msguard"})
