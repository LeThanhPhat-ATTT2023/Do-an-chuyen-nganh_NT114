import pytest
import torch

from graphslm_ids.offline.federated.aggregators import FedAvg, build_aggregator, weighted_mean


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
