import numpy as np
import pytest

from graphslm_ids.offline.federated.partition import (
    ClientPartition,
    make_partition,
    partition_dirichlet,
    partition_iid,
    subsample_partition,
)


def _labels(n=600, c=4, seed=0):
    return np.random.default_rng(seed).integers(0, c, n)


def test_iid_is_deterministic_disjoint_and_complete():
    train = np.arange(100, 400)
    a = partition_iid(train, 5, seed=1)
    b = partition_iid(train, 5, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a.client_flow_ids, b.client_flow_ids))
    assert np.array_equal(a.all_flow_ids(), train)
    assert sum(a.sizes()) == train.size
    assert max(a.sizes()) - min(a.sizes()) <= 1


def test_dirichlet_small_alpha_is_more_skewed_than_large_alpha():
    labels = _labels()
    train = np.arange(labels.size)

    def skew(alpha):
        part = partition_dirichlet(train, labels, 5, alpha, seed=3, min_flows_per_client=1)
        hist = part.class_histogram(labels, 4).astype(float)
        frac = hist / hist.sum(axis=1, keepdims=True)
        return float(frac.max(axis=1).mean())

    assert skew(0.1) > skew(100.0)


def test_dirichlet_keeps_a_disjoint_cover_of_train():
    labels = _labels()
    train = np.arange(0, labels.size, 2)
    part = partition_dirichlet(train, labels, 4, 0.5, seed=0, min_flows_per_client=5)
    assert np.array_equal(part.all_flow_ids(), np.sort(train))
    assert min(part.sizes()) >= 5


def test_dirichlet_raises_when_minimum_is_unreachable():
    labels = _labels(40)
    with pytest.raises(ValueError, match="alpha"):
        partition_dirichlet(np.arange(40), labels, 4, 0.5, seed=0, min_flows_per_client=30)


def test_subsample_keeps_every_class_and_respects_cap():
    labels = _labels()
    part = partition_iid(np.arange(labels.size), 2, seed=0)
    sub = subsample_partition(part, labels, 40, seed=0)
    assert max(sub.sizes()) <= 40
    for orig, small in zip(part.client_flow_ids, sub.client_flow_ids):
        assert set(small.tolist()) <= set(orig.tolist())
        assert set(labels[small].tolist()) == set(labels[orig].tolist())


def test_make_partition_dispatch_and_errors():
    labels = _labels()
    part = make_partition(np.arange(labels.size), labels, {"scheme": "iid", "num_clients": 3, "seed": 0, "max_flows_per_client": 50})
    assert part.num_clients == 3 and max(part.sizes()) <= 50
    with pytest.raises(ValueError):
        make_partition(np.arange(10), np.zeros(10, dtype=int), {"scheme": "bogus"})


def test_json_round_trip():
    part = partition_iid(np.arange(20), 3, seed=0)
    back = ClientPartition.from_json(part.to_json())
    assert back.scheme == "iid" and back.seed == 0
    assert all(np.array_equal(x, y) for x, y in zip(part.client_flow_ids, back.client_flow_ids))
