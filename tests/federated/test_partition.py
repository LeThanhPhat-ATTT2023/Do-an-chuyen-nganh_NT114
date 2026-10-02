import json

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


def test_json_round_trip_dirichlet_through_real_json():
    labels = _labels()
    part = partition_dirichlet(np.arange(labels.size), labels, 4, 0.5, seed=2, min_flows_per_client=5)
    back = ClientPartition.from_json(json.loads(json.dumps(part.to_json())))
    assert back.scheme == "dirichlet" and back.seed == 2
    assert isinstance(back.alpha, float) and back.alpha == 0.5
    assert all(np.array_equal(x, y) for x, y in zip(part.client_flow_ids, back.client_flow_ids))


def test_dirichlet_through_make_partition_is_deterministic_and_seed_sensitive():
    labels = _labels()
    train = np.arange(labels.size)
    cfg = {
        "scheme": "dirichlet",
        "alpha": 0.1,
        "num_clients": 4,
        "min_flows_per_client": 5,
        "max_flows_per_client": 60,
        "seed": 7,
    }
    a = make_partition(train, labels, cfg)
    b = make_partition(train, labels, cfg)
    assert all(np.array_equal(x, y) for x, y in zip(a.client_flow_ids, b.client_flow_ids))
    assert max(a.sizes()) <= 60

    p7 = partition_dirichlet(train, labels, 4, 0.1, seed=7, min_flows_per_client=5)
    p8 = partition_dirichlet(train, labels, 4, 0.1, seed=8, min_flows_per_client=5)
    assert not all(np.array_equal(x, y) for x, y in zip(p7.client_flow_ids, p8.client_flow_ids))


def test_subsample_rejects_non_positive_cap():
    labels = _labels()
    part = partition_iid(np.arange(labels.size), 2, seed=0)
    for bad in (0, -5):
        with pytest.raises(ValueError, match="max_flows_per_client"):
            subsample_partition(part, labels, bad, seed=0)


def test_iid_rejects_more_clients_than_flows():
    with pytest.raises(ValueError, match="num_clients"):
        partition_iid(np.arange(3), 5, seed=0)


def test_partitioners_reject_duplicate_train_ids():
    labels = _labels(40)
    dup = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9])
    with pytest.raises(ValueError, match="duplicate"):
        partition_iid(dup, 2, seed=0)
    with pytest.raises(ValueError, match="duplicate"):
        partition_dirichlet(dup, labels, 2, 0.5, seed=0)


def test_make_partition_rejects_cap_below_dirichlet_minimum():
    labels = _labels()
    cfg = {
        "scheme": "dirichlet",
        "num_clients": 3,
        "seed": 0,
        "min_flows_per_client": 20,
        "max_flows_per_client": 10,
    }
    with pytest.raises(ValueError, match="max_flows_per_client"):
        make_partition(np.arange(labels.size), labels, cfg)
