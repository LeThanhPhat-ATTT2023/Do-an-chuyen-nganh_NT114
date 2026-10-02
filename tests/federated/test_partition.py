import json

import numpy as np
import pytest

from graphslm_ids.offline.federated.partition import (
    ClientPartition,
    _attempt_rng,
    _dirichlet_attempt,
    _stratified_take,
    _subsample_rng,
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


def test_dirichlet_raises_when_minimum_is_unlucky_but_feasible():
    # 40 flows / 4 clients / min 10 is feasible only as a perfectly even split, which no
    # Dirichlet(0.5) draw for this seed produces, so the retries are exhausted.
    labels = _labels(40)
    with pytest.raises(ValueError, match="alpha"):
        partition_dirichlet(np.arange(40), labels, 4, 0.5, seed=0, min_flows_per_client=10)


def test_dirichlet_fails_fast_on_infeasible_configs():
    labels = _labels(40)
    train = np.arange(40)
    # 4 clients * 30 flows = 120 > 40 train flows: no amount of retrying can help.
    with pytest.raises(ValueError) as excinfo:
        partition_dirichlet(train, labels, 4, 0.5, seed=0, min_flows_per_client=30)
    msg = str(excinfo.value)
    assert "num_clients" in msg and "min_flows_per_client" in msg and "40" in msg
    assert "alpha" not in msg
    with pytest.raises(ValueError, match="min_flows_per_client") as excinfo:
        partition_dirichlet(train, labels, 4, 0.5, seed=0, min_flows_per_client=0)
    assert ">= 1" in str(excinfo.value)
    with pytest.raises(ValueError, match="min_flows_per_client"):
        partition_dirichlet(train, labels, 4, 0.5, seed=0, min_flows_per_client=-3)
    with pytest.raises(ValueError) as excinfo:
        partition_dirichlet(np.empty((0,), dtype=np.int64), labels, 4, 0.5, seed=0)
    assert "alpha" not in str(excinfo.value)


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

    c = make_partition(train, labels, {**cfg, "seed": 8})
    assert not all(np.array_equal(x, y) for x, y in zip(a.client_flow_ids, c.client_flow_ids))


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


def test_subsample_stream_is_independent_of_every_partitioner_stream():
    seed = 7
    first = lambda rng: tuple(rng.integers(0, 2**62, size=8).tolist())  # noqa: E731
    sub = first(_subsample_rng(seed))
    assert sub != first(np.random.default_rng(seed))
    for attempt in range(100):
        assert sub != first(_attempt_rng(seed, attempt)), f"collides with attempt {attempt}"
    # Attempt 0 must stay bit-identical to the original default_rng(seed) stream.
    assert first(_attempt_rng(seed, 0)) == first(np.random.default_rng(seed))


def test_dirichlet_retry_path_uses_next_attempt_stream():
    # 200 flows, 4 classes, alpha=0.3, min=25, seed=7: attempt 0 is too skewed, attempt 1 passes.
    labels = _labels(200)
    train = np.arange(200)
    y = labels[train]
    classes = np.unique(y)
    first = _dirichlet_attempt(train, y, classes, 4, 0.3, _attempt_rng(7, 0))
    second = _dirichlet_attempt(train, y, classes, 4, 0.3, _attempt_rng(7, 1))
    assert min(c.shape[0] for c in first) < 25 <= min(c.shape[0] for c in second)
    part = partition_dirichlet(train, labels, 4, 0.3, seed=7, min_flows_per_client=25)
    assert all(np.array_equal(x, y_) for x, y_ in zip(part.client_flow_ids, second))
    assert np.array_equal(part.all_flow_ids(), train)


def test_subsample_after_dirichlet_retry_is_deterministic_and_capped():
    labels = _labels(200)
    cfg = {
        "scheme": "dirichlet",
        "alpha": 0.3,
        "num_clients": 4,
        "min_flows_per_client": 25,
        "max_flows_per_client": 30,
        "seed": 7,
    }
    a = make_partition(np.arange(200), labels, cfg)
    b = make_partition(np.arange(200), labels, cfg)
    assert all(np.array_equal(x, y) for x, y in zip(a.client_flow_ids, b.client_flow_ids))
    assert all(25 <= s <= 30 for s in a.sizes())


@pytest.mark.parametrize("cap", [5, 6, 20, 77, 100, 150, 299])  # 5 == #classes, 299 == len(ids) - 1
def test_stratified_take_hits_cap_exactly_keeps_classes_and_is_proportional(cap):
    rng_labels = np.random.default_rng(1)
    labels = rng_labels.choice(5, size=300, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    ids = np.arange(0, 300)
    out = _stratified_take(ids, labels, cap, np.random.default_rng(0))
    assert out.dtype == np.int64
    assert out.shape[0] == cap
    assert np.array_equal(out, np.sort(out)) and np.unique(out).size == out.size
    assert set(out.tolist()) <= set(ids.tolist())
    classes, counts = np.unique(labels[ids], return_counts=True)
    got = np.bincount(labels[out], minlength=classes.size)
    assert (got >= 1).all()  # every class kept (cap >= #classes)
    exact = counts * cap / ids.size
    if (exact >= 1).all():
        assert (np.abs(got - exact) < 1).all()


def test_stratified_take_exact_for_odd_class_sizes_and_edge_caps():
    labels = np.array([0] * 37 + [1] * 29 + [2] * 11 + [3] * 3)
    ids = np.arange(labels.size)
    for cap in (4, 5, 17, 40, 79, 80):
        out = _stratified_take(ids, labels, cap, np.random.default_rng(3))
        assert out.shape[0] == cap
        assert set(labels[out].tolist()) == {0, 1, 2, 3}
    # cap == len(ids): nothing to drop.
    assert np.array_equal(_stratified_take(ids, labels, ids.size, np.random.default_rng(3)), ids)
    # Tiny classes next to a huge one: floor-of-1 bumps overshoot the cap and must be trimmed.
    skew = np.array([0] * 1000 + [1, 2, 3])
    out = _stratified_take(np.arange(skew.size), skew, 4, np.random.default_rng(0))
    assert out.shape[0] == 4 and set(skew[out].tolist()) == {0, 1, 2, 3}


def test_stratified_take_with_cap_below_class_count_drops_classes_but_keeps_k():
    labels = np.array([0] * 50 + [1] * 30 + [2] * 15 + [3] * 5)
    ids = np.arange(labels.size)
    for cap in (1, 2, 3):
        out = _stratified_take(ids, labels, cap, np.random.default_rng(0))
        assert out.shape[0] == cap and np.unique(out).size == cap


def test_subsample_partition_fills_cap_exactly():
    labels = _labels(1000, c=6)
    part = partition_iid(np.arange(1000), 3, seed=0)  # ~333 flows per client
    for cap in (6, 100, 200, 332):
        sub = subsample_partition(part, labels, cap, seed=0)
        assert sub.sizes() == [cap] * 3
        for orig, small in zip(part.client_flow_ids, sub.client_flow_ids):
            assert set(labels[small].tolist()) == set(labels[orig].tolist())


def test_make_partition_rejects_non_positive_cap_for_both_schemes():
    labels = _labels()
    train = np.arange(labels.size)
    for scheme in ("iid", "dirichlet"):
        for bad in (0, -5):
            cfg = {"scheme": scheme, "num_clients": 3, "seed": 0, "max_flows_per_client": bad}
            with pytest.raises(ValueError, match=r"max_flows_per_client must be >= 1, got"):
                make_partition(train, labels, cfg)
    # None means "no cap"; a missing key too.
    cfg = {"scheme": "iid", "num_clients": 3, "seed": 0, "max_flows_per_client": None}
    assert sum(make_partition(train, labels, cfg).sizes()) == train.size


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
