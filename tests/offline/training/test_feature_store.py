from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch")


def _load_downcast():
    script = Path(__file__).resolve().parents[3] / "scripts" / "tools" / "downcast_packet_x.py"
    spec = importlib.util.spec_from_file_location("downcast_packet_x_tool", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.downcast_packet_x


def test_downcast_packet_x_rewrites_npz_to_float16(tmp_path):
    downcast_packet_x = _load_downcast()

    src = tmp_path / "graph.npz"
    np.savez(
        str(src),
        flow_x=np.zeros((2, 4), dtype=np.float32),
        flow_y=np.array([0, 1], dtype=np.int64),
        packet_x=(np.arange(3 * 2323).reshape(3, 2323) % 100).astype(np.float32),
        host_x=np.zeros((1, 4), dtype=np.float32),
        technique_x=np.zeros((5, 8), dtype=np.float32),
    )
    out = tmp_path / "graph_fp16.npz"
    downcast_packet_x(src, out)

    with np.load(out) as loaded:
        assert loaded["packet_x"].dtype == np.float16
        assert loaded["flow_x"].dtype == np.float32
        assert loaded["flow_y"].tolist() == [0, 1]


def test_array_packet_source_gather(tmp_path):
    from graphslm_ids.offline.training.feature_store import ArrayPacketSource

    data = np.arange(5 * 3).reshape(5, 3).astype(np.float16)
    src = ArrayPacketSource(data)
    assert src.num_rows == 5
    assert src.dim == 3
    got = src.gather(np.array([4, 0, 2], dtype=np.int64))
    np.testing.assert_array_equal(got, data[[4, 0, 2]])


def test_memmap_packet_source_gather(tmp_path):
    from graphslm_ids.offline.training.feature_store import MemmapPacketSource

    data = np.arange(6 * 4).reshape(6, 4).astype(np.float16)
    bin_path = tmp_path / "packet_x.f16.bin"
    data.tofile(bin_path)

    src = MemmapPacketSource(bin_path, num_rows=6, dim=4, dtype="float16")
    assert src.num_rows == 6
    got = src.gather(np.array([5, 1], dtype=np.int64))
    np.testing.assert_array_equal(got, data[[5, 1]])


def test_compute_cache_capacity_cpu_is_zero():
    from graphslm_ids.offline.training.feature_store import compute_cache_capacity

    K = compute_cache_capacity(
        device=torch.device("cpu"),
        num_rows=1000,
        row_bytes=4646,
        free_bytes_override=None,
        model_reserve_bytes=0,
        cache_fraction=0.6,
    )
    assert K == 0


def test_compute_cache_capacity_caps_at_num_rows():
    from graphslm_ids.offline.training.feature_store import compute_cache_capacity

    K = compute_cache_capacity(
        device=torch.device("cpu"),
        num_rows=500,
        row_bytes=4646,
        free_bytes_override=10 * 1024**3,
        model_reserve_bytes=0,
        cache_fraction=1.0,
    )
    assert K == 500


def test_compute_cache_capacity_partial():
    from graphslm_ids.offline.training.feature_store import compute_cache_capacity

    free = 2 * 4646 * 1000  # *0.5 -> 1000 rows
    K = compute_cache_capacity(
        device=torch.device("cpu"),
        num_rows=100_000,
        row_bytes=4646,
        free_bytes_override=free,
        model_reserve_bytes=0,
        cache_fraction=0.5,
    )
    assert K == 1000


def _make_store(num_rows, dim, K, device="cpu"):
    from graphslm_ids.offline.training.feature_store import (
        ArrayPacketSource,
        TieredFeatureStore,
    )

    data = (np.arange(num_rows * dim).reshape(num_rows, dim) % 97).astype(np.float16)
    source = ArrayPacketSource(data)
    freq_order = np.arange(num_rows, dtype=np.int64)[::-1].copy()  # row N-1 hottest
    store = TieredFeatureStore(
        source=source,
        device=torch.device(device),
        freq_order=freq_order,
        capacity=K,
        cache_dtype="float32",  # cpu path
    )
    return data, store


def test_tiered_store_full_cache_all_hits():
    data, store = _make_store(num_rows=8, dim=3, K=8)
    out = store.gather(np.array([7, 0, 3], dtype=np.int64))
    assert out.device.type == "cpu"
    np.testing.assert_allclose(out.cpu().numpy(), data[[7, 0, 3]].astype(np.float32))


def test_tiered_store_zero_cache_all_misses():
    data, store = _make_store(num_rows=8, dim=3, K=0)
    out = store.gather(np.array([2, 5], dtype=np.int64))
    np.testing.assert_allclose(out.cpu().numpy(), data[[2, 5]].astype(np.float32))


def test_tiered_store_partial_cache_mixed():
    # K=4 -> the 4 hottest (freq_order[:4] = rows 7,6,5,4) are cached, rest miss.
    data, store = _make_store(num_rows=8, dim=3, K=4)
    ids = np.array([7, 0, 6, 1], dtype=np.int64)  # hit, miss, hit, miss
    out = store.gather(ids)
    np.testing.assert_allclose(out.cpu().numpy(), data[ids].astype(np.float32))


def test_compute_access_frequency_deterministic_and_counts():
    from graphslm_ids.offline.training.feature_store import compute_access_frequency

    class FakeSampler:
        def sample(self, seed_flow_ids):
            import numpy as _np
            base = int(seed_flow_ids[0])
            class B:
                local_to_global = {"packet": _np.array([base, base + 1], dtype=_np.int64)}
            return B()

    seeds = np.array([0, 2, 4, 6], dtype=np.int64)
    f1 = compute_access_frequency(
        sampler=FakeSampler(), seed_flow_ids=seeds, num_packets=10,
        batch_size=2, n_warmup_batches=2, seed=42,
    )
    f2 = compute_access_frequency(
        sampler=FakeSampler(), seed_flow_ids=seeds, num_packets=10,
        batch_size=2, n_warmup_batches=2, seed=42,
    )
    assert f1.shape == (10,)
    np.testing.assert_array_equal(f1, f2)
    assert int(f1.sum()) > 0
