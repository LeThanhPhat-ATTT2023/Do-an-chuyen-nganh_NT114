from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from graphslm_ids.offline.training.on_disk_graph_store import gather_csr_neighbors


def _random_csr(num_src, avg_degree, seed):
    rng = np.random.default_rng(seed)
    degrees = rng.integers(0, 2 * avg_degree + 1, size=num_src)
    indptr = np.empty((num_src + 1,), dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degrees, out=indptr[1:])
    num_edges = int(indptr[-1])
    indices = rng.integers(0, num_src, size=num_edges).astype(np.int64)
    attr = rng.random(num_edges).astype(np.float32)
    return indptr, indices, attr


def test_gather_csr_torch_matches_numpy_when_fanout_none():
    from graphslm_ids.offline.training.gpu_sampling import gather_csr_neighbors_torch

    indptr, indices, attr = _random_csr(30, 4, seed=0)
    src = np.array([3, 7, 11, 29, 0], dtype=np.int64)
    n_li, n_nb, n_w = gather_csr_neighbors(indptr, indices, attr, src, fanout=None)

    dev = torch.device("cpu")
    t_li, t_nb, t_w = gather_csr_neighbors_torch(
        torch.from_numpy(indptr).to(dev),
        torch.from_numpy(indices).to(dev),
        torch.from_numpy(attr).to(dev),
        torch.from_numpy(src).to(dev),
        fanout=None,
    )
    assert np.array_equal(t_li.cpu().numpy(), n_li)
    assert np.array_equal(t_nb.cpu().numpy(), n_nb)
    np.testing.assert_allclose(t_w.cpu().numpy(), n_w, rtol=1e-5)


def test_gather_csr_torch_fanout_caps_each_row():
    from graphslm_ids.offline.training.gpu_sampling import gather_csr_neighbors_torch

    indptr, indices, attr = _random_csr(40, 6, seed=1)
    src = np.arange(40, dtype=np.int64)
    dev = torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(123)
    t_li, t_nb, t_w = gather_csr_neighbors_torch(
        torch.from_numpy(indptr).to(dev),
        torch.from_numpy(indices).to(dev),
        torch.from_numpy(attr).to(dev),
        torch.from_numpy(src).to(dev),
        fanout=3, generator=gen,
    )
    degrees = (indptr[src + 1] - indptr[src]).astype(np.int64)
    expected_counts = np.minimum(degrees, 3)
    assert np.array_equal(np.diff(t_li.cpu().numpy()), expected_counts)
    li = t_li.cpu().numpy(); nb = t_nb.cpu().numpy()
    for row, node in enumerate(src.tolist()):
        row_nb = set(nb[li[row]:li[row + 1]].tolist())
        full = set(indices[indptr[node]:indptr[node + 1]].tolist())
        assert row_nb <= full


def test_gather_csr_torch_empty_inputs():
    from graphslm_ids.offline.training.gpu_sampling import gather_csr_neighbors_torch

    dev = torch.device("cpu")
    indptr = torch.zeros(11, dtype=torch.int64)
    indices = torch.empty(0, dtype=torch.int64)
    attr = torch.empty(0, dtype=torch.float32)
    src = torch.arange(10, dtype=torch.int64)
    li, nb, w = gather_csr_neighbors_torch(indptr, indices, attr, src, fanout=None)
    assert nb.numel() == 0 and w.numel() == 0
    assert li.shape == (11,)


def test_torch_local_map_matches_numpy_localmap():
    from graphslm_ids.offline.training.neighbor_sampling import _LocalMap
    from graphslm_ids.offline.training.gpu_sampling import TorchLocalMap

    dev = torch.device("cpu")
    batches = [
        np.array([5, 1, 5, 9], dtype=np.int64),
        np.array([1, 2, 9, 7], dtype=np.int64),
        np.array([7, 7, 3], dtype=np.int64),
    ]
    np_map = _LocalMap()
    t_map = TorchLocalMap(device=dev)
    for b in batches:
        np_new = np_map.add(b)
        t_new = t_map.add(torch.from_numpy(b).to(dev))
        assert np.array_equal(t_new.cpu().numpy(), np_new), f"add mismatch on batch {b.tolist()}"

    assert np.array_equal(t_map.globals_array().cpu().numpy(), np_map.globals_array())
    q = np.array([9, 100, 3, 5], dtype=np.int64)
    assert np.array_equal(
        t_map.lookup(torch.from_numpy(q).to(dev)).cpu().numpy(),
        np_map.lookup(q),
    )


def test_torch_local_map_empty_calls():
    from graphslm_ids.offline.training.gpu_sampling import TorchLocalMap

    dev = torch.device("cpu")
    m = TorchLocalMap(device=dev)
    out = m.add(torch.empty(0, dtype=torch.int64, device=dev))
    assert out.numel() == 0
    assert m.count == 0
    look = m.lookup(torch.tensor([1, 2], dtype=torch.int64, device=dev))
    assert look.tolist() == [-1, -1]
    assert m.globals_array().numel() == 0


@pytest.fixture
def tiny_backend():
    from graphslm_ids.offline.training.hetero_graph_artifact import HeteroGraphArtifact
    from graphslm_ids.offline.training.neighbor_sampling import InMemoryNeighborBackend

    nf = {
        "flow": np.random.RandomState(0).rand(4, 5).astype(np.float32),
        "packet": (np.arange(6 * 2323).reshape(6, 2323) % 50).astype(np.float32),
        "technique": np.random.RandomState(1).rand(3, 8).astype(np.float32),
        "tactic": np.zeros((2, 1), dtype=np.float32),
        "host": np.zeros((2, 4), dtype=np.float32),
    }
    ei = {
        ("flow", "contains", "packet"): np.array([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]], dtype=np.int64),
        ("packet", "next_packet", "packet"): np.array([[0, 1], [1, 2]], dtype=np.int64),
        ("technique", "belongs_to_tactic", "tactic"): np.array([[0, 1], [0, 1]], dtype=np.int64),
    }
    ea = {k: np.ones((v.shape[1],), dtype=np.float32) for k, v in ei.items()}
    art = HeteroGraphArtifact(
        node_features=nf, edge_index=ei, edge_attr=ea,
        flow_y=np.array([0, 1, 0, 1], dtype=np.int64), metadata={},
    )
    return InMemoryNeighborBackend(art)


def test_torch_sampler_matches_numpy_sampler_fanout_none(tiny_backend):
    from graphslm_ids.offline.training.neighbor_sampling import HeteroNeighborSampler
    from graphslm_ids.offline.training.gpu_sampling import (
        GpuNeighborBackend, TorchHeteroNeighborSampler,
    )

    np_sampler = HeteroNeighborSampler(
        backend=tiny_backend, hops=2, fanouts=None,
        always_include_all_techniques=False, always_include_all_tactics=False,
        standardize_flow_features=False,
    )
    np_batch = np_sampler.sample([0, 1])

    gpu_backend = GpuNeighborBackend(tiny_backend, device=torch.device("cpu"))
    t_sampler = TorchHeteroNeighborSampler(
        backend=gpu_backend, hops=2, fanouts=None,
        always_include_all_techniques=False, always_include_all_tactics=False,
        standardize_flow_features=False,
    )
    t_batch = t_sampler.sample([0, 1])

    # Same packet/flow node sets (first-occurrence order, deterministic with fanout=None).
    assert np.array_equal(
        t_batch.local_to_global["packet"].cpu().numpy(),
        np_batch.local_to_global["packet"],
    )
    assert np.array_equal(
        t_batch.local_to_global["flow"].cpu().numpy(),
        np_batch.local_to_global["flow"],
    )


def test_gpu_sampling_path_produces_batch_with_packet_store(tiny_backend):
    from graphslm_ids.offline.training.gpu_sampling import (
        GpuNeighborBackend, TorchHeteroNeighborSampler,
    )
    from graphslm_ids.offline.training.feature_store import (
        ArrayPacketSource, TieredFeatureStore,
    )
    from graphslm_ids.offline.training.train_hgt_flow_classifier import to_torch_batch

    dev = torch.device("cpu")
    gpu_backend = GpuNeighborBackend(tiny_backend, device=dev)
    sampler = TorchHeteroNeighborSampler(
        backend=gpu_backend, hops=2, fanouts={"contains": 4, "next_packet": 2},
        always_include_all_techniques=False, always_include_all_tactics=False,
    )
    batch = sampler.sample([0, 1])

    packet_x = tiny_backend.artifact.node_features["packet"].astype(np.float16)
    store = TieredFeatureStore(
        source=ArrayPacketSource(packet_x), device=dev,
        freq_order=np.arange(6, dtype=np.int64), capacity=6, cache_dtype="float32",
    )
    edge_types = list(batch.edge_index.keys())
    node_tensors, *_ = to_torch_batch(
        batch, edge_types, dev,
        use_semantic_edge_weights=False, packet_store=store,
    )
    pkt_ids = batch.local_to_global["packet"]
    assert node_tensors["packet"].shape[0] == int(pkt_ids.numel())
    assert node_tensors["packet"].dtype == torch.float32
