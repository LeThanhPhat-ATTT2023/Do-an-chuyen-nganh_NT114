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
