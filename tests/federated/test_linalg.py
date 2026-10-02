import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.linalg import (
    coordinate_median,
    coordinate_trimmed_mean,
    cosine_to,
    mean_excluding_self,
    median_excluding_self,
    pairwise_cosine,
    pairwise_sq_dists,
    robust_z,
    top_right_singular_vector,
)


def test_pairwise_sq_dists_matches_cdist():
    torch.manual_seed(0)
    x = torch.randn(5, 7)
    ref = torch.cdist(x.double(), x.double()).pow(2)
    assert torch.allclose(pairwise_sq_dists(x), ref, atol=1e-4)


def test_pairwise_cosine_and_cosine_to():
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    c = pairwise_cosine(x)
    assert c[0, 1].item() == pytest.approx(0.0, abs=1e-9)
    assert c[0, 2].item() == pytest.approx(-1.0)
    assert torch.allclose(c.diagonal(), torch.ones(3, dtype=torch.float64))
    got = cosine_to(x, torch.tensor([1.0, 0.0]))
    assert torch.allclose(got, torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64))


def test_coordinate_median_and_trimmed_mean_are_chunk_invariant():
    torch.manual_seed(1)
    x = torch.randn(6, 11)
    assert torch.allclose(coordinate_median(x, chunk=4), x.sort(0).values[2:4].mean(0))
    x5 = torch.randn(5, 11)
    assert torch.allclose(coordinate_median(x5, chunk=3), x5.median(0).values)
    assert torch.allclose(coordinate_trimmed_mean(x, 1, chunk=4), x.sort(0).values[1:5].mean(0))
    with pytest.raises(ValueError):
        coordinate_trimmed_mean(x, 3)


def test_robust_z_and_offdiag_helpers():
    z = robust_z([1.0, 2.0, 3.0, 100.0])
    assert z[3] > 10 and abs(z[1]) < 1
    m = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    assert median_excluding_self(m).tolist() == pytest.approx([0.3, 0.4, 0.5])
    assert mean_excluding_self(m).tolist() == pytest.approx([0.3, 0.4, 0.5])
    assert median_excluding_self(np.ones((1, 1))).tolist() == [1.0]


def test_top_right_singular_vector_finds_dominant_axis():
    x = torch.zeros(4, 3)
    x[:, 1] = torch.tensor([3.0, -3.0, 2.0, -2.0])
    v = top_right_singular_vector(x)
    assert abs(float(v[1])) == pytest.approx(1.0, abs=1e-5)
