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
    robust_center_scale,
    robust_z,
    row_norms,
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


def test_row_norms_and_cosine_to_are_precise_at_large_d():
    torch.manual_seed(0)
    x = torch.randn(3, 1_500_000) + 0.5
    norms = row_norms(x)
    assert norms.dtype == torch.float64
    assert torch.allclose(norms, x.double().norm(dim=1), rtol=1e-6)
    assert cosine_to(x, x[0])[0].item() == pytest.approx(1.0, abs=1e-6)


def test_zero_norm_rows_give_zero_cosine_not_nan():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    c = pairwise_cosine(x)
    assert not torch.isnan(c).any()
    assert c[0].tolist() == [0.0, 0.0]
    assert c[:, 0].tolist() == [0.0, 0.0]
    assert c[1, 1].item() == pytest.approx(1.0)
    got = cosine_to(x, torch.tensor([1.0, 0.0]))
    assert got.tolist() == pytest.approx([0.0, 1.0])
    assert cosine_to(x, torch.zeros(2)).tolist() == [0.0, 0.0]
    assert row_norms(x).tolist() == [0.0, 1.0]


def test_robust_center_scale_respects_axis_and_floor():
    v = np.array([[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [3.0, 5.0, 9.0], [10.0, 5.0, 1.0]])
    center, scale = robust_center_scale(v, axis=0)
    assert center.shape == (3,) and scale.shape == (3,)
    assert center.tolist() == pytest.approx([2.5, 5.0, 3.0])
    assert scale[1] == 1e-6  # constant column: MAD is 0, so the floor applies exactly
    assert scale[0] == pytest.approx(1.4826 * 1.0)

    m = np.array([[1.0, 2.0, 10.0], [4.0, 4.5, 9.0], [0.0, 7.0, 7.5]])
    c1, s1 = robust_center_scale(m, axis=1)
    assert c1.shape == (3,) and s1.shape == (3,)
    for i in range(3):
        ci, si = robust_center_scale(m[i])
        assert c1[i] == pytest.approx(float(ci))
        assert s1[i] == pytest.approx(float(si))

    c0, s0 = robust_center_scale([1.0, 2.0, 3.0, 100.0])
    assert np.ndim(c0) == 0 and np.ndim(s0) == 0


def test_pairwise_sq_dists_is_exactly_symmetric_with_zero_diagonal():
    torch.manual_seed(2)
    d = pairwise_sq_dists(torch.randn(6, 33))
    assert torch.equal(d, d.T)
    assert torch.equal(d.diagonal(), torch.zeros(6, dtype=torch.float64))


def test_top_right_singular_vector_preserves_float64_dtype():
    x = torch.zeros(4, 3, dtype=torch.float64)
    x[:, 1] = torch.tensor([3.0, -3.0, 2.0, -2.0], dtype=torch.float64)
    v = top_right_singular_vector(x)
    assert v.dtype == torch.float64
    assert (x @ v).shape == (4,)
    assert abs(float(v[1])) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("bad_chunk", [0, -1])
def test_chunked_helpers_reject_non_positive_chunk(bad_chunk):
    x = torch.randn(4, 5)
    with pytest.raises(ValueError):
        coordinate_median(x, chunk=bad_chunk)
    with pytest.raises(ValueError):
        coordinate_trimmed_mean(x, 1, chunk=bad_chunk)
    with pytest.raises(ValueError):
        row_norms(x, chunk=bad_chunk)


def test_coordinate_reduction_edge_cases():
    torch.manual_seed(3)
    with pytest.raises(ValueError):
        coordinate_median(torch.empty(0, 5))
    x1 = torch.randn(1, 5)
    assert torch.equal(coordinate_median(x1), x1[0])
    x = torch.randn(5, 9)
    assert torch.allclose(coordinate_trimmed_mean(x, 0, chunk=4), x.mean(0))
