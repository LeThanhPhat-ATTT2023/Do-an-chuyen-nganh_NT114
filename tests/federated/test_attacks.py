import pytest
import torch

from graphslm_ids.offline.federated.attacks import (
    ALIE,
    AttackContext,
    MinMax,
    MinSum,
    ScaleSign,
    SignFlip,
    alie_z,
    build_attack,
    krum_sign_counts,
    match_sign_counts,
    search_gamma,
)
from graphslm_ids.offline.federated.linalg import pairwise_sq_dists


def _ctx(benign, honest, knowledge="omniscient"):
    return AttackContext(
        honest=honest,
        benign=benign,
        num_participants=benign.shape[0] + honest.shape[0],
        knowledge=knowledge,
    )


def test_alie_z_matches_paper_example():
    assert alie_z(50, 24) == pytest.approx(1.75, abs=0.01)
    assert alie_z(10, 2) == pytest.approx(0.2533, abs=1e-3)


def test_sign_flip_negates_honest_deltas():
    honest = torch.randn(2, 5)
    out = SignFlip(scale=2.0).craft(_ctx(torch.randn(3, 5), honest), 2)
    assert torch.allclose(out, -2.0 * honest)


def test_alie_uses_benign_mean_and_std():
    benign = torch.randn(8, 6)
    out = ALIE(z=1.5).craft(_ctx(benign, torch.zeros(2, 6)), 2)
    expected = benign.mean(0) + 1.5 * benign.std(0, unbiased=False)
    assert out.shape == (2, 6)
    assert torch.allclose(out[0], expected) and torch.allclose(out[1], expected)


def test_partial_knowledge_uses_attacker_rows():
    benign = torch.randn(8, 6)
    honest = torch.randn(3, 6) + 5.0
    out = ALIE(z=0.0).craft(_ctx(benign, honest, "partial"), 3)
    assert torch.allclose(out[0], honest.mean(0))


def test_search_gamma_finds_threshold():
    assert search_gamma(lambda g: g <= 3.7, 10.0, 1e-6) == pytest.approx(3.7, abs=1e-4)


@pytest.mark.parametrize("cls", [MinMax, MinSum])
def test_gamma_attacks_sit_on_their_constraint_boundary(cls):
    torch.manual_seed(0)
    benign = torch.randn(10, 40)
    attack = cls(perturbation="std", gamma_init=10.0, tol=1e-6)
    v = attack.craft(_ctx(benign, torch.zeros(0, 40)), 3)[0]
    d = pairwise_sq_dists(benign)

    def measure(vec):
        dist = (benign - vec).pow(2).sum(1).double()
        return float(dist.max()) if cls is MinMax else float(dist.sum())

    bound = float(d.max()) if cls is MinMax else float(d.sum(1).max())
    assert attack.last_gamma > 0
    assert measure(v) <= bound * (1 + 1e-4)
    mu, p = benign.mean(0), -benign.std(0, unbiased=False)
    assert measure(mu + 1.01 * attack.last_gamma * p) > bound


def test_match_sign_counts_handles_both_directions():
    out = match_sign_counts(torch.tensor([3.0, 1.0, -2.0, -0.5, 0.0]), 1, 3)
    assert int((out > 0).sum()) == 1 and int((out < 0).sum()) == 3
    out2 = match_sign_counts(torch.tensor([-1.0, -2.0, 0.0, 0.0]), 2, 1)
    assert int((out2 > 0).sum()) == 2 and int((out2 < 0).sum()) == 1


def test_scale_sign_matches_target_sign_counts():
    torch.manual_seed(1)
    benign = torch.randn(8, 200)
    honest = torch.randn(3, 200)
    attack = ScaleSign(gamma_init=10.0, tol=1e-4)
    out = attack.craft(_ctx(benign, honest), 3)
    pos, _zero, neg = krum_sign_counts(honest)
    assert int((out[0] > 0).sum()) == pos and int((out[0] < 0).sum()) == neg
    assert attack.last_gamma >= 1.0
    assert torch.equal(out[0], out[2])


def test_build_attack_validates_names_and_params():
    assert build_attack({"name": "alie", "params": {"z": 1.0}}).z == 1.0
    assert build_attack({"name": "none"}).name == "none"
    with pytest.raises(ValueError):
        build_attack({"name": "bogus"})
    with pytest.raises(ValueError):
        build_attack({"name": "min_max", "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        build_attack({"name": "min_max", "params": {"perturbation": "bogus"}})


def test_reference_needs_rows_and_valid_knowledge():
    with pytest.raises(ValueError):
        AttackContext(torch.zeros(0, 3), torch.zeros(0, 3), 0).reference()
    with pytest.raises(ValueError):
        AttackContext(torch.zeros(1, 3), torch.zeros(1, 3), 2, knowledge="bogus").reference()


def test_scale_sign_gamma_stays_in_one_to_gamma_init():
    # Tiny honest deltas against a wide benign cloud: every gamma is feasible, so the
    # NDSS step schedule alone would walk past gamma_init (up to ~2x). Spec: [1, gamma_init].
    torch.manual_seed(2)
    benign = torch.randn(10, 40) * 5
    honest = torch.randn(3, 40) * 1e-3
    attack = ScaleSign(gamma_init=10.0, tol=1e-4)
    attack.craft(_ctx(benign, honest), 3)
    assert 1.0 <= attack.last_gamma <= 10.0
