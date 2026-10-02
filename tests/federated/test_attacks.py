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
    perturbation_direction,
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
    assert attack.info["saturated"] is True


@pytest.mark.parametrize("constraint", ["min_max", "min_sum"])
def test_scale_sign_gamma_sits_on_constraint_boundary(constraint):
    # Correlated clients: gamma* sits well below gamma_init / 8, where a search that
    # rejects g < 1 on its way down used to return 1.0 (a no-op attack).
    torch.manual_seed(3)
    common = torch.randn(400)
    benign = common + 0.1 * torch.randn(8, 400)
    honest = common + 0.1 * torch.randn(2, 400)
    attack = ScaleSign(gamma_init=10.0, tol=1e-6, constraint=constraint)
    attack.craft(_ctx(benign, honest), 2)
    gamma = attack.last_gamma
    assert 1.0 < gamma < 10.0 / 8
    gbar = honest.mean(0)
    pos, neg = gbar.clamp_min(0), gbar.clamp_max(0)
    d = pairwise_sq_dists(benign)
    bound = float(d.max()) if constraint == "min_max" else float(d.sum(1).max())

    def measure(g):
        dist = (benign - (g * pos + neg / g)).pow(2).sum(1).double()
        return float(dist.max()) if constraint == "min_max" else float(dist.sum())

    assert measure(gamma) <= bound * (1 + 1e-4)
    assert measure(1.01 * gamma) > bound


def test_scale_sign_oracle_is_accurate_with_large_common_offset():
    # |r|^2 ~ 4e9 swamps the ~20 squared distance: the origin-centred expansion loses
    # every digit in float32; the expansion around gbar does not.
    torch.manual_seed(5)
    common = 1000.0 + torch.randn(4000)
    benign = common + 0.05 * torch.randn(8, 4000)
    honest = common + 0.05 * torch.randn(2, 4000)
    a32, a64 = ScaleSign(tol=1e-6), ScaleSign(tol=1e-6)
    a32.craft(_ctx(benign, honest), 2)
    a64.craft(_ctx(benign.double(), honest.double()), 2)
    assert a32.last_gamma > 1.0
    assert a32.last_gamma == pytest.approx(a64.last_gamma, rel=1e-3)


@pytest.mark.parametrize("kind", ["unit", "sign"])
def test_unit_and_sign_perturbations_are_rescaled_to_sigma_norm(kind):
    torch.manual_seed(7)
    ref = torch.randn(10, 200) * 100.0 + 5.0
    p = perturbation_direction(ref, kind)
    sigma = ref.std(0, unbiased=False)
    assert float(p.double().norm()) == pytest.approx(float(sigma.double().norm()), rel=1e-5)
    mu = ref.mean(0)
    base = -mu if kind == "unit" else -torch.sign(mu)
    cos = (p.double() @ base.double()) / (p.double().norm() * base.double().norm())
    assert float(cos) == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(perturbation_direction(ref, "std"), -sigma)


@pytest.mark.parametrize("cls", [MinMax, MinSum])
@pytest.mark.parametrize("kind", ["unit", "sign"])
def test_unit_and_sign_perturbations_reach_boundary_on_scaled_data(cls, kind):
    torch.manual_seed(4)
    benign = torch.randn(10, 40) * 100.0 + 30.0
    attack = cls(perturbation=kind, gamma_init=10.0, tol=1e-6)
    v = attack.craft(_ctx(benign, torch.zeros(0, 40)), 3)[0]
    d = pairwise_sq_dists(benign)

    def measure(vec):
        dist = (benign - vec).pow(2).sum(1).double()
        return float(dist.max()) if cls is MinMax else float(dist.sum())

    bound = float(d.max()) if cls is MinMax else float(d.sum(1).max())
    mu, p = benign.mean(0), perturbation_direction(benign, kind)
    assert 0 < attack.last_gamma < 2 * attack.gamma_init - 1.0
    assert not attack.info["saturated"]
    assert measure(v) <= bound * (1 + 1e-4)
    assert measure(mu + 1.01 * attack.last_gamma * p) > bound


def test_gamma_attack_partial_knowledge_uses_honest_rows():
    torch.manual_seed(6)
    benign = torch.randn(8, 30)
    honest = torch.randn(3, 30) * 2.0 + 5.0
    attack = MinMax(tol=1e-6)
    out = attack.craft(_ctx(benign, honest, "partial"), 3)
    expected = honest.mean(0) + attack.last_gamma * perturbation_direction(honest, "std")
    assert torch.allclose(out[0], expected, atol=1e-5)
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])
    omniscient = MinMax(tol=1e-6).craft(_ctx(benign, honest), 3)
    assert not torch.allclose(out[0], omniscient[0])


def test_no_attack_returns_honest_copy():
    honest = torch.randn(2, 5)
    out = build_attack({"name": "none"}).craft(_ctx(torch.randn(3, 5), honest), 2)
    assert torch.equal(out, honest)
    assert out.data_ptr() != honest.data_ptr()


def test_partial_knowledge_without_honest_rows_raises():
    benign = torch.ones(2, 3)
    with pytest.raises(ValueError):
        AttackContext(torch.zeros(0, 3), benign, 2, knowledge="partial").reference()
    honest = torch.full((1, 3), 7.0)
    assert torch.equal(AttackContext(honest, torch.zeros(0, 3), 1).reference(), honest)


def test_build_attack_validates_knowledge():
    assert build_attack({"name": "alie", "knowledge": "partial"}).name == "alie"
    assert build_attack({"name": "alie", "knowledge": "omniscient"}).name == "alie"
    with pytest.raises(ValueError):
        build_attack({"name": "alie", "knowledge": "bogus"})


def test_attack_info_diagnostics():
    torch.manual_seed(8)
    benign = torch.randn(10, 40)
    alie = ALIE()
    alie.craft(_ctx(benign, torch.zeros(2, 40)), 2)
    assert alie.info["z"] == pytest.approx(alie_z(12, 2))
    assert alie.last_gamma is None

    mm = MinMax(tol=1e-6)
    mm.craft(_ctx(benign, torch.zeros(0, 40)), 2)
    assert mm.info["gamma"] == mm.last_gamma > 0
    assert mm.info["saturated"] is False
    # a tiny search range cannot reach the constraint: gamma saturates at its ceiling
    tiny = MinMax(gamma_init=0.01, tol=1e-6)
    tiny.craft(_ctx(benign, torch.zeros(0, 40)), 2)
    assert tiny.info["saturated"] is True

    honest = torch.randn(3, 40)
    ss = ScaleSign(tol=1e-4)
    ss.craft(_ctx(benign, honest), 3)
    pos, _zero, neg = krum_sign_counts(honest)
    assert ss.info["target_pos"] == pos and ss.info["target_neg"] == neg
    assert ss.info["gamma"] == ss.last_gamma
    ss.info["stale"] = 1
    ss.craft(_ctx(benign, honest), 3)
    assert "stale" not in ss.info


@pytest.mark.parametrize(
    "name,params",
    [
        ("min_max", {"gamma_init": 0.0}),
        ("min_max", {"gamma_init": float("inf")}),
        ("min_max", {"tol": -1.0}),
        ("min_sum", {"tol": 0.0}),
        ("min_sum", {"gamma_init": float("nan")}),
        ("scale_sign", {"gamma_init": 1.0}),
        ("scale_sign", {"gamma_init": 1.0005, "tol": 1e-3}),
        ("scale_sign", {"gamma_init": float("inf")}),
        ("scale_sign", {"tol": 0.0}),
        ("scale_sign", {"constraint": "bogus"}),
        ("sign_flip", {"scale": 0.0}),
        ("sign_flip", {"scale": -1.0}),
        ("sign_flip", {"scale": float("inf")}),
        ("alie", {"z": float("nan")}),
        ("alie", {"z": float("inf")}),
    ],
)
def test_constructors_reject_bad_numbers(name, params):
    with pytest.raises(ValueError):
        build_attack({"name": name, "params": params})


def test_scale_sign_info_tells_forced_gamma_from_tight_gamma():
    torch.manual_seed(3)
    common = torch.randn(400)
    benign = common + 0.1 * torch.randn(8, 400)
    close = ScaleSign(tol=1e-6)
    close.craft(_ctx(benign, common + 0.1 * torch.randn(2, 400)), 2)
    assert close.last_gamma > 1.0 and close.info["feasible"] is True

    # honest deltas far from every benign one: even gamma = 1 breaks the constraint,
    # so the search returns nothing and gamma = 1 is forced.
    torch.manual_seed(9)
    benign = torch.randn(8, 60)
    forced = ScaleSign(tol=1e-6)
    forced.craft(_ctx(benign, benign[:2] + 10.0), 2)
    assert forced.last_gamma == 1.0
    assert forced.info["feasible"] is False


@pytest.mark.parametrize("kind", ["std", "unit", "sign"])
@pytest.mark.parametrize("cls", [MinMax, MinSum])
def test_gamma_attacks_flag_a_zero_perturbation_as_degenerate(cls, kind):
    # One reference row: sigma = 0, so the perturbation vanishes and the output is mu.
    row = torch.randn(1, 10)
    ctx = AttackContext(row, torch.zeros(0, 10), 1, knowledge="partial")
    attack = cls(perturbation=kind)
    out = attack.craft(ctx, 2)
    assert attack.info["degenerate"] is True
    assert attack.info["saturated"] is False
    assert torch.equal(out[0], row[0])

    torch.manual_seed(10)
    normal = cls(perturbation=kind)
    normal.craft(_ctx(torch.randn(8, 10), torch.zeros(0, 10)), 2)
    assert normal.info["degenerate"] is False


def test_alie_info_z_is_the_override():
    alie = ALIE(z=1.5)
    alie.craft(_ctx(torch.randn(8, 6), torch.zeros(2, 6)), 2)
    assert alie.info["z"] == 1.5
