import copy

import numpy as np
import pytest

from graphslm_ids.offline.federated.config import load_fl_config
from graphslm_ids.offline.federated.partition import (
    ClientPartition,
    explicit_counts,
    explicit_fractions,
    group_members,
    make_partition,
    partition_explicit,
)
from graphslm_ids.offline.federated.simulation import select_malicious

NAMES = [
    "Benign", "DDoS-ACK_Fragmentation", "DDoS-ICMP_Flood", "DDoS-ICMP_Fragmentation", "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood", "Recon-HostDiscovery", "Recon-OSScan", "Recon-PingSweep", "Recon-PortScan",
    "VulnerabilityScan", "CommandInjection", "XSS", "SqlInjection", "Uploading_Attack", "BrowserHijacking",
    "Backdoor_Malware", "Mirai-udpplain",
]
LABEL_NAMES = dict(enumerate(NAMES))
C = {n: i for i, n in enumerate(NAMES)}


def _cfg(**kw):
    cfg = copy.deepcopy(load_fl_config(None)["partition"])
    cfg.update(kw)
    return cfg


def _data(per_class=400, extra=()):
    labels = np.repeat(np.arange(18 + len(extra)), per_class)
    return np.arange(labels.size, dtype=np.int64), labels


def test_default_partition_config_is_the_explicit_table():
    cfg = _cfg()
    assert cfg["scheme"] == "explicit" and cfg["dominance"] == 0.8 and cfg["num_clients"] == 20
    assert group_members(cfg["groups"], 20) == {
        "G1": [0, 1, 2, 3, 4], "G2": [5, 6, 7, 8, 9], "G3": [10, 11, 12, 13], "G4": [14, 15], "G5": [16, 17, 18, 19]
    }


@pytest.mark.parametrize(
    "p, ddos_in, web_in, other_in, out",
    [(0.8, 0.17, 0.21, 0.41, 0.01), (0.5, 0.125, 0.15, 0.275, 0.025), (0.0, 0.05, 0.05, 0.05, 0.05)],
)
def test_fraction_table_matches_spec(p, ddos_in, web_in, other_in, out):
    cfg = _cfg()
    f = explicit_fractions(LABEL_NAMES, 20, cfg["groups"], cfg["families"], p)
    assert f.shape == (20, 18) and np.allclose(f.sum(axis=0), 1.0)
    assert f[0, C["DDoS-ICMP_Flood"]] == pytest.approx(ddos_in)
    assert f[5, C["Recon-PortScan"]] == pytest.approx(ddos_in)
    assert f[10, C["XSS"]] == pytest.approx(web_in)
    assert f[14, C["Mirai-udpplain"]] == pytest.approx(other_in)
    assert f[19, C["DDoS-ICMP_Flood"]] == pytest.approx(out) and f[0, C["XSS"]] == pytest.approx(out)
    assert np.allclose(f[:, C["Benign"]], 0.05)


def test_counts_floor_plus_largest_remainder_with_low_index_ties():
    assert explicit_counts(np.array([[0.5], [0.25], [0.25]]), np.array([5])).tolist() == [[3], [1], [1]]
    assert explicit_counts(np.full((3, 1), 1 / 3), np.array([4])).tolist() == [[2], [1], [1]]


def test_partition_is_deterministic_and_matches_the_count_table():
    ids, labels = _data()
    a = partition_explicit(ids, labels, LABEL_NAMES, _cfg(seed=1))
    b = partition_explicit(ids, labels, LABEL_NAMES, _cfg(seed=2))
    assert all(np.array_equal(x, y) for x, y in zip(a.client_flow_ids, b.client_flow_ids))
    cfg = _cfg()
    expected = explicit_counts(explicit_fractions(LABEL_NAMES, 20, cfg["groups"], cfg["families"], 0.8), np.full(18, 400))
    assert np.array_equal(a.counts, expected) and np.array_equal(a.class_histogram(labels, 18), expected)
    flat = np.concatenate(a.client_flow_ids)
    assert flat.size == np.unique(flat).size and set(flat.tolist()) <= set(ids.tolist())


def test_clients_get_contiguous_time_segments():
    ids, labels = _data()
    order = -ids.astype(np.float64)  # later ids are earlier in time
    part = partition_explicit(ids, labels, LABEL_NAMES, _cfg(), order=order)
    for c in range(18):
        runs = [np.sort(k[labels[k] == c])[::-1] for k in part.client_flow_ids]
        assert np.array_equal(np.concatenate(runs), np.sort(ids[labels == c])[::-1])
    assert part.params["ordered_by"] == "flow_start_ts"


def test_zero_dominance_is_an_even_split():
    ids, labels = _data(per_class=203)
    part = partition_explicit(ids, labels, LABEL_NAMES, _cfg(dominance=0.0))
    assert (part.counts[:3] == 11).all() and (part.counts[3:] == 10).all()


def _in_group_share(counts, cfg, family):
    """Share of the family's flows held by the clients of the group whose family it is."""
    group = next(g for g, spec in cfg["groups"].items() if spec["family"] == family)
    members = group_members(cfg["groups"], 20)[group]
    cols = [C[name] for name in cfg["families"][family]]
    return counts[members][:, cols].sum() / counts[:, cols].sum()


def test_cap_shrinks_the_whole_table_uniformly():
    ids, labels = _data(per_class=2000)
    order = -ids.astype(np.float64)  # later ids are earlier in time
    full = partition_explicit(ids, labels, LABEL_NAMES, _cfg(), order=order)
    cfg = _cfg(max_flows_per_client=600)
    capped = partition_explicit(ids, labels, LABEL_NAMES, cfg, order=order)
    assert full.counts.sum(axis=1).max() > 600 and full.params["cap_scale"] == 1.0
    # (a) every client total is within the cap
    assert capped.sizes() == capped.counts.sum(axis=1).tolist() and max(capped.sizes()) <= 600
    # (b) the capped table is the same share table applied to floor(s * N_c)
    s = capped.params["cap_scale"]
    assert 0.0 < s < 1.0
    f = explicit_fractions(LABEL_NAMES, 20, cfg["groups"], cfg["families"], 0.8)
    shrunk = np.floor(s * np.full(18, 2000) + 1e-9).astype(np.int64)
    assert np.array_equal(capped.counts, explicit_counts(f, shrunk))
    assert np.array_equal(capped.class_histogram(labels, 18), capped.counts)
    # (c) the in-group family shares of the full table survive the cap
    for family in ("ddos", "recon", "web", "other"):
        assert _in_group_share(capped.counts, cfg, family) == pytest.approx(
            _in_group_share(full.counts, cfg, family), abs=0.02
        ), family
    # (d) client k keeps the head (earliest flows) of its full-data run of every class
    for f_ids, c_ids in zip(full.client_flow_ids, capped.client_flow_ids):
        for c in range(18):
            run = np.sort(f_ids[labels[f_ids] == c])[::-1]
            kept = np.sort(c_ids[labels[c_ids] == c])[::-1]
            assert np.array_equal(kept, run[: kept.size])


def test_cap_that_does_not_bind_changes_nothing():
    ids, labels = _data()
    full = partition_explicit(ids, labels, LABEL_NAMES, _cfg())
    loose = partition_explicit(ids, labels, LABEL_NAMES, _cfg(max_flows_per_client=int(full.counts.sum(axis=1).max())))
    assert np.array_equal(loose.counts, full.counts) and loose.params["cap_scale"] == 1.0
    assert all(np.array_equal(a, b) for a, b in zip(loose.client_flow_ids, full.client_flow_ids))


def test_cap_that_would_grow_a_cell_is_refused():
    # Largest remainder is not monotone in the class total (Alabama paradox): at p = 0.5 with 5 flows
    # per class, the table shrunk to the cap of 11 gives a client a flow its full-data run lacks.
    ids, labels = _data(per_class=5)
    with pytest.raises(AssertionError, match="max_flows_per_client"):
        partition_explicit(
            ids, labels, LABEL_NAMES, _cfg(dominance=0.5, max_flows_per_client=11, min_flows_per_client=1)
        )


def test_class_outside_every_family_is_split_evenly():
    names = {**LABEL_NAMES, 18: "Weird"}
    cfg = _cfg()
    f = explicit_fractions(names, 20, cfg["groups"], cfg["families"], 0.8)
    assert np.allclose(f[:, 18], 0.05)
    ids, labels = _data(extra=("Weird",))
    part = partition_explicit(ids, labels, names, cfg)
    assert part.counts[:, 18].sum() == 400


@pytest.mark.parametrize(
    "override",
    [
        {"groups": {"A": {"clients": [1, 10], "family": None}, "B": {"clients": [10, 20], "family": None}}},
        {"groups": {"A": {"clients": [1, 19], "family": None}}},
        {"groups": {"A": {"clients": [1, 20], "family": "nope"}}},
        {"dominance": 1.5},
        {"min_flows_per_client": 1000},
    ],
)
def test_invalid_explicit_config_raises(override):
    ids, labels = _data()
    with pytest.raises(ValueError):
        partition_explicit(ids, labels, LABEL_NAMES, _cfg(**override))


def test_make_partition_explicit_needs_label_names_and_json_round_trip():
    ids, labels = _data()
    with pytest.raises(ValueError):
        make_partition(ids, labels, _cfg())
    part = make_partition(ids, labels, _cfg(), label_names=LABEL_NAMES)
    back = ClientPartition.from_json(part.to_json())
    assert np.array_equal(back.counts, part.counts) and back.params["dominance"] == 0.8


def test_select_malicious_group_placement():
    groups = load_fl_config(None)["partition"]["groups"]
    assert select_malicious(20, 0.2, 0, placement="group:G2", groups=groups) == [5, 6, 7, 8]
    with pytest.raises(ValueError):
        select_malicious(20, 0.3, 0, placement="group:G2", groups=groups)
    with pytest.raises(ValueError):
        select_malicious(20, 0.2, 0, placement="group:G9", groups=groups)
    with pytest.raises(ValueError):
        select_malicious(20, 0.2, 0, placement="bogus")


def test_yaml_groups_and_families_replace_the_defaults_instead_of_merging(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "partition:\n  num_clients: 2\n  groups:\n    A: {clients: [1, 2], family: f}\n  families:\n    f: [XSS]\n",
        encoding="utf-8",
    )
    part = load_fl_config(path)["partition"]
    assert list(part["groups"]) == ["A"] and part["families"] == {"f": ["XSS"]}
    assert group_members(part["groups"], 2) == {"A": [0, 1]}


def test_class_listed_in_two_families_is_rejected():
    cfg = _cfg()
    cfg["families"]["dup"] = ["XSS"]  # XSS is already in "web"
    with pytest.raises(ValueError, match="both families"):
        explicit_fractions(LABEL_NAMES, 20, cfg["groups"], cfg["families"], 0.8)


@pytest.mark.parametrize(
    "override", [{"scheme": "bogus"}, {"dominance": -0.1}, {"groups": {"A": {"clients": [1, 5], "family": None}}}]
)
def test_check_partition_config_rejects_bad_sections_without_data(override):
    from graphslm_ids.offline.federated.partition import check_partition_config

    assert check_partition_config(_cfg()) == "explicit"
    with pytest.raises(ValueError):
        check_partition_config(_cfg(**override))
