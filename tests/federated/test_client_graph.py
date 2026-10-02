import copy
import logging
from dataclasses import replace

import numpy as np
import pytest

from graphslm_ids.offline.federated.client_graph import (
    CONTAINS,
    FROM_HOST,
    TO_HOST,
    build_client_artifact,
    packet_owner,
    recompute_host_features,
)
from graphslm_ids.offline.training.hetero_graph_artifact import HeteroGraphArtifact

CLIENT_GRAPH_LOGGER = "graphslm_ids.offline.federated.client_graph"


def _with_edges(art, key, edges):
    edge_index = dict(art.edge_index)
    edge_index[key] = np.asarray(edges)
    return replace(art, edge_index=edge_index)


def _with_node_features(art, **overrides):
    return replace(art, node_features={**art.node_features, **overrides})


def test_client_graph_keeps_only_owned_flow_and_packet_edges(tiny_artifact):
    art = tiny_artifact
    client = np.arange(0, art.flow_y.shape[0], 2)
    sub = build_client_artifact(art, client)
    owner = packet_owner(art)
    flows = set(client.tolist())
    assert sub.edge_index.keys() == art.edge_index.keys()
    for key, edges in sub.edge_index.items():
        for pos, ntype in ((0, key[0]), (1, key[2])):
            ids = edges[pos]
            if ntype == "flow":
                assert set(ids.tolist()) <= flows, key
            if ntype == "packet":
                assert set(owner[ids].tolist()) <= flows, key
        assert sub.edge_attr[key].shape[0] == edges.shape[1]
    burst = sub.edge_index[("flow", "burst_neighbor", "flow")]
    assert burst.shape[1] > 0 and np.all(burst % 2 == 0)


def test_client_graph_keeps_mitre_knowledge_edges(tiny_artifact):
    sub = build_client_artifact(tiny_artifact, np.array([0]))
    for key in (
        ("technique", "has_subtechnique", "technique"),
        ("technique", "belongs_to_tactic", "tactic"),
        ("tactic", "rev_belongs_to_tactic", "technique"),
    ):
        assert sub.edge_index[key].shape[1] == tiny_artifact.edge_index[key].shape[1]


def test_client_graph_shares_large_arrays(tiny_artifact):
    sub = build_client_artifact(tiny_artifact, np.arange(5))
    assert sub.node_features["packet"] is tiny_artifact.node_features["packet"]
    assert sub.node_features["flow"] is tiny_artifact.node_features["flow"]
    assert sub.flow_y is tiny_artifact.flow_y


def test_recompute_host_features_matches_hand_count():
    flow_x = np.array([[6, 1000, 80], [6, 1001, 80], [6, 1002, 22]], dtype=np.float32)
    art = HeteroGraphArtifact(
        node_features={
            "flow": flow_x,
            "packet": np.zeros((0, 2), dtype=np.float32),
            "host": np.full((3, 4), 99.0, dtype=np.float32),
            "technique": np.zeros((1, 2), dtype=np.float32),
            "tactic": np.arange(1)[:, None],
        },
        edge_index={
            FROM_HOST: np.array([[0, 1, 2], [0, 0, 1]]),
            TO_HOST: np.array([[0, 1, 2], [1, 2, 2]]),
        },
        edge_attr={},
        flow_y=np.zeros(3, dtype=np.int64),
        metadata={"flow_feature_names": ["proto", "src_port", "dst_port"]},
    )
    hx = recompute_host_features(art, np.array([0, 1]))
    assert hx.tolist() == [[2, 0, 1, 2], [0, 1, 0, 0], [0, 1, 0, 0]]


# --- host-feature parity -------------------------------------------------------------------


def test_recompute_host_features_over_all_flows_equals_global_host_x(tiny_artifact):
    art = tiny_artifact
    hx = recompute_host_features(art, np.arange(art.flow_y.shape[0]))
    assert hx.any(), "fixture host_x must be non-trivial for this parity check to mean anything"
    np.testing.assert_array_equal(hx, art.node_features["host"])


def test_recompute_host_features_matches_build_host_tier_per_client():
    """Cross-check against the production host-tier rule, globally and on client subsets."""
    pd = pytest.importorskip("pandas")
    from graphslm_ids.offline.preprocessing.graph_builder import _build_host_tier

    rng = np.random.default_rng(3)
    n, h = 60, 6
    ips = np.array([f"10.0.0.{i}" for i in range(h)])
    src = rng.integers(0, h, n)
    dst = rng.integers(0, h, n)
    port = rng.integers(1, 6, n)
    feats = pd.DataFrame({"src_ip": ips[src], "dst_ip": ips[dst], "dst_port": port})
    full_x, host_to_idx, *_ = _build_host_tier(feats, pd.DataFrame())
    assert len(host_to_idx) == h
    row = np.array([host_to_idx[ip] for ip in ips])
    flow_x = np.zeros((n, 3), dtype=np.float32)
    flow_x[:, 2] = port
    art = HeteroGraphArtifact(
        node_features={
            "flow": flow_x,
            "packet": np.zeros((0, 2), dtype=np.float32),
            "host": full_x,
        },
        edge_index={
            FROM_HOST: np.stack([np.arange(n), row[src]]),
            TO_HOST: np.stack([np.arange(n), row[dst]]),
        },
        edge_attr={},
        flow_y=np.zeros(n, dtype=np.int64),
        metadata={"flow_feature_names": ["proto", "src_port", "dst_port"]},
    )
    for ids in (np.arange(n), rng.choice(n, size=25, replace=False)):
        local_x, local_map, *_ = _build_host_tier(
            feats.iloc[np.sort(ids)].reset_index(drop=True), pd.DataFrame()
        )
        expected = np.zeros_like(full_x)
        for ip, i in local_map.items():
            expected[host_to_idx[ip]] = local_x[i]
        np.testing.assert_array_equal(recompute_host_features(art, ids), expected)


def test_recompute_host_features_dedupes_client_ids(tiny_artifact):
    clean = recompute_host_features(tiny_artifact, np.array([0, 2, 4]))
    dup = recompute_host_features(tiny_artifact, np.array([0, 2, 2, 4, 0, 4, 4]))
    np.testing.assert_array_equal(dup, clean)
    assert clean[:, 0].sum() == 3


def test_recompute_host_features_rejects_non_4d_host_x(tiny_artifact):
    bad = _with_node_features(tiny_artifact, host=np.zeros((6, 3), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(6, 3\).*recomputed per client"):
        recompute_host_features(bad, np.array([0, 1]))
    with pytest.raises(ValueError, match="recomputed per client"):
        build_client_artifact(bad, np.array([0, 1]))
    # Opting out of host recomputation never inspects host_x.
    sub = build_client_artifact(bad, np.array([0, 1]), recompute_host=False)
    assert sub.node_features["host"] is bad.node_features["host"]


def test_recompute_host_features_missing_dst_port_zeroes_column_and_warns(tiny_artifact, caplog):
    art = replace(tiny_artifact, metadata={**tiny_artifact.metadata, "flow_feature_names": ["proto", "src_port"]})
    ids = np.arange(0, art.flow_y.shape[0], 2)
    with caplog.at_level(logging.WARNING, logger=CLIENT_GRAPH_LOGGER):
        hx = recompute_host_features(art, ids)
    assert any("dst_port" in r.getMessage() for r in caplog.records)
    assert np.all(hx[:, 2] == 0)
    reference = recompute_host_features(tiny_artifact, ids)
    np.testing.assert_array_equal(hx[:, [0, 1, 3]], reference[:, [0, 1, 3]])
    assert reference[:, 2].any()


# --- input validation ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_ids", [np.array([0.0, 1.0]), np.array([0.5]), np.array([True, False])])
def test_client_ids_must_be_integer_typed(tiny_artifact, bad_ids):
    with pytest.raises(ValueError, match="integer"):
        build_client_artifact(tiny_artifact, bad_ids)
    with pytest.raises(ValueError, match="integer"):
        recompute_host_features(tiny_artifact, bad_ids)


@pytest.mark.parametrize("empty_ids", [np.array([]), [], np.array([], dtype=np.float32), np.array([], dtype=np.int32)])
def test_empty_client_ids_of_any_dtype_are_accepted(tiny_artifact, empty_ids):
    sub = build_client_artifact(tiny_artifact, empty_ids)
    for key in (CONTAINS, ("flow", "burst_neighbor", "flow"), ("packet", "next_packet", "packet")):
        assert sub.edge_index[key].shape == (2, 0)
    assert not sub.node_features["host"].any()
    assert not recompute_host_features(tiny_artifact, empty_ids).any()


@pytest.mark.parametrize("bad_ids", [np.array([-1, 0]), np.array([0, 48]), np.array([10**6])])
def test_client_ids_must_be_valid_flow_ids(tiny_artifact, bad_ids):
    with pytest.raises(ValueError, match="flow ids"):
        build_client_artifact(tiny_artifact, bad_ids)
    with pytest.raises(ValueError, match="flow ids"):
        recompute_host_features(tiny_artifact, bad_ids)


def test_client_ids_accept_other_integer_dtypes(tiny_artifact):
    ref = build_client_artifact(tiny_artifact, np.array([1, 3, 5]))
    for dtype in (np.int32, np.uint8, np.uint64):
        sub = build_client_artifact(tiny_artifact, np.array([1, 3, 5], dtype=dtype))
        for key, edges in ref.edge_index.items():
            np.testing.assert_array_equal(sub.edge_index[key], edges)


def test_owner_must_match_packet_count(tiny_artifact):
    num_packets = tiny_artifact.node_features["packet"].shape[0]
    with pytest.raises(ValueError, match="owner"):
        build_client_artifact(tiny_artifact, np.array([0]), owner=np.zeros(num_packets - 1, dtype=np.int64))
    with pytest.raises(ValueError, match="owner"):
        build_client_artifact(tiny_artifact, np.array([0]), owner=np.zeros((num_packets, 1), dtype=np.int64))
    with pytest.raises(ValueError, match="owner"):
        build_client_artifact(tiny_artifact, np.array([0]), owner=np.full(num_packets, 10**6))
    # A correct precomputed owner is equivalent to letting the function derive it.
    owner = packet_owner(tiny_artifact)
    a = build_client_artifact(tiny_artifact, np.array([0, 5]), owner=owner)
    b = build_client_artifact(tiny_artifact, np.array([0, 5]))
    for key, edges in b.edge_index.items():
        np.testing.assert_array_equal(a.edge_index[key], edges)


def test_packet_owner_rejects_packet_with_two_contains_edges(tiny_artifact):
    edges = tiny_artifact.edge_index[CONTAINS]
    bad = _with_edges(tiny_artifact, CONTAINS, np.concatenate([edges, [[1], [0]]], axis=1))
    with pytest.raises(ValueError, match="more than one"):
        packet_owner(bad)
    with pytest.raises(ValueError, match="more than one"):
        build_client_artifact(bad, np.array([0]))


@pytest.mark.parametrize("key", [FROM_HOST, TO_HOST])
def test_recompute_host_features_rejects_flow_with_two_host_edges(tiny_artifact, key):
    edges = tiny_artifact.edge_index[key]
    bad = _with_edges(tiny_artifact, key, np.concatenate([edges, [[0], [2]]], axis=1))
    attr = tiny_artifact.edge_attr[key]
    bad = replace(bad, edge_attr={**bad.edge_attr, key: np.concatenate([attr, attr[:1]])})
    with pytest.raises(ValueError, match="more than one"):
        recompute_host_features(bad, np.array([0, 1]))
    with pytest.raises(ValueError, match="more than one"):
        build_client_artifact(bad, np.array([0, 1]))


# --- view semantics ------------------------------------------------------------------------


def test_client_graph_edge_attr_values_follow_kept_edges(tiny_artifact):
    owner = packet_owner(tiny_artifact)
    edge_attr = {
        key: 100.0 + np.arange(edges.shape[1], dtype=np.float32)[:, None]
        for key, edges in tiny_artifact.edge_index.items()
    }
    art = replace(tiny_artifact, edge_attr=edge_attr)
    client = {0, 2, 3, 8, 9, 20}
    sub = build_client_artifact(art, np.array(sorted(client)))

    def owned(ntype, node):
        if ntype == "flow":
            return node in client
        if ntype == "packet":
            return int(owner[node]) in client
        return True

    partially_filtered = 0
    for key, edges in art.edge_index.items():
        kept = [
            j
            for j in range(edges.shape[1])
            if owned(key[0], int(edges[0, j])) and owned(key[2], int(edges[1, j]))
        ]
        partially_filtered += 0 < len(kept) < edges.shape[1]
        np.testing.assert_array_equal(sub.edge_index[key], edges[:, kept])
        np.testing.assert_array_equal(
            sub.edge_attr[key], 100.0 + np.array(kept, dtype=np.float32)[:, None]
        )
    assert partially_filtered >= 5


def test_client_build_does_not_mutate_global_artifact(tiny_artifact):
    art = tiny_artifact
    before = copy.deepcopy((art.node_features, art.edge_index, art.edge_attr, art.flow_y))
    sub = build_client_artifact(art, np.arange(0, art.flow_y.shape[0], 2))
    after = (art.node_features, art.edge_index, art.edge_attr, art.flow_y)
    for old, new in zip(before[:3], after[:3], strict=True):
        assert old.keys() == new.keys()
        for key in old:
            np.testing.assert_array_equal(old[key], new[key], err_msg=str(key))
    np.testing.assert_array_equal(before[3], after[3])
    # The client's host tier is genuinely local, not an alias of the global one.
    assert sub.node_features["host"] is not art.node_features["host"]
    assert not np.array_equal(sub.node_features["host"], art.node_features["host"])


def test_recompute_host_false_shares_host_by_reference(tiny_artifact):
    sub = build_client_artifact(tiny_artifact, np.arange(5), recompute_host=False)
    assert sub.node_features["host"] is tiny_artifact.node_features["host"]


def test_orphan_packet_edges_are_all_dropped(tiny_artifact):
    art = tiny_artifact
    orphan = 5  # owned by flow 2 in the fixture
    contains = art.edge_index[CONTAINS]
    keep = contains[1] != orphan
    art = replace(
        _with_edges(art, CONTAINS, contains[:, keep]),
        edge_attr={**art.edge_attr, CONTAINS: art.edge_attr[CONTAINS][keep]},
    )
    owner = packet_owner(art)
    assert owner[orphan] == -1
    link = ("packet", "next_packet", "packet")
    assert (art.edge_index[link] == orphan).any()
    sub = build_client_artifact(art, np.arange(art.flow_y.shape[0]))
    for key, edges in sub.edge_index.items():
        for pos, ntype in ((0, key[0]), (1, key[2])):
            if ntype == "packet":
                assert not np.any(edges[pos] == orphan), key
    assert sub.edge_index[link].shape[1] == art.edge_index[link].shape[1] - 1
