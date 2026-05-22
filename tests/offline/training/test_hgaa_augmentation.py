"""Tests for HGAA augmentation operators and selectors.

Reference paper: Zhao et al., HGAA (Symmetry 2025, 17, 1623), §3.1-3.3.

These tests verify the multiclass adaptation of HGAA augmentation logic
from the paper's binary anomaly detection paradigm to NT114's 12-13 class
intrusion detection setting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


# ---------- Test fixtures ----------


@dataclass
class FakeSubgraph:
    """Minimal subgraph for testing — mirrors MiniBatchSubgraph public fields."""

    node_features: dict[str, np.ndarray]
    edge_index: dict[tuple, np.ndarray]
    edge_attr: dict[tuple, np.ndarray]
    seed_mask: np.ndarray
    seed_labels: np.ndarray


def _make_fake_subgraph(seed: int = 0) -> FakeSubgraph:
    """Build a small heterogeneous subgraph for unit tests.

    Layout:
      - 10 flow nodes (8 features)
      - 30 packet nodes (8 features)
      - 5 technique nodes (4 features)
      - 3 tactic nodes
      - flow→packet: 50 edges
      - packet→packet: 80 edges
      - flow→technique: 15 edges
    """
    rng = np.random.default_rng(seed)
    return FakeSubgraph(
        node_features={
            "flow": rng.normal(size=(10, 8)).astype(np.float32),
            "packet": rng.normal(size=(30, 8)).astype(np.float32),
            "technique": rng.normal(size=(5, 4)).astype(np.float32),
            "tactic": np.arange(3, dtype=np.int64),
        },
        edge_index={
            ("flow", "contains", "packet"): np.stack(
                [rng.integers(0, 10, 50), rng.integers(0, 30, 50)]
            ).astype(np.int64),
            ("packet", "next_packet", "packet"): np.stack(
                [rng.integers(0, 30, 80), rng.integers(0, 30, 80)]
            ).astype(np.int64),
            ("flow", "matches_technique", "technique"): np.stack(
                [rng.integers(0, 10, 15), rng.integers(0, 5, 15)]
            ).astype(np.int64),
        },
        edge_attr={
            ("flow", "contains", "packet"): rng.uniform(0, 1, (50, 1)).astype(np.float32),
            ("packet", "next_packet", "packet"): rng.uniform(0, 1, (80, 1)).astype(np.float32),
            ("flow", "matches_technique", "technique"): rng.uniform(0, 1, (15, 1)).astype(np.float32),
        },
        seed_mask=np.array([True] * 5 + [False] * 5),
        seed_labels=np.array([0, 1, 10, 11, 7, -1, -1, -1, -1, -1], dtype=np.int64),
    )


# ---------- Task 1: Augmentation operators ----------


def test_edge_addition_preserves_other_edge_types_unchanged():
    """Edge addition on target type must not modify other edge types."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(42))
    target = ("flow", "contains", "packet")
    out = aug.edge_addition(sg, target_edge_type=target)

    # Other edge types untouched
    other = ("packet", "next_packet", "packet")
    np.testing.assert_array_equal(out.edge_index[other], sg.edge_index[other])


def test_edge_addition_grows_target_edge_count_by_eta_fraction():
    """Edge addition adds eta * E_orig new edges to target type."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    target = ("flow", "contains", "packet")
    orig_n = sg.edge_index[target].shape[1]
    eta = 0.2

    aug = GraphAugmentor(eta=eta, rng=np.random.default_rng(42))
    out = aug.edge_addition(sg, target_edge_type=target)

    expected_added = int(np.floor(eta * orig_n))
    actual_added = out.edge_index[target].shape[1] - orig_n
    assert actual_added == expected_added


def test_edge_addition_respects_src_dst_node_index_bounds():
    """Newly added edges must reference valid node indices."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    target = ("flow", "contains", "packet")
    aug = GraphAugmentor(eta=0.3, rng=np.random.default_rng(42))
    out = aug.edge_addition(sg, target_edge_type=target)

    src = out.edge_index[target][0]
    dst = out.edge_index[target][1]
    assert src.max() < sg.node_features["flow"].shape[0]
    assert dst.max() < sg.node_features["packet"].shape[0]
    assert src.min() >= 0 and dst.min() >= 0


def test_node_feature_swap_keeps_topology_unchanged():
    """Feature swap must NOT change edge_index or edge_attr."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    aug = GraphAugmentor(eta=0.3, rng=np.random.default_rng(42))
    same_class_mask = np.zeros(10, dtype=bool)
    same_class_mask[:4] = True  # first 4 flows are same class
    out = aug.node_feature_swap(sg, node_type="flow", same_class_mask=same_class_mask)

    for ek, ei in sg.edge_index.items():
        np.testing.assert_array_equal(out.edge_index[ek], ei)
    for ek, ea in sg.edge_attr.items():
        np.testing.assert_array_equal(out.edge_attr[ek], ea)


def test_node_feature_swap_only_swaps_within_same_class():
    """After swap, set of feature vectors for same-class nodes is permuted (not new)."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    aug = GraphAugmentor(eta=1.0, rng=np.random.default_rng(42))
    same_class_mask = np.array([True, True, True, True] + [False] * 6)
    out = aug.node_feature_swap(sg, node_type="flow", same_class_mask=same_class_mask)

    # Set of feature vectors for same-class indices is permutation of original
    orig_set = {tuple(row) for row in sg.node_features["flow"][same_class_mask]}
    new_set = {tuple(row) for row in out.node_features["flow"][same_class_mask]}
    assert orig_set == new_set

    # Non-same-class rows untouched
    np.testing.assert_array_equal(
        out.node_features["flow"][~same_class_mask],
        sg.node_features["flow"][~same_class_mask],
    )


def test_edge_direction_swap_preserves_edge_count():
    """Direction swap reverses fraction of edges — total count unchanged."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    target = ("packet", "next_packet", "packet")
    orig_n = sg.edge_index[target].shape[1]

    aug = GraphAugmentor(eta=0.3, rng=np.random.default_rng(42))
    out = aug.edge_direction_swap(sg, target_edge_type=target)

    assert out.edge_index[target].shape[1] == orig_n


def test_edge_direction_swap_reverses_eta_fraction():
    """Exactly floor(eta * E) edges should have src/dst swapped."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    target = ("packet", "next_packet", "packet")
    orig = sg.edge_index[target].copy()
    eta = 0.25
    expected_swapped = int(np.floor(eta * orig.shape[1]))

    aug = GraphAugmentor(eta=eta, rng=np.random.default_rng(42))
    out = aug.edge_direction_swap(sg, target_edge_type=target)

    # Count edges where (src_new, dst_new) is the reverse of (src_orig, dst_orig)
    src_orig, dst_orig = orig[0], orig[1]
    src_new, dst_new = out.edge_index[target][0], out.edge_index[target][1]
    swapped = ((src_new == dst_orig) & (dst_new == src_orig) & (src_orig != dst_orig)).sum()
    # Allow exact match — direction swap is deterministic given seed
    assert swapped >= expected_swapped - 1  # tolerate 1-edge off-by-one if self-loops chosen


def test_edge_perturbation_drops_and_adds_respecting_eta():
    """Perturbation = XOR-style: drop eta fraction, add eta fraction of new edges."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    target = ("flow", "matches_technique", "technique")
    orig_n = sg.edge_index[target].shape[1]
    eta = 0.2

    aug = GraphAugmentor(eta=eta, rng=np.random.default_rng(42))
    out = aug.edge_perturbation(sg, target_edge_type=target)

    # Total edge count: drop eta*N, add eta*N → net unchanged (±1 floor rounding)
    new_n = out.edge_index[target].shape[1]
    assert abs(new_n - orig_n) <= 1


def test_augmentor_records_operation_name():
    """Each augmentation result must tag the operation that produced it."""
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor

    sg = _make_fake_subgraph()
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(42))
    target = ("flow", "contains", "packet")
    out = aug.edge_addition(sg, target_edge_type=target)
    assert out.augmentation_op == "edge_addition"


# ---------- Task 2: AdaptiveOpSelector ----------


def test_adaptive_op_selector_uniform_at_init():
    """Before any record_success, p(op | class) is uniform across ops."""
    from graphslm_ids.offline.training.hgaa_augmentation import AdaptiveOpSelector

    ops = ["edge_addition", "node_feature_swap", "edge_direction_swap", "edge_perturbation"]
    sel = AdaptiveOpSelector(num_classes=13, op_names=ops, smoothing=1.0)
    dist = sel.get_distribution(class_id=5)
    expected = 1.0 / len(ops)
    for op in ops:
        assert abs(dist[op] - expected) < 1e-6


def test_adaptive_op_selector_shifts_toward_successful_ops():
    """After many successes for op X, p(X | class) should be higher than other ops."""
    from graphslm_ids.offline.training.hgaa_augmentation import AdaptiveOpSelector

    ops = ["op_a", "op_b"]
    sel = AdaptiveOpSelector(num_classes=3, op_names=ops, smoothing=1.0)
    for _ in range(50):
        sel.record_success(class_id=1, op="op_a", success=True)
        sel.record_success(class_id=1, op="op_b", success=False)
    dist = sel.get_distribution(class_id=1)
    assert dist["op_a"] > dist["op_b"]


def test_adaptive_op_selector_smoothing_prevents_zero():
    """Even with 100% failures, op probability stays > 0 due to +1 smoothing."""
    from graphslm_ids.offline.training.hgaa_augmentation import AdaptiveOpSelector

    ops = ["op_a", "op_b"]
    sel = AdaptiveOpSelector(num_classes=2, op_names=ops, smoothing=1.0)
    for _ in range(100):
        sel.record_success(class_id=0, op="op_a", success=False)
    dist = sel.get_distribution(class_id=0)
    assert dist["op_a"] > 0


def test_adaptive_op_selector_sample_op_returns_valid_op():
    """sample_op output must be one of the registered op names."""
    from graphslm_ids.offline.training.hgaa_augmentation import AdaptiveOpSelector

    ops = ["op_a", "op_b", "op_c"]
    sel = AdaptiveOpSelector(num_classes=3, op_names=ops)
    rng = np.random.default_rng(42)
    for _ in range(20):
        result = sel.sample_op(class_id=0, rng=rng)
        assert result in ops


# ---------- Task 3: detect_bias_classes + BiasAwareSelector ----------


def test_detect_bias_classes_top_k_returns_rarest_classes():
    """top_k method picks the k classes with smallest counts."""
    from graphslm_ids.offline.training.hgaa_augmentation import detect_bias_classes

    # Class 2 has only 5 samples, class 7 has 10 — these should be top-2 rarest
    labels = np.array([0] * 100 + [1] * 50 + [2] * 5 + [7] * 10 + [11] * 200)
    bias = detect_bias_classes(labels, method="top_k", top_k=2)
    assert set(bias) == {2, 7}


def test_detect_bias_classes_freq_threshold_method():
    """freq_threshold picks all classes below the freq cutoff."""
    from graphslm_ids.offline.training.hgaa_augmentation import detect_bias_classes

    # Total = 100 + 50 + 5 + 10 + 200 = 365. Threshold 0.05 → cutoff 18.25.
    # Classes below 18: class 2 (5), class 7 (10). Class 1 (50) above.
    labels = np.array([0] * 100 + [1] * 50 + [2] * 5 + [7] * 10 + [11] * 200)
    bias = detect_bias_classes(labels, method="freq_threshold", freq_threshold=0.05)
    assert set(bias) == {2, 7}


def test_detect_bias_classes_handles_13_classes_like_nt114():
    """NT114-style class distribution: pick 3 rarest from 13 classes."""
    from graphslm_ids.offline.training.hgaa_augmentation import detect_bias_classes

    # Counts from actual NT114 log
    counts = [5313, 404360, 6780, 8202, 23366, 119528, 116155, 3327, 99222, 11468, 2604, 700676, 6614]
    labels = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    bias = detect_bias_classes(labels, method="top_k", top_k=3)
    # 3 rarest: class 10 (2604), class 7 (3327), class 0 (5313)
    assert set(bias) == {10, 7, 0}


def test_bias_aware_selector_force_selects_preferred_op_when_bias_class():
    """For tail-class samples, with prob T, preferred_op is forced."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        AdaptiveOpSelector,
        BiasAwareSelector,
    )

    ops = ["edge_addition", "node_feature_swap"]
    base = AdaptiveOpSelector(num_classes=5, op_names=ops)
    bias = BiasAwareSelector(
        base_selector=base,
        bias_classes=[3, 4],
        bias_factor=1.0,  # always force
        preferred_op="node_feature_swap",
    )
    rng = np.random.default_rng(42)
    for _ in range(20):
        out = bias.sample_op(class_id=3, rng=rng)
        assert out == "node_feature_swap"


def test_bias_aware_selector_uses_base_for_non_bias_class():
    """Non-bias classes bypass force-select and use base distribution."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        AdaptiveOpSelector,
        BiasAwareSelector,
    )

    ops = ["op_a", "op_b"]
    base = AdaptiveOpSelector(num_classes=5, op_names=ops)
    bias = BiasAwareSelector(
        base_selector=base,
        bias_classes=[3],
        bias_factor=1.0,
        preferred_op="op_a",
    )
    rng = np.random.default_rng(42)
    # Class 0 is NOT in bias_classes → should sample from base (mixed)
    results = [bias.sample_op(class_id=0, rng=rng) for _ in range(50)]
    assert "op_a" in results and "op_b" in results


# ---------- Task 6: HGAAPipeline orchestrator ----------


def test_hgaa_pipeline_p_aug_zero_never_augments():
    """With p_aug=0, the pipeline must return the original subgraph unchanged."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        AdaptiveOpSelector,
        GraphAugmentor,
        HGAAPipeline,
    )

    sg = _make_fake_subgraph()
    pipeline = HGAAPipeline(
        augmentor=GraphAugmentor(eta=0.1, rng=np.random.default_rng(0)),
        selector=AdaptiveOpSelector(
            num_classes=13,
            op_names=["edge_addition", "node_feature_swap", "edge_direction_swap", "edge_perturbation"],
        ),
        p_aug=0.0,
        seed=42,
    )
    for _ in range(20):
        out = pipeline.maybe_augment(sg)
        assert out is sg  # returns the exact same object


def test_hgaa_pipeline_p_aug_one_always_augments():
    """With p_aug=1, the pipeline must produce an augmented subgraph every call."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        AdaptiveOpSelector,
        GraphAugmentor,
        HGAAPipeline,
    )

    sg = _make_fake_subgraph()
    pipeline = HGAAPipeline(
        augmentor=GraphAugmentor(eta=0.1, rng=np.random.default_rng(0)),
        selector=AdaptiveOpSelector(
            num_classes=13,
            op_names=["edge_addition", "node_feature_swap", "edge_direction_swap", "edge_perturbation"],
        ),
        p_aug=1.0,
        seed=42,
    )
    augmented_count = 0
    for _ in range(20):
        out = pipeline.maybe_augment(sg)
        # Same fields but different array identity in augmented dicts
        if out is not sg:
            augmented_count += 1
    assert augmented_count == 20


def test_hgaa_pipeline_handles_tensor_inputs_from_pin_memory():
    """Regression: MiniBatchSubgraph.pin_memory() converts batch fields to
    torch tensors; HGAA operators were silently failing with AttributeError
    ('Tensor' object has no attribute 'copy') and TypeError ('Cannot interpret
    torch.int64 as a data type') for every batch in real training.

    The pipeline must transparently handle a subgraph whose node_features,
    edge_index, edge_attr dicts hold torch tensors instead of numpy arrays.
    """
    torch = pytest.importorskip("torch")

    from graphslm_ids.offline.training.hgaa_augmentation import (
        AdaptiveOpSelector,
        GraphAugmentor,
        HGAAPipeline,
    )

    sg_np = _make_fake_subgraph()
    # Mimic what pin_memory() produces: int64 edge_index, float32 edge_attr,
    # bool seed_mask, int64 seed_labels, float32 node_features.
    sg_np.node_features = {k: torch.from_numpy(v) for k, v in sg_np.node_features.items()}
    sg_np.edge_index = {
        k: torch.from_numpy(v.astype(np.int64)) for k, v in sg_np.edge_index.items()
    }
    sg_np.edge_attr = {
        k: torch.from_numpy(v.astype(np.float32)) for k, v in sg_np.edge_attr.items()
    }
    sg_np.seed_mask = torch.from_numpy(sg_np.seed_mask.astype(bool))
    sg_np.seed_labels = torch.from_numpy(sg_np.seed_labels.astype(np.int64))

    pipeline = HGAAPipeline(
        augmentor=GraphAugmentor(eta=0.2, rng=np.random.default_rng(0)),
        selector=AdaptiveOpSelector(
            num_classes=13,
            op_names=["edge_addition", "node_feature_swap", "edge_direction_swap", "edge_perturbation"],
        ),
        p_aug=1.0,
        seed=42,
    )

    # 20 calls — none should fall back to raw_batch via the exception handler.
    # If any operator crashed, stats_snapshot()["augmented"] would be < 20.
    for _ in range(20):
        out = pipeline.maybe_augment(sg_np)
        assert out is not None
    snap = pipeline.stats_snapshot()
    assert snap["considered"] == 20
    assert snap["augmented"] == 20, (
        f"Expected all 20 augmentations to succeed, but got {snap['augmented']}. "
        f"op_counts={snap['op_counts']}"
    )


def test_hgaa_pipeline_preserves_seed_labels_and_seed_mask():
    """Augmentation must not alter labels — loss computation depends on them."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        AdaptiveOpSelector,
        GraphAugmentor,
        HGAAPipeline,
    )

    sg = _make_fake_subgraph()
    orig_labels = sg.seed_labels.copy()
    orig_mask = sg.seed_mask.copy()
    pipeline = HGAAPipeline(
        augmentor=GraphAugmentor(eta=0.3, rng=np.random.default_rng(0)),
        selector=AdaptiveOpSelector(
            num_classes=13,
            op_names=["edge_addition", "node_feature_swap", "edge_direction_swap", "edge_perturbation"],
        ),
        p_aug=1.0,
        seed=42,
    )
    out = pipeline.maybe_augment(sg)
    np.testing.assert_array_equal(out.seed_labels, orig_labels)
    np.testing.assert_array_equal(out.seed_mask, orig_mask)


def test_hgaa_pipeline_stats_track_augmentation_decisions():
    """Internal stats counter must reflect considered/augmented decisions."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        AdaptiveOpSelector,
        GraphAugmentor,
        HGAAPipeline,
    )

    sg = _make_fake_subgraph()
    pipeline = HGAAPipeline(
        augmentor=GraphAugmentor(eta=0.1, rng=np.random.default_rng(0)),
        selector=AdaptiveOpSelector(
            num_classes=13,
            op_names=["edge_addition", "node_feature_swap"],
        ),
        p_aug=0.5,
        seed=42,
    )
    for _ in range(100):
        pipeline.maybe_augment(sg)
    snap = pipeline.stats_snapshot()
    assert snap["considered"] == 100
    assert 30 <= snap["augmented"] <= 70  # ~50% with reasonable variance
    assert 0.0 <= snap["aug_rate"] <= 1.0


def test_build_pipeline_from_config_disabled_returns_none():
    """When `hgaa.enabled=false`, the factory must return None for clean gating."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        build_hgaa_pipeline_from_config,
    )

    config = {"train": {"hgaa": {"enabled": False}}}
    pipeline = build_hgaa_pipeline_from_config(
        config=config,
        train_labels=np.array([0, 1, 2, 1, 0]),
        num_classes=3,
    )
    assert pipeline is None


def test_build_pipeline_from_config_enabled_builds_pipeline():
    """Enabled config must produce a working pipeline with auto-detected bias classes."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        BiasAwareSelector,
        build_hgaa_pipeline_from_config,
    )

    config = {
        "train": {
            "hgaa": {
                "enabled": True,
                "ops": ["edge_addition", "node_feature_swap", "edge_direction_swap", "edge_perturbation"],
                "eta_per_op": 0.1,
                "aug_ratio_in_batch": 0.5,
                "bias_selection_method": "top_k",
                "bias_top_k_rarest": 2,
                "bias_factor_T": 0.5,
                "bias_preferred_op": "node_feature_swap",
            }
        }
    }
    # Counts: class 0=100, class 1=10, class 2=5 → top-2 rarest = {1, 2}
    labels = np.concatenate([np.zeros(100), np.ones(10), np.full(5, 2)]).astype(np.int64)
    pipeline = build_hgaa_pipeline_from_config(
        config=config, train_labels=labels, num_classes=3, seed=42
    )
    assert pipeline is not None
    assert isinstance(pipeline.selector, BiasAwareSelector)
    assert set(pipeline.selector.bias_classes) == {1, 2}


# ============================================================================
# BufferGraph operator tests (v8.5 — signal-dilution prevention)
# ============================================================================
from types import SimpleNamespace


def _make_buffer_sg(n_majority, n_minority, shared_tech_idx=0, with_reverse=False):
    """Subgraph where all flows connect to a single shared technique.

    All majority flows AND minority flows attach to technique[shared_tech_idx].
    This is the canonical "signal dilution" scenario.
    """
    n_flows = n_majority + n_minority
    flow_feats = np.random.rand(n_flows, 8).astype(np.float32)
    n_techs = max(2, shared_tech_idx + 1)
    tech_feats = np.random.rand(n_techs, 4).astype(np.float32)

    src = list(range(n_flows))
    dst = [shared_tech_idx] * n_flows
    fei = np.array([src, dst], dtype=np.int64)
    f_attr = np.random.rand(fei.shape[1]).astype(np.float32)

    sg = SimpleNamespace(
        node_features={"flow": flow_feats, "technique": tech_feats},
        edge_index={("flow", "matches_technique", "technique"): fei},
        edge_attr={("flow", "matches_technique", "technique"): f_attr},
    )
    if with_reverse:
        rev = np.array([fei[1], fei[0]], dtype=np.int64)
        sg.edge_index[("technique", "rev_matches_technique", "flow")] = rev
        sg.edge_attr[("technique", "rev_matches_technique", "flow")] = f_attr.copy()
    return sg


def _make_many_shared_sg(n_majority, n_minority, n_shared_techs):
    """Each flow connects to ALL n_shared_techs (high pressure on every tech)."""
    n_flows = n_majority + n_minority
    flow_feats = np.random.rand(n_flows, 8).astype(np.float32)
    tech_feats = np.random.rand(n_shared_techs, 4).astype(np.float32)
    src, dst = [], []
    for f in range(n_flows):
        for t in range(n_shared_techs):
            src.append(f); dst.append(t)
    fei = np.array([src, dst], dtype=np.int64)
    return SimpleNamespace(
        node_features={"flow": flow_feats, "technique": tech_feats},
        edge_index={("flow", "matches_technique", "technique"): fei},
        edge_attr={("flow", "matches_technique", "technique"): np.random.rand(fei.shape[1]).astype(np.float32)},
    )


def test_buffer_op_noop_when_no_minority():
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    sg = _make_buffer_sg(n_majority=4, n_minority=0)
    out = aug.buffer_node_insertion(
        sg, seed_labels=np.array([0, 0, 0, 0]),
        minority_classes=np.array([1]), pressure_threshold=2,
    )
    assert out.extra["n_buffers_created"] == 0
    assert out.node_features["technique"].shape == sg.node_features["technique"].shape


def test_buffer_op_creates_buffer_for_minority_under_pressure():
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    sg = _make_buffer_sg(n_majority=5, n_minority=1, shared_tech_idx=0)

    out = aug.buffer_node_insertion(
        sg, seed_labels=np.array([0, 0, 0, 0, 0, 1]),
        minority_classes=np.array([1]), pressure_threshold=3,
    )
    assert out.extra["n_buffers_created"] == 1
    assert out.node_features["technique"].shape[0] == sg.node_features["technique"].shape[0] + 1

    new_ei = out.edge_index[("flow", "matches_technique", "technique")]
    minority_dst = new_ei[1][new_ei[0] == 5].tolist()
    buffer_idx = sg.node_features["technique"].shape[0]
    assert buffer_idx in minority_dst, "minority flow not rerouted to buffer"
    assert 0 not in minority_dst, "minority flow still connects to diluted tech 0"


def test_buffer_op_leaves_majority_connections_intact():
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    sg = _make_buffer_sg(n_majority=5, n_minority=1, shared_tech_idx=0)
    out = aug.buffer_node_insertion(
        sg, seed_labels=np.array([0, 0, 0, 0, 0, 1]),
        minority_classes=np.array([1]), pressure_threshold=3,
    )
    new_ei = out.edge_index[("flow", "matches_technique", "technique")]
    for f in range(5):
        assert 0 in new_ei[1][new_ei[0] == f].tolist(), f"majority flow {f} lost tech 0"


def test_buffer_op_respects_max_buffers_cap():
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    sg = _make_many_shared_sg(n_majority=10, n_minority=5, n_shared_techs=4)
    out = aug.buffer_node_insertion(
        sg, seed_labels=np.array([0] * 10 + [1] * 5),
        minority_classes=np.array([1]), pressure_threshold=3, max_buffers=10,
    )
    assert out.extra["n_buffers_created"] == 10


def test_buffer_op_below_threshold_no_buffer():
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    sg = _make_buffer_sg(n_majority=2, n_minority=1, shared_tech_idx=0)
    out = aug.buffer_node_insertion(
        sg, seed_labels=np.array([0, 0, 1]),
        minority_classes=np.array([1]), pressure_threshold=5,
    )
    assert out.extra["n_buffers_created"] == 0
    assert out.node_features["technique"].shape == sg.node_features["technique"].shape


def test_buffer_op_reroutes_reverse_edges():
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    sg = _make_buffer_sg(n_majority=4, n_minority=1, shared_tech_idx=0, with_reverse=True)
    out = aug.buffer_node_insertion(
        sg, seed_labels=np.array([0, 0, 0, 0, 1]),
        minority_classes=np.array([1]), pressure_threshold=3,
    )
    assert out.extra["n_buffers_created"] == 1
    buffer_idx = sg.node_features["technique"].shape[0]
    rev_ei = out.edge_index[("technique", "rev_matches_technique", "flow")]
    src_into_minority = rev_ei[0][rev_ei[1] == 4].tolist()
    assert buffer_idx in src_into_minority
    assert 0 not in src_into_minority


def test_buffer_op_preserves_edge_attr():
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    sg = _make_buffer_sg(n_majority=4, n_minority=1, shared_tech_idx=0)
    pre_attr = sg.edge_attr[("flow", "matches_technique", "technique")].copy()
    out = aug.buffer_node_insertion(
        sg, seed_labels=np.array([0, 0, 0, 0, 1]),
        minority_classes=np.array([1]), pressure_threshold=3,
    )
    np.testing.assert_allclose(
        out.edge_attr[("flow", "matches_technique", "technique")], pre_attr,
    )


def test_buffer_op_handles_missing_technique_edge_type():
    from graphslm_ids.offline.training.hgaa_augmentation import GraphAugmentor
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    sg = SimpleNamespace(
        node_features={"flow": np.zeros((2, 4)), "technique": np.zeros((2, 4))},
        edge_index={},
        edge_attr={},
    )
    out = aug.buffer_node_insertion(
        sg, seed_labels=np.array([0, 1]),
        minority_classes=np.array([1]), pressure_threshold=2,
    )
    assert out.extra["n_buffers_created"] == 0
    assert out.extra["reason"] == "no_technique_edges"


def test_pipeline_applies_buffer_before_random_op():
    """When buffer_enabled and minority is in batch, buffer is applied even when p_aug=0."""
    from graphslm_ids.offline.training.hgaa_augmentation import (
        GraphAugmentor, BiasAwareSelector, AdaptiveOpSelector, HGAAPipeline,
    )
    aug = GraphAugmentor(eta=0.1, rng=np.random.default_rng(0))
    base_sel = AdaptiveOpSelector(
        num_classes=2,
        op_names=["edge_addition", "node_feature_swap", "edge_direction_swap", "edge_perturbation"],
    )
    selector = BiasAwareSelector(
        base_selector=base_sel, bias_classes=[1], preferred_op="node_feature_swap",
    )
    pipe = HGAAPipeline(
        augmentor=aug, selector=selector,
        p_aug=0.0,  # disable random op
        seed=0,
        buffer_enabled=True,
        buffer_minority_classes=np.array([1]),
        buffer_pressure_threshold=3,
        buffer_max_per_batch=100,
    )
    sg = _make_buffer_sg(n_majority=5, n_minority=1, shared_tech_idx=0)
    sg.seed_labels = np.array([0, 0, 0, 0, 0, 1])

    out = pipe.maybe_augment(sg)
    assert out.node_features["technique"].shape[0] == sg.node_features["technique"].shape[0] + 1
    assert pipe._stats["buffer_created_total"] == 1
