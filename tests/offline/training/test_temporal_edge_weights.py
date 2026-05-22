"""Unit tests for v8.5 Temporal Edge Weights (TEW) wiring.

TEW transforms next_packet edges' delta_time_seconds into edge_weight =
1/(Δt+ε), so the HGT layer's `scores += log(edge_weight)` becomes a
temporal-locality bias: small Δt amplifies attention, large Δt suppresses.
"""
from __future__ import annotations

import numpy as np
import torch
import pytest


def _make_fake_batch():
    """Minimal MiniBatchSubgraph-shaped object with next_packet edge attrs."""
    from types import SimpleNamespace
    return SimpleNamespace(
        node_features={
            "flow": np.zeros((1, 6), dtype=np.float32),
            "packet": np.zeros((3, 4), dtype=np.float32),
        },
        edge_index={
            ("packet", "next_packet", "packet"): np.array([[0, 1], [1, 2]], dtype=np.int64),
        },
        edge_attr={
            ("packet", "next_packet", "packet"): np.array(
                [[0.01], [10.0]], dtype=np.float32
            ),  # small Δt then large Δt
        },
        seed_mask=np.array([True], dtype=bool),
        seed_labels=np.array([0], dtype=np.int64),
        local_to_global={},
    )


def test_tew_disabled_does_not_wire_next_packet():
    """When _TEW_ENABLED=False, next_packet edges get NO edge_weight."""
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        to_torch_batch, _set_tew_state,
    )
    _set_tew_state(enabled=False)
    batch = _make_fake_batch()
    _, _, ew, _, _ = to_torch_batch(
        batch,
        edge_types=[("packet", "next_packet", "packet")],
        device=torch.device("cpu"),
        use_semantic_edge_weights=False,
    )
    # to_torch_batch returns `edge_weights or None`: with both semantic and TEW
    # off, the dict is empty → None. Either way, no next_packet weight is wired.
    assert ew is None or ("packet", "next_packet", "packet") not in ew


def test_tew_enabled_wires_inverse_delta_t():
    """edge_weight for next_packet == 1/(Δt+ε), matching the documented transform."""
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        to_torch_batch, _set_tew_state,
    )
    eps = 1.0e-3
    _set_tew_state(enabled=True, epsilon=eps)
    try:
        batch = _make_fake_batch()
        _, _, ew, _, _ = to_torch_batch(
            batch,
            edge_types=[("packet", "next_packet", "packet")],
            device=torch.device("cpu"),
            use_semantic_edge_weights=False,
        )
        key = ("packet", "next_packet", "packet")
        assert key in ew
        expected = 1.0 / (np.array([0.01, 10.0], dtype=np.float32) + eps)
        torch.testing.assert_close(ew[key], torch.tensor(expected), rtol=1e-5, atol=1e-6)
    finally:
        _set_tew_state(enabled=False)  # restore module state


def test_tew_small_delta_t_amplifies_attention_score():
    """log(1/(Δt+ε)) added to attention score must be LARGE positive for small Δt."""
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        to_torch_batch, _set_tew_state,
    )
    _set_tew_state(enabled=True, epsilon=1.0e-3)
    try:
        batch = _make_fake_batch()
        _, _, ew, _, _ = to_torch_batch(
            batch,
            edge_types=[("packet", "next_packet", "packet")],
            device=torch.device("cpu"),
            use_semantic_edge_weights=False,
        )
        key = ("packet", "next_packet", "packet")
        # HGT does scores += log(edge_weight).
        # Δt=0.01 → log(1/0.011) ≈ +4.5  (amplify)
        # Δt=10   → log(1/10.001) ≈ -2.3 (suppress)
        bias = ew[key].log()
        assert bias[0].item() > 4.0, f"expected strong amplification for small Δt, got {bias[0].item()}"
        assert bias[1].item() < -2.0, f"expected suppression for large Δt, got {bias[1].item()}"
    finally:
        _set_tew_state(enabled=False)


def test_tew_handles_zero_delta_t_without_divide_by_zero():
    """Δt=0 must clamp to ε, not produce inf."""
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        to_torch_batch, _set_tew_state,
    )
    _set_tew_state(enabled=True, epsilon=1.0e-3)
    try:
        from types import SimpleNamespace
        batch = SimpleNamespace(
            node_features={"packet": np.zeros((2, 4), dtype=np.float32)},
            edge_index={
                ("packet", "next_packet", "packet"): np.array([[0], [1]], dtype=np.int64),
            },
            edge_attr={
                ("packet", "next_packet", "packet"): np.array([[0.0]], dtype=np.float32),
            },
            seed_mask=np.zeros(1, dtype=bool),
            seed_labels=np.zeros(1, dtype=np.int64),
            local_to_global={},
        )
        _, _, ew, _, _ = to_torch_batch(
            batch,
            edge_types=[("packet", "next_packet", "packet")],
            device=torch.device("cpu"),
            use_semantic_edge_weights=False,
        )
        key = ("packet", "next_packet", "packet")
        assert torch.isfinite(ew[key]).all(), "TEW produced non-finite weights for Δt=0"
        # 1/(0+0.001) = 1000
        assert ew[key][0].item() == pytest.approx(1000.0, rel=1e-3)
    finally:
        _set_tew_state(enabled=False)


def test_tew_state_isolated_from_semantic_weights():
    """TEW and matches_technique weights can coexist (different edge types)."""
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        to_torch_batch, _set_tew_state,
    )
    _set_tew_state(enabled=True, epsilon=1.0e-3)
    try:
        from types import SimpleNamespace
        batch = SimpleNamespace(
            node_features={
                "packet": np.zeros((2, 4), dtype=np.float32),
                "flow": np.zeros((1, 6), dtype=np.float32),
                "technique": np.zeros((2, 4), dtype=np.float32),
            },
            edge_index={
                ("packet", "next_packet", "packet"): np.array([[0], [1]], dtype=np.int64),
                ("flow", "matches_technique", "technique"): np.array([[0], [0]], dtype=np.int64),
            },
            edge_attr={
                ("packet", "next_packet", "packet"): np.array([[0.1]], dtype=np.float32),
                ("flow", "matches_technique", "technique"): np.array([0.85], dtype=np.float32),
            },
            seed_mask=np.array([True], dtype=bool),
            seed_labels=np.array([0], dtype=np.int64),
            local_to_global={},
        )
        _, _, ew, _, _ = to_torch_batch(
            batch,
            edge_types=[
                ("packet", "next_packet", "packet"),
                ("flow", "matches_technique", "technique"),
            ],
            device=torch.device("cpu"),
            use_semantic_edge_weights=True,
        )
        assert ("packet", "next_packet", "packet") in ew
        assert ("flow", "matches_technique", "technique") in ew
        # Semantic weight unchanged (0.85), TEW = 1/(0.1+0.001) ≈ 9.9
        torch.testing.assert_close(
            ew[("flow", "matches_technique", "technique")],
            torch.tensor([0.85], dtype=torch.float32),
            rtol=1e-5, atol=1e-6,
        )
        assert ew[("packet", "next_packet", "packet")][0].item() == pytest.approx(
            1.0 / 0.101, rel=1e-3,
        )
    finally:
        _set_tew_state(enabled=False)
