"""HGAA — Heterogeneous Graph Adaptive Augmentation for multiclass IDS.

Adapted from Zhao et al., "HGAA: A Heterogeneous Graph Adaptive Augmentation
Method for Asymmetric Datasets", Symmetry 17(10):1623, 2025.

Adaptation notes (binary anomaly → multiclass classification):
  - Paper's filter network distinguishes pseudo-anomaly from normal.
    Ours validates that augmented samples remain in the intended class
    (per-class plausibility, see hgaa_filter_network.ClassCentroidBank).
  - Paper's bias-aware selection prioritizes rare anomaly TYPES.
    Ours auto-detects the K rarest CLASSES from the train label
    distribution (no hardcoded class IDs — DC-2 in spec §1.5).
  - Type swap operator is intentionally disabled — swapping node/edge
    types between {flow, packet, technique, tactic} would break the
    semantic hierarchy of the NT114 three-tier graph.

Public API:
  - GraphAugmentor: 4 augmentation operators with shared eta parameter
  - AdaptiveOpSelector: per-class success-rate-weighted op sampling
  - BiasAwareSelector: force preferred op for auto-detected tail classes
  - detect_bias_classes: dataset-portable rare-class identification
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


EdgeKey = tuple[str, str, str]


def _ensure_numpy_subgraph(subgraph: Any) -> Any:
    """Return a shallow-copy subgraph with node/edge dicts materialized as numpy.

    ``MiniBatchSubgraph.pin_memory()`` (called by the DataLoader pin thread)
    converts ``node_features``, ``edge_index``, ``edge_attr`` to pinned torch
    tensors. The HGAA operators in this module are written for numpy arrays
    (use ``.copy()``, ``np.dtype``, ``np.concatenate``...) and crash with
    ``AttributeError: 'Tensor' object has no attribute 'copy'`` or
    ``Cannot interpret 'torch.int64' as a data type`` when given tensors.

    We convert each dict's values back to numpy via ``.cpu().numpy()`` only
    when augmentation actually fires (after the probabilistic gate), so the
    non-augmented fast-path keeps the pin_memory optimization intact.

    Downstream ``to_torch_batch`` accepts both numpy and torch via
    ``_to_device`` so re-transferring augmented numpy back to GPU works.
    """
    out = copy(subgraph)

    def _dict_to_numpy(d: dict[Any, Any]) -> dict[Any, np.ndarray]:
        result: dict[Any, np.ndarray] = {}
        for key, value in d.items():
            if hasattr(value, "cpu"):
                result[key] = value.cpu().numpy()
            else:
                result[key] = np.asarray(value)
        return result

    out.node_features = _dict_to_numpy(subgraph.node_features)
    out.edge_index = _dict_to_numpy(subgraph.edge_index)
    out.edge_attr = _dict_to_numpy(subgraph.edge_attr)
    return out


@dataclass
class AugmentedSubgraph:
    """Result of an augmentation op.

    Mirrors the public layout of ``MiniBatchSubgraph`` so the same
    downstream code paths (collator, model forward) work unchanged.
    """

    node_features: dict[str, np.ndarray]
    edge_index: dict[EdgeKey, np.ndarray]
    edge_attr: dict[EdgeKey, np.ndarray]
    augmentation_op: str
    target_edge_type: EdgeKey | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class GraphAugmentor:
    """Paper §3.1 augmentation operators, adapted for NT114 heterogeneous graphs.

    Each operator takes a subgraph (any object exposing ``node_features``,
    ``edge_index``, ``edge_attr``) and returns an :class:`AugmentedSubgraph`.

    Args:
        eta: Fraction of edges/nodes affected per operation (paper default 0.1).
        rng: Random number generator. Pass a seeded ``np.random.Generator`` for
            reproducibility; defaults to ``np.random.default_rng()``.
    """

    def __init__(self, eta: float = 0.1, rng: np.random.Generator | None = None) -> None:
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1], got {eta}")
        self.eta = float(eta)
        self.rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def _copy_dicts(subgraph: Any) -> tuple[dict, dict, dict]:
        """Shallow-copy the three container dicts (values still alias arrays)."""
        return (
            dict(subgraph.node_features),
            dict(subgraph.edge_index),
            dict(subgraph.edge_attr),
        )

    # ----- Operator 1: edge addition (paper §3.1.1) -----
    def edge_addition(
        self,
        subgraph: Any,
        target_edge_type: EdgeKey,
    ) -> AugmentedSubgraph:
        """Add ``eta * |E_target|`` new random edges of the target type.

        Other edge types and node features are untouched.
        """
        node_features, edge_index, edge_attr = self._copy_dicts(subgraph)

        src_type, _, dst_type = target_edge_type
        n_src = node_features[src_type].shape[0]
        n_dst = node_features[dst_type].shape[0]
        orig = edge_index[target_edge_type]
        n_add = int(np.floor(self.eta * orig.shape[1]))

        if n_add > 0 and n_src > 0 and n_dst > 0:
            new_src = self.rng.integers(0, n_src, n_add).astype(orig.dtype)
            new_dst = self.rng.integers(0, n_dst, n_add).astype(orig.dtype)
            new_edges = np.stack([new_src, new_dst])
            edge_index[target_edge_type] = np.concatenate([orig, new_edges], axis=1)

            # Append matching edge_attr rows. Use mean of existing attr as a
            # neutral prior — paper §3.1.1 doesn't specify; this avoids
            # destabilizing the attention scores.
            orig_attr = edge_attr[target_edge_type]
            if orig_attr.ndim == 1:
                attr_fill = np.full(n_add, float(orig_attr.mean()), dtype=orig_attr.dtype)
            else:
                attr_fill = np.tile(
                    orig_attr.mean(axis=0, keepdims=True), (n_add, 1)
                ).astype(orig_attr.dtype)
            edge_attr[target_edge_type] = np.concatenate([orig_attr, attr_fill], axis=0)

        return AugmentedSubgraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            augmentation_op="edge_addition",
            target_edge_type=target_edge_type,
        )

    # ----- Operator 2: node feature swap (paper §3.1.4) -----
    def node_feature_swap(
        self,
        subgraph: Any,
        node_type: str,
        same_class_mask: np.ndarray,
    ) -> AugmentedSubgraph:
        """Permute feature vectors among same-class nodes of the given type.

        Multiclass adaptation: paper swaps arbitrary nodes; we restrict to
        swaps among nodes carrying the same class label, so the swap acts
        as in-class oversampling rather than cross-class corruption.

        Args:
            subgraph: Source subgraph.
            node_type: Which node type to mutate (typically "flow").
            same_class_mask: Boolean mask of length ``num_nodes(node_type)``
                marking the nodes that share the target class label. Only
                these positions participate in the swap.
        """
        node_features, edge_index, edge_attr = self._copy_dicts(subgraph)
        feats = node_features[node_type].copy()
        mask_idx = np.flatnonzero(same_class_mask)

        if mask_idx.size >= 2:
            n_swap = max(1, int(np.floor(self.eta * mask_idx.size)))
            # Pick `n_swap` distinct positions and permute their rows.
            chosen = self.rng.choice(mask_idx, size=min(n_swap, mask_idx.size), replace=False)
            perm = self.rng.permutation(chosen)
            feats[chosen] = node_features[node_type][perm]

        node_features[node_type] = feats
        return AugmentedSubgraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            augmentation_op="node_feature_swap",
            extra={"node_type": node_type, "n_swappable": int(mask_idx.size)},
        )

    # ----- Operator 3: edge direction swap (paper §3.1.5) -----
    def edge_direction_swap(
        self,
        subgraph: Any,
        target_edge_type: EdgeKey,
    ) -> AugmentedSubgraph:
        """Reverse the direction of ``eta * |E|`` randomly-selected edges.

        Edge count is preserved; only the (src, dst) pairs are flipped.
        Edge attributes follow their original edge index — they are NOT
        shuffled, since direction swap doesn't change which packet/flow
        a transition concerns.
        """
        node_features, edge_index, edge_attr = self._copy_dicts(subgraph)
        ei = edge_index[target_edge_type].copy()
        n_edges = ei.shape[1]
        n_swap = int(np.floor(self.eta * n_edges))

        if n_swap > 0 and n_edges > 0:
            chosen = self.rng.choice(n_edges, size=n_swap, replace=False)
            ei[:, chosen] = ei[::-1, chosen]
        edge_index[target_edge_type] = ei

        return AugmentedSubgraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            augmentation_op="edge_direction_swap",
            target_edge_type=target_edge_type,
        )

    # ----- Operator 4: heterogeneous edge perturbation (paper §3.1.3) -----
    def edge_perturbation(
        self,
        subgraph: Any,
        target_edge_type: EdgeKey,
    ) -> AugmentedSubgraph:
        """XOR-style perturbation: drop ``eta * |E|`` edges, add same count
        of fresh random edges of the target type.

        Net edge count ≈ original (off-by-one tolerated due to floor).
        """
        node_features, edge_index, edge_attr = self._copy_dicts(subgraph)
        orig_ei = edge_index[target_edge_type]
        orig_attr = edge_attr[target_edge_type]
        n_edges = orig_ei.shape[1]
        n_perturb = int(np.floor(self.eta * n_edges))

        if n_perturb > 0 and n_edges > 0:
            # Drop
            keep_mask = np.ones(n_edges, dtype=bool)
            dropped = self.rng.choice(n_edges, size=n_perturb, replace=False)
            keep_mask[dropped] = False
            kept_ei = orig_ei[:, keep_mask]
            kept_attr = orig_attr[keep_mask] if orig_attr.ndim > 0 else orig_attr

            # Add
            src_type, _, dst_type = target_edge_type
            n_src = node_features[src_type].shape[0]
            n_dst = node_features[dst_type].shape[0]
            new_src = self.rng.integers(0, n_src, n_perturb).astype(orig_ei.dtype)
            new_dst = self.rng.integers(0, n_dst, n_perturb).astype(orig_ei.dtype)
            new_edges = np.stack([new_src, new_dst])
            edge_index[target_edge_type] = np.concatenate([kept_ei, new_edges], axis=1)

            if orig_attr.ndim == 1:
                attr_fill = np.full(n_perturb, float(orig_attr.mean()), dtype=orig_attr.dtype)
                edge_attr[target_edge_type] = np.concatenate([kept_attr, attr_fill])
            elif orig_attr.ndim >= 2:
                attr_fill = np.tile(
                    orig_attr.mean(axis=0, keepdims=True), (n_perturb, 1)
                ).astype(orig_attr.dtype)
                edge_attr[target_edge_type] = np.concatenate([kept_attr, attr_fill], axis=0)

        return AugmentedSubgraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            augmentation_op="edge_perturbation",
            target_edge_type=target_edge_type,
        )

    # ----- Operator 5: buffer node insertion (BufferGraph 2025, v8.5) -----
    def buffer_node_insertion(
        self,
        subgraph: Any,
        seed_labels: np.ndarray,
        minority_classes: np.ndarray,
        pressure_threshold: int = 3,
        max_buffers: int = 100,
        technique_edge_type: EdgeKey = ("flow", "matches_technique", "technique"),
        reverse_edge_type: EdgeKey | None = ("technique", "rev_matches_technique", "flow"),
    ) -> AugmentedSubgraph:
        """Insert buffer technique nodes for minority flows under majority pressure.

        Blocks majority→minority signal dilution through shared technique nodes.
        For each (minority_flow F_m, tech T) pair where T is also connected to
        >= ``pressure_threshold`` other flows in the batch, create a private
        buffer technique node B (features = copy of T) and reroute F_m's edge
        from T to B. F_m then aggregates clean technique signal from B (only
        F_m as neighbor) instead of majority-dominated T.

        Majority flow connections to T are NOT modified.
        """
        node_features, edge_index, edge_attr = self._copy_dicts(subgraph)

        if technique_edge_type not in edge_index:
            return AugmentedSubgraph(
                node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
                augmentation_op="buffer_node_insertion",
                extra={"n_buffers_created": 0, "reason": "no_technique_edges"},
            )
        fei = np.asarray(edge_index[technique_edge_type])
        if fei.shape[1] == 0:
            return AugmentedSubgraph(
                node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
                augmentation_op="buffer_node_insertion",
                extra={"n_buffers_created": 0, "reason": "empty_technique_edges"},
            )

        seed_labels_arr = np.asarray(seed_labels)
        minority_arr = np.asarray(minority_classes)
        if minority_arr.size == 0 or seed_labels_arr.size == 0:
            return AugmentedSubgraph(
                node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
                augmentation_op="buffer_node_insertion",
                extra={"n_buffers_created": 0, "reason": "no_minority_input"},
            )

        is_minority = np.isin(seed_labels_arr, minority_arr)
        minority_flow_indices = np.flatnonzero(is_minority)
        if minority_flow_indices.size == 0:
            return AugmentedSubgraph(
                node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
                augmentation_op="buffer_node_insertion",
                extra={"n_buffers_created": 0, "reason": "no_minority_in_batch"},
            )

        # Build tech → set(flow_indices) from valid seed flows only.
        valid_seeds = set(np.flatnonzero(seed_labels_arr >= 0).tolist())
        tech_to_flows: dict[int, set[int]] = {}
        for i in range(fei.shape[1]):
            f_idx, t_idx = int(fei[0, i]), int(fei[1, i])
            if f_idx in valid_seeds:
                tech_to_flows.setdefault(t_idx, set()).add(f_idx)

        # Collect unique (F_m, T) pairs exceeding pressure threshold (FIFO cap).
        minority_set = set(minority_flow_indices.tolist())
        seen: set[tuple[int, int]] = set()
        buffer_candidates: list[tuple[int, int]] = []
        for i in range(fei.shape[1]):
            f_idx, t_idx = int(fei[0, i]), int(fei[1, i])
            if f_idx not in minority_set:
                continue
            key = (f_idx, t_idx)
            if key in seen:
                continue
            if t_idx not in tech_to_flows:
                continue
            if len(tech_to_flows[t_idx]) < pressure_threshold:
                continue
            seen.add(key)
            buffer_candidates.append(key)
            if len(buffer_candidates) >= max_buffers:
                break

        if not buffer_candidates:
            return AugmentedSubgraph(
                node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
                augmentation_op="buffer_node_insertion",
                extra={"n_buffers_created": 0, "reason": "no_pressure_exceeded"},
            )

        # Extend technique features with buffer rows.
        tech_feats = node_features["technique"]
        n_existing = int(tech_feats.shape[0])
        n_buffers = len(buffer_candidates)
        buffer_indices = np.arange(n_existing, n_existing + n_buffers, dtype=fei.dtype)
        buffer_feats = np.stack(
            [tech_feats[t] for _, t in buffer_candidates], axis=0,
        ).astype(tech_feats.dtype)
        node_features["technique"] = np.concatenate([tech_feats, buffer_feats], axis=0)

        pair_to_buffer: dict[tuple[int, int], int] = {
            pair: int(buf) for pair, buf in zip(buffer_candidates, buffer_indices)
        }

        # Reroute forward edges F_m → T  ⟶  F_m → buffer.
        new_fei = fei.copy()
        for i in range(new_fei.shape[1]):
            key = (int(new_fei[0, i]), int(new_fei[1, i]))
            if key in pair_to_buffer:
                new_fei[1, i] = pair_to_buffer[key]
        edge_index[technique_edge_type] = new_fei

        # Reroute reverse edges T → F_m  ⟶  buffer → F_m, if present.
        if reverse_edge_type is not None and reverse_edge_type in edge_index:
            rev = np.asarray(edge_index[reverse_edge_type])
            if rev.shape[1] > 0:
                new_rev = rev.copy()
                for i in range(new_rev.shape[1]):
                    key = (int(new_rev[1, i]), int(new_rev[0, i]))  # (flow, tech)
                    if key in pair_to_buffer:
                        new_rev[0, i] = pair_to_buffer[key]
                edge_index[reverse_edge_type] = new_rev

        return AugmentedSubgraph(
            node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
            augmentation_op="buffer_node_insertion",
            extra={
                "n_buffers_created": n_buffers,
                "n_minority_in_batch": int(minority_flow_indices.size),
                "pressure_threshold": int(pressure_threshold),
            },
        )


# =========================================================================
# Adaptive op selector — paper §3.3.1, multiclass extension
# =========================================================================


class AdaptiveOpSelector:
    """Per-class adaptive sampling of augmentation operators.

    Maintains a Laplace-smoothed success count ``S[class, op]`` and a total
    count ``T[class, op]``. Sampling distribution for class c:

        p(op | class=c) = (S[c, op] + α) / Σ_o (S[c, o] + α)

    Smoothing α (default 1.0) prevents zero-probability ops even after
    100% failure streaks.

    Args:
        num_classes: Total number of classes (data-driven, see DC-1).
        op_names: List of operator name strings.
        smoothing: Laplace smoothing constant α.
    """

    def __init__(
        self,
        num_classes: int,
        op_names: list[str],
        smoothing: float = 1.0,
    ) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if not op_names:
            raise ValueError("op_names must be non-empty")
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")
        self.num_classes = int(num_classes)
        self.op_names = list(op_names)
        self.smoothing = float(smoothing)
        self._success = np.zeros((num_classes, len(op_names)), dtype=np.float64)
        self._total = np.zeros((num_classes, len(op_names)), dtype=np.float64)
        self._op_index = {name: i for i, name in enumerate(op_names)}

    def record_success(self, class_id: int, op: str, success: bool) -> None:
        """Update counts for a single observation."""
        c = int(class_id)
        if not 0 <= c < self.num_classes:
            return  # silently ignore out-of-range — keeps caller defensive
        if op not in self._op_index:
            raise KeyError(f"unknown op {op!r}; registered: {self.op_names}")
        i = self._op_index[op]
        if success:
            self._success[c, i] += 1.0
        self._total[c, i] += 1.0

    def get_distribution(self, class_id: int) -> dict[str, float]:
        """Return p(op | class) as a dict keyed by op name."""
        c = int(class_id)
        smoothed = self._success[c] + self.smoothing
        denom = smoothed.sum()
        probs = smoothed / denom if denom > 0 else smoothed
        return {self.op_names[i]: float(probs[i]) for i in range(len(self.op_names))}

    def sample_op(
        self,
        class_id: int,
        rng: np.random.Generator,
    ) -> str:
        """Sample one op for the given class using current distribution."""
        dist = self.get_distribution(class_id)
        names = list(dist.keys())
        probs = np.array([dist[n] for n in names], dtype=np.float64)
        probs = probs / probs.sum()  # ensure normalized
        return str(rng.choice(names, p=probs))


# =========================================================================
# Bias-aware selector + tail-class auto-detection
# =========================================================================


def detect_bias_classes(
    train_labels: np.ndarray,
    method: str = "top_k",
    top_k: int = 3,
    freq_threshold: float = 0.01,
) -> list[int]:
    """Auto-detect tail (rare) classes from the training label distribution.

    Dataset-portable per spec §1.5 DC-2: returns class IDs based on
    runtime label counts; never hardcodes IDs.

    Args:
        train_labels: 1D array of integer class labels for the train split.
        method: ``"top_k"`` picks the K classes with smallest counts;
            ``"freq_threshold"`` picks all classes whose frequency falls
            below ``freq_threshold``.
        top_k: K for ``top_k`` method.
        freq_threshold: Cutoff for ``freq_threshold`` method (fraction of
            total).

    Returns:
        Sorted list of class IDs flagged as bias targets.
    """
    if train_labels.ndim != 1:
        train_labels = train_labels.reshape(-1)
    labels = train_labels[train_labels >= 0]  # drop -1 padding if present
    if labels.size == 0:
        return []
    counts = np.bincount(labels)
    if method == "top_k":
        # Only consider classes that actually appear (count > 0)
        present = np.where(counts > 0)[0]
        sorted_by_count = present[np.argsort(counts[present])]
        return sorted(int(c) for c in sorted_by_count[: int(top_k)])
    if method == "freq_threshold":
        total = counts.sum()
        if total == 0:
            return []
        freqs = counts / total
        below = np.where((freqs > 0) & (freqs < freq_threshold))[0]
        return sorted(int(c) for c in below)
    raise ValueError(f"unknown method {method!r}; expected 'top_k' or 'freq_threshold'")


class BiasAwareSelector:
    """Wraps an :class:`AdaptiveOpSelector` with force-selection for tail classes.

    For tail-class samples (``class_id in bias_classes``):
        - with probability ``bias_factor`` (T in paper), force the
          ``preferred_op``;
        - with probability ``1 - bias_factor``, fall back to base sampling.

    For non-tail-class samples, always uses the base sampler.

    Args:
        base_selector: Underlying ``AdaptiveOpSelector``.
        bias_classes: Class IDs to prioritize. Use :func:`detect_bias_classes`
            to compute at training startup.
        bias_factor: T in (0, 1]. 1.0 means always force.
        preferred_op: Op name to force-select (must be in base_selector.op_names).
    """

    def __init__(
        self,
        base_selector: AdaptiveOpSelector,
        bias_classes: list[int],
        bias_factor: float = 0.5,
        preferred_op: str = "node_feature_swap",
    ) -> None:
        if not 0.0 <= bias_factor <= 1.0:
            raise ValueError(f"bias_factor must be in [0, 1], got {bias_factor}")
        if preferred_op not in base_selector.op_names:
            raise ValueError(
                f"preferred_op {preferred_op!r} not in base selector ops "
                f"{base_selector.op_names}"
            )
        self.base = base_selector
        self.bias_classes = set(int(c) for c in bias_classes)
        self.bias_factor = float(bias_factor)
        self.preferred_op = str(preferred_op)

    def sample_op(self, class_id: int, rng: np.random.Generator) -> str:
        c = int(class_id)
        if c in self.bias_classes and rng.random() < self.bias_factor:
            return self.preferred_op
        return self.base.sample_op(c, rng=rng)


# =========================================================================
# HGAAPipeline — orchestrator wired into training loop
# =========================================================================


# Default mapping from operator name → which edge type it perturbs.
# These are dataset-portable defaults that match the NT114 three-tier graph
# (flow → packet → technique → tactic). When swapping datasets with different
# edge schemas, override via config.
DEFAULT_OP_EDGE_MAP: dict[str, EdgeKey] = {
    "edge_addition": ("flow", "contains", "packet"),
    "edge_direction_swap": ("packet", "next_packet", "packet"),
    "edge_perturbation": ("flow", "matches_technique", "technique"),
    # node_feature_swap doesn't act on edges — sentinel handled in pipeline.
}


class HGAAPipeline:
    """Per-batch orchestrator for HGAA adaptive augmentation.

    Wires together the augmentation operators, adaptive op selector, and
    bias-aware selector. The training loop calls a single method
    :meth:`maybe_augment` per batch — the pipeline decides whether to
    augment (probabilistically, gated by ``p_aug``), picks an operator
    based on the rarest class present in the batch's seed labels, and
    returns a (possibly augmented) subgraph.

    Filter network is intentionally NOT wired here for the v7-p2 first
    cut — see spec §3.2.2; it will be added once v7-p1 produces a usable
    teacher checkpoint.

    Args:
        augmentor: :class:`GraphAugmentor` instance.
        selector: A :class:`AdaptiveOpSelector` or :class:`BiasAwareSelector`.
            The pipeline only calls ``.sample_op(class_id, rng)``.
        p_aug: Per-batch probability of applying augmentation (paper's
            "aug_ratio_in_batch" — default 0.5).
        op_edge_map: Override the default operator → edge-type mapping.
        feature_swap_node_type: Which node type ``node_feature_swap``
            operates on (default ``"flow"`` — the seed node type for NT114).
        seed: Optional integer seed for reproducible op sampling.
    """

    def __init__(
        self,
        augmentor: GraphAugmentor,
        selector: AdaptiveOpSelector | BiasAwareSelector,
        p_aug: float = 0.5,
        op_edge_map: dict[str, EdgeKey] | None = None,
        feature_swap_node_type: str = "flow",
        seed: int | None = None,
        # BufferGraph pre-step (v8.5 — deterministic, applied before stochastic op).
        buffer_enabled: bool = False,
        buffer_minority_classes: np.ndarray | None = None,
        buffer_pressure_threshold: int = 3,
        buffer_max_per_batch: int = 100,
    ) -> None:
        if not 0.0 <= p_aug <= 1.0:
            raise ValueError(f"p_aug must be in [0, 1], got {p_aug}")
        self.augmentor = augmentor
        self.selector = selector
        self.p_aug = float(p_aug)
        self.op_edge_map = dict(op_edge_map) if op_edge_map is not None else dict(DEFAULT_OP_EDGE_MAP)
        self.feature_swap_node_type = str(feature_swap_node_type)
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.buffer_enabled = bool(buffer_enabled)
        self.buffer_minority_classes = (
            np.asarray(buffer_minority_classes, dtype=np.int64)
            if buffer_minority_classes is not None
            else np.array([], dtype=np.int64)
        )
        self.buffer_pressure_threshold = int(buffer_pressure_threshold)
        self.buffer_max_per_batch = int(buffer_max_per_batch)
        self._stats = {
            "considered": 0, "augmented": 0, "op_counts": {},
            "buffer_created_total": 0, "buffer_applied_batches": 0,
        }

    def _pick_target_class(self, seed_labels: np.ndarray) -> int | None:
        """Pick the rarest class present in this batch (excluding pad -1).

        Returns ``None`` if no positive-labeled seeds exist (all -1).
        """
        valid = seed_labels[seed_labels >= 0]
        if valid.size == 0:
            return None
        counts = np.bincount(valid)
        present = np.where(counts > 0)[0]
        if present.size == 0:
            return None
        # Rarest class in THIS batch = smallest count
        return int(present[np.argmin(counts[present])])

    def maybe_augment(self, subgraph: Any) -> Any:
        """Possibly augment the input subgraph.

        Returns either the original subgraph (no augmentation this batch)
        or a fresh :class:`AugmentedSubgraph` with the same fields plus
        the chosen op recorded in ``augmentation_op``.

        Args:
            subgraph: A ``MiniBatchSubgraph`` (or any object exposing
                ``node_features``, ``edge_index``, ``edge_attr``,
                ``seed_labels``).
        """
        self._stats["considered"] += 1
        # The seed_labels may be a numpy array or a torch tensor depending
        # on whether ``pin_memory`` was already called.
        labels = subgraph.seed_labels
        if hasattr(labels, "cpu"):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.asarray(labels)

        # --- BufferGraph pre-step (v8.5 — deterministic, runs before stochastic gate) ---
        if self.buffer_enabled and self.buffer_minority_classes.size > 0:
            subgraph_np = _ensure_numpy_subgraph(subgraph)
            buffered = self.augmentor.buffer_node_insertion(
                subgraph_np,
                seed_labels=labels_np,
                minority_classes=self.buffer_minority_classes,
                pressure_threshold=self.buffer_pressure_threshold,
                max_buffers=self.buffer_max_per_batch,
            )
            n_created = int(buffered.extra.get("n_buffers_created", 0))
            if n_created > 0:
                subgraph = self._merge_back(subgraph, buffered)
                self._stats["buffer_created_total"] += n_created
                self._stats["buffer_applied_batches"] += 1

        # --- Stochastic gate for the remaining HGAA ops ---
        if self._rng.random() >= self.p_aug:
            return subgraph
        target_class = self._pick_target_class(labels_np)
        if target_class is None:
            return subgraph
        op = self.selector.sample_op(class_id=target_class, rng=self._rng)
        # Operators are written for numpy arrays (use .copy(), np.dtype, etc.).
        # MiniBatchSubgraph.pin_memory() converts node_features / edge_index /
        # edge_attr to pinned torch tensors in the DataLoader pin thread. We
        # need to materialize them back as numpy for the operators; downstream
        # ``to_torch_batch`` accepts both numpy and tensor and re-transfers.
        subgraph_for_op = _ensure_numpy_subgraph(subgraph)
        try:
            aug = self._apply_op(subgraph_for_op, op=op, target_class=target_class, labels=labels_np)
        except Exception as exc:  # pragma: no cover - augmentation should not fail in practice
            # Fail-open: log and proceed with un-augmented batch rather than crash training.
            print(f"[hgaa] augmentation '{op}' failed: {exc} — proceeding without augmentation",
                  flush=True)
            return subgraph
        self._stats["augmented"] += 1
        self._stats["op_counts"][op] = self._stats["op_counts"].get(op, 0) + 1
        # Wrap result back in a subgraph-compatible object: we preserve all
        # other fields of the original subgraph and override the augmented
        # tensors. This keeps `seed_mask`, `seed_labels`, `seed_flow_ids`,
        # `local_to_global`, and any extra metadata intact for downstream
        # loss computation.
        return self._merge_back(subgraph, aug)

    def _apply_op(
        self,
        subgraph: Any,
        op: str,
        target_class: int,
        labels: np.ndarray,
    ) -> AugmentedSubgraph:
        if op == "node_feature_swap":
            same_class_mask = labels == target_class
            return self.augmentor.node_feature_swap(
                subgraph,
                node_type=self.feature_swap_node_type,
                same_class_mask=same_class_mask,
            )
        edge_type = self.op_edge_map.get(op)
        if edge_type is None:
            raise KeyError(f"no edge type mapped for op {op!r}; check op_edge_map")
        if edge_type not in subgraph.edge_index:
            raise KeyError(
                f"edge type {edge_type} not present in subgraph; "
                f"available: {list(subgraph.edge_index.keys())}"
            )
        if op == "edge_addition":
            return self.augmentor.edge_addition(subgraph, target_edge_type=edge_type)
        if op == "edge_direction_swap":
            return self.augmentor.edge_direction_swap(subgraph, target_edge_type=edge_type)
        if op == "edge_perturbation":
            return self.augmentor.edge_perturbation(subgraph, target_edge_type=edge_type)
        raise ValueError(f"unsupported op {op!r}")

    @staticmethod
    def _merge_back(original: Any, aug: AugmentedSubgraph) -> Any:
        """Overlay augmented fields back onto a copy of the original subgraph.

        The original may be a :class:`MiniBatchSubgraph` with extra fields
        (seed_mask, seed_labels, ...) — we keep those, just swap the three
        mutated dicts.
        """
        # Shallow copy to avoid mutating the dataloader's batch object.
        out = copy(original)
        out.node_features = aug.node_features
        out.edge_index = aug.edge_index
        out.edge_attr = aug.edge_attr
        if hasattr(out, "stats") and isinstance(out.stats, dict):
            out.stats = dict(out.stats)
            out.stats["hgaa_op"] = aug.augmentation_op
        return out

    def stats_snapshot(self) -> dict[str, Any]:
        """Return a copy of internal augmentation stats for logging."""
        snap = {
            "considered": int(self._stats["considered"]),
            "augmented": int(self._stats["augmented"]),
            "op_counts": dict(self._stats["op_counts"]),
        }
        if snap["considered"] > 0:
            snap["aug_rate"] = snap["augmented"] / snap["considered"]
        else:
            snap["aug_rate"] = 0.0
        return snap


def build_hgaa_pipeline_from_config(
    config: dict,
    train_labels: np.ndarray,
    num_classes: int,
    seed: int | None = None,
) -> HGAAPipeline | None:
    """Convenience factory: read ``config["train"]["hgaa"]`` and build a pipeline.

    Returns ``None`` if ``config["train"]["hgaa"]["enabled"]`` is False
    or the block is missing — caller can use this to gate integration.

    Args:
        config: The full training config dict.
        train_labels: 1D array of train-split labels for tail-class detection.
        num_classes: Data-driven number of classes.
        seed: RNG seed for reproducible op sampling.
    """
    hgaa_cfg = (config.get("train") or {}).get("hgaa") or {}
    if not hgaa_cfg.get("enabled", False):
        return None

    ops = list(hgaa_cfg.get("ops") or [
        "edge_addition",
        "node_feature_swap",
        "edge_direction_swap",
        "edge_perturbation",
    ])
    eta = float(hgaa_cfg.get("eta_per_op", 0.1))
    p_aug = float(hgaa_cfg.get("aug_ratio_in_batch", 0.5))

    augmentor = GraphAugmentor(eta=eta, rng=np.random.default_rng(seed))
    base_selector = AdaptiveOpSelector(
        num_classes=num_classes,
        op_names=ops,
        smoothing=float(hgaa_cfg.get("smoothing", 1.0)),
    )

    bias_method = str(hgaa_cfg.get("bias_selection_method", "top_k"))
    bias_classes = detect_bias_classes(
        train_labels,
        method=bias_method,
        top_k=int(hgaa_cfg.get("bias_top_k_rarest", 3)),
        freq_threshold=float(hgaa_cfg.get("bias_min_class_freq", 0.01)),
    )

    selector: AdaptiveOpSelector | BiasAwareSelector
    bias_factor = float(hgaa_cfg.get("bias_factor_T", 0.0))
    if bias_factor > 0 and bias_classes:
        preferred = str(hgaa_cfg.get("bias_preferred_op", "node_feature_swap"))
        if preferred not in ops:
            preferred = ops[0]
        selector = BiasAwareSelector(
            base_selector=base_selector,
            bias_classes=bias_classes,
            bias_factor=bias_factor,
            preferred_op=preferred,
        )
    else:
        selector = base_selector

    print(
        f"[hgaa] enabled — ops={ops} eta={eta} p_aug={p_aug} "
        f"bias_classes={bias_classes} bias_factor={bias_factor}",
        flush=True,
    )

    # BufferGraph (v8.5) — re-use the same bias_classes already detected above
    # as the minority pool. Off by default; turned on via `buffer_enabled` flag.
    buffer_enabled = bool(hgaa_cfg.get("buffer_enabled", False))
    buffer_minority_arr = (
        np.asarray(bias_classes, dtype=np.int64) if buffer_enabled else None
    )
    if buffer_enabled:
        print(
            f"[hgaa] BufferGraph enabled — minority={bias_classes} "
            f"pressure={int(hgaa_cfg.get('buffer_pressure_threshold', 3))} "
            f"max_per_batch={int(hgaa_cfg.get('buffer_max_per_batch', 100))}",
            flush=True,
        )

    return HGAAPipeline(
        augmentor=augmentor,
        selector=selector,
        p_aug=p_aug,
        seed=seed,
        buffer_enabled=buffer_enabled,
        buffer_minority_classes=buffer_minority_arr,
        buffer_pressure_threshold=int(hgaa_cfg.get("buffer_pressure_threshold", 3)),
        buffer_max_per_batch=int(hgaa_cfg.get("buffer_max_per_batch", 100)),
    )
