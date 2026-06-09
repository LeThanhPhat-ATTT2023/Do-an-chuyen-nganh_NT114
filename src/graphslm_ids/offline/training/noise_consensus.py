"""Noise-robust training core for EG-HGT — single-model online self-detection of
instance-dependent label noise.

Design: docs/superpowers/specs/2026-06-09-neighbor-consensus-noise-robust-hgt-design.md

The web-attack classes in CIC-IoT-2023 are ~95-98% background IoT->cloud traffic
mislabeled as the attack (per-pcap labeling). This is INSTANCE-DEPENDENT noise, so
loss-based detectors (small-loss / GMM, DivideMix, SAT) fail: the mislabeled flows
are the EASY, low-loss ones and the true attacks are the HARD, high-loss ones.

We instead detect noise with two label-free, loss-free signals and fold them into a
per-sample loss weight, so a single model becomes more noise-robust as it trains:

  * Signal 1 (PRIMARY) — ``evidence_support``: does the flow carry a MITRE evidence
    edge consistent with its (claimed) attack label? Grounded in actual payload
    attack tokens (MSEE / Tầng-3). A flow labeled an attack but with NO matching
    evidence is a prime noise candidate. Independent of model, neighbors, and loss.
  * Signal 2 (SUPPORTING) — ``neighbor_consensus``: do the flow's graph neighbors'
    predicted distributions support its label? Uses the burst_neighbor / shared-host
    flow-flow edges that GNN4ID (a per-graph classifier) structurally lacks.

``combine_clean_confidence`` blends them; ``curriculum_weight`` applies warmup + clamp;
``EMAConsensusBuffer`` smooths the score across epochs. All pure & unit-tested.
"""
from __future__ import annotations

import torch

__all__ = [
    "neighbor_consensus",
    "evidence_support",
    "combine_clean_confidence",
    "curriculum_weight",
    "EMAConsensusBuffer",
    "build_evidence_by_flow",
]


def build_evidence_by_flow(
    contains_edge_index: torch.Tensor,
    evidence_per_family: dict[int, tuple[torch.Tensor, torch.Tensor]],
    num_flows: int,
    num_families: int,
) -> torch.Tensor:
    """Aggregate packet-level MITRE evidence edges up to a per-flow, per-family table.

    Signal 1 (Tầng-3 grounding) needs, for each flow, the strongest matching
    evidence-edge weight in each attack family. Evidence edges are
    ``packet -> evidence_{family} -> technique``; a flow "has" family-F evidence if any
    of the packets it contains is the source of an ``evidence_F`` edge. We take the MAX
    weight (a single strong attack token is enough to ground the flow).

    Args:
        contains_edge_index: ``(2, E)`` ``flow -> contains -> packet`` edges
            (row 0 = flow id, row 1 = packet id).
        evidence_per_family: ``{family_col: (packet_ids, weights)}`` — for each attack
            family, the source packet ids of its evidence edges and their weights.
        num_flows: total flow count (table rows).
        num_families: total attack-family count (table cols).

    Returns:
        ``(num_flows, num_families)`` float tensor; entry ``[i, f]`` is the max
        evidence weight of family ``f`` over the packets contained in flow ``i`` (``0``
        if none).
    """
    flow_of_packet_size = int(contains_edge_index[1].max().item()) + 1 if contains_edge_index.numel() else 0
    # packet -> flow lookup (a packet belongs to one flow via contains)
    n_packets = max(flow_of_packet_size, 0)
    out = torch.zeros((num_flows, num_families), dtype=torch.float32)
    if contains_edge_index.numel() == 0:
        return out
    packet_to_flow = torch.full((n_packets,), -1, dtype=torch.long)
    packet_to_flow[contains_edge_index[1]] = contains_edge_index[0]

    for fam, (pkt_ids, weights) in evidence_per_family.items():
        if pkt_ids.numel() == 0:
            continue
        pkt_ids = pkt_ids.to(torch.long)
        weights = weights.to(torch.float32)
        # drop evidence packets that are out of range / not contained in any flow
        in_range = pkt_ids < n_packets
        pkt_ids, weights = pkt_ids[in_range], weights[in_range]
        flows = packet_to_flow[pkt_ids]
        valid = flows >= 0
        flows, weights = flows[valid], weights[valid]
        if flows.numel() == 0:
            continue
        # scatter-max weight into (flow, fam)
        col = out[:, fam]
        col.scatter_reduce_(0, flows, weights, reduce="amax", include_self=True)
        out[:, fam] = col
    return out


def neighbor_consensus(
    pred_probs: torch.Tensor,
    edge_index: torch.Tensor,
    seed_idx: torch.Tensor,
    seed_labels: torch.Tensor,
) -> torch.Tensor:
    """Soft graph-neighbor label agreement for each seed flow (Signal 2).

    For seed flow ``i`` with label ``y_i``, average over its neighbors ``j`` the
    predicted probability mass they place on ``y_i``::

        consensus_i = mean_{j in N(i)} pred_probs[j, y_i]

    High when neighbors' predictions support the flow's label (likely clean), low
    when they don't (the flow's label conflicts with its neighborhood -> likely
    noise). A flow with NO neighbors cannot be judged and defaults to ``1.0`` (we
    never penalise a flow merely for being unconnected).

    Args:
        pred_probs: ``(N, C)`` softmax over ALL flow nodes in the subgraph.
        edge_index: ``(2, E)`` undirected flow-flow edges in local node ids. Each
            column ``(a, b)`` contributes ``b`` as a neighbor of ``a`` (pass both
            directions, or this function will only see one).
        seed_idx: ``(S,)`` local ids of the seed flows to score.
        seed_labels: ``(S,)`` labels of the seed flows.

    Returns:
        ``(S,)`` consensus in ``[0, 1]``.
    """
    device = pred_probs.device
    num_nodes = pred_probs.shape[0]
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    seed_idx = seed_idx.to(device)
    seed_labels = seed_labels.to(device)
    S = seed_idx.shape[0]

    # Map each seed's source node id -> its position k in the output, so an edge
    # whose src is a seed contributes its dst's prob-mass to that seed. Non-seed
    # sources map to -1 and are dropped. Then scatter-sum + count for a mean.
    node_to_k = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    node_to_k[seed_idx] = torch.arange(S, device=device)

    k_of_edge = node_to_k[src]                      # (E,) seed-position or -1
    keep = k_of_edge >= 0
    k_e = k_of_edge[keep]
    dst_e = dst[keep]
    # prob mass each neighbor places on ITS seed's label
    label_e = seed_labels[k_e]                       # (Ekeep,)
    mass_e = pred_probs[dst_e, label_e]              # (Ekeep,)

    sums = torch.zeros(S, dtype=pred_probs.dtype, device=device)
    counts = torch.zeros(S, dtype=pred_probs.dtype, device=device)
    sums.scatter_add_(0, k_e, mass_e)
    counts.scatter_add_(0, k_e, torch.ones_like(mass_e))

    # mean where there are neighbors; default 1.0 where there are none
    out = torch.ones(S, dtype=pred_probs.dtype, device=device)
    has_nbr = counts > 0
    out[has_nbr] = sums[has_nbr] / counts[has_nbr]
    return out


def evidence_support(
    labels: torch.Tensor,
    evidence_by_flow: torch.Tensor,
    class_to_family: torch.Tensor,
) -> torch.Tensor:
    """MITRE-evidence grounding of each flow's (claimed) attack label (Signal 1).

    A flow labeled with an attack class is supported only if it carries a positive
    matching evidence-edge weight for that class's attack family. Non-attack classes
    (``class_to_family == -1``) are trusted by construction (they are not subject to
    the web-attack label pollution).

    Args:
        labels: ``(S,)`` flow labels.
        evidence_by_flow: ``(S, F)`` summed matching evidence-edge weight per flow
            per attack family ``F`` (>= 0). Column ordering matches ``class_to_family``
            family indices.
        class_to_family: ``(C,)`` map class id -> family column index, or ``-1`` for
            non-attack classes that need no grounding.

    Returns:
        ``(S,)`` support in ``[0, 1]`` (``1`` if grounded or non-attack, ``0`` if the
        label claims an attack but no matching evidence is present).
    """
    fam = class_to_family[labels]                      # (S,) family col or -1
    is_attack = fam >= 0
    out = torch.ones(labels.shape[0], dtype=evidence_by_flow.dtype, device=evidence_by_flow.device)
    if is_attack.any():
        idx = torch.nonzero(is_attack, as_tuple=True)[0]
        cols = fam[idx]
        w = evidence_by_flow[idx, cols]                # (n_attack,)
        out[idx] = (w > 0).to(evidence_by_flow.dtype)
    return out


def combine_clean_confidence(
    evidence: torch.Tensor, consensus: torch.Tensor, lam: float
) -> torch.Tensor:
    """Geometric blend of the two signals: ``evidence^lam * consensus^(1-lam)``.

    ``lam`` in ``[0, 1]`` weights the evidence (Signal 1) vs neighbor consensus
    (Signal 2). A near-zero value in EITHER signal drives the blend toward zero, so a
    flow is fully trusted only when both its grounded evidence AND its neighborhood
    support its label.
    """
    ev = evidence.clamp_min(0.0)
    co = consensus.clamp_min(0.0)
    return (ev ** lam) * (co ** (1.0 - lam))


def curriculum_weight(
    clean_conf: torch.Tensor, epoch: int, warmup_epochs: int, w_min: float
) -> torch.Tensor:
    """Per-sample loss weight: identity during warmup, then clamp to ``[w_min, 1]``.

    During warmup (``epoch < warmup_epochs``) every flow is fully weighted — the model
    must learn basic structure before its predictions are trustworthy enough to judge
    neighbors. Afterward the clean-confidence is clamped into ``[w_min, 1]`` so even a
    suspected-noisy flow keeps a small gradient (soft down-weight, never hard drop).
    """
    if epoch < warmup_epochs:
        return torch.ones_like(clean_conf)
    return clean_conf.clamp(min=w_min, max=1.0)


class EMAConsensusBuffer:
    """Persistent per-flow EMA of the clean-confidence across epochs (temporal
    consistency, SAT-style), so the weight does not jitter epoch-to-epoch.

    The first time a flow is updated, the raw value is stored (no stale-init bias);
    subsequent updates blend ``decay * old + (1 - decay) * new``. Flows never updated
    keep ``init``.
    """

    def __init__(self, num_flows: int, decay: float = 0.9, init: float = 1.0) -> None:
        self.decay = float(decay)
        self.values = torch.full((num_flows,), float(init))
        self.seen = torch.zeros(num_flows, dtype=torch.bool)

    def update(self, idx: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        """EMA-update flows ``idx`` with ``vals``; return the updated values for ``idx``."""
        idx = idx.to(torch.long).cpu()
        vals = vals.detach().to(self.values.dtype).cpu()
        seen = self.seen[idx]
        old = self.values[idx]
        blended = torch.where(seen, self.decay * old + (1.0 - self.decay) * vals, vals)
        self.values[idx] = blended
        self.seen[idx] = True
        return blended

    def get(self, idx: torch.Tensor) -> torch.Tensor:
        return self.values[idx.to(torch.long).cpu()]
