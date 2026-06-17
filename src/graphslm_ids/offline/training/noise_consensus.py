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

The previous Evidence-Prediction Contradiction (EPC) + 2-component EM soft-relabel path
was measured HARMFUL in a full training run and has been removed (2026-06-11).  It is
replaced by EACS (Evidence-Anchored Candidate-Set self-relabeling): suspect flows
(labeled with a polluted web-attack class AND carrying no matching MITRE evidence) receive
a soft target restricted to {own label, Benign}, disambiguated by the model's own
prediction, optionally modulated by graph-neighbor consensus, EMA-smoothed.  Anchors
(same classes WITH evidence) and all other classes keep hard labels.
See: docs/superpowers/specs/2026-06-11-eacs-noise-robust-design.md
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "neighbor_consensus",
    "evidence_support",
    "combine_clean_confidence",
    "curriculum_weight",
    "EMAConsensusBuffer",
    "build_evidence_by_flow",
    "static_evidence_weight",
    "build_class_to_family",
    "soft_target_focal_loss",
    "EACSController",
    "build_eacs_controller",
    "evidence_table_from_artifact",
    "class_to_family_from_csvs",
]


def build_class_to_family(
    class_to_tech: dict[str, list[tuple[str, float]]],
    tech_to_family: dict[str, str],
    family_to_col: dict[str, int],
    label_mapping: dict[str, int],
    num_classes: int,
) -> torch.Tensor:
    """Map each class id -> its dominant MITRE attack-family column, or -1 (non-attack).

    A class points at one or more techniques (with weights); each technique belongs to a
    family. We sum the class's technique weights per family and pick the family with the
    largest mass. Classes with no technique (e.g. Benign) or whose families are unknown
    map to ``-1`` so the evidence signal treats them as non-attack (always trusted).
    """
    out = torch.full((num_classes,), -1, dtype=torch.long)
    for class_name, cls_idx in label_mapping.items():
        if cls_idx < 0 or cls_idx >= num_classes:
            continue
        pairs = class_to_tech.get(class_name) or []
        fam_mass: dict[int, float] = {}
        for tech_id, weight in pairs:
            fam_name = tech_to_family.get(tech_id)
            if fam_name is None:
                continue
            col = family_to_col.get(fam_name)
            if col is None:
                continue
            fam_mass[col] = fam_mass.get(col, 0.0) + float(weight)
        if fam_mass:
            out[cls_idx] = max(fam_mass, key=fam_mass.get)
    return out


def static_evidence_weight(
    contains_edge_index: torch.Tensor,
    evidence_per_family: dict[int, tuple[torch.Tensor, torch.Tensor]],
    labels: torch.Tensor,
    class_to_family: torch.Tensor,
    num_families: int,
    w_min: float = 0.2,
) -> torch.Tensor:
    """Static per-flow loss weight from Signal 1 (MITRE evidence grounding) alone.

    Computes ``evidence_support`` for every flow (1 if its attack label is grounded by
    a matching MITRE evidence edge or the class is non-attack; 0 if it claims an attack
    with no matching evidence) and maps it to a loss weight in ``[w_min, 1]``:

        w_i = 1.0           if supported / non-attack
        w_i = w_min         if attack-labeled but ungrounded (prime noise candidate)

    This is independent of the model and of training epoch, so it can be precomputed
    once over ALL flows and indexed per batch — the least-invasive way to make the
    trainer noise-robust. Signal 2 (neighbor consensus) can refine this later.
    """
    num_flows = labels.shape[0]
    ev = build_evidence_by_flow(
        contains_edge_index, evidence_per_family, num_flows, num_families
    )
    support = evidence_support(labels, ev, class_to_family)   # (num_flows,) in {0,1}
    # map support 1 -> weight 1, support 0 -> weight w_min
    return w_min + (1.0 - w_min) * support


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


# Canonical MITRE evidence family order (must match graph_builder._EVIDENCE_FAMILIES
# and the family-head column order).
_FAMILY_ORDER: tuple[str, ...] = (
    "injection", "command_exec", "file_upload", "recon", "c2_beacon",
)


def evidence_table_from_artifact(artifact) -> torch.Tensor:
    """Per-flow, per-family MITRE evidence table (num_flows, 5) from a loaded
    graph artifact (flow->contains->packet + packet->evidence_*->technique)."""
    ei = artifact.edge_index
    ea = artifact.edge_attr
    family_to_col = {fam: i for i, fam in enumerate(_FAMILY_ORDER)}
    contains = torch.as_tensor(
        np.asarray(ei[("flow", "contains", "packet")], dtype=np.int64)
    )
    evidence_per_family: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for fam in _FAMILY_ORDER:
        key = ("packet", f"evidence_{fam}", "technique")
        if key not in ei:
            continue
        edge = np.asarray(ei[key], dtype=np.int64)
        pkt_ids = torch.as_tensor(edge[0])
        attr = ea.get(key)
        if attr is None:
            weights = torch.ones(pkt_ids.shape[0], dtype=torch.float32)
        else:
            attr = np.asarray(attr, dtype=np.float32).reshape(pkt_ids.shape[0], -1)
            weights = torch.as_tensor(attr[:, 0])
        evidence_per_family[family_to_col[fam]] = (pkt_ids, weights)
    num_flows = int(artifact.node_features["flow"].shape[0])
    return build_evidence_by_flow(
        contains, evidence_per_family, num_flows, len(_FAMILY_ORDER)
    )


def class_to_family_from_csvs(
    mitre_dir: str, label_mapping: dict[str, int], num_classes: int
) -> torch.Tensor:
    """class id -> dominant MITRE family column (or -1) from the two MITRE CSVs."""
    import csv as _csv
    from pathlib import Path as _Path

    family_to_col = {fam: i for i, fam in enumerate(_FAMILY_ORDER)}
    mdir = _Path(mitre_dir)
    class_to_tech: dict[str, list[tuple[str, float]]] = {}
    ctm = mdir / "class_technique_map.csv"
    if ctm.exists():
        for row in _csv.DictReader(ctm.open(encoding="utf-8")):
            cls = (row.get("class") or "").strip()
            tech = (row.get("technique") or "").strip()
            if not cls or not tech:
                continue
            try:
                w = float(row.get("weight") or 1.0)
            except ValueError:
                w = 1.0
            class_to_tech.setdefault(cls, []).append((tech, w))
    tech_to_family: dict[str, str] = {}
    tf = mdir / "technique_family.csv"
    if tf.exists():
        for row in _csv.DictReader(tf.open(encoding="utf-8")):
            t = (row.get("technique") or "").strip()
            f = (row.get("family") or "").strip()
            if t and f:
                tech_to_family[t] = f
    return build_class_to_family(
        class_to_tech, tech_to_family, family_to_col, label_mapping, num_classes
    )


def soft_target_focal_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    loss_type: str = "ce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Soft-target counterpart of the trainer's ``_compute_train_loss``.

    Reduces EXACTLY to the hard-label focal loss when ``soft_targets`` is one-hot
    (same smoothing blend, focal modulation, per-class alpha, mean reduction) —
    so warmup epochs and non-suspect flows reproduce the baseline loss bit-for-bit.
    The focal factor and alpha are keyed on the soft target's EXPECTATION::

        p_t     = sum_c t_c * p_c
        alpha_t = sum_c t_c * weight_c

    so a flow whose target has shifted toward Benign is focal-weighted and
    alpha-weighted as (mostly) a Benign sample, not as its polluted attack label.
    (For ``loss_type='ce'`` with ``weight`` the reduction differs from
    ``F.cross_entropy``'s weight-normalised mean; the trainer only routes
    focal/cb_focal through here.)
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0:
        uniform = torch.full_like(log_probs, 1.0 / log_probs.shape[-1])
        t = (1.0 - label_smoothing) * soft_targets + label_smoothing * uniform
    else:
        t = soft_targets
    ce = -(t * log_probs).sum(dim=-1)
    if loss_type in ("focal", "cb_focal"):
        p_t = (soft_targets * log_probs.exp()).sum(dim=-1)
        ce = (1.0 - p_t).pow(focal_gamma) * ce
    if weight is not None:
        alpha_t = soft_targets @ weight.to(soft_targets.dtype)
        ce = alpha_t * ce
    return ce.mean()


class EACSController:
    """Evidence-Anchored Candidate-Set self-relabeling (design 2026-06-11).

    Flow groups, precomputed once over ALL flows:
      * suspect  — labeled with a polluted web class AND carrying no matching-family
        MITRE evidence: the prime noise candidates. Candidate set {y, Benign}.
      * anchor   — same classes WITH matching evidence: grounded true attacks,
        beta=1 forever (they teach the model the real attack pattern).
      * everything else — untouched, beta=1.

    Per epoch after warmup, for suspect flows only::

        beta_raw = p[y] / (p[y] + p[Benign])          # 2-way disambiguation
        beta     = beta_raw^lam * consensus^(1-lam)   # optional neighbor modulation
        beta     = EMA(beta)                          # per-flow, across epochs
        target   = beta*onehot(y) + (1-beta)*onehot(Benign)

    The candidate-set restriction structurally bounds confirmation bias: a suspect
    can only move between its own label and Benign, never to a third class.
    """

    def __init__(
        self,
        *,
        suspect_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        benign_class_id: int,
        num_classes: int,
        warmup_epochs: int = 5,
        ema_decay: float = 0.9,
        lambda_disambig: float = 0.7,
    ) -> None:
        self.suspect_mask = suspect_mask.to(torch.bool).cpu()
        self.anchor_mask = anchor_mask.to(torch.bool).cpu()
        self.benign_class_id = int(benign_class_id)
        self.num_classes = int(num_classes)
        self.warmup_epochs = int(warmup_epochs)
        self.lambda_disambig = float(lambda_disambig)
        self.beta_buffer = EMAConsensusBuffer(
            num_flows=int(suspect_mask.shape[0]), decay=ema_decay, init=1.0
        )
        self._stats = self._fresh_stats()

    @staticmethod
    def _fresh_stats() -> dict:
        return {"suspects_seen": 0, "relabeled": 0, "beta_sum": 0.0, "per_class": {}}

    def epoch_stats(self, reset: bool = False) -> dict:
        s = self._stats
        out = {
            "suspects_seen": s["suspects_seen"],
            "relabeled": s["relabeled"],
            "mean_beta": (s["beta_sum"] / s["suspects_seen"]) if s["suspects_seen"] else 1.0,
            "per_class": dict(s["per_class"]),
        }
        if reset:
            self._stats = self._fresh_stats()
        return out

    def soft_targets(
        self,
        class_logits: torch.Tensor,
        seed_global_ids: torch.Tensor,
        seed_labels: torch.Tensor,
        epoch: int,
        consensus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(S, C) soft training targets for the seed flows. Pure one-hot during
        warmup and for every non-suspect row."""
        device = class_logits.device
        p = torch.softmax(class_logits.detach().float(), dim=1)
        onehot = torch.zeros_like(p)
        onehot[torch.arange(seed_labels.shape[0], device=device), seed_labels] = 1.0
        if epoch <= self.warmup_epochs:
            return onehot
        sus = self.suspect_mask.to(device)[seed_global_ids]
        if not bool(sus.any()):
            return onehot
        p_y = p.gather(1, seed_labels.view(-1, 1)).squeeze(1)
        p_b = p[:, self.benign_class_id]
        beta_raw = p_y / (p_y + p_b).clamp_min(1e-12)
        if consensus is not None and self.lambda_disambig < 1.0:
            lam = self.lambda_disambig
            beta_raw = beta_raw.clamp(1e-8, 1.0).pow(lam) * (
                consensus.to(device).to(beta_raw.dtype).clamp(1e-8, 1.0).pow(1.0 - lam)
            )
        sus_idx = torch.nonzero(sus, as_tuple=False).reshape(-1)
        beta_sus = self.beta_buffer.update(
            seed_global_ids[sus_idx], beta_raw[sus_idx]
        ).to(device).to(p.dtype)
        beta = torch.ones_like(beta_raw)
        beta[sus_idx] = beta_sus
        benign_onehot = torch.zeros_like(p)
        benign_onehot[:, self.benign_class_id] = 1.0
        targets = beta.unsqueeze(1) * onehot + (1.0 - beta).unsqueeze(1) * benign_onehot
        targets = torch.where(sus.unsqueeze(1), targets, onehot)
        # diagnostics
        self._stats["suspects_seen"] += int(sus_idx.numel())
        self._stats["relabeled"] += int((beta_sus < 0.5).sum().item())
        self._stats["beta_sum"] += float(beta_sus.sum().item())
        sus_labels = seed_labels[sus_idx]
        for cid in sus_labels.unique().tolist():
            pc = self._stats["per_class"].setdefault(int(cid), {"seen": 0, "relabeled": 0})
            cmask = sus_labels == cid
            pc["seen"] += int(cmask.sum().item())
            pc["relabeled"] += int((beta_sus[cmask] < 0.5).sum().item())
        return targets


def build_eacs_controller(
    *,
    artifact,
    num_classes: int,
    label_mapping: dict[str, int],
    suspect_classes: list[str],
    warmup_epochs: int = 5,
    ema_decay: float = 0.9,
    lambda_disambig: float = 0.7,
    mitre_dir: str = "data/mitre",
    anchor_mask: np.ndarray | None = None,
) -> EACSController:
    """Assemble an EACSController from a loaded graph artifact + MITRE CSVs.

    ``anchor_mask`` (bool, num_flows) overrides the default evidence-weight
    anchor rule. The default — any matching-family MSEE weight > 0 — measures
    16% precision against the clean answer key (5.4k background flows anchored
    to wrong hard attack labels); a precomputed procedure-literal mask
    (scripts/tools/extract_eacs_anchor_mask.py) measures 95% / recall ~1.0.
    """
    if "Benign" not in label_mapping:
        raise ValueError("EACS requires a 'Benign' class in label_mapping")
    flow_labels = torch.as_tensor(np.asarray(artifact.flow_y, dtype=np.int64))
    sus_ids = sorted(
        label_mapping[c] for c in suspect_classes if c in label_mapping
    )
    if not sus_ids:
        raise ValueError(f"none of suspect_classes {suspect_classes!r} in label_mapping")
    in_suspect_class = torch.isin(flow_labels, torch.tensor(sus_ids, dtype=torch.long))
    if anchor_mask is not None:
        if anchor_mask.shape[0] != flow_labels.shape[0]:
            raise ValueError(
                f"anchor_mask has {anchor_mask.shape[0]} flows, artifact has "
                f"{flow_labels.shape[0]}"
            )
        has_matching_ev = torch.as_tensor(np.asarray(anchor_mask, dtype=bool))
    else:
        evidence_by_flow = evidence_table_from_artifact(artifact)
        class_to_family = class_to_family_from_csvs(
            mitre_dir, label_mapping, num_classes
        )
        fam = class_to_family[flow_labels]
        has_matching_ev = torch.zeros_like(in_suspect_class)
        ok = fam >= 0
        idx_ok = torch.arange(flow_labels.shape[0])[ok]
        has_matching_ev[idx_ok] = evidence_by_flow[idx_ok, fam[ok]] > 0
    return EACSController(
        suspect_mask=in_suspect_class & ~has_matching_ev,
        anchor_mask=in_suspect_class & has_matching_ev,
        benign_class_id=int(label_mapping["Benign"]),
        num_classes=num_classes,
        warmup_epochs=warmup_epochs,
        ema_decay=ema_decay,
        lambda_disambig=lambda_disambig,
    )
