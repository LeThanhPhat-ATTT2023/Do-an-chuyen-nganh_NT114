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
    "evidence_prediction_contradiction",
    "em_clean_confidence",
    "soft_relabel_target",
    "build_class_to_family",
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


def evidence_prediction_contradiction(
    q: torch.Tensor, e: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Evidence-Prediction Contradiction (EPC) — the DYNAMIC noise signal.

    ``q`` (N, F) is the model's CURRENT predicted attack-family distribution for each
    flow (evolves every epoch as the model learns); ``e`` (N, F) is the grounded MITRE
    evidence mass per family (Tầng-3, from the graph). EPC is ``1 - cos(q, e)``: low
    when what the model now thinks the flow is agrees with its grounded evidence (likely
    clean), high when they contradict (label suspect).

    A flow with NO evidence (``||e|| == 0``, background traffic) cannot be judged by
    evidence and returns a NEUTRAL ``0.5`` — the EM step + label handle it; we neither
    punish nor reward purely for missing evidence.
    """
    q_norm = q.norm(dim=1)
    e_norm = e.norm(dim=1)
    dot = (q * e).sum(dim=1)
    cos = dot / (q_norm * e_norm).clamp_min(eps)
    epc = 1.0 - cos
    # neutral 0.5 where there is no evidence to contradict
    no_evidence = e_norm <= eps
    epc = torch.where(no_evidence, torch.full_like(epc, 0.5), epc)
    return epc.clamp(0.0, 1.0)


def em_clean_confidence(epc: torch.Tensor, n_iter: int = 30, eps: float = 1e-6) -> torch.Tensor:
    """Per-flow clean-confidence ``beta`` via a 2-component 1-D Gaussian mixture on EPC.

    The clean component has the SMALLER mean EPC (predictions agree with evidence). We
    fit means/variances/mixing-weights by EM (recomputed each epoch, so beta tracks the
    model), then return the posterior probability of the clean component as ``beta_i``.
    This is the "model decides which labels to trust, and gets better each epoch" step —
    no hand-set threshold (DivideMix/ICGNN style).
    """
    x = epc.detach().float().reshape(-1)
    n = x.numel()
    if n == 0:
        return x
    # init: clean mean at the low end, noisy mean at the high end. All init tensors
    # are created on x.device so the EM runs entirely on the input's device (cuda/cpu).
    dev = x.device
    lo, hi = x.min(), x.max()
    mu = torch.stack([lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)])
    var = torch.full((2,), float(((hi - lo) ** 2).clamp_min(eps) / 4 + eps), device=dev)
    pi = torch.tensor([0.5, 0.5], device=dev)

    for _ in range(n_iter):
        # E-step: responsibilities
        diff2 = (x[:, None] - mu[None, :]) ** 2
        log_g = -0.5 * (diff2 / var[None, :].clamp_min(eps) + torch.log(2 * torch.pi * var[None, :].clamp_min(eps)))
        log_w = torch.log(pi.clamp_min(eps))[None, :] + log_g
        log_w = log_w - log_w.logsumexp(dim=1, keepdim=True)
        r = log_w.exp()                                   # (n, 2)
        # M-step
        nk = r.sum(dim=0).clamp_min(eps)
        pi = nk / n
        mu = (r * x[:, None]).sum(dim=0) / nk
        var = (r * (x[:, None] - mu[None, :]) ** 2).sum(dim=0) / nk
        var = var.clamp_min(eps)

    # clean component = the one with the smaller mean EPC
    clean_k = int(torch.argmin(mu).item())
    diff2 = (x[:, None] - mu[None, :]) ** 2
    log_g = -0.5 * (diff2 / var[None, :].clamp_min(eps) + torch.log(2 * torch.pi * var[None, :].clamp_min(eps)))
    log_w = torch.log(pi.clamp_min(eps))[None, :] + log_g
    log_w = log_w - log_w.logsumexp(dim=1, keepdim=True)
    beta = log_w.exp()[:, clean_k]
    return beta.clamp(0.0, 1.0)


def soft_relabel_target(
    labels: torch.Tensor, q: torch.Tensor, beta: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Soft training target blending the given (noisy) label with the model prediction.

    ``target_i = beta_i · onehot(y_i) + (1 - beta_i) · q_i`` (ICGNN Eq. 6). When the
    label is believed clean (``beta→1``) the target is the given label; when believed
    noisy (``beta→0``) the model is trained toward its own family-aware prediction
    instead of the wrong label. ``q`` here is the per-CLASS prediction (N, num_classes).
    Rows sum to 1.
    """
    onehot = torch.zeros((labels.shape[0], num_classes), dtype=q.dtype, device=q.device)
    onehot[torch.arange(labels.shape[0]), labels] = 1.0
    b = beta.to(q.dtype).reshape(-1, 1)
    return b * onehot + (1.0 - b) * q


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


class NoiseRobustController:
    """Ties the dynamic noise-detection pieces together for the trainer.

    Holds the global per-flow MITRE evidence table and the class->family map (built
    once from the artifact), plus a persistent EMA buffer of the per-flow clean-
    confidence ``beta``. Each training step calls :meth:`soft_targets`, which:

      1. turns the model's family logits into a prediction ``q`` (dynamic, this epoch),
      2. gathers each seed flow's grounded family evidence ``e`` (Tầng-3, static),
      3. computes the Evidence-Prediction Contradiction ``EPC = 1 - cos(q, e)``,
      4. fits a 2-component EM on the batch EPC -> per-flow ``beta`` (clean posterior),
         EMA-smoothed across epochs,
      5. returns the soft-relabel target ``beta*onehot(label) + (1-beta)*p`` where ``p``
         is the model's CLASS prediction.

    During warmup (``epoch < warmup_epochs``) it returns the hard one-hot label so the
    model first learns basic structure before it is trusted to relabel.
    """

    def __init__(
        self,
        evidence_by_flow: torch.Tensor,
        class_to_family: torch.Tensor,
        num_classes: int,
        num_families: int,
        warmup_epochs: int = 5,
        ema_decay: float = 0.9,
        em_iters: int = 20,
    ) -> None:
        self.evidence_by_flow = evidence_by_flow            # (num_flows, num_families)
        self.class_to_family = class_to_family              # (num_classes,)
        self.num_classes = int(num_classes)
        self.num_families = int(num_families)
        self.warmup_epochs = int(warmup_epochs)
        self.em_iters = int(em_iters)
        self.beta_buffer = EMAConsensusBuffer(
            num_flows=evidence_by_flow.shape[0], decay=ema_decay, init=1.0
        )

    def soft_targets(
        self,
        class_logits: torch.Tensor,
        family_logits: torch.Tensor,
        seed_global_ids: torch.Tensor,
        seed_labels: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        device = class_logits.device
        p = torch.softmax(class_logits.float(), dim=1)           # (S, C) class pred
        if epoch < self.warmup_epochs or family_logits is None:
            # hard label during warmup -> identical to standard training
            onehot = torch.zeros_like(p)
            onehot[torch.arange(seed_labels.shape[0], device=device), seed_labels] = 1.0
            return onehot

        q = torch.softmax(family_logits.float(), dim=1)          # (S, F) family pred
        e = self.evidence_by_flow.to(device)[seed_global_ids]    # (S, F) evidence
        epc = evidence_prediction_contradiction(q, e)            # (S,)
        beta_batch = em_clean_confidence(epc, n_iter=self.em_iters)  # (S,)
        # EMA-smooth beta per flow across epochs for stability
        beta = self.beta_buffer.update(seed_global_ids, beta_batch).to(device)
        return soft_relabel_target(seed_labels, p, beta, self.num_classes)


__all__.append("NoiseRobustController")


# Canonical MITRE evidence family order (must match graph_builder._EVIDENCE_FAMILIES
# and the family-head column order).
_FAMILY_ORDER: tuple[str, ...] = (
    "injection", "command_exec", "file_upload", "recon", "c2_beacon",
)


def build_noise_robust_controller(
    *,
    artifact,
    num_classes: int,
    label_mapping: dict[str, int],
    warmup_epochs: int = 5,
    ema_decay: float = 0.9,
    mitre_dir: str = "data/mitre",
) -> "NoiseRobustController":
    """Assemble a :class:`NoiseRobustController` from a loaded graph artifact + MITRE CSVs.

    Reads the flow->contains->packet edges and the 5 packet->evidence_{family}->technique
    edges from the artifact to build the per-flow evidence table, and the
    class_technique_map.csv + technique_family.csv to build the class->family map.
    """
    import csv as _csv
    from pathlib import Path as _Path

    family_to_col = {fam: i for i, fam in enumerate(_FAMILY_ORDER)}
    num_families = len(_FAMILY_ORDER)

    # --- per-flow evidence table from artifact edges ---
    ei = artifact.edge_index
    ea = artifact.edge_attr
    contains = torch.as_tensor(
        np.asarray(ei[("flow", "contains", "packet")], dtype=np.int64)
    )
    evidence_per_family: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for fam in _FAMILY_ORDER:
        key = ("packet", f"evidence_{fam}", "technique")
        if key not in ei:
            continue
        edge = np.asarray(ei[key], dtype=np.int64)
        pkt_ids = torch.as_tensor(edge[0])                       # packet is src
        attr = ea.get(key)
        if attr is None:
            weights = torch.ones(pkt_ids.shape[0], dtype=torch.float32)
        else:
            attr = np.asarray(attr, dtype=np.float32).reshape(pkt_ids.shape[0], -1)
            weights = torch.as_tensor(attr[:, 0])                # first attr col = weight
        evidence_per_family[family_to_col[fam]] = (pkt_ids, weights)
    num_flows = int(artifact.node_features["flow"].shape[0])
    evidence_by_flow = build_evidence_by_flow(
        contains, evidence_per_family, num_flows, num_families
    )

    # --- class -> family map from the two CSVs ---
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
    class_to_family = build_class_to_family(
        class_to_tech, tech_to_family, family_to_col, label_mapping, num_classes
    )

    return NoiseRobustController(
        evidence_by_flow=evidence_by_flow,
        class_to_family=class_to_family,
        num_classes=num_classes,
        num_families=num_families,
        warmup_epochs=warmup_epochs,
        ema_decay=ema_decay,
    )


__all__.append("build_noise_robust_controller")
