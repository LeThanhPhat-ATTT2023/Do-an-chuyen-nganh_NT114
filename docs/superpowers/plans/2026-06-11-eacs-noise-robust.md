# EACS Noise-Robust Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the measurably-harmful EPC/EM soft-relabel path with EACS (Evidence-Anchored Candidate-Set Self-relabeling) and add the standard-LNL clean-answer-key evaluation, targeting `val_macro_f1_clean ≥ 0.90`.

**Architecture:** Three components per `docs/superpowers/specs/2026-06-11-eacs-noise-robust-design.md`: (C1) an eval-only clean answer key derived from the existing `flow_attack_labeler` signatures, (C2) an `EACSController` that soft-relabels ONLY suspect flows (web-attack label, no MITRE evidence) within the 2-way candidate set {own label, Benign}, (C3) per-epoch diagnostics + dual val metrics + end-of-run noise-detection ROC-AUC.

**Tech Stack:** PyTorch (CPU local / CUDA remote), numpy, pytest. Python: `D:\v\nt114\Scripts\python.exe` locally; `~/venv/bin/python` on hgt-aws.

**Parallelism:** Task 1 and Task 2 touch disjoint files — dispatch in parallel. Task 3 depends on Task 1. Task 4 depends on 1-3 and is run by the orchestrator (needs ssh).

**Key facts an implementer needs (verified against the codebase):**
- Trainer: `src/graphslm_ids/offline/training/train_hgt_flow_classifier.py` (~3000 lines).
  - `_nr_cfg` / `_nr_enabled` / `_nr_num_families` defined at lines 1893-1895.
  - Controller build block: lines 2164-2198 (after `label_names` at 2162).
  - Train-step soft-relabel block to replace: lines 2321-2362 (inside the autocast).
  - `logits` at line 2317/2319 covers ALL flow nodes in the subgraph (classifier over
    every flow embedding), on BOTH the gcl and non-gcl paths; `sm` is the seed mask,
    `sl` the seed labels, `ei` the edge-index dict, `batch.seed_flow_ids` the seeds'
    GLOBAL flow ids (used at line 2331-2333).
  - `_compute_train_loss` at 960, `_weighted_mean` at 945, `metrics_from_predictions`
    at 1444, `_metrics_from_counts` at 1404, `_per_class_counts_tensor` at 1381.
  - `evaluate_neighbor_sampling` at 1579 (DDP branch ~1630-1686, single-device branch
    ~1687 onward); val call site at 2725-2741; epoch summary print at 2841-2869;
    monitor whitelist at 2200-2205; `_compute_monitor_score` at 896; `[hgaa]` epoch
    log block at 2782-2790 (model for the `[eacs]` log placement); `history`/metrics
    json at 2234 / 2942.
  - Artifact labels: `backend.artifact.flow_y` (np.int64, all flows).
  - Epochs are 1-based in the train loop.
- `noise_consensus.py`: keep `build_evidence_by_flow`, `neighbor_consensus`,
  `evidence_support`, `combine_clean_confidence`, `curriculum_weight`,
  `static_evidence_weight`, `EMAConsensusBuffer`, `build_class_to_family`,
  `_FAMILY_ORDER`. DELETE `evidence_prediction_contradiction`, `em_clean_confidence`,
  `soft_relabel_target`, `family_supervision_loss`, `NoiseRobustController`,
  `build_noise_robust_controller` (the EPC/EM path — measured harmful).
- Flow id format in `graph.meta.json::flow_id_order`:
  `"<Label>|<lo_ip:port>|<hi_ip:port>|<proto>#<seg>.<dir>"` e.g.
  `Backdoor_Malware|0.0.0.0:68|255.255.255.255:67|1#1.0`. The canonical key used by
  `flow_attack_labeler._canon_key` is the middle part: `lo|hi|proto`.
- `label_pcap_flows(pcap_path, original_label)` (in
  `src/graphslm_ids/offline/preprocessing/flow_attack_labeler.py`) returns
  `(flow_key -> new_label, audit_counts)`; for web classes the keys are canonical
  5-tuple keys; a key maps to the attack label iff a matching HTTP signature was seen.
- `label_mapping` (graph.meta.json) is name→id, 18 classes, `Benign`=1.
- Remote: host `hgt-aws`, repo at `~/Do-an-chuyen-nganh_NT114`, venv `~/venv`,
  graph at `outputs/v3_ob/`. Local pcaps: `data/raw/14gb/<class>/*.pcap`.

---

### Task 1: Rewrite `noise_consensus.py` — delete EPC/EM, add EACS

**Files:**
- Modify: `src/graphslm_ids/offline/training/noise_consensus.py`
- Modify: `tests/test_noise_consensus.py` (delete tests of removed functions)
- Create: `tests/test_eacs_controller.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eacs_controller.py`:

```python
"""EACS — Evidence-Anchored Candidate-Set self-relabeling (unit, CPU, pure)."""
import numpy as np
import pytest
import torch

from graphslm_ids.offline.training.noise_consensus import (
    EACSController,
    soft_target_focal_loss,
)

NUM_CLASSES = 4
BENIGN = 1  # class ids: 0=AttackA, 1=Benign, 2=AttackB, 3=Other


def _make_controller(**kw):
    # 6 flows: 0,1 suspects (AttackA label, no evidence); 2 anchor (AttackA + evidence);
    # 3 benign; 4 other-class; 5 suspect (AttackB label, no evidence).
    suspect = torch.tensor([True, True, False, False, False, True])
    anchor = torch.tensor([False, False, True, False, False, False])
    defaults = dict(
        suspect_mask=suspect,
        anchor_mask=anchor,
        benign_class_id=BENIGN,
        num_classes=NUM_CLASSES,
        warmup_epochs=2,
        ema_decay=0.0,  # no smoothing -> assertions are exact
        lambda_disambig=1.0,  # no consensus term unless a test passes one
    )
    defaults.update(kw)
    return EACSController(**defaults)


def _logits_for(p_y: float, p_benign: float, y: int) -> torch.Tensor:
    """Single-row logits whose softmax puts ~p_y on y and ~p_benign on BENIGN."""
    rest = (1.0 - p_y - p_benign) / (NUM_CLASSES - 2)
    probs = torch.full((1, NUM_CLASSES), rest)
    probs[0, y] = p_y
    probs[0, BENIGN] = p_benign
    return probs.clamp_min(1e-9).log()


def test_warmup_returns_pure_onehot():
    ctrl = _make_controller(warmup_epochs=3)
    logits = _logits_for(0.1, 0.8, y=0)
    tgt = ctrl.soft_targets(logits, torch.tensor([0]), torch.tensor([0]), epoch=3)
    assert torch.allclose(tgt, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_non_suspect_rows_stay_onehot_after_warmup():
    ctrl = _make_controller()
    # flow 2 = anchor, flow 3 = benign, flow 4 = other class
    logits = torch.cat([_logits_for(0.05, 0.9, y=0)] * 3, dim=0)
    gids = torch.tensor([2, 3, 4])
    labels = torch.tensor([0, 1, 3])
    tgt = ctrl.soft_targets(logits, gids, labels, epoch=5)
    expect = torch.zeros(3, NUM_CLASSES)
    expect[torch.arange(3), labels] = 1.0
    assert torch.allclose(tgt, expect)


def test_suspect_beta_is_two_way_disambiguation():
    ctrl = _make_controller()
    # p[y]=0.2, p[Benign]=0.6 -> beta = 0.2/0.8 = 0.25
    logits = _logits_for(0.2, 0.6, y=0)
    tgt = ctrl.soft_targets(logits, torch.tensor([0]), torch.tensor([0]), epoch=5)
    beta = 0.2 / (0.2 + 0.6)
    expect = torch.zeros(1, NUM_CLASSES)
    expect[0, 0] = beta
    expect[0, BENIGN] = 1.0 - beta
    assert torch.allclose(tgt, expect, atol=1e-5)
    # rows sum to 1
    assert torch.allclose(tgt.sum(dim=1), torch.ones(1))


def test_confident_in_label_keeps_label():
    ctrl = _make_controller()
    logits = _logits_for(0.9, 0.02, y=0)
    tgt = ctrl.soft_targets(logits, torch.tensor([0]), torch.tensor([0]), epoch=5)
    assert tgt[0, 0].item() > 0.95


def test_consensus_modulates_beta_geometrically():
    ctrl = _make_controller(lambda_disambig=0.5)
    logits = _logits_for(0.5, 0.5, y=0)  # beta_raw = 0.5
    cons = torch.tensor([0.125])
    tgt = ctrl.soft_targets(
        logits, torch.tensor([0]), torch.tensor([0]), epoch=5, consensus=cons
    )
    # beta = 0.5^0.5 * 0.125^0.5 = sqrt(0.0625) = 0.25
    assert abs(tgt[0, 0].item() - 0.25) < 1e-4


def test_ema_smooths_beta_across_calls():
    ctrl = _make_controller(ema_decay=0.5)
    g = torch.tensor([0])
    y = torch.tensor([0])
    ctrl.soft_targets(_logits_for(0.8, 0.1, y=0), g, y, epoch=5)  # beta1=0.8/0.9
    tgt2 = ctrl.soft_targets(_logits_for(0.1, 0.8, y=0), g, y, epoch=5)
    b1 = 0.8 / 0.9
    b2_raw = 0.1 / 0.9
    expect = 0.5 * b1 + 0.5 * b2_raw
    assert abs(tgt2[0, 0].item() - expect) < 1e-4


def test_epoch_stats_counts_suspects_and_relabels():
    ctrl = _make_controller()
    logits = torch.cat([_logits_for(0.1, 0.8, y=0), _logits_for(0.9, 0.02, y=2)])
    ctrl.soft_targets(logits, torch.tensor([0, 5]), torch.tensor([0, 2]), epoch=5)
    stats = ctrl.epoch_stats(reset=True)
    assert stats["suspects_seen"] == 2
    assert stats["relabeled"] == 1  # only the beta<0.5 one
    assert 0.0 < stats["mean_beta"] < 1.0
    # reset worked
    assert ctrl.epoch_stats(reset=False)["suspects_seen"] == 0


def test_mask_construction_via_builder_logic():
    # mirrors build_eacs_controller's mask derivation (pure tensor logic)
    flow_labels = torch.tensor([0, 0, 0, 1, 3, 2])
    suspect_ids = torch.tensor([0, 2])
    class_to_family = torch.tensor([0, -1, 1, -1])  # AttackA->fam0, AttackB->fam1
    evidence = torch.zeros(6, 2)
    evidence[2, 0] = 0.9  # flow 2 has matching family evidence
    in_sus = torch.isin(flow_labels, suspect_ids)
    fam = class_to_family[flow_labels]
    has_ev = torch.zeros(6, dtype=torch.bool)
    ok = fam >= 0
    has_ev[ok] = evidence[torch.arange(6)[ok], fam[ok]] > 0
    assert (in_sus & ~has_ev).tolist() == [True, True, False, False, False, True]
    assert (in_sus & has_ev).tolist() == [False, False, True, False, False, False]


# ---------- soft_target_focal_loss ----------

def test_soft_loss_equals_hard_focal_when_onehot():
    from graphslm_ids.offline.training.train_hgt_flow_classifier import (
        _compute_train_loss,
    )
    torch.manual_seed(0)
    logits = torch.randn(16, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (16,))
    weight = torch.rand(NUM_CLASSES) + 0.5
    onehot = torch.nn.functional.one_hot(labels, NUM_CLASSES).float()
    for ls in (0.0, 0.05):
        hard = _compute_train_loss(
            logits, labels, weight,
            loss_type="focal", label_smoothing=ls, focal_gamma=1.5,
        )
        soft = soft_target_focal_loss(
            logits, onehot, weight,
            loss_type="focal", label_smoothing=ls, focal_gamma=1.5,
        )
        assert torch.allclose(hard, soft, atol=1e-6), (ls, hard, soft)


def test_soft_loss_moves_toward_benign_target():
    logits = _logits_for(0.1, 0.8, y=0).repeat(4, 1)
    onehot = torch.zeros(4, NUM_CLASSES)
    onehot[:, 0] = 1.0
    soft = onehot.clone()
    soft[:, 0] = 0.2
    soft[:, BENIGN] = 0.8
    l_hard = soft_target_focal_loss(logits, onehot, None, loss_type="focal", focal_gamma=1.5)
    l_soft = soft_target_focal_loss(logits, soft, None, loss_type="focal", focal_gamma=1.5)
    # prediction already favors Benign -> soft target must give LOWER loss
    assert l_soft.item() < l_hard.item()
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/test_eacs_controller.py -q`
Expected: FAIL / ERROR with `ImportError: cannot import name 'EACSController'`.

- [ ] **Step 3: Edit `noise_consensus.py` — delete the EPC/EM pieces**

Delete these definitions entirely (and their `__all__` entries):
`evidence_prediction_contradiction`, `em_clean_confidence`, `soft_relabel_target`,
`family_supervision_loss`, `class NoiseRobustController`,
`build_noise_robust_controller`, and the two trailing `__all__.append(...)` lines.
KEEP: module docstring (update per Step 4), `build_class_to_family`,
`build_evidence_by_flow`, `neighbor_consensus`, `evidence_support`,
`combine_clean_confidence`, `curriculum_weight`, `static_evidence_weight`,
`EMAConsensusBuffer`, `_FAMILY_ORDER`.

Extract the artifact/CSV plumbing of the deleted `build_noise_robust_controller`
into two shared helpers (same code, new names):

```python
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
```

- [ ] **Step 4: Add EACS to `noise_consensus.py`**

Update the module docstring's last paragraph to describe EACS (anchor/suspect/
untouched groups, 2-way candidate set {y, Benign}, neighbor-consensus modulation,
EMA smoothing; EPC/EM removed — see the 2026-06-11 design doc). Then append:

```python
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
) -> EACSController:
    """Assemble an EACSController from a loaded graph artifact + MITRE CSVs."""
    if "Benign" not in label_mapping:
        raise ValueError("EACS requires a 'Benign' class in label_mapping")
    evidence_by_flow = evidence_table_from_artifact(artifact)
    class_to_family = class_to_family_from_csvs(mitre_dir, label_mapping, num_classes)
    flow_labels = torch.as_tensor(np.asarray(artifact.flow_y, dtype=np.int64))
    sus_ids = sorted(
        label_mapping[c] for c in suspect_classes if c in label_mapping
    )
    if not sus_ids:
        raise ValueError(f"none of suspect_classes {suspect_classes!r} in label_mapping")
    in_suspect_class = torch.isin(flow_labels, torch.tensor(sus_ids, dtype=torch.long))
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
```

Add to `__all__`: `"soft_target_focal_loss"`, `"EACSController"`,
`"build_eacs_controller"`, `"evidence_table_from_artifact"`,
`"class_to_family_from_csvs"`.

- [ ] **Step 5: Prune `tests/test_noise_consensus.py`**

Delete every test that imports or exercises the removed names
(`evidence_prediction_contradiction`, `em_clean_confidence`, `soft_relabel_target`,
`family_supervision_loss`, `NoiseRobustController`, `build_noise_robust_controller`)
and remove those names from its imports. Keep all tests of the surviving functions.

- [ ] **Step 6: Run the module's tests**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/test_eacs_controller.py tests/test_noise_consensus.py -q`
Expected: ALL PASS.

NOTE on `EMAConsensusBuffer` semantics used by `test_ema_smooths_beta_across_calls`:
the first update stores the raw value (`seen` flag), the second blends
`decay*old + (1-decay)*new`. With `decay=0.5` the expectation in the test is exact.

- [ ] **Step 7: Commit**

```bash
git add src/graphslm_ids/offline/training/noise_consensus.py tests/test_eacs_controller.py tests/test_noise_consensus.py
git commit -m "feat(training): EACS controller + soft-target focal loss; delete harmful EPC/EM path"
```

---

### Task 2: Clean answer key extraction script (parallel with Task 1)

**Files:**
- Create: `scripts/tools/extract_clean_eval_labels.py`
- Create: `tests/test_clean_eval_labels.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clean_eval_labels.py`:

```python
"""Keying + relabel logic of the clean eval answer key (pure, no pcaps)."""
import numpy as np

from scripts.tools.extract_clean_eval_labels import (
    canonical_key_of_flow_id,
    clean_labels_from_attack_keys,
    label_of_flow_id,
)

LABEL_MAPPING = {"Benign": 1, "CommandInjection": 3, "XSS": 17, "Recon-OSScan": 11}


def test_flow_id_parsing():
    fid = "CommandInjection|10.0.0.1:80|10.0.0.2:5555|6#2.1"
    assert label_of_flow_id(fid) == "CommandInjection"
    assert canonical_key_of_flow_id(fid) == "10.0.0.1:80|10.0.0.2:5555|6"


def test_web_flow_with_attack_key_keeps_label():
    order = ["CommandInjection|a:1|b:2|6#1.0"]
    keys = {"CommandInjection": {"a:1|b:2|6"}}
    out, audit = clean_labels_from_attack_keys(order, LABEL_MAPPING, keys)
    assert out.tolist() == [3]
    assert audit == {}


def test_web_flow_without_attack_key_demoted_to_benign():
    order = ["CommandInjection|a:1|b:2|6#1.0", "XSS|c:3|d:4|6#1.0"]
    keys = {"CommandInjection": set(), "XSS": set()}
    out, audit = clean_labels_from_attack_keys(order, LABEL_MAPPING, keys)
    assert out.tolist() == [1, 1]
    assert audit == {"CommandInjection": 1, "XSS": 1}


def test_non_web_classes_untouched():
    order = ["Recon-OSScan|a:1|b:2|6#1.0", "Benign|x:1|y:2|17#1.0"]
    out, audit = clean_labels_from_attack_keys(order, LABEL_MAPPING, {})
    assert out.tolist() == [11, 1]
    assert audit == {}


def test_segment_suffix_does_not_leak_into_key():
    # multiple segments of the same 5-tuple share the canonical key
    order = [
        "XSS|a:1|b:2|6#1.0",
        "XSS|a:1|b:2|6#2.0",
    ]
    keys = {"XSS": {"a:1|b:2|6"}}
    out, _ = clean_labels_from_attack_keys(order, LABEL_MAPPING, keys)
    assert out.tolist() == [17, 17]


def test_dtype_is_int64():
    out, _ = clean_labels_from_attack_keys(
        ["Benign|x:1|y:2|17#1.0"], LABEL_MAPPING, {}
    )
    assert out.dtype == np.int64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/test_clean_eval_labels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.tools.extract_clean_eval_labels'`.
(If `scripts/` or `scripts/tools/` lack `__init__.py`, check how existing tests
import from scripts — `tests/test_eval_reporting.py` imports `scripts.eval.eval_reporting`,
so the pattern exists; mirror it. Add empty `scripts/tools/__init__.py` if missing.)

- [ ] **Step 3: Write the script**

Create `scripts/tools/extract_clean_eval_labels.py`:

```python
"""Build the eval-only CLEAN answer key for the LNL protocol.

Runs the existing signature-based attack-flow isolation
(`flow_attack_labeler.label_pcap_flows`) over the web-attack pcaps and maps the
resulting attack-flow keys onto a graph artifact's `flow_id_order`, producing a
clean label vector ALIGNED to `flow_y`. Flows of the 4 web classes whose
canonical 5-tuple carries a matching HTTP attack signature keep their label;
the rest of those classes' flows are demoted to Benign (they are background
IoT->cloud traffic — see docs/reports/2026-06-06-web-attack-encryption-ceiling.md
§3d). All other classes pass through unchanged.

The output is the GRADING KEY of the standard learning-with-noisy-labels
protocol (train on noisy, evaluate on clean). It is never used in training.

Usage (local, pcaps + meta required):
    python scripts/tools/extract_clean_eval_labels.py \
        --graph-meta outputs/v3_ob/graph.meta.json \
        --raw-root data/raw/14gb \
        --out-npy outputs/v3_ob/clean_eval_labels.npy \
        --out-audit outputs/v3_ob/clean_eval_labels.audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphslm_ids.offline.preprocessing.flow_attack_labeler import (
    BENIGN_LABEL,
    WEB_ATTACK_CLASSES,
    label_pcap_flows,
)


def label_of_flow_id(flow_id: str) -> str:
    """'Label|lo|hi|proto#seg.dir' -> 'Label'."""
    return flow_id.split("|", 1)[0]


def canonical_key_of_flow_id(flow_id: str) -> str:
    """'Label|lo|hi|proto#seg.dir' -> 'lo|hi|proto' (matches _canon_key)."""
    core = flow_id.split("|", 1)[1]
    return core.rsplit("#", 1)[0]


def clean_labels_from_attack_keys(
    flow_id_order: list[str],
    label_mapping: dict[str, int],
    attack_keys_by_class: dict[str, set[str]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Clean label vector aligned to flow_id_order + per-class demotion audit.

    Only classes present in ``attack_keys_by_class`` are relabel-eligible; for
    those, a flow keeps its label iff its canonical key is in the class's attack
    key set, else it becomes Benign. Every other class passes through.
    """
    benign_id = label_mapping[BENIGN_LABEL]
    out = np.empty(len(flow_id_order), dtype=np.int64)
    audit: dict[str, int] = {}
    for i, fid in enumerate(flow_id_order):
        name = label_of_flow_id(fid)
        cls_id = label_mapping[name]
        keys = attack_keys_by_class.get(name)
        if keys is None:
            out[i] = cls_id
        elif canonical_key_of_flow_id(fid) in keys:
            out[i] = cls_id
        else:
            out[i] = benign_id
            audit[name] = audit.get(name, 0) + 1
    return out, audit


def collect_attack_keys(raw_root: Path) -> dict[str, set[str]]:
    """Run the signature isolation over every web-attack pcap under raw_root."""
    keys: dict[str, set[str]] = {}
    for cls in sorted(WEB_ATTACK_CLASSES):
        cls_dir = raw_root / cls
        pcaps = sorted(cls_dir.glob("*.pcap")) if cls_dir.exists() else []
        if not pcaps:
            print(f"[warn] no pcaps for {cls} under {cls_dir} — class skipped")
            continue
        cls_keys: set[str] = set()
        for pcap in pcaps:
            mapping, audit = label_pcap_flows(pcap, cls)
            cls_keys |= {k for k, v in mapping.items() if v == cls}
            print(f"[{cls}] {pcap.name}: {len(cls_keys)} attack keys, audit={audit}")
        keys[cls] = cls_keys
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-meta", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out-npy", type=Path, required=True)
    ap.add_argument("--out-audit", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads(args.graph_meta.read_text(encoding="utf-8"))
    flow_id_order = meta["flow_id_order"]
    label_mapping = meta["label_mapping"]

    attack_keys = collect_attack_keys(args.raw_root)
    clean, audit = clean_labels_from_attack_keys(
        flow_id_order, label_mapping, attack_keys
    )

    orig = np.array([label_mapping[label_of_flow_id(f)] for f in flow_id_order])
    n_changed = int((clean != orig).sum())
    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, clean)
    summary = {
        "n_flows": len(flow_id_order),
        "n_demoted_to_benign": n_changed,
        "demoted_per_class": audit,
        "attack_keys_per_class": {k: len(v) for k, v in attack_keys.items()},
        "graph_meta": str(args.graph_meta),
    }
    args.out_audit.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/test_clean_eval_labels.py -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/tools/extract_clean_eval_labels.py tests/test_clean_eval_labels.py
git commit -m "feat(eval): clean answer key extraction for the LNL protocol (eval-only)"
```

---

### Task 3: Trainer integration (after Task 1)

**Files:**
- Modify: `src/graphslm_ids/offline/training/train_hgt_flow_classifier.py`

All line numbers refer to the file BEFORE edits; re-locate with the quoted code.

- [ ] **Step 1: Config parse + family head removal (lines 1893-1895)**

Replace:
```python
    _nr_cfg = config["train"].get("noise_robust") or {}
    _nr_enabled = bool(_nr_cfg.get("enabled", False))
    _nr_num_families = 5 if _nr_enabled else 0
```
with:
```python
    _nr_cfg = config["train"].get("noise_robust") or {}
    _nr_enabled = bool(_nr_cfg.get("enabled", False))
    _nr_mode = str(_nr_cfg.get("mode", "eacs")).lower()
    if _nr_enabled and _nr_mode != "eacs":
        raise ValueError(
            f"train.noise_robust.mode={_nr_mode!r} unsupported — the EPC/EM path "
            "was removed (measured harmful, see 2026-06-11 EACS design). Use 'eacs'."
        )
    # EACS does not use the auxiliary family head (EPC removed).
    _nr_num_families = 0
```

- [ ] **Step 2: Clean answer key loading (insert right after `label_names = label_name_mapping(...)`, line 2162)**

```python
    # Eval-only CLEAN answer key (standard LNL protocol: train on noisy labels,
    # grade on clean ones). Never enters the training loss.
    clean_eval_labels_t: torch.Tensor | None = None
    _clean_eval_path = config["train"].get("clean_eval_labels")
    if _clean_eval_path:
        _cl = np.load(_clean_eval_path)
        _n_flows_total = int(np.asarray(backend.artifact.flow_y).shape[0])
        if int(_cl.shape[0]) != _n_flows_total:
            raise ValueError(
                f"clean_eval_labels has {_cl.shape[0]} rows but the artifact has "
                f"{_n_flows_total} flows — wrong answer key for this graph."
            )
        clean_eval_labels_t = torch.as_tensor(np.asarray(_cl, dtype=np.int64))
        if rank == 0:
            _n_diff = int((np.asarray(_cl) != np.asarray(backend.artifact.flow_y)).sum())
            print(
                f"[clean_eval] answer key loaded from {_clean_eval_path} "
                f"({_n_diff} flows differ from the noisy labels)",
                flush=True,
            )
```
NOTE: `backend.artifact` only exists on the in-memory backend; `noise_robust`
already requires it (the old build used `backend.artifact` at line 2183). Guard:
if `not hasattr(backend, "artifact")`, raise ValueError telling the user
clean_eval_labels requires the in-memory v3 backend.

- [ ] **Step 3: Replace the controller build block (lines 2164-2198)**

Replace the whole `if _nr_enabled:` block (including the
`_family_supervision_loss = None` line above it and the gcl_enabled ValueError)
with:

```python
    if _nr_enabled:
        from graphslm_ids.offline.training.noise_consensus import (
            build_eacs_controller,
        )
        _suspect_classes = list(_nr_cfg.get(
            "suspect_classes",
            ["CommandInjection", "XSS", "SqlInjection", "Uploading_Attack"],
        ))
        noise_robust_ctrl = build_eacs_controller(
            artifact=backend.artifact,
            num_classes=num_classes,
            label_mapping={v: k for k, v in label_names.items()},
            suspect_classes=_suspect_classes,
            warmup_epochs=int(_nr_cfg.get("warmup_epochs", 5)),
            ema_decay=float(_nr_cfg.get("ema_decay", 0.9)),
            lambda_disambig=float(_nr_cfg.get("lambda_disambig", 0.7)),
            mitre_dir=str(_nr_cfg.get("mitre_dir", "data/mitre")),
        )
        if rank == 0:
            print(
                f"[noise_robust] ENABLED mode=eacs warmup={noise_robust_ctrl.warmup_epochs} "
                f"ema_decay={_nr_cfg.get('ema_decay', 0.9)} "
                f"lambda_disambig={noise_robust_ctrl.lambda_disambig} "
                f"suspect_classes={_suspect_classes} "
                f"suspects={int(noise_robust_ctrl.suspect_mask.sum())} "
                f"anchors={int(noise_robust_ctrl.anchor_mask.sum())}",
                flush=True,
            )
```

Also import `neighbor_consensus` at the same site (used in Step 4):
add `neighbor_consensus,` to the import list.

- [ ] **Step 4: Replace the train-step soft-relabel block (lines 2321-2362)**

Replace everything from `seed_logits = logits[sm].float()` through the
`primary_loss = primary_loss + 0.3 * _fam_loss` line (keeping the `else:` that
falls back to `_compute_train_loss`) with:

```python
                seed_logits = logits[sm].float()
                # EACS noise-robust self-learning: suspect flows (web-attack label,
                # no MITRE evidence) get a soft target inside {y, Benign}; anchors
                # and all other flows keep the hard label. Warmup epochs return pure
                # one-hot, so early training matches the baseline exactly.
                if noise_robust_ctrl is not None:
                    _seed_gids = torch.as_tensor(
                        np.asarray(batch.seed_flow_ids, dtype=np.int64), device=device
                    )
                    _cons = None
                    if noise_robust_ctrl.lambda_disambig < 1.0:
                        _ff_parts = []
                        for _ff_key in (
                            ("flow", "burst_neighbor", "flow"),
                            ("flow", "rev_burst_neighbor", "flow"),
                        ):
                            _ff = ei.get(_ff_key)
                            if _ff is not None and _ff.numel() > 0:
                                _ff_parts.append(_ff)
                        if _ff_parts:
                            _seed_idx_local = torch.nonzero(sm, as_tuple=False).reshape(-1)
                            _cons = neighbor_consensus(
                                torch.softmax(logits.detach().float(), dim=1),
                                torch.cat(_ff_parts, dim=1),
                                _seed_idx_local,
                                sl,
                            )
                    soft_tgt = noise_robust_ctrl.soft_targets(
                        seed_logits, _seed_gids, sl, epoch=epoch, consensus=_cons,
                    )
                    primary_loss = soft_target_focal_loss(
                        seed_logits, soft_tgt, weight,
                        loss_type=loss_type,
                        label_smoothing=label_smoothing,
                        focal_gamma=focal_gamma,
                    )
                else:
                    primary_loss = _compute_train_loss(
                        seed_logits, sl, weight,
                        loss_type=loss_type,
                        label_smoothing=label_smoothing,
                        focal_gamma=focal_gamma,
                    )
```

`soft_target_focal_loss` import: add it to the `if _nr_enabled:` import in Step 3
and bind a module-level name before the train loop, e.g. right after the
controller build add `from graphslm_ids.offline.training.noise_consensus import soft_target_focal_loss`
(one import line with the others is fine).
The old EPC code referenced `x_dict`/`family_logits`/`_family_supervision_loss`;
all such references must be gone after this step. EACS works on BOTH the gcl and
non-gcl paths (logits cover all flow nodes either way) — there is no longer a
gcl_enabled requirement for noise_robust.

- [ ] **Step 5: Dual metric in `evaluate_neighbor_sampling` (line 1579)**

1. Add parameter `clean_labels: torch.Tensor | None = None` (after `packet_store`).
2. After `counts_acc_tau = ...` (line 1614), add:
   ```python
   counts_acc_clean = torch.zeros((num_classes, 4), dtype=torch.int64, device=device)
   ```
3. In BOTH loops (DDP ~line 1631, single-device ~line 1688), right after the
   `_logits_raw = seed_logits.detach().float()` line, add:
   ```python
                if clean_labels is not None:
                    _gids = torch.as_tensor(
                        np.asarray(batch.seed_flow_ids, dtype=np.int64)
                    )
                    _clean_y = clean_labels[_gids].to(device)
                    counts_acc_clean += _per_class_counts_tensor(
                        _logits_raw.argmax(dim=1), _clean_y, num_classes
                    )
   ```
4. DDP branch: next to the `tau_normalized` all_reduce/attach (lines 1678-1685), add:
   ```python
            if clean_labels is not None:
                dist.all_reduce(counts_acc_clean, op=dist.ReduceOp.SUM)
                metrics["clean"] = _metrics_from_counts(
                    counts_acc_clean.cpu().numpy(), label_names, None,
                    int(examples_t.item()),
                )
   ```
5. Single-device branch: immediately before its final `return metrics`, add:
   ```python
            if clean_labels is not None:
                metrics["clean"] = _metrics_from_counts(
                    counts_acc_clean.cpu().numpy(), label_names, None,
                    int(counts_acc_clean[:, 3].sum().item()) if counts_acc_clean.shape[1] > 3 else 0,
                )
   ```
   IMPORTANT: read `_per_class_counts_tensor` (line 1381) first — if its 4 columns
   are not (tp, fp, fn, support), compute the example count instead as the sum of
   per-class supports, mirroring how `_metrics_from_counts` derives counts. Match
   whatever the DDP branch uses (`examples_t` equivalent: total seed count —
   accumulate a local `int` counter `n_seen += int(seed_labels.numel())` in the
   loop and pass that, which is the simplest correct option for both branches).
6. Val call site (lines 2725-2741): add `clean_labels=clean_eval_labels_t,` to the
   `evaluate_neighbor_sampling(...)` call.

- [ ] **Step 6: Monitor + epoch print + [eacs] log**

1. Monitor whitelist (lines 2200-2205): add `"val_macro_f1_clean"` to the set and
   to the error message.
2. `_compute_monitor_score` (line 896): add before the final raise:
   ```python
    if monitor == "val_macro_f1_clean":
        clean = val_metrics.get("clean") if isinstance(val_metrics, dict) else None
        if not clean or clean.get("macro_f1") is None:
            return float("nan")
        return float(clean["macro_f1"])
   ```
   (Also update its docstring list.)
3. Epoch summary print (lines 2841-2861): after the `_tn_str` definition add:
   ```python
            _cl = val_metrics.get("clean") if isinstance(val_metrics, dict) else None
            _cl_str = (
                f" | CLEAN: val_acc={_cl['accuracy']:.4f} val_macro_f1_clean={_cl['macro_f1']:.4f}"
                if isinstance(_cl, dict) and _cl.get("macro_f1") is not None else ""
            )
   ```
   and append `{_cl_str}` into the f-string right after `{_tn_str}`.
4. `[eacs]` per-epoch log — directly after the `[hgaa]` block (lines 2782-2790):
   ```python
        if noise_robust_ctrl is not None and rank == 0:
            _es = noise_robust_ctrl.epoch_stats(reset=True)
            _pc_named = {
                label_names.get(cid, str(cid)): v for cid, v in _es["per_class"].items()
            }
            print(
                f"[eacs] epoch={epoch} suspects_seen={_es['suspects_seen']} "
                f"mean_beta={_es['mean_beta']:.3f} relabeled={_es['relabeled']} "
                f"per_class={_pc_named}",
                flush=True,
            )
   ```

- [ ] **Step 7: End-of-run noise-detection ROC-AUC**

After the train loop ends and the best checkpoint is finalized (near the metrics
json dump at ~line 2942 — locate `"history": history`), add on rank 0:

```python
    if noise_robust_ctrl is not None and clean_eval_labels_t is not None and rank == 0:
        # Headline detector metric: did the model self-discover the noise?
        # Oracle = clean answer key disagrees with the noisy label. Score = 1-beta.
        # Restricted to TRAIN-seen suspect flows (beta_buffer.seen).
        from sklearn.metrics import roc_auc_score
        _sus_np = noise_robust_ctrl.suspect_mask.cpu().numpy()
        _seen_np = noise_robust_ctrl.beta_buffer.seen.cpu().numpy()
        _beta_np = noise_robust_ctrl.beta_buffer.values.cpu().numpy()
        _flow_y_np = np.asarray(backend.artifact.flow_y, dtype=np.int64)
        _oracle = (clean_eval_labels_t.cpu().numpy() != _flow_y_np)
        _m = _sus_np & _seen_np
        _det = {"n_scored": int(_m.sum())}
        if _m.any() and len(np.unique(_oracle[_m])) == 2:
            _det["roc_auc"] = float(roc_auc_score(_oracle[_m], 1.0 - _beta_np[_m]))
        print(f"[eacs] noise-detection vs oracle: {_det}", flush=True)
        with open(output_dir / "eacs_noise_detection.json", "w", encoding="utf-8") as fh:
            json.dump(_det, fh, indent=2)
```
(`json` is already imported at module top — verify; if not, add the import.)

- [ ] **Step 8: Verify the whole test suite still passes**

Run: `D:\v\nt114\Scripts\python.exe -m pytest tests/ -q`
Expected: ALL PASS (no test enables noise_robust against the trainer; the lazy
import inside `if _nr_enabled:` keeps disabled runs untouched).

- [ ] **Step 9: CPU smoke (mechanism on, no clean key)**

Run (uses the local v3 artifact; eacs on, 2 epochs, warmup_epochs=0 so the EACS
path actually executes):

```bat
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.training.train_hgt_flow_classifier ^
  --config configs/eg_hgt.yaml --device cpu --epochs 2 ^
  --set train.noise_robust.enabled=true ^
  --set train.noise_robust.warmup_epochs=0
```
First check `parse_args`/`apply_cli_overrides` (lines 276/346) for the override
flag syntax; if `--set` is not supported, create a throwaway yaml copy of
`configs/eg_hgt.yaml` with the `noise_robust` block instead (do NOT commit it).
Expected: `[noise_robust] ENABLED mode=eacs ...` banner with nonzero
suspects/anchors, `[eacs] epoch=...` lines, finite loss, run completes.

- [ ] **Step 10: Commit**

```bash
git add src/graphslm_ids/offline/training/train_hgt_flow_classifier.py
git commit -m "feat(training): wire EACS into the trainer + clean-answer-key dual val metric"
```

---

### Task 4: Config, answer key generation, remote 50-epoch launch (orchestrator)

**Files:**
- Create: `configs/eg_hgt_v6_ob_eacs.yaml`
- Create: `outputs/v3_ob/clean_eval_labels.npy` (artifact, gitignored)
- Create (remote): `~/Do-an-chuyen-nganh_NT114/run_eacs_full.sh`

- [ ] **Step 1: Write the run config**

Create `configs/eg_hgt_v6_ob_eacs.yaml` — copy `configs/eg_hgt_v6_ob_noiserobust.yaml`
verbatim, then change ONLY:

```yaml
experiment:
  source_name: "EG-HGT v6 EACS (evidence-anchored candidate-set self-relabeling, LNL clean-key eval, 50ep)"
  rationale: >
    EACS replaces the EPC/EM soft-relabel (measured harmful: relabeled the
    evidence-grounded TRUE attacks instead of the no-evidence background noise).
    Suspect flows (web-attack label, no MITRE evidence) get a model-driven soft
    target inside {own label, Benign}; anchors and all other classes keep hard
    labels. Val is graded BOTH on original labels (fair GNN4ID compare) and on
    the signature-isolation clean answer key (standard LNL protocol; target
    val_macro_f1_clean >= 0.90). See docs/superpowers/specs/2026-06-11-eacs-*.

train:
  output_dir: outputs/v3_ob_eacs
  monitor: val_macro_f1_clean
  clean_eval_labels: outputs/v3_ob/clean_eval_labels.npy

  noise_robust:
    enabled: true
    mode: eacs
    warmup_epochs: 5
    ema_decay: 0.9
    lambda_disambig: 0.7
    suspect_classes: [CommandInjection, XSS, SqlInjection, Uploading_Attack]
    mitre_dir: data/mitre
```
(Everything else — graph paths, model, focal/cb knobs, gcl, hgaa, sampler,
dataloader, feature_store — stays identical to the noiserobust yaml.)

- [ ] **Step 2: Generate the answer key locally**

```bat
scp hgt-aws:~/Do-an-chuyen-nganh_NT114/outputs/v3_ob/graph.meta.json outputs/v3_ob/graph.meta.json
D:\v\nt114\Scripts\python.exe scripts/tools/extract_clean_eval_labels.py ^
  --graph-meta outputs/v3_ob/graph.meta.json ^
  --raw-root data/raw/14gb ^
  --out-npy outputs/v3_ob/clean_eval_labels.npy ^
  --out-audit outputs/v3_ob/clean_eval_labels.audit.json
```
Expected: audit shows demotions concentrated in the 4 web classes, each ~90-97%
of the class (per §3d ~95% of CommandInjection is background), other classes 0.
Sanity-check against the class sizes from the audit JSON; if a web class shows
<50% demoted, STOP and investigate before training.

- [ ] **Step 3: Push + sync remote**

```bash
git push origin feat/model-optim-faircompare
ssh hgt-aws "cd ~/Do-an-chuyen-nganh_NT114 && git pull"
scp outputs/v3_ob/clean_eval_labels.npy hgt-aws:~/Do-an-chuyen-nganh_NT114/outputs/v3_ob/
scp outputs/v3_ob/clean_eval_labels.audit.json hgt-aws:~/Do-an-chuyen-nganh_NT114/outputs/v3_ob/
```

- [ ] **Step 4: Remote launch (nohup pattern proven on this box — tmux is broken)**

Confirm the old NR run has finished (`pgrep -af train_hgt` empty / log shows exit),
then:

```bash
ssh hgt-aws "cat > ~/Do-an-chuyen-nganh_NT114/run_eacs_full.sh <<'EOF'
#!/bin/bash
cd ~/Do-an-chuyen-nganh_NT114
source ~/venv/bin/activate
export PYTHONPATH=src
echo \"=== EACS FULL 50ep start \$(date) ===\"
python -m graphslm_ids.offline.training.train_hgt_flow_classifier \
  --config configs/eg_hgt_v6_ob_eacs.yaml --device cuda
echo \"=== EACS FULL exit \$? \$(date) ===\"
rm -rf /tmp/v3_edges_* 2>/dev/null || true
EOF
chmod +x ~/Do-an-chuyen-nganh_NT114/run_eacs_full.sh
cd ~/Do-an-chuyen-nganh_NT114 && nohup bash run_eacs_full.sh > outputs/eacs_full.log 2>&1 & disown"
```

- [ ] **Step 5: Verify launch (epoch 1)**

```bash
ssh hgt-aws "sleep 90 && tail -n 40 ~/Do-an-chuyen-nganh_NT114/outputs/eacs_full.log && nvidia-smi --query-gpu=memory.used --format=csv"
```
Expected: `[noise_robust] ENABLED mode=eacs ... suspects=... anchors=...`,
`[clean_eval] answer key loaded ...`, GPU memory allocated, epoch 1 progressing.
From epoch 1 summary on, the line must show BOTH `val_macro_f1=` and
`CLEAN: ... val_macro_f1_clean=`. From epoch 6 (post-warmup), `[eacs]` lines must
show nonzero `suspects_seen` and a falling `mean_beta` trend across epochs.

- [ ] **Step 6: Commit config + monitor**

```bash
git add configs/eg_hgt_v6_ob_eacs.yaml
git commit -m "feat(config): EACS 50ep run config (clean-key monitor, LNL protocol)"
git push origin feat/model-optim-faircompare
```
Then schedule monitoring (orchestrator: ScheduleWakeup ~30 min cadence; check
`val_macro_f1_clean` trajectory, `[eacs]` mean_beta trend, and at the end the
ROC-AUC in `outputs/v3_ob_eacs/eacs_noise_detection.json`).

---

## Self-review checklist (done at plan-writing time)

- Spec coverage: C1=Task 2, C2=Tasks 1+3, C3=Task 3 steps 6-7; protocol+config=Task 4. ✓
- No placeholders: every step has full code or an exact command + expected output. ✓
- Type consistency: `soft_targets(class_logits, seed_global_ids, seed_labels, epoch, consensus)` used identically in Task 1 tests and Task 3 step 4; `soft_target_focal_loss(logits, soft_targets, weight, *, loss_type, label_smoothing, focal_gamma)` identical in Task 1 and Task 3; `epoch_stats(reset=...)` identical in Task 1 and Task 3 step 6. ✓
- Known judgment points called out inline: scripts package import pattern (Task 2 step 2), `_per_class_counts_tensor` column semantics (Task 3 step 5), CLI override syntax (Task 3 step 9). Executors must verify these against the code as instructed rather than assume. ✓
