# Design: EACS — Evidence-Anchored Candidate-Set Self-relabeling

**Date:** 2026-06-11
**Status:** APPROVED (user, 2026-06-11)
**Supersedes:** the EPC/EM soft-relabel mechanism of
`2026-06-09-neighbor-consensus-noise-robust-hgt-design.md` (kept for history).

---

## 1. Why the previous mechanism failed (diagnosis, measured)

The EPC/EM soft-relabel run (`outputs/nr_full.log`, 50 ep) landed at
val_macro_f1 0.852 vs baseline 0.857 at the same epoch — and the classes it was
built to rescue got WORSE (CommandInjection 0.402→0.324, Uploading 0.343→0.318).
Two structural causes:

1. **Inverted gradable set.** Commit `3876f34` confined soft-relabel to flows WITH
   MITRE evidence. But the noise is background flows mislabeled as web attacks —
   and those carry NO evidence. So the actual noise candidates kept beta=1 (full
   label trust, never relabeled) while the evidence-carrying flows — mostly TRUE
   attacks with clean labels — were the only ones the 2-component EM judged. The EM
   always splits its input into two clusters, even when the EPC distribution is
   unimodal-clean, so a fraction of true attacks was soft-relabeled toward the
   model's background-dominated prediction. The mechanism eroded exactly the signal
   it was meant to protect.
2. **Metric blindness.** val/test labels are the SAME per-pcap polluted labels as
   train. A model that correctly learns "this background flow is Benign" gets
   *penalized* on val. Measured on noisy labels, a perfect noise-robust learner
   scores WORSE than a noise-overfitting one. The standard Learning-with-Noisy-
   Labels (LNL) protocol is train-on-noisy / evaluate-on-clean (Friends and Foes,
   arXiv 2103.15055; NoisyGL, NeurIPS 2024; BeGIN, arXiv 2506.12468). Our current
   protocol cannot show ANY noise-robust gain, no matter how good the mechanism.

Secondary defects fixed in passing: per-batch EM on small gradable subsets
(unstable), focal modulation using the HARD label's p_t even when the target is
soft (amplifies the wrong-direction loss on relabeled samples), and a single
banner log line with zero per-epoch visibility.

## 2. Goal & success criteria

Train on the ORIGINAL noisy labels with a single model, single online run, no
manual relabeling in training. The model self-detects and self-corrects the label
noise, getting better each epoch.

- **Primary:** `val_macro_f1_clean` (measured against the signature-isolation
  answer key, eval-only) **≥ 0.90** on the 18-class problem. Expectation grounded
  in `docs/reports/2026-06-06-web-attack-encryption-ceiling.md` §3d-3e: true-attack
  flows are separable at ~0.8-0.95 (char-ngram LR already gets 0.80 web-cluster);
  web cluster 0.4→~0.85-0.90 lifts 18-class macro-F1 to ~0.90+.
- **Secondary:** noise-detection ROC-AUC of (1−beta) vs the oracle noise indicator
  ≥ 0.75 (the design-doc headline for "the model self-discovers the noise").
- **Honesty:** keep reporting plain `val_macro_f1` on original labels (fair
  comparison vs GNN4ID, expected to stay ~0.85-0.86 — information-bounded).

## 3. Mechanism — EACS

Three precomputed, label-structure-aware flow groups (from the graph + config):

| Group | Definition | Treatment |
|---|---|---|
| **Anchor** | label ∈ `suspect_classes` AND has matching-family MITRE evidence | beta=1 always; these are the true attacks that teach the model the attack pattern |
| **Suspect** | label ∈ `suspect_classes` AND no matching evidence | candidate label set **{y, Benign}**; model disambiguates online |
| **Untouched** | every other class | beta=1 always (their per-pcap labels are correct; DDoS/Recon/Mirai whole-capture = attack) |

`suspect_classes` defaults to the 4 web classes the isolation study proved
polluted: CommandInjection, XSS, SqlInjection, Uploading_Attack
(`flow_attack_labeler.WEB_ATTACK_CLASSES`).

Per epoch, after `warmup_epochs` (=5), for suspect flows in the batch:

```
p        = softmax(class_logits)                  # (S, C), this epoch's belief
beta_raw = p[y] / (p[y] + p[Benign])              # two-way disambiguation
cons     = neighbor_consensus(p, flow-flow edges) # existing pure fn, support of y
beta_i   = beta_raw^λ · cons^(1-λ)                # λ = lambda_disambig = 0.7
beta_i   = EMA(beta_i)                            # existing per-flow buffer, decay 0.9
target_i = beta_i·onehot(y) + (1-beta_i)·onehot(Benign)
```

Loss: identical focal/CE recipe as baseline, with the focal factor computed
against the soft target: `p_t = Σ_c target_c · p_c` (reduces exactly to standard
focal when beta=1, so warmup and non-suspect flows reproduce the baseline loss
bit-for-bit). Per-class alpha weight and label smoothing unchanged.

Removed (vs the EPC/EM version): the family head + 0.3·family_supervision_loss
term in the train loss, the per-batch 2-component EM, and EPC itself. The
candidate-set restriction replaces all of them with one safer primitive.

**Why this gets smarter every epoch (the self-improvement loop):** warmup teaches
basic structure from all labels. Anchors — grounded true attacks — keep teaching
the real attack pattern with full weight. As the model learns it, its p[Benign]
for background suspects rises → beta falls → their soft target shifts toward
Benign → the conflicting gradient on the attack classes shrinks → the attack
pattern gets cleaner → suspects separate even faster. Confirmation-bias collapse
is structurally bounded: a suspect can only move between its own label and
Benign, never to a third class, and anchors + untouched classes (97% of flows)
are immutable ground truth.

## 4. Components

### C1 — Clean answer key (eval-only artifact)
`scripts/tools/extract_clean_eval_labels.py`:
1. For each of the 4 web-attack pcaps under `data/raw/14gb/`, run the existing
   `flow_attack_labeler.label_pcap_flows` → set of canonical 5-tuple keys
   (`lo|hi|proto`) that carry the true attack signature.
2. Parse `outputs/v3_ob/graph.meta.json::flow_id_order`
   (format `label|lo|hi|proto#seg.dir`) → for each flow labeled with a web class,
   clean_label = original label if its canonical key is in the attack set, else
   Benign. All other flows: clean_label = original label.
3. Write `outputs/v3_ob/clean_eval_labels.npy` (int64, aligned to `flow_y`) plus
   a JSON audit (per-class demoted counts, expected ≈95% of each web class).

Note: cross-graph alignment with `outputs/v3_ob_clean` was investigated and
rejected — its `flow_id_order` embeds the label in the id and re-assigns segment
counters, so row order is label-dependent and not joinable. Re-deriving the key
set from the pcaps via the same labeler is exact and cheap (4 pcaps, CPU).

### C2 — Trainer integration
- `noise_consensus.py`: new `EACSController` (suspect/anchor masks precomputed
  from artifact evidence edges + label_mapping; `soft_targets()` implementing the
  formula above; reuses `neighbor_consensus`, `EMAConsensusBuffer`).
- `train_hgt_flow_classifier.py`: `noise_robust.mode: eacs` selects the new
  controller; the EPC/EM path is deleted (it is measurably harmful). Soft-target
  focal fix as in §3. Config-gated, default OFF → baseline runs untouched.
- New config key `train.clean_eval_labels` (path). When set, every val pass also
  computes `val_macro_f1_clean` (same predictions, clean answer key) and logs it
  alongside the original metric; `train.monitor` may select it for checkpointing.

### C3 — Diagnostics (no more flying blind)
- Per-epoch line:
  `[eacs] epoch=K suspects_seen=N mean_beta=… relabeled(beta<0.5)=M per_class={…}`
- End of training: ROC-AUC of (1−beta_final) vs oracle noise indicator
  (clean_eval_labels != flow_y) over suspect flows; written into the metrics JSON.
- Both metrics curves (`val_macro_f1`, `val_macro_f1_clean`) in the history dump.

## 5. Config

```yaml
train:
  clean_eval_labels: outputs/v3_ob/clean_eval_labels.npy   # optional; eval-only
  monitor: val_macro_f1_clean
  noise_robust:
    enabled: true
    mode: eacs                 # eacs | off
    warmup_epochs: 5
    ema_decay: 0.9
    lambda_disambig: 0.7
    suspect_classes: [CommandInjection, XSS, SqlInjection, Uploading_Attack]
```

New run config `configs/eg_hgt_v6_ob_eacs.yaml` = the de-inflated recipe +
the block above (gcl unchanged; hgaa unchanged; 50 ep, same budget as GNN4ID).

## 6. Testing

- **Unit (TDD, CPU, pure):** suspect/anchor mask construction on a toy artifact;
  two-way beta math incl. extremes; soft-target composition; focal-with-soft-target
  == baseline focal when beta=1 (exact); answer-key script keying logic on
  synthetic `flow_id_order` strings; consensus blend.
- **Integration:** 2-epoch CPU smoke with `mode: eacs` — finite loss, `[eacs]`
  lines present, `val_macro_f1_clean` computed; with `enabled: false` the loss
  path is byte-identical to baseline.
- **Full run (L40S):** 50 ep, compare three rows × two metrics:
  baseline / EPC-EM (nr_full) / EACS, on original-label F1 AND clean-key F1.

## 7. Honesty & defensibility

- The clean answer key is **never** an input to training — it is the grading key,
  exactly the standard LNL protocol (train noisy, evaluate clean).
- The suspect-set evidence (MSEE: PMI + L1-LR + procedure match) and the eval
  oracle (handcrafted HTTP signatures in `flow_attack_labeler`) are different
  sources; overlap is disclosed in the thesis but the oracle is eval-only.
- Original-label metrics keep being reported: the GNN4ID comparison stays fair
  and unchanged.
- Soft relabeling is restricted to {own label, Benign} for 4 explicitly listed
  classes backed by the §3d isolation study — no open-ended self-labeling.

## 8. Out of scope (YAGNI)

- No second network / co-teaching; single model, single run.
- No graph rewiring, no retraining on a self-cleaned artifact (two-stage).
- No change to GCL, HGAA, sampler, or the artifact itself.
- The temporal-split protocol and the 2×2 noise-quantification table are separate
  experiments and unaffected.

## 9. References

- Friends and Foes in Learning from Noisy Labels — arXiv 2103.15055 (clean-test
  protocol).
- NoisyGL: benchmark for GNNs under label noise — NeurIPS 2024, arXiv 2406.04299.
- BeGIN: instance-dependent graph label noise benchmark — arXiv 2506.12468 (2025).
- SilentSentinel: graph-based sample selection & purification for NIDS label
  noise — Scientific Reports, 2026 (s41598-026-45988-y).
- PiCO / partial-label disambiguation line of work (candidate-set restriction).
- ICGNN soft-relabel (arXiv 2601.17469) — the beta·onehot + (1−beta)·q template,
  here restricted to a 2-way candidate set.
