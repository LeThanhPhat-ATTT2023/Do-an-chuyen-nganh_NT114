# EACS vs Baselines — Fair-Comparison Results (2026-06-13)

All three models trained on the **same** ordered-byte v3_ob graph (18 classes,
211,851 flows, original per-pcap labels), same random split, graded on **two**
label sets:

- **noisy** — the per-pcap CIC-IoT-2023 labels (what prior work reports on).
- **clean** — the signature-isolated answer key (LNL protocol): a web-attack
  flow keeps its label iff its 5-tuple carries a matching HTTP attack signature,
  else it is demoted to Benign. 14,276 / 211,851 flows differ from the noisy
  labels (the dataset's per-pcap labeling noise, quantified).

The clean key is **eval-only**: it never enters any loss or gradient. EACS
self-relabels using MITRE evidence + the model's own predictions only.

## Headline table (TEST split, macro-F1)

| Model | noisy | clean (raw) | clean (calibrated) | noisy − clean |
|---|---|---|---|---|
| GNN4ID (retrained on v3 dist.) | **0.8588** | 0.7294 | — | **+0.129** |
| HGT de-inflated (no EACS) | 0.8520 | 0.7224 | 0.7753 | **+0.130** |
| **HGT + EACS v2** | 0.7228 | **0.8582** | 0.8518 | **−0.135** |

VAL-split clean macro for EACS v2: **0.8856 raw → 0.9269 calibrated**.

### Reading the sign of (noisy − clean)

- **Positive** (GNN4ID, de-inflated): noisy > clean ⇒ the model scores *higher*
  when graded on the wrong labels than on the truth ⇒ it **memorized the label
  noise** (campaign fingerprint), exactly what XG-NID/GNN4ID-style pipelines do.
- **Negative** (EACS): clean > noisy ⇒ the model scores *higher* on the truth ⇒
  it **filtered the label noise**. Same HGT backbone, same data; the only changed
  variable is the EACS self-relabeling controller.

The sign flip is the contribution.

## The decisive metric — binary web-attack detection (TEST)

"Did the flow carry a real web attack?" pooled over CmdInj/SQLi/Upload/XSS,
graded on the clean key (107 true attacks among 21,186 test flows):

| Model | TP | FP | FN | recall | precision | F1 |
|---|---|---|---|---|---|---|
| HGT de-inflated | 106 | **1667** | 1 | 0.991 | 0.060 | 0.113 |
| **HGT + EACS v2** | 106 | **42** | 1 | 0.991 | **0.716** | **0.831** |

Both models **find** the attacks (identical recall: 106/107, one miss). The
difference is false alarms: the de-inflated model flags **1,667** benign flows as
web attacks because it learned "anything resembling the XSS/SQLi campaign = attack";
EACS flags **42**. A **97.5 % reduction in false positives at equal recall** — the
operational value a SOC actually feels.

## Self-supervision quality (no answer-key peeking)

EACS noise-detection ROC-AUC vs the clean oracle, over train-seen suspect flows:

| | anchor rule | anchored flows | anchor precision | noise-detection AUC |
|---|---|---|---|---|
| EACS v1 | MSEE evidence weight > 0 | 6,439 | 0.16 | 0.609 |
| **EACS v2** | HTTP-request + MITRE procedure literal | 1,272 | **0.952** | **0.760** |

The v1→v2 fix: the collapsed MSEE scalar anchored 5,406 background flows to wrong
hard attack labels (16 % precision), drowning the ~1k real attacks 5:1. Anchoring
on a high-precision procedure-literal match (MSEE source 2, train-legitimate)
restored 95 % anchor precision and lifted every downstream number.

## Per-class clean F1 (EACS v2, TEST) — where the residual gap is

10 DDoS/Recon classes: 0.95–1.00. SqlInjection (n=93): **0.876**. Benign: ~0.91.
The three classes still scoring low — XSS (n=5), CommandInjection (n=7),
Uploading_Attack (n=2) — have **single-digit test support**: one false positive
swings F1 by 0.1–0.3, so these are measurement noise of the *metric*, not the
model (cf. the 0.991 pooled recall). A 60/20/20 attack-key re-split would give
these classes enough test support to score them stably.

## Artifacts

- `results/2026-06-13/eacs_v2_*.json` — EACS v2 training summary, calibration,
  noise-detection AUC.
- `results/2026-06-13/deinflated_*.json` — de-inflated rerun (the noise-memorizer
  control).
- `results/2026-06-13/gnn4id_{noisy,clean}.json` — GNN4ID baseline, both gradings.
- Checkpoints live on metis under `outputs/v3_ob_eacs_v2/` and
  `outputs/v3_ob_focal_deinflated_rerun/` (gitignored, 2 GB graph + weights).

## Reproduce

```bash
# EACS v2 (procedure-anchor)
python scripts/tools/extract_eacs_anchor_mask.py --graph-meta outputs/v3_ob/graph.meta.json \
  --raw-root data/raw --out-npy outputs/v3_ob/eacs_anchor_mask.npy --out-audit /tmp/a.json
python -m graphslm_ids.offline.training.train_hgt_flow_classifier \
  --config configs/eg_hgt_v6_ob_eacs_v2.yaml --device cuda
python scripts/eval/calibrate_thresholds.py --config configs/eg_hgt_v6_ob_eacs_v2.yaml \
  --checkpoint outputs/v3_ob_eacs_v2/hgt_flow_best.pt \
  --training-summary outputs/v3_ob_eacs_v2/training_summary.json \
  --clean-labels outputs/v3_ob/clean_eval_labels.npy \
  --out outputs/v3_ob_eacs_v2/confusion_calibrated.json --device cuda

# GNN4ID baseline + clean re-grade
python baselines/gnn4id/train_imbalanced.py --device cuda  # noisy
python baselines/gnn4id/regrade_clean.py --device cuda      # clean
```
