# v5 Phase 0 — v4 confusion-matrix findings (decisive)

**Date:** 2026-06-07
**Script:** `scripts/diagnostics/v4_confusion_analysis.py` (reuses the production
`evaluate_neighbor_sampling` path; reproduces v4 test macro-F1 0.863 vs summary 0.865 — valid).
**Artifact:** `outputs/v3/hgt_v4/v4_confusion.json`

## TL;DR — the five "draggers" are mostly ONE problem

It is **not** five independent representation gaps. Four of the five low-F1 classes share a
single failure mode: **minority classes are over-predicted (low precision), and Backdoor_Malware
is the main precision sink that eats Benign.**

## The smoking gun

| | count |
|---|---|
| Flows predicted as `Backdoor_Malware` | **784** |
| …actually Backdoor (correct) | 298 |
| …actually **Benign** (false positive) | **481** |
| Benign total (test) | 1838 → **26% leak into Backdoor** |

This single confusion creates BOTH low F1s:
- **Benign**: recall 0.60 (37% missed, mostly → Backdoor) → F1 0.76.
- **Backdoor_Malware**: precision 0.38 (298/784) → F1 0.57. (Its recall is 0.98 — the model
  rarely *misses* Backdoor; it over-*assigns* it.)

Same pattern (high recall, low precision = over-prediction) on the other draggers:
- `DDoS-ICMP_Fragmentation`: R 0.84 / **P 0.32** → F1 0.45.
- `Recon-PingSweep`: R 0.98 / **P 0.55** → F1 0.70.
- `SqlInjection`: R 1.00 / **P 0.66** → F1 0.80.

## Root cause (recipe, not representation)

v4's recipe over-weights rare classes, so the model dumps uncertain flows into them:
- `class_weight: cb`, `cb_beta 0.9999`, `class_weight_cap 15.0` → rare classes up to 15× weight.
- `focal_gamma 2.0`.
- `logit_adjustment: 0.0` — **disabled at inference**, so the train-time prior shift is never
  corrected back at test time.

After the v4 relabel, Benign absorbed ~14k heterogeneous background flows, widening its overlap
with stealthy/rare classes — and the aggressive minority weighting then resolves that overlap in
favor of the rare class.

## Implication for the v5 plan (re-prioritized)

1. **Biggest + cheapest lever = precision calibration (no graph rebuild):** re-enable
   `logit_adjustment` at inference, lower `class_weight_cap` (15 → ~6-8), soften `cb_beta`.
   Estimated: fixing Benign↔Backdoor alone ≈ Benign 0.76→~0.92, Backdoor 0.57→~0.95 ⇒ **+0.028
   macro** (→ ~0.89) from a config change + retrain.
2. **Ordered-byte feature (Phase 1) + windowing (Phase 2) are now SECONDARY** — they still help
   Mirai/SqlInj/ICMP_Frag where ordered payload is the signal, and push from ~0.89 toward ≥0.93,
   but they are not the main fix for the precision sinks.

**Sequencing:** retrain with the recipe fix FIRST (fast, isolates its contribution as a clean
ablation), THEN rebuild with ordered-byte + windowing for the final push.
