# NT114 — GraphSLM IDS (Đồ án chuyên ngành)

## Python environment (USE THIS)
- Dedicated venv: **`D:\v\nt114`** (Windows).
- Interpreter: **`D:\v\nt114\Scripts\python.exe`** (Python 3.13.13).
- Installed: numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, torch 2.11.0+cpu, scikit-learn 1.8.0, dpkt, scapy, pytest 9.0.3.
- Always run project scripts with this interpreter, e.g.:
  `D:\v\nt114\Scripts\python.exe scripts/diagnostics/v2_ceiling_test.py`
- Machine: 16 logical CPUs, ~16 GB RAM. CPU-only torch locally; full HGT training runs on Kaggle / remote L40S VM.

## Local data
- `data/raw/14gb/<class>/*.pcap` — 13 PCAPs (5.6 GB total), one per attack class.
- `data/interim/payload_dataset_14gb/metadata.csv` — 5.26M packets (legacy v1).
- `data/interim/payload_dataset_14gb/payload_256.npy` — (5261944, 256) uint8 payload bytes (legacy v1).
- `data/processed/graph_artifact_3tier_14gb.npz` (+ `_packet_semantic_x.npy`, 16 GB sidecar) — legacy v1 artifact.

Dataset: CIC-IoT-2023 style, **13 classes**, 1.5M flows (v1) / 612k bidirectional flows (v2). Labels assigned per-PCAP-file via `infer_label_from_path`.

## v3 pipeline (CURRENT, since 2026-05-24 evening) — Smart-BOTH Hybrid

**Design doc:** `docs/superpowers/specs/2026-05-24-v3-smart-both-design.md`
**Plan:** `docs/superpowers/plans/2026-05-24-v3-smart-both-hybrid.md`

End-to-end (reuses v2 stages 1-2, replaces 3-6):
1. `v2/extractor.py` (kept) — all packets + TCP flags + IP len + direction.
2. `v2/flows.py` (kept) — bidirectional 5-tuple flows + ~80 CICFlowMeter features.
3. `v3/split.py` — temporal AND random stratified splits (both produced from same packets).
4. `v3/tokenizer.py` — deterministic byte n-gram + HTTP/text tokens (no training).
5. `v3/pmi_learner.py` — PMI candidate generation + L1-multinomial LR refinement on TRAIN subsample.
6. `v3/procedure_matcher.py` — Aho-Corasick over MITRE STIX `enterprise-attack.json` procedure literals.
7. `v3/flow_consensus.py` — wraps `v2/signatures.match_flow_signatures` for behavioral boost.
8. `v3/ensemble.py` — 3-source aggregation (PMI + procedure + flow consensus) → final edge weights.
9. `v3/edge_writers.py` — memmap streaming edge writers (out-of-core for ~5GB graph).
10. `v3/graph_builder.py` — assemble v3 artifact: flow + packet + host + technique + tactic nodes, typed edges, hierarchy edges.
11. `v3/cli.py` — single-command orchestrator local CPU.
12. `train_hgt_flow_classifier.py` — reads `data.artifact_version: v3`, calls `load_v3_artifact()`, adds **GCL auxiliary loss** with positive pairs from class→technique map.
13. `scripts/eval/v3_eval_both_splits.py` — run identical model on random + temporal splits, report GAP.

### v3 design principles (do not deviate)
- **Zero learned encoders besides HGT itself**. PMI = counting + convex L1-LR. Procedure matcher = Aho-Corasick string match.
- All Stage 3-11 deterministic given seed 42.
- Pipeline 3-11 runs on local CPU (16 GB RAM). Only Stage 12 (HGT train) runs on L40S server.
- Eval BOTH random + temporal splits. The gap is the contribution.

### Run commands
```bat
:: Build v3 graph artifact (local CPU, ~2.5 hours)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.preprocessing.v3.cli ^
  --raw-root data/raw/14gb ^
  --interim-root data/interim/payload_dataset_14gb ^
  --out-npz outputs/v3/graph.npz ^
  --out-meta outputs/v3/graph.meta.json ^
  --pmi-subsample-per-class 25000 ^
  --temporal-train-frac 0.80 ^
  --temporal-val-frac 0.10 ^
  --workers 14

:: Sanity audit BEFORE upload
D:\v\nt114\Scripts\python.exe scripts/diagnostics/v3_artifact_audit.py outputs/v3/graph.npz

:: Smoke train HGT v3 on CPU (verify trainer integration)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.training.train_hgt_flow_classifier ^
  --config configs/eg_hgt_v3.yaml --device cpu --epochs 2

:: Eval both splits AFTER server training
D:\v\nt114\Scripts\python.exe scripts/eval/v3_eval_both_splits.py ^
  --checkpoint-random outputs/v3/checkpoint_random.pt ^
  --checkpoint-temporal outputs/v3/checkpoint_temporal.pt ^
  --graph outputs/v3/graph.npz

:: All v3 unit tests
D:\v\nt114\Scripts\python.exe -m pytest tests/v3/ -v
```

## v2 pipeline (KEPT for ablation, do not delete)

v2 files in `src/graphslm_ids/offline/preprocessing/v2/` remain functional and serve as the ablation baseline (rule-based edges vs v3 hybrid PMI+procedure+consensus). v2 docs:
- Design: `docs/superpowers/specs/2026-05-24-evidence-grounded-graph-ids-design.md`
- Plan: `docs/superpowers/plans/2026-05-24-evidence-grounded-graph-ids.md`

v2 ceiling (random split, HGBM on ~85 flow features): **macro-F1 = 0.9768**, accuracy = 0.9930. This is the random-split ceiling v3 should approach or exceed while ALSO providing honest temporal-split numbers.

## Diagnosis history (v1, May 2026)
- v1 HGT kept plateauing at macro-F1 ≈ 0.12 across all recipe variants. Root cause was NOT recipe; it was the input pipeline:
  - **Extract (raw→CSV)**: v1 dropped every TCP control packet (SYN/ACK/RST/FIN, no payload) and all ICMP, and never recorded TCP flags. Scan/recon/flood lost their signature.
  - **Flow features**: only 6 coarse stats.
  - **Cosine `matches_technique` edges**: SecureBERT embedding of payload-hex vs MITRE text — semantically meaningless.
- `val_acc=nan` in v1 epoch logs was benign: just the `eval_every` sparse-validation skip.
- v2 fixes the three root causes (all-packets+flags extractor; ~65 CICFlowMeter features; evidence-grounded edges from OWASP CRS + flow signatures) and adds **zero learned encoders** — every HGT input is deterministic; HGT is the only model that trains.

## Academic novelty (Q1 framing, v3)
The v3 contribution has three layers:

1. **Multi-Source Evidence Ensemble (MSEE)** for packet→technique edges: replaces both hand-crafted signatures (v2/CRS) and semantically-invalid embedding cosine (v1) with a three-source statistical ensemble — PMI candidate generation, L1-regularized multinomial logistic refinement, and MITRE procedure substring matching. Each edge carries token-level provenance suitable for SOC audit. No learned encoder.
2. **Typed heterogeneous schema**: 5 typed evidence edge types per attack family (injection / command_exec / file_upload / recon / c2_beacon), flow-flow homophily edges, host tier — HGT's multi-head attention now operates on 22 edge types vs v1's effectively-homogeneous 5. PMI provides PRIOR weights; HGT attention REFINES; GCL auxiliary loss SUPERVISES.
3. **Smart-BOTH evaluation protocol**: report both random-stratified (matches prior work) and temporal split (deployment-realistic). The gap between random and temporal F1 is itself a finding — small gap means model captures attack-intrinsic patterns rather than dataset-specific campaign fingerprints.

This is what XG-NID (post-hoc LLM narrative, no knowledge graph) and PacketCLIP (learned contrastive encoder, single-source) structurally cannot match.
