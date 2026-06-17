# NT114 — GraphSLM IDS (Đồ án chuyên ngành)

A heterogeneous-graph intrusion detection system on CIC-IoT-2023-style traffic.
Raw PCAPs → an evidence-grounded knowledge graph (flow / packet / host / technique /
tactic nodes) → an HGT flow classifier trained with a GCL auxiliary loss **and an
EACS noise-robust self-relabeling controller**, evaluated on **both** random and
temporal splits **and** on a signature-isolated clean answer key (LNL protocol).

Current artifact: **`v3_ob`** (ordered-byte graph, **18 classes**, 211,851 flows).
The headline result is the **sign of (noisy − clean) macro-F1**: every baseline
memorizes the dataset's per-pcap label noise (positive gap), only HGT+EACS filters
it (negative gap). See `docs/reports/2026-06-13-eacs-vs-baselines.md`.

## Python environment (USE THIS)
- Dedicated venv: **`D:\v\nt114`** (Windows).
- Interpreter: **`D:\v\nt114\Scripts\python.exe`** (Python 3.13.13).
- Installed: numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, torch 2.11.0+cpu, scikit-learn 1.8.0, dpkt, scapy, pytest 9.0.3.
- Always run project scripts with this interpreter, e.g.:
  `D:\v\nt114\Scripts\python.exe -m pytest tests/ -q`
- Machine: 16 logical CPUs, ~16 GB RAM. CPU-only torch locally; full HGT training runs on a remote L40S (48 GB) VM.

## Local data
- `data/raw/<class>/*.pcap` — 18 PCAPs, one per attack class.
- `data/mitre/` — MITRE ATT&CK CSVs + STIX `enterprise-attack.json` (technique embeddings, technique↔tactic edges, class→technique map, technique families).
- `outputs/v3_ob/graph.npz` (+ `graph.meta.json`, `splits.json`, `pmi_table.parquet`) — the current built graph artifact (gitignored).
- `outputs/v3_ob/clean_eval_labels.npy` + `eacs_anchor_mask.npy` — eval-only clean answer key and EACS anchor mask (built from pcaps via `scripts/tools/`).

Dataset: CIC-IoT-2023 style, **18 classes**, **211,851 bidirectional flows** (`v3_ob`), control packets kept. Labels assigned per-PCAP-file via `infer_label_from_path` — this per-capture labeling is the source of the asymmetric label noise on the 4 web-attack classes (CommandInjection / XSS / SqlInjection / Uploading_Attack), quantified at 14,276 flows by the clean key.

`packet_x` (the dominant packet feature array) is stored on disk as **float16** to halve footprint; the loader upcasts to float32 by default (or keeps float16 with `packet_dtype="preserve"`). See the tiered feature store below.

## Preprocessing pipeline — Smart-BOTH Hybrid

**Design doc:** `docs/superpowers/specs/2026-05-24-v3-smart-both-design.md`

One flat package: `src/graphslm_ids/offline/preprocessing/`. End-to-end on local CPU (~2.5h for the 14 GB subset):

1. `extractor.py` — parse pcaps → per-packet metadata (ALL packets + TCP flags + IP len + direction).
2. `flows.py` — bidirectional 5-tuple flows + ~80 CICFlowMeter features.
3. `split.py` — temporal AND random stratified splits (both from the same packets).
4. `tokenizer.py` — deterministic byte n-gram + HTTP/text tokens (no training).
5. `pmi_learner.py` — PMI candidate generation + L1-multinomial LR refinement on a TRAIN subsample.
6. `procedure_matcher.py` — Aho-Corasick over MITRE STIX `enterprise-attack.json` procedure literals.
7. `flow_consensus.py` — `signatures.match_flow_signatures` behavioral boost.
8. `ensemble.py` — 3-source aggregation (PMI + procedure + flow consensus) → final edge weights.
9. `edge_writers.py` — memmap streaming edge writers (out-of-core for the ~5 GB graph).
10. `graph_builder.py` — assemble the artifact: flow + packet + host + technique + tactic nodes, typed evidence edges, hierarchy edges. (`payload_features.py` + `signatures.py` are shared feature/rule modules.)
11. `cli.py` — single-command orchestrator (local CPU).

Then training + eval:
- `training/train_hgt_flow_classifier.py` — reads `data.artifact_version: v3`, calls `load_v3_artifact()`, adds a **GCL auxiliary loss** (positive pairs from the class→technique map) and an optional **EACS** controller (`train.noise_robust.mode: eacs`): suspect web-attack flows without matching MITRE evidence get a model-driven soft target in `{own label, Benign}`; evidence-anchored true attacks (`eacs_anchor_mask.npy`) and all other classes keep hard labels.
- `scripts/eval/calibrate_thresholds.py` — grade a checkpoint on **both** noisy and clean labels, compute the web-binary metric, calibrate a per-class additive logit bias on clean VAL and apply to TEST (decision-rule only, no retrain/relabel).
- `scripts/eval/v3_eval_both_splits.py` — run the identical model on random + temporal splits, report the GAP.

### Design principles (do not deviate)
- **Zero learned encoders besides HGT itself**. PMI = counting + convex L1-LR. Procedure matcher = Aho-Corasick string match.
- Preprocessing stages 3-11 are deterministic given seed 42.
- Preprocessing runs on local CPU (16 GB RAM). Only HGT training runs on the L40S server.
- Eval BOTH random + temporal splits. The gap is the contribution.

## Tiered feature store + GPU sampling (training memory/throughput)

**Design:** `docs/superpowers/specs/2026-05-25-tiered-feature-store-design.md`
**Plan:** `docs/superpowers/plans/2026-05-25-tiered-feature-store.md`

Auto-scaling memory hierarchy for `packet_x` so training adapts to hardware:
- `training/feature_store.py` — `TieredFeatureStore` (GPU hot cache → CPU RAM → disk memmap), static frequency-based cache placement, auto-scale capacity from measured VRAM. One code path: full GPU-resident on L40S (zero feature transfer when everything fits), spills to RAM/disk on larger data, runs CPU-only for smoke tests.
- `training/gpu_sampling.py` — optional GPU-side neighbor sampling (`gather_csr_neighbors_torch`, `TorchLocalMap`, `GpuNeighborBackend`, `TorchHeteroNeighborSampler`), numpy-parity tested.
- `scripts/tools/downcast_packet_x.py` — one-time tool to downcast an existing artifact's `packet_x` fp32→fp16 (avoids a full rebuild).

Both are config-gated and OFF by default (`feature_store.enabled`, `gpu_sampling.enabled`).

### Run commands
```bat
:: Build the graph artifact (local CPU)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.preprocessing.cli ^
  --raw-root data/raw ^
  --out-npz outputs/v3_ob/graph.npz ^
  --out-meta outputs/v3_ob/graph.meta.json ^
  --pmi-table outputs/v3_ob/pmi_table.parquet ^
  --pmi-meta outputs/v3_ob/pmi.meta.json ^
  --splits-json outputs/v3_ob/splits.json ^
  --pmi-subsample-per-class 25000 ^
  --temporal-train-frac 0.80 ^
  --temporal-val-frac 0.10

:: Sanity audit BEFORE upload
D:\v\nt114\Scripts\python.exe scripts/diagnostics/v3_artifact_audit.py outputs/v3_ob/graph.npz

:: Build eval-only clean answer key + EACS anchor mask (need pcaps)
D:\v\nt114\Scripts\python.exe scripts/tools/extract_clean_eval_labels.py ^
  --graph-meta outputs/v3_ob/graph.meta.json --raw-root data/raw ^
  --out-npy outputs/v3_ob/clean_eval_labels.npy --out-audit outputs/v3_ob/clean_eval_labels.audit.json
D:\v\nt114\Scripts\python.exe scripts/tools/extract_eacs_anchor_mask.py ^
  --graph-meta outputs/v3_ob/graph.meta.json --raw-root data/raw ^
  --out-npy outputs/v3_ob/eacs_anchor_mask.npy --out-audit outputs/v3_ob/anchor.audit.json

:: (Optional) downcast an existing fp32 artifact to fp16 without rebuilding
D:\v\nt114\Scripts\python.exe scripts/tools/downcast_packet_x.py outputs/v3_ob/graph.npz outputs/v3_ob/graph.npz

:: Smoke train HGT on CPU (verify trainer integration)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.training.train_hgt_flow_classifier ^
  --config configs/eg_hgt_v6_ob_eacs_v2.yaml --device cpu --epochs 2

:: Train HGT + EACS v2 on the server
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.training.train_hgt_flow_classifier ^
  --config configs/eg_hgt_v6_ob_eacs_v2.yaml --device cuda

:: Grade noisy + clean (+ calibration on clean val) AFTER training
D:\v\nt114\Scripts\python.exe scripts/eval/calibrate_thresholds.py ^
  --config configs/eg_hgt_v6_ob_eacs_v2.yaml ^
  --checkpoint outputs/v3_ob_eacs_v2/hgt_flow_best.pt ^
  --training-summary outputs/v3_ob_eacs_v2/training_summary.json ^
  --clean-labels outputs/v3_ob/clean_eval_labels.npy ^
  --out outputs/v3_ob_eacs_v2/confusion_calibrated.json --device cuda

:: Eval both splits (Smart-BOTH gap)
D:\v\nt114\Scripts\python.exe scripts/eval/v3_eval_both_splits.py ^
  --checkpoint-random outputs/v3_ob_eacs_v2/hgt_flow_best.pt ^
  --checkpoint-temporal outputs/v3_ob_eacs_v2_temporal/hgt_flow_best.pt ^
  --graph outputs/v3_ob/graph.npz

:: All unit tests
D:\v\nt114\Scripts\python.exe -m pytest tests/ -q
```

The tiered feature store is already enabled in `configs/eg_hgt_v6_ob_eacs_v2.yaml`. To enable it in another config, add:
```yaml
feature_store:
  enabled: true
  cache_fraction: 0.6
  model_reserve_gb: 4.0
  n_warmup_batches: 200
# gpu_sampling: { enabled: true }   # only if CPU sampling is the throughput bottleneck
```

## Diagnosis history (v1, May 2026)
v1 HGT plateaued at macro-F1 ≈ 0.12. Root cause was the input pipeline, not the recipe:
- **Extract**: v1 dropped every TCP control packet (SYN/ACK/RST/FIN) and all ICMP, and never recorded TCP flags — scan/recon/flood lost their signature.
- **Flow features**: only 6 coarse stats.
- **`matches_technique` edges**: cosine of SecureBERT(payload-hex) vs SecureBERT(MITRE-text) — semantically meaningless.

The current pipeline fixes all three (all-packets+flags extractor; ~80 CICFlowMeter features; evidence-grounded MSEE edges) and adds **zero learned encoders** — every HGT input is deterministic; HGT is the only model that trains. (The legacy three-tier v1 loader, `load_three_tier_graph_artifact`, is kept only for the on-disk graph store path.)

## Academic novelty (Q1 framing)
Four layers:

1. **Multi-Source Evidence Ensemble (MSEE)** for packet→technique edges: replaces both hand-crafted signature rules and semantically-invalid embedding cosine with a three-source statistical ensemble — PMI candidate generation, L1-regularized multinomial logistic refinement, and MITRE procedure substring matching. Each edge carries token-level provenance suitable for SOC audit. No learned encoder.
2. **Typed heterogeneous schema**: 5 typed evidence edge types per attack family (injection / command_exec / file_upload / recon / c2_beacon), flow-flow homophily edges, host tier — HGT's multi-head attention operates on many typed relations. PMI provides PRIOR weights; HGT attention REFINES; GCL auxiliary loss SUPERVISES.
3. **Smart-BOTH evaluation protocol**: report both random-stratified (matches prior work) and temporal split (deployment-realistic). The gap between random and temporal F1 is itself a finding — a small gap means the model captures attack-intrinsic patterns rather than dataset-specific campaign fingerprints.
4. **EACS + clean-key (LNL) grading**: quantify CIC-IoT-2023's per-capture label noise with a signature-isolated answer key (eval-only, never in any loss), then prove noise-filtering via the **sign of (noisy − clean) macro-F1**. EACS self-relabels suspect web-attack flows using MITRE evidence + the model's own predictions (procedure-literal anchor, 95.2% precision). Same HGT backbone, same data, same split — the only changed variable is the EACS controller; baselines memorize the noise (positive gap), EACS filters it (negative gap), with ~97% fewer false web-attack alarms at equal recall.

This is what XG-NID (post-hoc LLM narrative, no knowledge graph) and PacketCLIP (learned contrastive encoder, single-source) structurally cannot match.
