# NT114 — GraphSLM IDS (Đồ án chuyên ngành)

A heterogeneous-graph intrusion detection system on CIC-IoT-2023-style traffic.
Raw PCAPs → an evidence-grounded knowledge graph (flow / packet / host / technique /
tactic nodes) → an HGT flow classifier trained with a GCL auxiliary loss, evaluated
on **both** random and temporal splits.

## Python environment (USE THIS)
- Dedicated venv: **`D:\v\nt114`** (Windows).
- Interpreter: **`D:\v\nt114\Scripts\python.exe`** (Python 3.13.13).
- Installed: numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, torch 2.11.0+cpu, scikit-learn 1.8.0, dpkt, scapy, pytest 9.0.3.
- Always run project scripts with this interpreter, e.g.:
  `D:\v\nt114\Scripts\python.exe -m pytest tests/ -q`
- Machine: 16 logical CPUs, ~16 GB RAM. CPU-only torch locally; full HGT training runs on a remote L40S (48 GB) VM.

## Local data
- `data/raw/14gb/<class>/*.pcap` — 13 PCAPs (5.6 GB total), one per attack class.
- `data/mitre/` — MITRE ATT&CK CSVs + STIX `enterprise-attack.json` (technique embeddings, technique↔tactic edges, class→technique map, technique families).
- `outputs/v3/graph.npz` (+ `graph.meta.json`, `splits.json`, `pmi_table.parquet`) — the built graph artifact (gitignored).

Dataset: CIC-IoT-2023 style, **13 classes**, ~612k bidirectional flows / ~600k packet nodes (control packets kept). Labels assigned per-PCAP-file via `infer_label_from_path`.

`packet_x` (the dominant feature array, `(n_packets, 2323)`) is stored on disk as **float16** to halve footprint; the loader upcasts to float32 by default (or keeps float16 with `packet_dtype="preserve"`). See the tiered feature store below.

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
- `training/train_hgt_flow_classifier.py` — reads `data.artifact_version: v3`, calls `load_v3_artifact()`, adds a **GCL auxiliary loss** with positive pairs from the class→technique map.
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
:: Build the graph artifact (local CPU, ~2.5 hours)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.preprocessing.cli ^
  --raw-root data/raw/14gb ^
  --out-npz outputs/v3/graph.npz ^
  --out-meta outputs/v3/graph.meta.json ^
  --pmi-table outputs/v3/pmi_table.parquet ^
  --pmi-meta outputs/v3/pmi.meta.json ^
  --splits-json outputs/v3/splits.json ^
  --pmi-subsample-per-class 25000 ^
  --temporal-train-frac 0.80 ^
  --temporal-val-frac 0.10

:: Sanity audit BEFORE upload
D:\v\nt114\Scripts\python.exe scripts/diagnostics/v3_artifact_audit.py outputs/v3/graph.npz

:: (Optional) downcast an existing fp32 artifact to fp16 without rebuilding
D:\v\nt114\Scripts\python.exe scripts/tools/downcast_packet_x.py outputs/v3/graph.npz outputs/v3/graph.npz

:: Smoke train HGT on CPU (verify trainer integration)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.training.train_hgt_flow_classifier ^
  --config configs/eg_hgt.yaml --device cpu --epochs 2

:: Eval both splits AFTER server training
D:\v\nt114\Scripts\python.exe scripts/eval/v3_eval_both_splits.py ^
  --checkpoint-random outputs/v3/checkpoint_random.pt ^
  --checkpoint-temporal outputs/v3/checkpoint_temporal.pt ^
  --graph outputs/v3/graph.npz

:: All unit tests
D:\v\nt114\Scripts\python.exe -m pytest tests/ -q
```

To enable the tiered feature store on the L40S, add to `configs/eg_hgt.yaml`:
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
Three layers:

1. **Multi-Source Evidence Ensemble (MSEE)** for packet→technique edges: replaces both hand-crafted signature rules and semantically-invalid embedding cosine with a three-source statistical ensemble — PMI candidate generation, L1-regularized multinomial logistic refinement, and MITRE procedure substring matching. Each edge carries token-level provenance suitable for SOC audit. No learned encoder.
2. **Typed heterogeneous schema**: 5 typed evidence edge types per attack family (injection / command_exec / file_upload / recon / c2_beacon), flow-flow homophily edges, host tier — HGT's multi-head attention operates on many typed relations. PMI provides PRIOR weights; HGT attention REFINES; GCL auxiliary loss SUPERVISES.
3. **Smart-BOTH evaluation protocol**: report both random-stratified (matches prior work) and temporal split (deployment-realistic). The gap between random and temporal F1 is itself a finding — a small gap means the model captures attack-intrinsic patterns rather than dataset-specific campaign fingerprints.

This is what XG-NID (post-hoc LLM narrative, no knowledge graph) and PacketCLIP (learned contrastive encoder, single-source) structurally cannot match.
