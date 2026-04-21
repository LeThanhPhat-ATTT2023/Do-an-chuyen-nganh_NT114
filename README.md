# Context-Rich Explainable IDS (NT114 Thesis)

This repository bootstraps the thesis project:

Context-rich and explainable network intrusion detection based on heterogeneous graph learning, MITRE ATT&CK integration, and embedding-level LLM reasoning.

## Problem and direction

Many graph-based IDS pipelines still push high-dimensional raw payload bytes directly into GNN layers. This hurts real-time viability and introduces noise.

This project follows a pragmatic strategy:

1. Use only the first 256 payload bytes per packet.
2. Distill a transformer teacher (for example SecureBERT) into a compact 1D-CNN student.
3. Build heterogeneous graph representations with flow, packet, and MITRE semantic context.
4. Split deployment into fast detection path and slower explainability path.

## Current bootstrap scope

This repository already contains runnable scaffolding for:

1. Extracting fixed-size payload vectors from PCAP files.
2. Creating teacher embedding targets with a transformer model.
3. Training a compact student 1D-CNN by distillation.

The graph/HGT and slow-path XAI modules are the next implementation stages.

## Repository layout

```text
.
|-- configs/
|   `-- pipeline.example.yaml
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- mitre/
|-- docs/
|   `-- feasibility_assessment_vi.md
|-- scripts/
|   |-- extract_payload_dataset.py
|   |-- build_teacher_targets.py
|   `-- train_student_cnn.py
|-- src/graphslm_ids/
|   |-- data/pcap_payload_extractor.py
|   |-- models/student_cnn.py
|   `-- utils/io.py
|-- tests/
|   `-- test_payload_extractor.py
|-- requirements.txt
|-- requirements-ml.txt
`-- pyproject.toml
```

## Quickstart (Windows PowerShell)

### 1) Create and activate environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

For phase 1 only:

```powershell
pip install -r requirements.txt
```

For distillation/training phases:

```powershell
pip install -r requirements-ml.txt
```

## Phase 0: Prepare MITRE Knowledge Base (for semantic layer)

Download official ATT&CK Enterprise STIX data:

```powershell
$outDir = "data/mitre"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json" -OutFile "$outDir/enterprise-attack.json"
```

Build MITRE node/edge CSV files from STIX:

```powershell
python scripts/prepare_mitre_knowledge_base.py --input-json "data/mitre/enterprise-attack.json"
```

Outputs:

1. data/mitre/mitre_techniques.csv
2. data/mitre/mitre_tactics.csv
3. data/mitre/mitre_technique_tactic_edges.csv
4. data/mitre/mitre_export_stats.json

For CIC-IoT2023 bootstrap mapping, use the seed file:

1. configs/cic_iot2023_to_mitre_seed.csv

Detailed step-by-step (Vietnamese):

1. docs/mitre_setup_cic_iot2023_vi.md

## Phase 1: Extract payload dataset (256 bytes)

Put your .pcap/.pcapng files under data/raw/ then run:

```powershell
python scripts/extract_payload_dataset.py --input-glob "data/raw/**/*.pcap" "data/raw/**/*.pcapng" --output-dir "data/interim/payload_dataset" --payload-length 256
```

Optional (review/debug only): export graph-review CSV files:

```powershell
python scripts/extract_payload_dataset.py --input-glob "data/raw/**/*.pcap" "data/raw/**/*.pcapng" --output-dir "data/interim/payload_dataset" --payload-length 256 --export-graph-csv --graph-flow-timeout-seconds 30 --graph-max-packets-per-flow 20
```

Expected outputs:

1. data/interim/payload_dataset/payload_256.npy
2. data/interim/payload_dataset/metadata.csv
3. data/interim/payload_dataset/stats.json
4. data/interim/payload_dataset/flow_nodes.csv (when --export-graph-csv is enabled)
5. data/interim/payload_dataset/packet_nodes.csv (when --export-graph-csv is enabled)
6. data/interim/payload_dataset/contain_edges.csv (when --export-graph-csv is enabled)
7. data/interim/payload_dataset/link_edges.csv (when --export-graph-csv is enabled)

## Phase 1.5: Build Graph Artifact Directly (no 4 CSV required)

This is the recommended path when you only need graph tensors/artifacts for training.

```powershell
python scripts/build_graph_artifact.py --metadata-csv "data/interim/payload_dataset/metadata.csv" --payload-npy "data/interim/payload_dataset/payload_256.npy" --output-npz "data/processed/graph_artifact.npz" --flow-timeout-seconds 30 --max-packets-per-flow 20
```

Outputs:

1. data/processed/graph_artifact.npz
2. data/processed/graph_artifact.meta.json

## Phase 2: Build teacher targets

```powershell
python scripts/build_teacher_targets.py --payload-npy "data/interim/payload_dataset/payload_256.npy" --metadata-csv "data/interim/payload_dataset/metadata.csv" --output-path "data/processed/teacher_targets.npy" --model-name "ehsanaghaei/SecureBERT" --batch-size 32
```

Notes:

1. If RAM is tight, lower --batch-size to 8 or 16.
2. Use --max-rows for quick smoke testing before full run.
3. metadata.csv is recommended to preserve the real payload length for each packet.

## Phase 3: Train student 1D-CNN

```powershell
python scripts/train_student_cnn.py --payload-npy "data/interim/payload_dataset/payload_256.npy" --teacher-npy "data/processed/teacher_targets.npy" --output-dir "outputs/student_cnn" --batch-size 256 --epochs 30
```

Expected outputs:

1. outputs/student_cnn/student_cnn_best.pt
2. outputs/student_cnn/training_summary.json

## Phase 3.5: Evaluate student embedding quality

```powershell
python scripts/evaluate_student_cnn.py --payload-npy "data/interim/payload_dataset/payload_256.npy" --teacher-npy "data/processed/teacher_targets.npy" --metadata-csv "data/interim/payload_dataset/metadata.csv" --checkpoint "outputs/student_cnn/student_cnn_best.pt" --output-path "outputs/student_cnn/evaluation_summary.json" --batch-size 512 --val-ratio 0.1 --seed 42
```

Expected output:

1. outputs/student_cnn/evaluation_summary.json

The evaluation report includes:

1. Overall MSE and cosine metrics for all/train/val splits.
2. Per-label metrics from metadata labels.
3. Top-5 labels with lowest cosine similarity for error analysis.

## Phase 4: Export student model to ONNX

```powershell
python scripts/export_student_onnx.py --checkpoint "outputs/student_cnn/student_cnn_best.pt" --output-path "outputs/student_cnn/student_cnn.onnx" --input-length 256 --opset 17 --verify
```

Expected outputs:

1. outputs/student_cnn/student_cnn.onnx
2. outputs/student_cnn/student_cnn.meta.json

## Phase 4.5: Export student embeddings for all packets

```powershell
python scripts/export_student_embeddings.py --payload-npy "data/interim/payload_dataset/payload_256.npy" --checkpoint "outputs/student_cnn/student_cnn_best.pt" --output-path "data/processed/student_embeddings.npy" --batch-size 1024 --device auto
```

Expected outputs:

1. data/processed/student_embeddings.npy
2. data/processed/student_embeddings.meta.json

## Phase 4.6: Build MITRE technique embeddings

```powershell
python scripts/build_mitre_technique_embeddings.py --techniques-csv "data/mitre/mitre_techniques.csv" --output-path "data/mitre/mitre_techniques_embeddings.npy" --teacher-meta-json "data/processed/teacher_targets.meta.json" --device auto
```

Expected outputs:

1. data/mitre/mitre_techniques_embeddings.npy
2. data/mitre/mitre_techniques_embeddings.meta.json

## Phase 5: Build Three-Tier Graph Artifact (flow, packet, MITRE)

```powershell
python scripts/build_three_tier_graph_artifact.py --metadata-csv "data/interim/payload_dataset/metadata.csv" --payload-npy "data/interim/payload_dataset/payload_256.npy" --student-embedding-npy "data/processed/student_embeddings.npy" --mitre-techniques-csv "data/mitre/mitre_techniques.csv" --mitre-technique-embeddings-npy "data/mitre/mitre_techniques_embeddings.npy" --mitre-technique-tactic-edges-csv "data/mitre/mitre_technique_tactic_edges.csv" --output-npz "data/processed/graph_artifact_3tier.npz" --similarity-threshold 0.85 --packet-top-k 3 --flow-top-k 3
```

Expected outputs:

1. data/processed/graph_artifact_3tier.npz
2. data/processed/graph_artifact_3tier.meta.json

This artifact keeps the original flow-packet structure and adds semantic tactical edges:

1. packet -> MITRE technique edges by cosine similarity.
2. flow -> MITRE technique edges by aggregated packet similarity.
3. MITRE technique -> tactic edges from ATT&CK knowledge base.

## Test

```powershell
pytest
```

## Thesis tracking notes

1. Feasibility and risk notes are tracked in docs/feasibility_assessment_vi.md.
2. Pipeline defaults are tracked in configs/pipeline.example.yaml.

## Important implementation reminders

1. Keep teacher computation offline; do not use transformer teacher in online detection path.
2. Keep online embedding path small and exportable (PyTorch -> ONNX Runtime).
3. Introduce MITRE tactical edge only after validating embedding quality of the student.

## License

This project follows the repository LICENSE file.