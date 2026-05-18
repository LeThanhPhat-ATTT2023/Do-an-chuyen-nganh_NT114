# Context-Rich Explainable IDS — NT114

Hệ thống phát hiện xâm nhập mạng có ngữ cảnh, kết hợp payload embedding, đồ thị dị thể, MITRE ATT&CK và lớp giải thích bằng SLM. Repo được tổ chức theo hai luồng:

- **Offline**: xử lý dữ liệu PCAP, sinh embedding qua teacher transformer, distill sang student 1D-CNN, xây graph dị thể 3-tier và huấn luyện HGT flow classifier.
- **Runtime**: nhận packet, trích payload online, chạy student ONNX, gắn ngữ cảnh MITRE, dựng hot graph, suy luận HGT, áp policy và đẩy alert sang slow path để sinh báo cáo XAI.

Ý tưởng cốt lõi: **không đưa raw payload trực tiếp vào GNN**. Payload được chuẩn hóa thành vector ngữ nghĩa trước, sau đó mới dùng làm đặc trưng packet trong graph. MITRE technique/tactic được đưa vào graph như tầng tri thức bổ sung để mô hình không chỉ dự đoán nhãn tấn công mà còn có đường dẫn ngữ cảnh phục vụ giải thích.

## Mục Lục

- [Trạng thái hiện tại](#trạng-thái-hiện-tại)
- [Dataset CIC IoT 2023](#dataset-cic-iot-2023)
- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt](#cài-đặt)
- [Pipeline offline từ đầu](#pipeline-offline-từ-đầu)
- [Chạy runtime fast path và slow path](#chạy-runtime-fast-path-và-slow-path)
- [Cấu hình quan trọng](#cấu-hình-quan-trọng)
- [Console scripts](#console-scripts)
- [Notebooks](#notebooks)
- [Kiểm thử](#kiểm-thử)
- [Artifact và dữ liệu](#artifact-và-dữ-liệu)
- [Tài liệu liên quan](#tài-liệu-liên-quan)
- [Giới hạn hiện tại](#giới-hạn-hiện-tại)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Trạng Thái Hiện Tại

### Offline pipeline

| Bước | Trạng thái | Artifact |
|---|---|---|
| Trích xuất payload 256B từ PCAP | ✅ Xong | `data/interim/payload_dataset_14gb/` |
| Sinh teacher targets (SecureBERT) | ✅ Xong | `data/processed/teacher_targets.npy` |
| Train student 1D-CNN | ✅ Xong | `outputs/student_cnn/student_cnn_best.pt` |
| Export student embeddings | ✅ Xong | `data/processed/student_embeddings_14gb.npy` |
| Chuẩn bị MITRE ATT&CK KB | ✅ Xong | `data/mitre/` |
| Xây graph 3-tier | ✅ Xong | `data/processed/graph_artifact_3tier_14gb.*` |
| Export student sang ONNX | ⏳ Chưa làm | `outputs/student_cnn/student_cnn.onnx` |
| Convert graph sang on-disk CSR store | ⏳ Chưa làm | `data/graph_store_14gb/` |
| Train HGT flow classifier | ⏳ Chưa làm | `outputs/hgt_flow_classifier_14gb/` |

### Runtime

Chưa sẵn sàng chạy end-to-end — cần hoàn thành 3 bước pending ở trên trước (student ONNX, graph store, HGT checkpoint). Toàn bộ mã nguồn runtime đã implement đầy đủ.

### Kết quả student 1D-CNN (14gb dataset)

| Metric | Giá trị |
|---|---:|
| Samples tổng | 5,000,000 |
| Samples train / val | 4,500,000 / 500,000 |
| Embedding dim | 768 |
| Best epoch | 2 |
| Best val loss | 0.00111 |

## Dataset CIC IoT 2023

Dataset gốc: [CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html). Workspace hiện dùng 14 GB PCAP gồm 13 lớp:

```text
Backdoor_Malware
Benign
BrowserHijacking
CommandInjection
DDoS-ICMP_Fragmentation
Recon-HostDiscovery
Recon-OSScan
Recon-PingSweep
Recon-PortScan
SqlInjection
Uploading_Attack
VulnerabilityScan
XSS
```

Thống kê graph artifact hiện có (`graph_artifact_3tier_14gb`):

| Thành phần | Giá trị |
|---|---:|
| Payload length | 256 byte |
| Embedding dim | 768 |
| Similarity threshold | 0.82 |
| Packet top-k MITRE | 5 |
| Flow top-k MITRE | 5 |
| Số flow | 1,507,615 |
| Số packet | 5,261,944 |
| Cạnh flow→packet | 5,261,944 |
| Cạnh packet→technique | 5,514,523 |
| Cạnh flow→technique | 1,586,463 |
| Cạnh packet→packet | 3,754,329 |
| Cạnh technique→tactic | 887 |
| Số MITRE technique | 691 |
| Số MITRE tactic | 14 |

## Kiến Trúc Tổng Quan

```text
Offline
-------
Raw PCAP (14 GB)
  → PayloadExtractor        → payload_256.npy + metadata.csv
  → SecureBERT teacher      → teacher_targets.npy
  → Student 1D-CNN distill  → student_cnn_best.pt
  → Export embeddings       → student_embeddings_14gb.npy
  → MITRE ATT&CK KB         → mitre_techniques_embeddings.npy
  → Build 3-tier graph      → graph_artifact_3tier_14gb.npz
  → Convert CSR store       → graph_store_14gb/
  → Train HGT               → hgt_flow_best.pt  ← [CHƯA XONG]

Runtime
-------
Packet stream
  → FlowTracker
  → PayloadExtractor online
  → StudentRuntime (ONNX)
  → MitreIndex top-k
  → SIGC edge filter
  → PersistentGraphStore (source of truth)
  → HotGraphBuffer (RAM cache)
  → SubgraphBuilder (DLG-IDS Top-N)
  → HGTRuntime
  → PolicyEngine
  → AlertDispatcher → SlowPathWorker → XAI report
```

Các module chính:

| Package | Vai trò |
|---|---|
| `graphslm_ids.offline.preprocessing` | Trích payload, MITRE KB, teacher target, xây graph |
| `graphslm_ids.offline.training` | Train/evaluate/export student, neighbor sampling, train HGT |
| `graphslm_ids.models` | `StudentCNN` và `HeteroGraphTransformer` |
| `graphslm_ids.runtime.fast_path` | Data plane: flow tracking, ONNX inference, MITRE index, SIGC, hot graph, Top-N subgraph, policy |
| `graphslm_ids.runtime.slow_path` | Evidence builder, ranker, SLM report generator, validator, fallback |
| `graphslm_ids.runtime.pipeline` | Orchestration: config, persistent graph store, cold-store, counterfactual, main pipeline |

## Cấu Trúc Thư Mục

```text
.
├── configs/
│   ├── hgt.example.yaml               # template HGT config
│   ├── hgt_t082_k5_l3_d01.yaml        # config cũ (cần cập nhật paths cho 14gb)
│   ├── hgt_paper_variants/            # 6 config biến thể cho so sánh
│   ├── pipeline.example.yaml          # runtime pipeline config
│   ├── cic_iot2023_to_mitre_seed.csv
│   └── mitre_techniques_template.csv
├── data/
│   ├── raw/14gb/                      # 13 PCAP attack classes
│   ├── interim/payload_dataset_14gb/  # payload_256.npy + metadata.csv
│   ├── processed/                     # teacher targets, student embeddings, graph artifact
│   └── mitre/                         # MITRE STIX JSON, CSV, embeddings
├── docs/
│   ├── architecture/                  # system flows, feasibility, fast-slow bridge, graph strategy
│   ├── runtime/                       # MITRE setup, HGT runtime streaming
│   ├── training/                      # HGT config guide, scalable training, student CNN report
│   ├── xai/                           # slow path XAI design
│   └── notebooks_vi.md
├── models/SecureBERT/                 # teacher model weights (local)
├── notebooks/
│   ├── train_hgt_official_full_pipeline_kaggle.ipynb
│   └── training/
│       ├── kaggle/                    # 01_distill_student_cnn, 02a_smoke, 02b_full_preview
│       └── local/                    # 01_extract_payload, 02_export_embeddings, 03_build_graph
├── outputs/
│   └── student_cnn/                   # student_cnn_best.pt, training_summary.json
├── src/graphslm_ids/
│   ├── models/                        # hgt.py, student_cnn.py
│   ├── offline/
│   │   ├── preprocessing/             # extract_payload, build_graph, MITRE KB, ...
│   │   └── training/                  # train_hgt, train_student, neighbor_sampling, ...
│   ├── runtime/
│   │   ├── fast_path/                 # flow_tracker, hgt_runtime, student_runtime, ...
│   │   ├── slow_path/                 # evidence_builder, report_generator, slm_client, ...
│   │   └── pipeline/                  # runtime_pipeline, pipeline_config, graph_store, ...
│   └── utils/
├── tests/                             # 13 test files
├── requirements.txt
├── requirements-ml.txt
└── pyproject.toml
```

## Yêu Cầu Môi Trường

- Python `>= 3.10`.
- Windows PowerShell được dùng trong các ví dụ lệnh.
- Để train teacher/student/HGT: cần GPU và các thư viện trong `requirements-ml.txt`.
- Để chạy slow path với SLM local: cần Ollama hoặc backend tương thích.

Dependencies nền (`requirements.txt`):

```text
numpy
pandas
scapy
tqdm
pyyaml
```

Dependencies ML (`requirements-ml.txt`):

```text
torch
transformers
onnxruntime
```

## Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ml.txt
pip install -e .
```

Kiểm tra:

```powershell
graphslm-train-hgt --help
graphslm-run-runtime --help
```

## Pipeline Offline Từ Đầu

Workspace hiện đã hoàn thành **bước 1–7**. Nếu clone mới, chạy lại từ đầu theo thứ tự sau.

### 1. Chuẩn bị MITRE ATT&CK

```powershell
New-Item -ItemType Directory -Path "data/mitre" -Force | Out-Null
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json" `
  -OutFile "data/mitre/enterprise-attack.json"

graphslm-prepare-mitre --input-json "data/mitre/enterprise-attack.json"
```

Output: `data/mitre/mitre_techniques.csv`, `mitre_tactics.csv`, `mitre_technique_tactic_edges.csv`.

Tài liệu: `docs/runtime/mitre_setup_cic_iot2023_vi.md`.

### 2. Trích xuất payload dataset

Đặt PCAP vào `data/raw/`. Cấu trúc khuyến nghị: `data/raw/<ClassName>/<file>.pcap`.

```powershell
graphslm-extract-payload `
  --input-glob "data/raw/**/*.pcap" "data/raw/**/*.pcapng" `
  --output-dir "data/interim/payload_dataset_14gb" `
  --payload-length 256
```

Output: `payload_256.npy`, `metadata.csv`, `stats.json`.

### 3. Sinh teacher targets

```powershell
graphslm-build-teacher `
  --payload-npy "data/interim/payload_dataset_14gb/payload_256.npy" `
  --metadata-csv "data/interim/payload_dataset_14gb/metadata.csv" `
  --output-path "data/processed/teacher_targets.npy" `
  --model-name "ehsanaghaei/SecureBERT" `
  --batch-size 32 `
  --device auto
```

Teacher transformer chỉ dùng offline — runtime không gọi lại.

### 4. Train student 1D-CNN

```powershell
graphslm-train-student `
  --payload-npy "data/interim/payload_dataset_14gb/payload_256.npy" `
  --teacher-npy "data/processed/teacher_targets.npy" `
  --output-dir "outputs/student_cnn" `
  --batch-size 256 `
  --epochs 30 `
  --device auto
```

Đánh giá:

```powershell
graphslm-eval-student `
  --payload-npy "data/interim/payload_dataset_14gb/payload_256.npy" `
  --teacher-npy "data/processed/teacher_targets.npy" `
  --metadata-csv "data/interim/payload_dataset_14gb/metadata.csv" `
  --checkpoint "outputs/student_cnn/student_cnn_best.pt" `
  --output-path "outputs/student_cnn/evaluation_summary.json" `
  --device auto
```

### 5. Export student sang ONNX

```powershell
graphslm-export-student-onnx `
  --checkpoint "outputs/student_cnn/student_cnn_best.pt" `
  --output-path "outputs/student_cnn/student_cnn.onnx" `
  --input-length 256 `
  --opset 17 `
  --verify `
  --device cpu
```

ONNX artifact này được `StudentRuntime` dùng trong fast path.

### 6. Export student embeddings

```powershell
graphslm-export-student-emb `
  --payload-npy "data/interim/payload_dataset_14gb/payload_256.npy" `
  --checkpoint "outputs/student_cnn/student_cnn_best.pt" `
  --output-path "data/processed/student_embeddings_14gb.npy" `
  --batch-size 1024 `
  --device auto
```

Hoặc dùng script PowerShell có sẵn:

```powershell
.\run_02_export_embeddings.ps1
```

### 7. Tạo MITRE technique embeddings

```powershell
graphslm-build-mitre-emb `
  --techniques-csv "data/mitre/mitre_techniques.csv" `
  --output-path "data/mitre/mitre_techniques_embeddings.npy" `
  --teacher-meta-json "data/processed/teacher_targets.meta.json" `
  --batch-size 64 `
  --device auto
```

### 8. Xây graph 3-tier

```powershell
graphslm-build-three-tier-graph `
  --metadata-csv "data/interim/payload_dataset_14gb/metadata.csv" `
  --payload-npy "data/interim/payload_dataset_14gb/payload_256.npy" `
  --student-embedding-npy "data/processed/student_embeddings_14gb.npy" `
  --mitre-techniques-csv "data/mitre/mitre_techniques.csv" `
  --mitre-technique-embeddings-npy "data/mitre/mitre_techniques_embeddings.npy" `
  --mitre-technique-tactic-edges-csv "data/mitre/mitre_technique_tactic_edges.csv" `
  --output-npz "data/processed/graph_artifact_3tier_14gb.npz" `
  --similarity-threshold 0.82 `
  --packet-top-k 5 `
  --flow-top-k 5
```

Output: `graph_artifact_3tier_14gb.npz` và `graph_artifact_3tier_14gb.meta.json`.

### 9. Convert graph sang on-disk CSR store

Graph 14gb quá lớn để load toàn bộ vào RAM. Chuyển sang CSR store một lần:

```powershell
graphslm-convert-graph-store `
  --graph-npz "data/processed/graph_artifact_3tier_14gb.npz" `
  --graph-meta-json "data/processed/graph_artifact_3tier_14gb.meta.json" `
  --output-root "data/graph_store_14gb"
```

Output:

```text
data/graph_store_14gb/manifest.json
data/graph_store_14gb/nodes/flow/features.f32
data/graph_store_14gb/nodes/flow/labels.i64
data/graph_store_14gb/edges/<edge_name>/indptr.i64
data/graph_store_14gb/edges/<edge_name>/indices.i64
data/graph_store_14gb/splits/train_flow_ids.i64
```

### 10. Train HGT flow classifier

**Lưu ý**: Các file config hiện tại (`hgt_t082_k5_l3_d01.yaml`, `hgt.example.yaml`) vẫn trỏ đến artifact cũ `t082_k5`. Trước khi train, cần cập nhật `data.graph_store_root` và `data.graph_npz` trong config sang đường dẫn 14gb:

```yaml
data:
  source: graph_store
  graph_store_root: data/graph_store_14gb
  graph_npz: data/processed/graph_artifact_3tier_14gb.npz
  graph_meta_json: data/processed/graph_artifact_3tier_14gb.meta.json

train:
  output_dir: outputs/hgt_flow_classifier_14gb
  device: cuda
  amp: true
```

Chạy train:

```powershell
graphslm-train-hgt --config "configs/hgt_t082_k5_l3_d01.yaml"
```

Hoặc trên Kaggle GPU T4 x2 chạy trước `notebooks/training/kaggle/02a_train_hgt_smoke_test.ipynb` (3 epochs), sau đó chạy `notebooks/training/kaggle/02b_train_hgt_full_preview.ipynb`.

Output:

```text
outputs/hgt_flow_classifier_14gb/hgt_flow_best.pt
outputs/hgt_flow_classifier_14gb/training_summary.json
```

## Chạy Runtime Fast Path Và Slow Path

Runtime cần đủ các artifact sau:

```text
outputs/student_cnn/student_cnn.onnx                    ← bước 5
data/mitre/mitre_techniques.csv
data/mitre/mitre_technique_tactic_edges.csv
data/mitre/mitre_techniques_embeddings.npy               ← bước 7
outputs/hgt_flow_classifier_14gb/hgt_flow_best.pt       ← bước 10
```

Cập nhật `configs/pipeline.example.yaml` để trỏ đúng checkpoint và embedding.

Chạy replay không bật slow worker:

```powershell
graphslm-run-runtime `
  --config "configs/pipeline.example.yaml" `
  --input "data/raw/14gb/Recon-PortScan/Recon-PortScan.pcap" `
  --max-packets 500 `
  --no-worker
```

Chạy có slow worker:

```powershell
graphslm-run-runtime `
  --config "configs/pipeline.example.yaml" `
  --input "data/raw/14gb/Recon-PortScan/Recon-PortScan.pcap" `
  --max-packets 500
```

Khi bật slow worker, pipeline:

1. Fast path xử lý từng packet → `FlowTracker` → `StudentRuntime` → `MitreIndex`.
2. Ghi vào `PersistentGraphStore` (source of truth) và `HotGraphBuffer` (RAM cache).
3. `SubgraphBuilder` dựng K-hop subgraph quanh flow.
4. `HGTRuntime` suy luận, `PolicyEngine` quyết định alert.
5. `AlertDispatcher` tạo `SlowPathJob` khi đạt ngưỡng.
6. `SlowPathWorker` hydrate context, build evidence, gọi SLM, validate report.

### Cấu hình Ollama

```yaml
slm:
  backend: ollama
  model: qwen2.5:3b-instruct-q4_k_m
  endpoint: http://localhost:11434
```

```powershell
ollama serve
ollama pull qwen2.5:3b-instruct-q4_k_m
```

Dùng `--no-worker` để bỏ qua slow path khi smoke test.

## Cấu Hình Quan Trọng

### `configs/hgt.example.yaml` / `configs/hgt_t082_k5_l3_d01.yaml`

Điều khiển train HGT. Các key quan trọng:

```yaml
data:
  source: graph_store          # mmap CSR — không load full graph vào RAM
  graph_store_root: data/graph_store_14gb
  graph_npz: data/processed/graph_artifact_3tier_14gb.npz
  graph_meta_json: data/processed/graph_artifact_3tier_14gb.meta.json

train:
  output_dir: outputs/hgt_flow_classifier_14gb
  epochs: 200
  lr: 0.001
  weight_decay: 0.00005
  batch_seed_flows: 256
  grad_accum_steps: 2
  scheduler: onecycle
  scheduler_pct_start: 0.05
  class_weight: balanced
  monitor: val_macro_f1
  device: cuda
  amp: true

sampler:
  fanouts:
    flow__contains__packet: 20
    packet__next_packet__packet: 4
    packet__matches_technique__technique: 5
    flow__matches_technique__technique: 5
    technique__belongs_to_tactic__tactic: 1
  always_include_all_tactics: true
  always_include_all_techniques: true
```

### `configs/hgt_paper_variants/`

Chứa 6 config biến thể HGT dùng để so sánh trong báo cáo:

| File | Mô tả |
|---|---|
| `hgt_t082_k5_ahgt_dfd_funnel_l3_h128_h4.yaml` | AHGT-DFD Funnel |
| `hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml` | DLG-IDS Sparse |
| `hgt_t082_k5_gatransformer_deep_l6_h256_h8.yaml` | GAT Transformer Deep |
| `hgt_t082_k5_one2_iov_l1_h64_h2.yaml` | ONE2 IoV Lightweight |
| `hgt_t082_k5_relgt_multi_token_l3_h128_h8.yaml` | RelGT Multi-Token |
| `hgt_t082_k5_xgnid_dual_modal_l1_h32_h4.yaml` | XGNID Dual-Modal |

### `configs/pipeline.example.yaml`

Điều khiển runtime end-to-end:

| Section | Vai trò |
|---|---|
| `fast_path` | Student ONNX path, MITRE top-k, payload length |
| `hot_graph` | TTL, giới hạn packet/flow/event trong RAM |
| `preprocessor` | SIGC Local Contribution Score, lọc cạnh yếu |
| `graph_store` | Persistent Graph Store, shard seal, retention, disk quota |
| `subgraph_builder` | DLG-IDS Top-N edge selection cho K-hop subgraph |
| `hgt_runtime` | Checkpoint và meta dùng khi inference |
| `policy` | Alert threshold, benign/attack labels |
| `slow_path` | Queue, top-k evidence, counterfactual |
| `cold_store` | Fallback JSONL khi tắt `graph_store` |
| `slm` | Backend/model sinh báo cáo |
| `validator` | Ngưỡng kiểm tra grounding/hallucination |

## Console Scripts

| Lệnh | Module | Mục đích |
|---|---|---|
| `graphslm-extract-payload` | `offline.preprocessing.extract_payload_dataset` | Trích payload từ PCAP |
| `graphslm-prepare-mitre` | `offline.preprocessing.prepare_mitre_knowledge_base` | Tạo MITRE CSV từ STIX JSON |
| `graphslm-build-teacher` | `offline.preprocessing.build_teacher_targets` | Sinh teacher targets |
| `graphslm-build-mitre-emb` | `offline.preprocessing.build_mitre_technique_embeddings` | Sinh MITRE technique embeddings |
| `graphslm-build-graph` | `offline.preprocessing.build_graph_artifact` | Xây graph flow-packet (cũ) |
| `graphslm-build-three-tier-graph` | `offline.preprocessing.build_three_tier_graph_artifact` | Xây graph 3-tier (flow-packet-technique-tactic) |
| `graphslm-train-student` | `offline.training.train_student_cnn` | Train student 1D-CNN |
| `graphslm-eval-student` | `offline.training.evaluate_student_cnn` | Đánh giá student |
| `graphslm-export-student-onnx` | `offline.training.export_student_onnx` | Export student sang ONNX |
| `graphslm-export-student-emb` | `offline.training.export_student_embeddings` | Export student embeddings |
| `graphslm-train-hgt` | `offline.training.train_hgt_flow_classifier` | Train HGT classifier |
| `graphslm-convert-graph-store` | `offline.training.on_disk_graph_store` | Chuyển NPZ sang on-disk CSR store |
| `graphslm-run-runtime` | `runtime.pipeline.run_runtime_pipeline` | Replay PCAP qua runtime pipeline |

## Notebooks

| Notebook | Môi trường | Mục đích |
|---|---|---|
| `notebooks/training/local/01_extract_payload_from_pcap.ipynb` | Local | Trích payload từ PCAP trên máy local |
| `notebooks/training/local/02_export_student_embeddings.ipynb` | Local | Export student embeddings sau khi train |
| `notebooks/training/local/03_build_three_tier_graph.ipynb` | Local | Xây graph 3-tier từ artifact có sẵn |
| `notebooks/training/kaggle/01_distill_student_cnn.ipynb` | Kaggle GPU | Train student 1D-CNN distillation |
| `notebooks/training/kaggle/02a_train_hgt_smoke_test.ipynb` | Kaggle GPU | Smoke test HGT 3 epochs từ graph artifact hoặc graph store |
| `notebooks/training/kaggle/02b_train_hgt_full_preview.ipynb` | Kaggle GPU T4x2 | Full preview HGT từ graph artifact hoặc graph store |
| `notebooks/train_hgt_official_full_pipeline_kaggle.ipynb` | Kaggle GPU T4x2 | Pipeline đầy đủ: PCAP → HGT, hoặc chỉ train HGT từ graph NPZ có sẵn |

Notebook Kaggle full pipeline hỗ trợ hai mode (chỉnh `PIPELINE_MODE`):

| `PIPELINE_MODE` | Mô tả |
|---|---|
| `full_from_pcap` | Chạy toàn bộ từ PCAP → graph → HGT |
| `existing_graph_npz` | Bỏ qua tiền xử lý, chuyển graph store rồi train HGT |

Và hai mode HGT (chỉnh `HGT_RUN_MODE`):

| `HGT_RUN_MODE` | Mô tả |
|---|---|
| `deployment` | Train 1 config production |
| `paper_variants` | Train song song 7 variant trên 2 GPU để so sánh |

## Kiểm Thử

```powershell
pytest
```

Các nhóm test chính:

```powershell
pytest tests/test_payload_extractor.py
pytest tests/test_graph_artifact_builder.py tests/test_three_tier_graph_artifact.py
pytest tests/test_hgt_model.py
pytest tests/test_hgt_neighbor_sampling.py tests/test_neighbor_sampling_vectorized.py
pytest tests/test_on_disk_graph_store_training.py
pytest tests/test_persistent_graph_store.py
pytest tests/test_fast_slow_bridge.py tests/test_slow_path.py
pytest tests/test_hgt_ddp_smoke.py
```

Nếu test không import được package:

```powershell
pip install -e .
```

## Artifact Và Dữ Liệu

| Thư mục | Nội dung |
|---|---|
| `data/raw/14gb/` | 13 PCAP theo lớp tấn công |
| `data/interim/payload_dataset_14gb/` | `payload_256.npy`, `metadata.csv`, `stats.json` |
| `data/processed/` | Teacher targets, student embeddings 14gb, graph artifact 3tier 14gb |
| `data/mitre/` | STIX JSON, CSV, technique embeddings |
| `data/graph_store_14gb/` | On-disk CSR store (cần tạo — bước 9) |
| `data/runtime/` | Fallback cold store JSONL khi tắt `graph_store` |
| `outputs/student_cnn/` | `student_cnn_best.pt`, `training_summary.json` |
| `outputs/hgt_flow_classifier_14gb/` | Checkpoint và training summary HGT (cần train — bước 10) |

Artifact lớn bị ignore bởi `.gitignore` (`*.pt`, `*.onnx`, `*.npz`, `*.npy`, `data/raw/*`, `data/interim/*`, `data/processed/*`, `data/mitre/*`). Khi clone mới cần tự tái tạo theo pipeline trên, hoặc copy từ nơi lưu trữ riêng.

## Tài Liệu Liên Quan

| Tài liệu | Nội dung |
|---|---|
| `docs/architecture/feasibility_assessment_vi.md` | Đánh giá khả thi của hướng tiếp cận |
| `docs/architecture/system_execution_flows.md` | Mermaid diagram cho offline/runtime flow |
| `docs/architecture/fast_slow_bridge_design_vi.md` | Thiết kế lớp nối fast path và slow path |
| `docs/architecture/unified_graph_growth_strategy_vi.md` | Chiến lược mở rộng graph online |
| `docs/runtime/mitre_setup_cic_iot2023_vi.md` | Chuẩn bị MITRE ATT&CK cho CIC IoT 2023 |
| `docs/runtime/streaming_hgt_runtime_v3_vi.md` | Chiến lược hot graph/runtime streaming |
| `docs/training/hgt_graph_threshold_selection_vi.md` | Chọn threshold/top-k cho graph HGT |
| `docs/training/hgt_train_config_recommendation_vi.md` | Hướng dẫn cấu hình train HGT |
| `docs/training/scalable_hgt_training_design_vi.md` | Thiết kế neighbor sampling cho graph lớn |
| `docs/training/student_1dcnn_distillation_report_vi.md` | Báo cáo student CNN distillation |
| `docs/xai/slm_slow_path_xai_design_vi.md` | Thiết kế slow path XAI bằng SLM |
| `docs/notebooks_vi.md` | Hướng dẫn sử dụng notebooks |

## Giới Hạn Hiện Tại

- HGT chưa được train trên dataset 14gb — chưa có kết quả baseline.
- Hai HGT config hiện có vẫn trỏ đến artifact cũ (`t082_k5`); cần cập nhật paths trước khi train.
- Runtime chưa thể chạy end-to-end vì thiếu student ONNX và HGT checkpoint.
- Mapping packet/flow sang MITRE technique dựa trên cosine similarity ngữ nghĩa — nên diễn giải là tương đồng ngữ nghĩa, không phải bằng chứng TTP xác định.
- Runtime hỗ trợ replay PCAP; live capture/inline blocking chưa phải mục tiêu chính.
- Slow path phụ thuộc chất lượng SLM local. Validator giảm hallucination nhưng không thay thế kiểm chứng của analyst.

## Troubleshooting

### `ModuleNotFoundError: graphslm_ids`

```powershell
pip install -e .
```

### Thiếu `torch`, `transformers` hoặc `onnxruntime`

```powershell
pip install -r requirements-ml.txt
```

### Runtime báo thiếu artifact

- Thiếu `student_cnn.onnx`: chạy `graphslm-export-student-onnx` (bước 5).
- Thiếu `mitre_techniques_embeddings.npy`: chạy `graphslm-build-mitre-emb` (bước 7).
- Thiếu `graph_artifact_3tier_14gb.npz`: chạy `graphslm-build-three-tier-graph` (bước 8).
- Thiếu `hgt_flow_best.pt`: chạy `graphslm-train-hgt` (bước 10).

### OOM khi train HGT

Đảm bảo đã chạy bước 9 (convert graph store) và config đặt:

```yaml
data:
  source: graph_store
  graph_store_root: data/graph_store_14gb
```

Train sẽ đọc từng chunk qua memory-mapped file — không load toàn bộ graph vào RAM.

### Ollama không phản hồi

```powershell
ollama list
ollama serve
```

Hoặc chạy runtime với `--no-worker` để bỏ qua slow path.

### Runtime chạy nhưng không có alert

Kiểm tra `policy.benign_labels` và `policy.alert_threshold` trong `configs/pipeline.example.yaml`:

```yaml
policy:
  alert_threshold: 0.70
  benign_labels:
    - Benign
```

### PowerShell không cho activate virtualenv

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## License

Xem file `LICENSE`.
