# Context-Rich Explainable IDS - NT114

Dự án xây dựng hệ thống phát hiện xâm nhập mạng có ngữ cảnh, kết hợp payload embedding, đồ thị dị thể, MITRE ATT&CK và lớp giải thích bằng SLM. Repo được tổ chức theo hai luồng chính:

- **Offline path**: xử lý dữ liệu PCAP, sinh embedding bằng teacher transformer, distill sang student 1D-CNN, xây graph dị thể và huấn luyện HGT flow classifier.
- **Runtime path**: replay/nhận packet, trích payload online, chạy student ONNX, gắn ngữ cảnh MITRE, dựng hot graph, suy luận HGT, áp policy và đẩy alert sang slow path để sinh báo cáo XAI.

Ý tưởng cốt lõi là **không đưa raw payload trực tiếp vào GNN**. Payload được chuẩn hóa thành vector ngữ nghĩa trước, sau đó mới dùng làm đặc trưng packet trong graph. MITRE technique/tactic được đưa vào graph như tầng tri thức bổ sung để mô hình không chỉ dự đoán nhãn tấn công mà còn có đường dẫn ngữ cảnh phục vụ giải thích.

## Mục Lục

- [Trạng thái hiện tại](#trạng-thái-hiện-tại)
- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt](#cài-đặt)
- [Quick start](#quick-start)
- [Pipeline offline từ đầu](#pipeline-offline-từ-đầu)
- [Chạy runtime fast path và slow path](#chạy-runtime-fast-path-và-slow-path)
- [Cấu hình quan trọng](#cấu-hình-quan-trọng)
- [Console scripts](#console-scripts)
- [Kiểm thử](#kiểm-thử)
- [Artifact và dữ liệu](#artifact-và-dữ-liệu)
- [Tài liệu liên quan](#tài-liệu-liên-quan)
- [Giới hạn hiện tại](#giới-hạn-hiện-tại)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Trạng Thái Hiện Tại

Repo hiện có đầy đủ mã nguồn cho:

- Trích xuất payload 256 byte từ PCAP/PCAPNG.
- Sinh teacher targets bằng transformer, mặc định dùng `ehsanaghaei/SecureBERT`.
- Huấn luyện student 1D-CNN để xấp xỉ embedding teacher.
- Export student sang ONNX cho online inference.
- Sinh embedding MITRE technique và cạnh technique-tactic.
- Xây graph dị thể gồm `flow`, `packet`, `technique`, `tactic`.
- Huấn luyện HGT flow classifier với hai chế độ: `full` (full-graph, nhỏ) và `neighbor_sampling` (mini-batch BFS, scale với graph lớn).
- On-disk CSR graph store (`OnDiskHeteroGraphStore`) đọc qua memory-mapped file — không cần load toàn bộ graph vào RAM.
- Runtime replay PCAP qua fast path.
- `PersistentGraphStore`: append-only source of truth cho runtime/training/slow path.
- Hot graph buffer chỉ còn là cache RAM cho fast path; JSONL `ColdStore` chỉ là fallback khi tắt `graph_store`.
- SIGC edge filter và DLG-IDS Top-N subgraph selection theo thiết kế Lean Optimal.
- Slow path tạo evidence bundle, gọi SLM qua Ollama, validate grounding và fallback report.
- Unit test cho graph builders, HGT, neighbor sampling, on-disk store, persistent graph store, payload extractor, fast-slow bridge và slow path.

Baseline artifact đang dùng trong workspace local:

```text
data/processed/graph_artifact_3tier_t082_k5.npz
data/processed/graph_artifact_3tier_t082_k5.meta.json
outputs/hgt_flow_classifier_t082_k5_l3_d01/hgt_flow_best.pt
outputs/student_cnn/student_cnn_best.pt
outputs/student_cnn/student_cnn.onnx
```

Thông số graph baseline:

| Thành phần | Giá trị |
|---|---:|
| Payload length | 256 byte |
| Embedding dim | 768 |
| Similarity threshold | 0.82 |
| Packet top-k MITRE | 5 |
| Flow top-k MITRE | 5 |
| Số PCAP đã xử lý | 12 |
| Số packet | 86,548 |
| Số flow | 27,541 |
| Số MITRE technique | 691 |
| Số MITRE tactic | 14 |

Kết quả student 1D-CNN:

| Metric | Giá trị |
|---|---:|
| Samples total | 86,548 |
| Embedding dim | 768 |
| Best epoch | 30 |
| Best validation loss | 0.0020009385 |
| Mean cosine similarity toàn tập | 0.99355167 |
| Mean MSE toàn tập | 0.0000371204 |

Kết quả HGT flow classifier baseline:

| Metric | Giá trị |
|---|---:|
| Best epoch | 143 |
| Validation macro-F1 | 0.351063 |
| Test macro-F1 | 0.363932 |
| Test accuracy | 0.347621 |
| Train/val/test split | 22,035 / 2,753 / 2,753 |

Các nhãn tấn công hiện có:

```text
Backdoor
BrowserHijacking
CommandInjection
DDoS
Recon
SqlInjection
Uploading
VulnerabilityScan
XSS
```

## Kiến Trúc Tổng Quan

```text
Offline training
--------------
Raw PCAP
  -> PayloadExtractor
  -> payload_256.npy + metadata.csv
  -> SecureBERT teacher embeddings
  -> Student 1D-CNN distillation
  -> student embeddings
  -> MITRE technique embeddings
  -> heterogeneous graph
  -> HGT flow classifier
  -> runtime artifacts

Runtime replay / online path
----------------------------
Packet stream
  -> FlowTracker
  -> online PayloadExtractor
  -> StudentRuntime ONNX
  -> MitreIndex top-k
  -> SIGC edge filter
  -> PersistentGraphStore (write-through source of truth)
  -> HotGraphBuffer cache
  -> SubgraphBuilder + DLG-IDS Top-N
  -> HGTRuntime
  -> PolicyEngine
  -> AlertDispatcher
  -> SlowPathWorker
  -> XAI report + graph store
```

Các module chính:

| Package | Vai trò |
|---|---|
| `graphslm_ids.offline_path.preprocessing` | Chuẩn bị dataset, MITRE KB, teacher target, graph artifact |
| `graphslm_ids.offline_path.training` | Train/evaluate/export student và train HGT |
| `graphslm_ids.models` | Mô hình `StudentCNN` và `HeteroGraphTransformer` |
| `graphslm_ids.fast_path` | Data plane online: flow tracking, ONNX inference, MITRE index, SIGC, hot graph, Top-N subgraph, policy |
| `graphslm_ids.runtime` | Control plane: config loader, pipeline orchestrator, persistent graph store, cold-store fallback, counterfactual |
| `graphslm_ids.slow_path` | Evidence builder, ranker, SLM report generator, validator, fallback report |

## Cấu Trúc Thư Mục

```text
.
|-- configs/
|   |-- hgt.example.yaml
|   |-- hgt_t082_k5_l3_d01.yaml
|   |-- pipeline.example.yaml
|   |-- cic_iot2023_to_mitre_seed.csv
|   `-- mitre_techniques_template.csv
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- mitre/
|-- docs/
|   |-- fast_slow_bridge_design_vi.md
|   |-- feasibility_assessment_vi.md
|   |-- hgt_graph_threshold_selection_vi.md
|   |-- mitre_setup_cic_iot2023_vi.md
|   |-- slm_slow_path_xai_design_vi.md
|   |-- streaming_hgt_runtime_strategy_vi.md
|   `-- system_execution_flows.md
|-- outputs/
|   |-- hgt_flow_classifier_t082_k5_l3_d01/
|   `-- student_cnn/
|-- src/graphslm_ids/
|   |-- fast_path/
|   |-- models/
|   |-- offline_path/
|   |-- runtime/
|   |-- slow_path/
|   `-- utils/
|-- tests/
|-- requirements.txt
|-- requirements-ml.txt
|-- pyproject.toml
`-- README.md
```

Lưu ý: `data/` và `outputs/` chứa dữ liệu lớn/artifact sinh ra khi chạy pipeline nên được ignore khỏi git, trừ các file `.gitkeep`.

## Yêu Cầu Môi Trường

- Python `>=3.10`.
- Windows PowerShell được dùng trong các ví dụ lệnh dưới đây.
- `pip` mới đủ để cài editable package.
- Dung lượng trống cho `.npy`, `.npz`, checkpoint `.pt`, model `.onnx`.
- Nếu chạy teacher/embedding/HGT training: cần PyTorch và các thư viện ML trong `requirements-ml.txt`.
- Nếu chạy slow path bằng SLM local: cần Ollama hoặc backend tương thích với config trong `configs/pipeline.example.yaml`.

Dependencies nền trong `requirements.txt`:

```text
numpy
pandas
scapy
tqdm
pyyaml
```

Dependencies ML trong `requirements-ml.txt`:

```text
torch
transformers
sentence-transformers
faiss-cpu
onnxruntime
```

## Cài Đặt

Tạo virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài bản đầy đủ để chạy cả ML pipeline và runtime:

```powershell
pip install -r requirements-ml.txt
pip install -e .
```

Nếu chỉ cần chạy phần code nền không dùng teacher/HGT/ONNX:

```powershell
pip install -r requirements.txt
pip install -e .
```

Kiểm tra console scripts đã được nhận:

```powershell
graphslm-extract-payload --help
graphslm-train-hgt --help
graphslm-run-runtime --help
```

## Quick Start

Nếu workspace đã có sẵn artifact trong `data/processed/`, `data/mitre/` và `outputs/`, có thể chạy nhanh một replay không bật slow worker:

```powershell
graphslm-run-runtime `
  --config "configs/pipeline.example.yaml" `
  --input "data/raw/Recon-PortScan.pcap" `
  --max-packets 100 `
  --no-worker
```

Kết quả sẽ in số packet đã xử lý, số alert và đường dẫn graph store. Dòng `[ALERT]` xuất hiện khi label dự đoán không nằm trong `policy.benign_labels` (hoặc nằm trong `policy.alert_labels` nếu bạn cấu hình whitelist) và confidence vượt `policy.alert_threshold`:

```text
[ALERT] ...
[OK] Processed packets=<n> alerts=<m>
[OK] Graph store: data/graph_store_v1
```

Nếu fresh clone chưa có artifact, hãy chạy pipeline offline ở phần tiếp theo trước.

## Pipeline Offline Từ Đầu

Các bước dưới đây tái tạo dataset, embedding, graph và checkpoint từ PCAP.

### 1. Chuẩn bị MITRE ATT&CK

Tạo thư mục MITRE:

```powershell
New-Item -ItemType Directory -Path "data/mitre" -Force | Out-Null
```

Tải Enterprise ATT&CK STIX JSON:

```powershell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json" `
  -OutFile "data/mitre/enterprise-attack.json"
```

Sinh CSV technique, tactic và cạnh technique-tactic:

```powershell
graphslm-prepare-mitre `
  --input-json "data/mitre/enterprise-attack.json"
```

Output chính:

```text
data/mitre/mitre_techniques.csv
data/mitre/mitre_tactics.csv
data/mitre/mitre_technique_tactic_edges.csv
data/mitre/mitre_export_stats.json
```

Tài liệu chi tiết: `docs/mitre_setup_cic_iot2023_vi.md`.

### 2. Chuẩn bị PCAP

Đặt các file `.pcap` hoặc `.pcapng` vào:

```text
data/raw/
```

Tên file có thể được dùng để suy ra nhãn nếu metadata không có nhãn riêng. Dataset local hiện dùng các lớp như `Backdoor`, `Recon`, `SqlInjection`, `XSS`, ...

### 3. Trích xuất payload dataset

```powershell
graphslm-extract-payload `
  --input-glob "data/raw/**/*.pcap" "data/raw/**/*.pcapng" `
  --output-dir "data/interim/payload_dataset" `
  --payload-length 256
```

Output chính:

```text
data/interim/payload_dataset/payload_256.npy
data/interim/payload_dataset/metadata.csv
data/interim/payload_dataset/stats.json
```

Tùy chọn hữu ích:

```powershell
graphslm-extract-payload `
  --input-glob "data/raw/**/*.pcap" `
  --output-dir "data/interim/payload_dataset_smoke" `
  --payload-length 256 `
  --max-packets-per-file 1000 `
  --shuffle `
  --seed 42
```

### 4. Sinh teacher targets

```powershell
graphslm-build-teacher `
  --payload-npy "data/interim/payload_dataset/payload_256.npy" `
  --metadata-csv "data/interim/payload_dataset/metadata.csv" `
  --output-path "data/processed/teacher_targets.npy" `
  --model-name "ehsanaghaei/SecureBERT" `
  --batch-size 32 `
  --device auto
```

Output:

```text
data/processed/teacher_targets.npy
data/processed/teacher_targets.meta.json
```

Teacher transformer chỉ dùng offline để tạo target/embedding. Runtime không gọi transformer này.

### 5. Train student 1D-CNN

```powershell
graphslm-train-student `
  --payload-npy "data/interim/payload_dataset/payload_256.npy" `
  --teacher-npy "data/processed/teacher_targets.npy" `
  --output-dir "outputs/student_cnn" `
  --batch-size 256 `
  --epochs 30 `
  --device auto
```

Output:

```text
outputs/student_cnn/student_cnn_best.pt
outputs/student_cnn/training_summary.json
```

Đánh giá student:

```powershell
graphslm-eval-student `
  --payload-npy "data/interim/payload_dataset/payload_256.npy" `
  --teacher-npy "data/processed/teacher_targets.npy" `
  --metadata-csv "data/interim/payload_dataset/metadata.csv" `
  --checkpoint "outputs/student_cnn/student_cnn_best.pt" `
  --output-path "outputs/student_cnn/evaluation_summary.json" `
  --batch-size 512 `
  --val-ratio 0.1 `
  --seed 42 `
  --device auto
```

### 6. Export student ONNX

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

Output:

```text
outputs/student_cnn/student_cnn.onnx
outputs/student_cnn/student_cnn.meta.json
```

### 7. Export student embeddings

```powershell
graphslm-export-student-emb `
  --payload-npy "data/interim/payload_dataset/payload_256.npy" `
  --checkpoint "outputs/student_cnn/student_cnn_best.pt" `
  --output-path "data/processed/student_embeddings.npy" `
  --batch-size 1024 `
  --device auto
```

Output:

```text
data/processed/student_embeddings.npy
data/processed/student_embeddings.meta.json
```

### 8. Tạo MITRE technique embeddings

```powershell
graphslm-build-mitre-emb `
  --techniques-csv "data/mitre/mitre_techniques.csv" `
  --output-path "data/mitre/mitre_techniques_embeddings.npy" `
  --teacher-meta-json "data/processed/teacher_targets.meta.json" `
  --batch-size 64 `
  --device auto
```

Output:

```text
data/mitre/mitre_techniques_embeddings.npy
data/mitre/mitre_techniques_embeddings.meta.json
```

### 9. Xây graph dị thể 3-tier baseline

Baseline hiện dùng threshold `0.82`, packet top-k `5`, flow top-k `5`.

```powershell
graphslm-build-three-tier-graph `
  --metadata-csv "data/interim/payload_dataset/metadata.csv" `
  --payload-npy "data/interim/payload_dataset/payload_256.npy" `
  --student-embedding-npy "data/processed/student_embeddings.npy" `
  --mitre-techniques-csv "data/mitre/mitre_techniques.csv" `
  --mitre-technique-embeddings-npy "data/mitre/mitre_techniques_embeddings.npy" `
  --mitre-technique-tactic-edges-csv "data/mitre/mitre_technique_tactic_edges.csv" `
  --output-npz "data/processed/graph_artifact_3tier_t082_k5.npz" `
  --similarity-threshold 0.82 `
  --packet-top-k 5 `
  --flow-top-k 5
```

Output:

```text
data/processed/graph_artifact_3tier_t082_k5.npz
data/processed/graph_artifact_3tier_t082_k5.meta.json
```

Graph này có các loại cạnh chính:

- Flow chứa packet.
- Packet liên kết với MITRE technique qua cosine similarity.
- Flow liên kết với MITRE technique qua tổng hợp top-k.
- Technique thuộc tactic.
- Reverse edges được thêm trong bước train/runtime nếu cấu hình bật `add_reverse_edges`.

### 10. Train HGT flow classifier

Với graph lớn (hàng chục nghìn flow, hàng trăm nghìn packet), cần chuyển NPZ sang on-disk CSR store trước để tránh OOM khi load toàn bộ graph vào RAM:

```powershell
graphslm-convert-graph-store `
  --graph-npz "data/processed/graph_artifact_3tier_t082_k5.npz" `
  --graph-meta-json "data/processed/graph_artifact_3tier_t082_k5.meta.json" `
  --output-root "data/graph_store_v1"
```

Output:

```text
data/graph_store_v1/manifest.json
data/graph_store_v1/nodes/flow/features.f32
data/graph_store_v1/nodes/flow/labels.i64
data/graph_store_v1/edges/<edge_name>/indptr.i64
data/graph_store_v1/edges/<edge_name>/indices.i64
data/graph_store_v1/edges/<edge_name>/attr.f32
data/graph_store_v1/splits/train_flow_ids.i64
```

Bước này chỉ cần làm một lần. Sau đó train với `source: graph_store` và `batch_mode: neighbor_sampling` (mặc định trong `hgt.example.yaml`), toàn bộ graph sẽ được đọc qua memory-mapped file — không load đầy vào RAM:

Cấu hình baseline:

```text
configs/hgt_t082_k5_l3_d01.yaml
```

Chạy train:

```powershell
graphslm-train-hgt --config "configs/hgt_t082_k5_l3_d01.yaml"
```

Output:

```text
outputs/hgt_flow_classifier_t082_k5_l3_d01/hgt_flow_best.pt
outputs/hgt_flow_classifier_t082_k5_l3_d01/training_summary.json
```

Có thể override một số tham số từ CLI:

```powershell
graphslm-train-hgt `
  --config "configs/hgt.example.yaml" `
  --epochs 50 `
  --device cpu `
  --output-dir "outputs/hgt_debug"
```

## Chạy Runtime Fast Path Và Slow Path

Runtime dùng config tổng hợp:

```text
configs/pipeline.example.yaml
```

Các artifact cần có trước khi chạy:

```text
outputs/student_cnn/student_cnn.onnx
data/mitre/mitre_techniques.csv
data/mitre/mitre_technique_tactic_edges.csv
data/mitre/mitre_techniques_embeddings.npy
data/processed/graph_artifact_3tier_t082_k5.meta.json
outputs/hgt_flow_classifier_t082_k5_l3_d01/hgt_flow_best.pt
```

Chạy replay không bật slow worker:

```powershell
graphslm-run-runtime `
  --config "configs/pipeline.example.yaml" `
  --input "data/raw/Recon-PortScan.pcap" `
  --max-packets 500 `
  --no-worker
```

Chạy replay có slow worker:

```powershell
graphslm-run-runtime `
  --config "configs/pipeline.example.yaml" `
  --input "data/raw/Recon-PortScan.pcap" `
  --max-packets 500
```

Khi bật slow worker, pipeline sẽ:

1. Chạy fast path cho từng packet.
2. Đưa packet/flow/technique vào `HotGraphBuffer`.
3. Dựng subgraph quanh flow hiện tại.
4. Chạy HGT để lấy logits và attention.
5. Dùng `PolicyEngine` để quyết định alert.
6. Nếu alert đạt ngưỡng, `AlertDispatcher` tạo `SlowPathJob`.
7. `SlowPathWorker` hydrate context, build evidence, gọi SLM và validate report.
8. Snapshot/report được ghi vào `data/runtime/events.jsonl`.

### Cấu hình Ollama cho slow path

Mặc định `configs/pipeline.example.yaml` dùng:

```yaml
slm:
  backend: ollama
  model: qwen2.5:3b-instruct-q4_k_m
  endpoint: http://localhost:11434
```

Khởi động Ollama và pull model tương ứng, hoặc sửa `slm.model` sang model bạn đang có:

```powershell
ollama serve
ollama pull qwen2.5:3b-instruct-q4_k_m
```

Nếu không muốn gọi SLM khi smoke test, dùng `--no-worker`.

## Cấu Hình Quan Trọng

### `configs/hgt_t082_k5_l3_d01.yaml` / `configs/hgt.example.yaml`

Điều khiển bước train HGT. Các key quan trọng:

```yaml
data:
  # "graph_store" = mmap CSR (khuyến nghị, không load toàn bộ graph vào RAM)
  # "npz"         = load NPZ đầy vào RAM (chỉ dùng khi graph nhỏ)
  source: graph_store
  graph_store_root: data/graph_store_v1
  graph_npz: data/processed/graph_artifact_3tier_t082_k5.npz
  graph_meta_json: data/processed/graph_artifact_3tier_t082_k5.meta.json
  packet_feature: semantic
  add_reverse_edges: true
  standardize_flow_features: true
  use_semantic_edge_weights: true

model:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1

train:
  # "neighbor_sampling" = mini-batch BFS, bắt buộc khi graph lớn
  # "full"              = full-graph (OOM với graph lớn)
  batch_mode: neighbor_sampling
  batch_seed_flows: 256
  grad_accum_steps: 4
  epochs: 150
  lr: 0.001
  weight_decay: 0.00005
  class_weight: balanced
  monitor: val_macro_f1

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

### `configs/pipeline.example.yaml`

Điều khiển runtime end-to-end:

| Section | Vai trò |
|---|---|
| `data` | Input glob, output dir, payload length |
| `teacher` | Model teacher offline |
| `mitre` | Đường dẫn MITRE CSV/embedding và ngưỡng similarity |
| `fast_path` | Student ONNX, MITRE top-k, payload length |
| `hot_graph` | TTL, giới hạn packet/flow/event trong RAM |
| `preprocessor` | SIGC Local Contribution Score, lọc cạnh yếu trước khi ghi store |
| `graph_store` | Persistent Graph Store, shard seal, retention, disk quota |
| `subgraph_builder` | DLG-IDS Top-N edge selection cho K-hop runtime subgraph |
| `hgt` | Tham số train HGT |
| `hgt_runtime` | Checkpoint và meta dùng khi inference |
| `policy` | Alert threshold và các label được xem là alert |
| `slow_path` | Queue, số evidence top-k, counterfactual |
| `cold_store` | Fallback JSONL khi `graph_store.enabled: false` |
| `slm` | Backend/model sinh báo cáo |
| `validator` | Ngưỡng kiểm tra grounding/hallucination |

Lưu ý quan trọng: mặc định nên dùng `policy.benign_labels`. Bất kỳ nhãn nào không nằm trong danh sách benign và vượt threshold sẽ bắn alert. Chỉ dùng `policy.alert_labels` khi muốn whitelist chính xác các nhãn cần alert. Với baseline hiện tại, các nhãn tấn công là `Backdoor`, `BrowserHijacking`, `CommandInjection`, `DDoS`, `Recon`, `SqlInjection`, `Uploading`, `VulnerabilityScan`, `XSS`.

## Console Scripts

Các lệnh được khai báo trong `pyproject.toml`:

| Lệnh | Module | Mục đích |
|---|---|---|
| `graphslm-extract-payload` | `offline_path.preprocessing.extract_payload_dataset` | Trích payload từ PCAP |
| `graphslm-prepare-mitre` | `offline_path.preprocessing.prepare_mitre_knowledge_base` | Tạo MITRE CSV từ STIX JSON |
| `graphslm-build-teacher` | `offline_path.preprocessing.build_teacher_targets` | Sinh teacher targets |
| `graphslm-build-mitre-emb` | `offline_path.preprocessing.build_mitre_technique_embeddings` | Sinh MITRE technique embeddings |
| `graphslm-build-graph` | `offline_path.preprocessing.build_graph_artifact` | Xây graph flow-packet cũ |
| `graphslm-build-three-tier-graph` | `offline_path.preprocessing.build_three_tier_graph_artifact` | Xây graph flow-packet-technique-tactic |
| `graphslm-train-student` | `offline_path.training.train_student_cnn` | Train student 1D-CNN |
| `graphslm-eval-student` | `offline_path.training.evaluate_student_cnn` | Đánh giá student |
| `graphslm-export-student-onnx` | `offline_path.training.export_student_onnx` | Export student sang ONNX |
| `graphslm-export-student-emb` | `offline_path.training.export_student_embeddings` | Export student embeddings |
| `graphslm-train-hgt` | `offline_path.training.train_hgt_flow_classifier` | Train HGT classifier |
| `graphslm-convert-graph-store` | `offline_path.training.on_disk_graph_store` | Chuyển NPZ graph sang on-disk CSR store (cần chạy một lần trước khi train với `source: graph_store`) |
| `graphslm-run-runtime` | `runtime.run_runtime_pipeline` | Replay PCAP qua runtime pipeline |

## Kiểm Thử

Chạy toàn bộ test:

```powershell
pytest
```

Chạy nhóm test chính:

```powershell
pytest tests/test_payload_extractor.py
pytest tests/test_graph_artifact_builder.py tests/test_three_tier_graph_artifact.py
pytest tests/test_hgt_model.py
pytest tests/test_hgt_neighbor_sampling.py
pytest tests/test_on_disk_graph_store_training.py
pytest tests/test_persistent_graph_store.py
pytest tests/test_fast_slow_bridge.py tests/test_slow_path.py
```

Nếu test không import được package, hãy đảm bảo đã chạy:

```powershell
pip install -e .
```

## Artifact Và Dữ Liệu

Các thư mục dữ liệu được dùng theo quy ước:

| Thư mục | Nội dung |
|---|---|
| `data/raw/` | PCAP/PCAPNG gốc |
| `data/interim/` | Dataset trung gian, ví dụ `payload_256.npy`, `metadata.csv` |
| `data/processed/` | Teacher targets, student embeddings, graph artifact |
| `data/mitre/` | MITRE STIX JSON, CSV và technique embeddings |
| `data/runtime/` | Fallback cold store JSONL khi tắt `graph_store` |
| `data/graph_store_v1/` | Persistent Graph Store runtime/training |
| `outputs/student_cnn/` | Checkpoint/evaluation/ONNX của student |
| `outputs/hgt_flow_classifier_t082_k5_l3_d01/` | Checkpoint và training summary HGT baseline |

Những artifact này có thể rất lớn và đang bị ignore bởi `.gitignore`:

```text
outputs/
*.pt
*.onnx
*.npz
*.npy
/data/raw/*
/data/interim/*
/data/processed/*
/data/mitre/*
```

Vì vậy khi clone repo mới, cần tự tạo lại artifact bằng pipeline offline hoặc copy artifact từ nơi lưu trữ riêng của nhóm.

## Tài Liệu Liên Quan

| Tài liệu | Nội dung |
|---|---|
| `docs/feasibility_assessment_vi.md` | Đánh giá khả thi của hướng tiếp cận |
| `docs/mitre_setup_cic_iot2023_vi.md` | Chuẩn bị MITRE ATT&CK cho CIC IoT 2023 |
| `docs/hgt_graph_threshold_selection_vi.md` | Chọn threshold/top-k cho graph HGT |
| `docs/slm_slow_path_xai_design_vi.md` | Thiết kế slow path XAI bằng SLM |
| `docs/streaming_hgt_runtime_strategy_vi.md` | Chiến lược hot graph/runtime streaming |
| `docs/fast_slow_bridge_design_vi.md` | Thiết kế lớp nối fast path và slow path |
| `docs/system_execution_flows.md` | Mermaid diagram cho offline/runtime flow |

## Giới Hạn Hiện Tại

- HGT baseline hiện mới đạt test macro-F1 khoảng `0.364`, phù hợp mức prototype/nghiên cứu, chưa phải IDS production.
- Mapping packet/flow sang MITRE technique dựa trên cosine similarity giữa embedding, nên chỉ nên diễn giải là tương đồng ngữ nghĩa, không phải bằng chứng chắc chắn về TTP.
- Runtime hiện hỗ trợ replay PCAP qua `graphslm-run-runtime`; live capture/inline blocking chưa phải mục tiêu chính trong repo này.
- Slow path phụ thuộc chất lượng evidence bundle và SLM local. Validator giúp giảm hallucination nhưng không thay thế kiểm chứng của analyst.
- Các file dữ liệu lớn không được commit. README mô tả đường dẫn artifact theo workspace hiện tại và pipeline tái tạo lại chúng.

## Troubleshooting

### `ModuleNotFoundError: graphslm_ids`

Cài package ở chế độ editable:

```powershell
pip install -e .
```

### Thiếu `torch`, `transformers` hoặc `onnxruntime`

Cài full ML dependencies:

```powershell
pip install -r requirements-ml.txt
```

### Runtime báo thiếu `.onnx`, `.pt`, `.npy`, `.npz`

Chạy lại các bước offline tương ứng:

- Thiếu `student_cnn.onnx`: chạy `graphslm-export-student-onnx`.
- Thiếu `mitre_techniques_embeddings.npy`: chạy `graphslm-build-mitre-emb`.
- Thiếu `graph_artifact_3tier_t082_k5.npz`: chạy `graphslm-build-three-tier-graph`.
- Thiếu `hgt_flow_best.pt`: chạy `graphslm-train-hgt`.

### Ollama không phản hồi

Kiểm tra Ollama server và model:

```powershell
ollama list
ollama serve
```

Hoặc chạy runtime với `--no-worker` để bỏ qua slow path.

### Runtime chạy nhưng không có alert

Kiểm tra `policy.benign_labels` và `policy.alert_labels` trong `configs/pipeline.example.yaml`. Mặc định chỉ cần cấu hình benign:

```yaml
policy:
  alert_threshold: 0.70
  benign_labels:
    - Benign
```

Nếu bật `alert_labels`, danh sách đó trở thành whitelist và phải khớp chính xác label trong checkpoint HGT.

### OOM khi train HGT (graph quá lớn để load vào RAM)

Dùng on-disk CSR store thay vì load NPZ trực tiếp. Chạy một lần:

```powershell
graphslm-convert-graph-store `
  --graph-npz "data/processed/graph_artifact_3tier_t082_k5.npz" `
  --graph-meta-json "data/processed/graph_artifact_3tier_t082_k5.meta.json" `
  --output-root "data/graph_store_v1"
```

Sau đó trong config HGT đặt:

```yaml
data:
  source: graph_store
  graph_store_root: data/graph_store_v1
train:
  batch_mode: neighbor_sampling
```

Train sẽ đọc từng chunk qua mmap — không load toàn bộ graph.

### PowerShell không cho activate virtualenv

Nếu gặp lỗi execution policy, mở PowerShell bằng quyền phù hợp và chạy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Sau đó activate lại:

```powershell
.\.venv\Scripts\Activate.ps1
```

## License

Xem file `LICENSE`.
