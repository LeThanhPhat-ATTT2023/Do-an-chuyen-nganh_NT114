# Context-Rich Explainable IDS - NT114

Dự án xây dựng hệ thống phát hiện xâm nhập mạng dựa trên payload embedding, đồ thị dị thể và ngữ cảnh MITRE ATT&CK.

Ý tưởng chính là không đưa trực tiếp payload raw vào GNN. Pipeline hiện tại dùng transformer teacher để tạo embedding, distill sang student 1D-CNN nhẹ hơn, sau đó dùng embedding này để tạo graph 3 tầng gồm `flow`, `packet`, `MITRE technique/tactic` và train HGT flow classifier.

## Trạng Thái Hiện Tại

Baseline đang được giữ trong repo là cấu hình HGT với graph:

`data/processed/graph_artifact_3tier_t082_k5.npz`

Thông số graph chính:

| Tham số | Giá trị |
|---|---:|
| similarity threshold | 0.82 |
| packet top-k | 5 |
| flow top-k | 5 |
| num flows | 27,541 |
| num packets | 86,548 |
| num techniques | 691 |
| num tactics | 14 |

Kết quả train tốt nhất hiện tại:

| Metric | Giá trị |
|---|---:|
| best epoch | 143 |
| validation macro-F1 | 0.351063 |
| test macro-F1 | 0.363932 |
| test accuracy | 0.347621 |

Checkpoint chính:

`outputs/hgt_flow_classifier_t082_k5_l3_d01/hgt_flow_best.pt`

## Cấu Trúc Repo

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
|-- outputs/
|   |-- hgt_flow_classifier_t082_k5_l3_d01/
|   `-- student_cnn/
|-- scripts/
|-- src/graphslm_ids/
|-- tests/
|-- requirements.txt
|-- requirements-ml.txt
`-- pyproject.toml
```

`outputs/`, `data/raw/`, `data/interim/`, `data/processed/` và `data/mitre/` là dữ liệu sinh ra trong quá trình chạy nên đã được ignore khỏi git. Repo chỉ giữ source, config, test và tài liệu.

## Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

## Pipeline Chạy Lại Từ Đầu

### 1. Chuẩn bị MITRE ATT&CK

Tải Enterprise ATT&CK STIX:

```powershell
$outDir = "data/mitre"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json" -OutFile "$outDir/enterprise-attack.json"
```

Sinh CSV technique/tactic:

```powershell
python scripts/prepare_mitre_knowledge_base.py --input-json "data/mitre/enterprise-attack.json"
```

Tài liệu chi tiết bằng tiếng Việt:

`docs/mitre_setup_cic_iot2023_vi.md`

### 2. Trích Xuất Payload Dataset

Đặt file `.pcap` hoặc `.pcapng` vào `data/raw/`, sau đó chạy:

```powershell
python scripts/extract_payload_dataset.py --input-glob "data/raw/**/*.pcap" "data/raw/**/*.pcapng" --output-dir "data/interim/payload_dataset" --payload-length 256
```

Output chính:

`data/interim/payload_dataset/payload_256.npy`  
`data/interim/payload_dataset/metadata.csv`  
`data/interim/payload_dataset/stats.json`

### 3. Tạo Teacher Targets

```powershell
python scripts/build_teacher_targets.py --payload-npy "data/interim/payload_dataset/payload_256.npy" --metadata-csv "data/interim/payload_dataset/metadata.csv" --output-path "data/processed/teacher_targets.npy" --model-name "ehsanaghaei/SecureBERT" --batch-size 32
```

Output:

`data/processed/teacher_targets.npy`  
`data/processed/teacher_targets.meta.json`

### 4. Train Student 1D-CNN

```powershell
python scripts/train_student_cnn.py --payload-npy "data/interim/payload_dataset/payload_256.npy" --teacher-npy "data/processed/teacher_targets.npy" --output-dir "outputs/student_cnn" --batch-size 256 --epochs 30
```

Output:

`outputs/student_cnn/student_cnn_best.pt`  
`outputs/student_cnn/training_summary.json`

Đánh giá student:

```powershell
python scripts/evaluate_student_cnn.py --payload-npy "data/interim/payload_dataset/payload_256.npy" --teacher-npy "data/processed/teacher_targets.npy" --metadata-csv "data/interim/payload_dataset/metadata.csv" --checkpoint "outputs/student_cnn/student_cnn_best.pt" --output-path "outputs/student_cnn/evaluation_summary.json" --batch-size 512 --val-ratio 0.1 --seed 42
```

Export ONNX:

```powershell
python scripts/export_student_onnx.py --checkpoint "outputs/student_cnn/student_cnn_best.pt" --output-path "outputs/student_cnn/student_cnn.onnx" --input-length 256 --opset 17 --verify
```

### 5. Export Student Embeddings

```powershell
python scripts/export_student_embeddings.py --payload-npy "data/interim/payload_dataset/payload_256.npy" --checkpoint "outputs/student_cnn/student_cnn_best.pt" --output-path "data/processed/student_embeddings.npy" --batch-size 1024 --device auto
```

Output:

`data/processed/student_embeddings.npy`  
`data/processed/student_embeddings.meta.json`

### 6. Tạo MITRE Technique Embeddings

```powershell
python scripts/build_mitre_technique_embeddings.py --techniques-csv "data/mitre/mitre_techniques.csv" --output-path "data/mitre/mitre_techniques_embeddings.npy" --teacher-meta-json "data/processed/teacher_targets.meta.json" --device auto
```

Output:

`data/mitre/mitre_techniques_embeddings.npy`  
`data/mitre/mitre_techniques_embeddings.meta.json`

### 7. Tạo Graph 3 Tầng Đã Chọn

Baseline hiện tại dùng threshold `0.82` và top-k `5`.

```powershell
python scripts/build_three_tier_graph_artifact.py --metadata-csv "data/interim/payload_dataset/metadata.csv" --payload-npy "data/interim/payload_dataset/payload_256.npy" --student-embedding-npy "data/processed/student_embeddings.npy" --mitre-techniques-csv "data/mitre/mitre_techniques.csv" --mitre-technique-embeddings-npy "data/mitre/mitre_techniques_embeddings.npy" --mitre-technique-tactic-edges-csv "data/mitre/mitre_technique_tactic_edges.csv" --output-npz "data/processed/graph_artifact_3tier_t082_k5.npz" --similarity-threshold 0.82 --packet-top-k 5 --flow-top-k 5
```

Output:

`data/processed/graph_artifact_3tier_t082_k5.npz`  
`data/processed/graph_artifact_3tier_t082_k5.meta.json`

### 8. Train HGT Flow Classifier

Cấu hình baseline:

`configs/hgt_t082_k5_l3_d01.yaml`

Chạy train:

```powershell
python scripts/train_hgt_flow_classifier.py --config "configs/hgt_t082_k5_l3_d01.yaml"
```

Output:

`outputs/hgt_flow_classifier_t082_k5_l3_d01/hgt_flow_best.pt`  
`outputs/hgt_flow_classifier_t082_k5_l3_d01/training_summary.json`

`configs/hgt.example.yaml` cũng đang trỏ về cùng baseline `t082` để tiện chạy nhanh.

## Artifact Đang Giữ

Sau khi dọn repo, các artifact quan trọng còn lại là:

```text
data/processed/
|-- graph_artifact_3tier_t082_k5.npz
|-- graph_artifact_3tier_t082_k5.meta.json
|-- student_embeddings.npy
|-- student_embeddings.meta.json
|-- teacher_targets.npy
`-- teacher_targets.meta.json

outputs/
|-- hgt_flow_classifier_t082_k5_l3_d01/
`-- student_cnn/
```

Các graph thử nghiệm không được chọn, graph default cũ, cache Python và output HGT smoke/default đã được xóa để repo gọn hơn.

## Test

```powershell
pytest
```

## Ghi Chú Triển Khai

1. Teacher transformer chỉ dùng offline để sinh target/embedding.
2. Online detection path nên dùng student CNN hoặc ONNX runtime.
3. HGT hiện là flow classifier trên graph dị thể, chưa phải module giải thích cuối cùng.
4. Slow-path XAI và reasoning layer là giai đoạn tiếp theo.

## Tài Liệu Liên Quan

`docs/feasibility_assessment_vi.md`  
`docs/mitre_setup_cic_iot2023_vi.md`  
`docs/system_execution_flows.md`

## License

Xem file `LICENSE`.
