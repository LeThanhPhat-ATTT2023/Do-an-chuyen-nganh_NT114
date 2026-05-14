# Notebooks

Ba notebook này chạy trên **Kaggle** (GPU T4 x2). Mỗi notebook tự động bỏ qua các stage đã có output.

---

## 1. `build_graph_npz_from_zipped_pcaps_kaggle.ipynb`

**Mục đích:** Xây dựng artifact đồ thị three-tier từ các file PCAP thô.

**Pipeline (7 stage, mỗi stage bỏ qua nếu output đã tồn tại):**

```
raw.rar / .pcap
  → Stage 1: payload_256.npy + metadata.csv          (trích xuất payload)
  → Stage 2: teacher_targets.npy                     (SecureBERT embedding)
  → Stage 3: student_cnn_best.pt                     (train student 1D-CNN)
  → Stage 4: student_embeddings.npy                  (export embedding)
  → Stage 5: mitre_techniques.csv + mitre_tactics.csv (build MITRE knowledge base)
  → Stage 6: mitre_techniques_embeddings.npy          (MITRE technique embedding)
  → Stage 7: graph_artifact_3tier_t082_k5.npz         (xây đồ thị three-tier)
```

**Artifact nên upload trước để tiết kiệm thời gian:**

| Kaggle Dataset | File | Tiết kiệm |
|---|---|---|
| `nt114-pcap-dataset` | `raw.rar` hoặc `.pcap` trực tiếp | Bắt buộc (trừ khi đã có payload) |
| `nt114-student-model` | `student_cnn_best.pt` | ~3–6 giờ (bỏ Stage 2–4) |
| `nt114-mitre` | `mitre_techniques.csv`, `mitre_tactics.csv`, `mitre_technique_tactic_edges.csv` | ~30 phút (bỏ Stage 5) |
| `nt114-mitre` | `mitre_techniques_embeddings.npy` | ~30 phút (bỏ Stage 6) |
| `securebert` | folder model SecureBERT | Tránh download ~400 MB |

**Output:** `graph_npz_artifact.zip` chứa `graph_artifact_3tier_t082_k5.npz` và metadata.

---

## 2. `train_hgt_official_full_pipeline_kaggle.ipynb`

**Mục đích:** Pipeline đầy đủ — có thể bắt đầu từ PCAP hoặc từ graph NPZ đã có, sau đó train HGT.

**Hai chế độ (chỉnh `PIPELINE_MODE` trong cell cấu hình):**

| `PIPELINE_MODE` | Mô tả |
|---|---|
| `full_from_pcap` | Chạy toàn bộ từ PCAP → HGT |
| `existing_graph_npz` | Bỏ qua tiền xử lý, chỉ convert graph store + train HGT |

**Hai chế độ HGT (chỉnh `HGT_RUN_MODE`):**

| `HGT_RUN_MODE` | Mô tả |
|---|---|
| `deployment` | Train 1 config production (`hgt_t082_k5_l3_d01`) |
| `paper_variants` | Train song song 7 variant trên 2 GPU |

**Output:** `hgt_training_results_kaggle.zip` chứa checkpoint, `training_summary.json`, và bảng so sánh CSV.

---

## 3. `train_hgt_existing_graph_pipeline_kaggle.ipynb`

**Mục đích:** Train HGT khi artifact đồ thị đã có sẵn — bỏ qua toàn bộ tiền xử lý.

**Yêu cầu upload (3 file):**
1. `graph_artifact_3tier_t082_k5.npz`
2. `graph_artifact_3tier_t082_k5.meta.json`
3. `graph_artifact_3tier_t082_k5_packet_semantic_x.npy.zst` (nén Zstandard, hoặc `.npy` thường)

**Điểm khác biệt so với notebook 2:**
- Hỗ trợ decompres tự động `.npy.zst` / `.npy.gz` / `.npy.zip` vào `/tmp` (không ảnh hưởng quota 19 GB `/kaggle/working`)
- `GITHUB_PULL_IF_CLONED=True`: tự `git pull` code mới nhất khi WORK_DIR đã là git clone
- Nhẹ hơn, ít biến cấu hình hơn so với notebook 2

**Output:** `hgt_existing_graph_results.zip` chứa checkpoint, log, và comparison CSV.

---

## Thứ tự chạy khuyến nghị

```
Lần đầu:   notebook 1  →  notebook 3   (hoặc notebook 2 với existing_graph_npz)
Retrain:   notebook 3 (upload NPZ cũ, train config mới)
Từ đầu:    notebook 2 với full_from_pcap
```
