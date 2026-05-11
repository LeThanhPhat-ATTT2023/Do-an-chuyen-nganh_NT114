# Báo Cáo Cấu Hình Train HGT Khuyến Nghị

Tài liệu này chỉ giữ hai nhóm cấu hình còn dùng:

- Baseline ban đầu của project: `configs/hgt_t082_k5_l3_d01.yaml`.
- Các cấu hình HGT/IDS mới từ giai đoạn 2024-2026 trong `configs/hgt_paper_variants/`.

Các cấu hình HGT cũ 2020-2023 đã được loại khỏi repo để tránh train lệch mục tiêu hiện tại. Baseline ban đầu vẫn được giữ nguyên để làm mốc so sánh thực nghiệm.

## Kết Luận Nhanh

Cấu hình dùng làm mốc:

```text
configs/hgt_t082_k5_l3_d01.yaml
```

Checkpoint và báo cáo tương ứng:

```text
outputs/hgt_flow_classifier_t082_k5_l3_d01/hgt_flow_best.pt
outputs/hgt_flow_classifier_t082_k5_l3_d01/training_summary.json
```

Theo `docs/hgt_graph_threshold_selection_vi.md`, graph `t082_k5` từng đạt:

| Metric | Giá trị |
|---|---:|
| Best epoch | 143 |
| Val macro-F1 | 0.3511 |
| Test macro-F1 | 0.3639 |
| Test accuracy | 0.3476 |

## Cấu Hình Mới Từ 2024-2026

Các file YAML còn lại trong `configs/hgt_paper_variants/`:

| File | Nguồn cảm hứng | Mục tiêu thử |
|---|---|---|
| `hgt_t082_k5_xgnid_dual_modal_l1_h32_h4.yaml` | XG-NID, 2024 | Backbone rất hẹp để kiểm tra giới hạn dung lượng mô hình |
| `hgt_t082_k5_one2_iov_l1_h64_h2.yaml` | One^2 IoV, 2025 | HGT 1 layer siêu nhẹ cho edge IDS |
| `hgt_t082_k5_relgt_multi_token_l3_h128_h8.yaml` | RelGT, 2025 | 3 layer, nhiều head hơn để mô phỏng relational tokenization |
| `hgt_t082_k5_gatransformer_deep_l6_h256_h8.yaml` | GATransformer, 2025 | Stack sâu để kiểm tra pattern dài hơn, có AMP/checkpointing |
| `hgt_t082_k5_ahgt_dfd_funnel_l3_h128_h4.yaml` | AHGT-DFD, 2026 | 3-hop funnel backbone, bỏ phần continual learning nâng cao |
| `hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml` | DLG-IDS, 2026 | Sparse-friendly backbone đi cùng SIGC + Top-N edge selection |

Tất cả cấu hình mới đều dùng:

```yaml
data:
  source: graph_store
  graph_store_root: data/graph_store_v1
  read_sealed_only: true

train:
  batch_mode: neighbor_sampling
```

Lý do: thiết kế scalable hiện tại không còn khuyến nghị full-batch cho nhóm cấu hình mới. Graph được đọc qua on-disk CSR store, sampler lấy K-hop quanh seed flow, static MITRE/tactic giữ theo global index.

## So Sánh Thực Nghiệm

Mốc so sánh:

1. Chạy baseline ban đầu: `configs/hgt_t082_k5_l3_d01.yaml`.
2. Chạy 6 cấu hình mới 2024-2026.
3. So sánh trên cùng graph `t082_k5`, cùng split nếu dùng `graph_store`.

Các metric cần ghi:

| Nhóm | Metric |
|---|---|
| Chất lượng | best epoch, val macro-F1, test macro-F1, test accuracy |
| Chi phí | thời gian/epoch, tổng thời gian train, peak VRAM |
| Sampler | avg subgraph nodes/edges theo relation |
| Ổn định | OOM, fallback CPU, early stopping |

## Lưu Ý Về Phạm Vi

Các YAML 2024-2026 chỉ chuyển phần backbone và lịch train tương thích với trainer hiện tại. Những thành phần chưa nằm trong code vẫn không được giả vờ là đã có:

- XG-NID: chưa có LLM-fusion head.
- One^2 IoV: chưa có optimizer/scheduler riêng của paper.
- RelGT: chưa có hop/time token trong `HeteroGraphTransformer`.
- GATransformer: rolling time window nằm ở runtime/store, không phải config train.
- AHGT-DFD: chưa có EWC, Dirichlet prior, Lipschitz constraint.
- DLG-IDS: phần đã triển khai trong runtime là SIGC + Top-N edge selection; localized temporal attention sâu hơn chưa bật.

## Lệnh Chạy

Tạo graph store một lần:

```powershell
graphslm-convert-graph-store `
  --graph-npz "data/processed/graph_artifact_3tier_t082_k5.npz" `
  --graph-meta-json "data/processed/graph_artifact_3tier_t082_k5.meta.json" `
  --output-root "data/graph_store_v1"
```

Train một cấu hình mới:

```powershell
graphslm-train-hgt --config "configs/hgt_paper_variants/hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml"
```

Train trên Kaggle dùng:

```text
notebooks/train_hgt_2024_variants_kaggle.py
notebooks/train_hgt_paper_variants_kaggle.ipynb
```
