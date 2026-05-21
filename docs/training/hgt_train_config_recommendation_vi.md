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
  epochs: 200
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

## v7 Phase 1 — Speed-up Configs (2026-05)

Hai config mới được thêm cho lineage v7:

| File | Mục tiêu | Khác v6 |
|---|---|---|
| `configs/hgt_smoke_v7_p1.yaml` | Smoke 5 epochs validate Phase 1 speed-ups | compile on; batch 512; workers 16; prefetch 16 |
| `configs/hgt_t082_k5_l3_d01_server_v7.yaml` | Full 100 epochs với Phase 1 speed-ups | (same as smoke + ema_enabled, drw_start_pct=0.7) |

Phase 1 KHÔNG thay model logic. Mục tiêu: giảm wall time 25-40h → 18-25h trên L40S.

Speed-ups áp dụng:

- Bật `torch.compile(mode="default", dynamic=True)` qua flag `train.compile: true`. Infrastructure đã sẵn ở [_maybe_compile](../../src/graphslm_ids/offline/training/train_hgt_flow_classifier.py#L429-L451) — chỉ flip config flag. Compile mode `default` (không phải `reduce-overhead`) để xử lý variable subgraph shapes từ neighbor sampling.
- Tăng `batch_seed_flows` 256 → 512: L40S 48GB còn dư VRAM, batch lớn → step ít hơn → wall time giảm.
- Tăng `dataloader.num_workers` 8 → 16, `prefetch_factor` 10 → 16: cải thiện H2D overlap.
- Thêm log `[diag] epoch=N | wall=X.Xs | peak_vram_gb=Y.YY` mỗi epoch để track speed gains + verify VRAM ≤ 42GB.

Phase 1 KHÔNG đụng vào [hgt.py](../../src/graphslm_ids/models/hgt.py) — FP32 fallback trong post-aggregation block được giữ lại pending CUDA verification trên L40S. Có characterization test ở `tests/models/test_hgt_amp_numerics.py` làm baseline để future engineer có thể remove FP32 fallback nếu cần thêm ~15-25% speed.

Acceptance criteria (smoke pass trước khi full run):

- val_macro_f1 ở epoch 5 trong khoảng ±0.02 của v6 smoke baseline.
- Wall time per epoch giảm ≥ 25%.
- Peak VRAM ≤ 42GB.
- Không có NaN losses.

Reference: `docs/superpowers/specs/2026-05-21-hgaa-multiclass-hgt-v7-design.md` §3.1.

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

Train một cấu hình mới — chọn lệnh theo môi trường:

```powershell
# 1) Local CPU laptop / Kaggle 1 GPU / local 1 GPU — single-process
graphslm-train-hgt --config "configs/hgt_paper_variants/hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml"

# 2) Kaggle 2x T4 hoặc server 1 node nhiều GPU — DDP (khuyến cáo)
#    LOCAL_RANK do torchrun set; trainer tự bật NCCL/Gloo + DistributedSampler.
torchrun --standalone --nproc_per_node=2 `
  -m graphslm_ids.offline.training.train_hgt_flow_classifier `
  --config "configs/hgt_paper_variants/hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml"

# 2b) Kaggle 2 GPU nếu không thể dùng torchrun trong notebook — legacy multi-GPU
graphslm-train-hgt `
  --config "configs/hgt_paper_variants/hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml" `
  --multi-gpu

# 3) Server multi-node — DDP cross-node
torchrun --nnodes=$WORLD_NODES --nproc_per_node=8 `
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 `
  -m graphslm_ids.offline.training.train_hgt_flow_classifier `
  --config "configs/hgt_paper_variants/hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml"
```

Các cờ CLI bổ sung:

| Cờ | Ý nghĩa | Mặc định |
|---|---|---|
| `--compile` | Bọc HGT bằng `torch.compile(mode="reduce-overhead", dynamic=True)` | tắt |
| `--no-tf32` | Tắt TF32 (chỉ dùng khi cần FP32 strict cho repro) | TF32 bật |
| `--amp` | Mixed precision (chỉ CUDA) | theo config |
| `--multi-gpu` / `--no-multi-gpu` | Legacy fallback (chỉ áp dụng khi không launch qua torchrun) | true |
| `--device {auto,cpu,cuda,cuda:N}` | Device override khi single-process | auto |

Train trên Kaggle dùng:

```text
notebooks/train_hgt_existing_graph_pipeline_kaggle.ipynb
notebooks/train_hgt_official_full_pipeline_kaggle.ipynb
```

Trong đó:

- `train_hgt_official_full_pipeline_kaggle.ipynb`: luồng thực thi chính thức cho retrain sau này, từ PCAP đến graph 3 tầng, convert mmap/CSR rồi train HGT.
- `train_hgt_existing_graph_pipeline_kaggle.ipynb`: luồng train khi đã có sẵn `graph_artifact_3tier_t082_k5.npz` và `.meta.json`, bỏ qua teacher/student/graph rebuild, chỉ convert sang mmap/CSR rồi train HGT.
