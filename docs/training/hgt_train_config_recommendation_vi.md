# Báo Cáo Cấu Hình Train HGT Khuyến Nghị

> ⚠️ **CẬP NHẬT (v3_ob, 2026-06):** Khuyến nghị về HGT (layers/heads/hidden, GCL
> augmentation, neighbor sampling) còn dùng được, nhưng **tên edge type đã đổi**:
> v3_ob không còn `matches_technique`; cạnh packet/flow→technique giờ là
> `evidence_<family>` (injection/command_exec/file_upload/recon/c2_beacon). Khi áp
> dụng `edge_perturbation` / flow-flow connectivity, thay `matches_technique` bằng
> các `evidence_<family>`. "teacher" trong doc là continual-learning/distillation
> (HERO/checkpoint trước), **không** phải SecureBERT teacher đã bị bỏ. Nguồn chuẩn:
> `CLAUDE.md`.

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

## v7-final — Unified Training Config (2026-05)

Lần train chính thức của thesis dùng **2 config duy nhất**:

| File | Mục tiêu | Thời gian |
|---|---|---|
| `configs/hgt_smoke_v7_final.yaml` | Smoke 5 epochs validate Phase 1 + 2 | ~30 phút |
| `configs/hgt_t082_k5_l3_d01_server_v7_final.yaml` | Full 100 epochs convergence | ~28-36h trên L40S |

Quyết định gộp các ablation v7-p1 / v7-p2 / v7-p3 → 1 unified run vì:
- Tiết kiệm compute ($18 vs $54 trên AWS L40S @ $0.60/h)
- Thesis ablation đơn giản hơn: 2 configs (v6 baseline vs v7-final) thay vì 5
- Pipeline end-to-end integrity được verify qua smoke

Các config v7-p1/p2/p3 đã bị xóa khỏi repo (commit 2026-05-21).

## v7 Phase 3 — HPE Laplacian PE: DEFERRED cho run này

**Status**: code đã implement đầy đủ ([laplacian_pe.py](../../src/graphslm_ids/offline/training/laplacian_pe.py), [precompute_laplacian_pe.py](../../src/graphslm_ids/offline/training/precompute_laplacian_pe.py), tests 9/9 pass) nhưng **không activate** trong v7-final.

**Lý do defer**: NT114 graph schema có quan hệ packet→flow là 1:1 (mỗi packet thuộc đúng 1 flow). Hệ quả: định nghĩa "flow-flow connectivity qua shared packets" trong [precompute_laplacian_pe.py:33](../../src/graphslm_ids/offline/training/precompute_laplacian_pe.py#L33) tạo ra **zero edges** trên dataset thật (verified 2026-05-21 trên L40S: `[pe] flow-flow edges: shape=(2, 0)`). Laplacian degenerate → eigenvectors meaningless → PE file = noise.

**Cần research thêm cho v8**: định nghĩa flow-flow connectivity phù hợp với IDS schema. Các phương án tiềm năng:
- Shared MITRE technique: `flow → matches_technique → technique ← matches_technique → flow`
- Shared 5-tuple subset (src_ip, dst_ip, proto)
- Temporal window co-occurrence
- 2-hop walks qua packet→next_packet→packet chains

**Cấu hình hiện tại**: `data.laplacian_pe_path: null` trong cả 2 v7-final configs. Code path tại [train_hgt_flow_classifier.py:1561](../../src/graphslm_ids/offline/training/train_hgt_flow_classifier.py#L1561) detect null → skip PE load → per-batch concat hook tại line 1032 trở thành no-op. Model input dim không thay đổi.

**Module vẫn giữ trong repo** cho v8 tương lai — không cần xóa, chỉ disable qua config.

## v7 Phase 2 — HGAA Adaptive Augmentation (2026-05)

HGAA được activate trong v7-final configs (`train.hgaa.enabled: true`):

**HGAA** (Heterogeneous Graph Adaptive Augmentation) là adaptation của Zhao et al. *Symmetry* 2025, 17, 1623 từ binary anomaly detection sang multiclass IDS. 4 operators:

1. **edge_addition** trên `flow__contains__packet`
2. **node_feature_swap** giữa flows cùng class (in-class oversampling)
3. **edge_direction_swap** trên `packet__next_packet__packet`
4. **edge_perturbation** trên `flow__matches_technique__technique`

(Skip type-swap vì sẽ phá ngữ nghĩa flow ↔ packet ↔ technique ↔ tactic.)

**Adaptive selection** (paper §3.3.1): track success rate per (class, op) → sample ops theo phân phối normalized. Laplace smoothing prevents zero-probability ops.

**Bias-aware** (paper §3.3.2, multiclass adaptation): auto-detect 3 rarest classes từ class distribution lúc training startup (KHÔNG hardcode class IDs — dataset-portable per spec §1.5 DC-2). Với prob `bias_factor_T=0.5`, tail-class samples force-select `node_feature_swap`.

**Filter network**: defer cho v7-p3 (cần v7-p1 checkpoint làm teacher). v7-p2 chạy không filter — adaptive op success-rate tự gating chất lượng qua feedback loop.

**Module ở**: [hgaa_augmentation.py](../../src/graphslm_ids/offline/training/hgaa_augmentation.py), [hgaa_filter_network.py](../../src/graphslm_ids/offline/training/hgaa_filter_network.py).

**Acceptance criteria (smoke v7-final)**:
- val_macro_f1 epoch 5 ≥ v6 smoke baseline (0.114).
- `[hgaa] enabled — ...` line log đầu training: confirm `bias_classes` auto-detected (kỳ vọng `{10, 7, 0}` cho NT114 13 classes).
- `[hgaa] epoch=N considered=X augmented=Y aug_rate≈0.5` mỗi epoch log.
- KHÔNG có log `[hpe] Loaded Laplacian PE` (Phase 3 đã defer).
- Không có NaN.

## v7 Phase 1 — Speed-up Configs (2026-05)

Phase 1 speed-ups (torch.compile, batch 512, workers 16, prefetch 16) được activate trong v7-final configs. KHÔNG thay model logic. Mục tiêu: giảm wall time 25-40h → 18-25h trên L40S.

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
