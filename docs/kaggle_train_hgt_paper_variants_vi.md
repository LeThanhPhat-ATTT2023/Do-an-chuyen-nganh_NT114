# Hướng Dẫn Train HGT 2024-2026 Trên Kaggle

File train chính:

```text
notebooks/train_hgt_2024_variants_kaggle.py
```

Notebook tương đương:

```text
notebooks/train_hgt_paper_variants_kaggle.ipynb
```

Kịch bản Kaggle hiện chỉ train 7 run:

1. Baseline ban đầu: `configs/hgt_t082_k5_l3_d01.yaml`.
2. XG-NID 2024: `hgt_t082_k5_xgnid_dual_modal_l1_h32_h4.yaml`.
3. One^2 IoV 2025: `hgt_t082_k5_one2_iov_l1_h64_h2.yaml`.
4. RelGT 2025: `hgt_t082_k5_relgt_multi_token_l3_h128_h8.yaml`.
5. GATransformer 2025: `hgt_t082_k5_gatransformer_deep_l6_h256_h8.yaml`.
6. AHGT-DFD 2026: `hgt_t082_k5_ahgt_dfd_funnel_l3_h128_h4.yaml`.
7. DLG-IDS 2026: `hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml`.

Nhóm cấu hình HGT 2020-2023 đã bị loại khỏi repo. Baseline chỉ giữ để so sánh với cấu hình ban đầu đã dùng.

## Chuẩn Bị Kaggle Dataset

Dataset Kaggle cần có:

```text
data/processed/graph_artifact_3tier_t082_k5.npz
data/processed/graph_artifact_3tier_t082_k5.meta.json
```

Nếu dùng GitHub cho code, sửa trong file train:

```python
GITHUB_REPO_URL = "https://github.com/username/Do-an-chuyen-nganh_NT114.git"
GITHUB_BRANCH = ""  # hoặc "main"
```

Nếu không dùng GitHub, upload repo thành Kaggle Dataset. Script sẽ tìm repo trong `/kaggle/input` rồi copy sang `/kaggle/working/nt114_hgt_work`.

## Luồng Train

Script tự làm các bước:

1. Tìm/copy repo vào `/kaggle/working/nt114_hgt_work`.
2. Tìm file graph `.npz` và `.meta.json` trong repo hoặc `/kaggle/input`.
3. Chạy `pip install -e .`.
4. Convert NPZ sang `data/graph_store_v1` nếu chưa có `manifest.json`.
5. Train baseline ban đầu.
6. Train 6 cấu hình 2024-2026 bằng `neighbor_sampling`.
7. Ghi bảng so sánh vào `outputs/hgt_kaggle_comparison.csv` và `.md`.
8. Nén kết quả thành `/kaggle/working/hgt_training_results_kaggle.zip`.

## Bật GPU

Trong Kaggle:

```text
Settings -> Accelerator -> GPU T4 x2 hoặc GPU P100/T4
```

Script sẽ dừng sớm nếu CUDA không khả dụng. Nếu một run bị CUDA OOM, script retry run đó trên CPU và ghi log hậu tố `_cpu_fallback.log`.

## Output

Sau khi chạy xong:

```text
/kaggle/working/nt114_hgt_work/outputs/
/kaggle/working/hgt_training_results_kaggle.zip
```

Trong zip có:

```text
outputs/hgt_flow_classifier_t082_k5_l3_d01/
outputs/hgt_paper_variants/
outputs/hgt_kaggle_comparison.csv
outputs/hgt_kaggle_comparison.md
outputs/hgt_kaggle_logs/
configs/hgt_t082_k5_l3_d01.yaml
configs/hgt_paper_variants/*.yaml
```

Mỗi run có:

```text
training_summary.json
hgt_flow_best.pt
log .txt/.log
```
