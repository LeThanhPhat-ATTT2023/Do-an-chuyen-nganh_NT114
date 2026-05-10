# Hướng Dẫn Train HGT Trên Kaggle GPU Và Tải Một Gói Kết Quả

Notebook Kaggle mẫu:

```text
notebooks/train_hgt_paper_variants_kaggle.ipynb
```

Notebook này train tổng cộng 7 cấu hình:

1. Baseline project: `configs/hgt_t082_k5_l3_d01.yaml`
2. WWW 2020 / OAG-style
3. pyHGT OGB-MAG-style
4. GraphStorm-style
5. Stanford CS224W Locomotion-style
6. HOPE HGT-backbone-style
7. GPT-GNN KDD 2020 backbone-style

Sau khi chạy xong, notebook tạo một file zip duy nhất:

```text
/kaggle/working/hgt_training_results_kaggle.zip
```

Bạn tải file này ở phần Output của Kaggle.

## Chuẩn Bị Kaggle Dataset

Bạn có 2 cách đưa **code** lên Kaggle:

### Cách A: Dùng GitHub Cho Code

Được, đây là cách gọn nhất nếu repo của bạn đã push lên GitHub.

Trong notebook Kaggle, ở cell đầu tiên sửa:

```python
GITHUB_REPO_URL = 'https://github.com/username/Do-an-chuyen-nganh_NT114.git'
GITHUB_BRANCH = ''  # hoặc 'main'
```

Sau đó notebook sẽ `git clone` code vào:

```text
/kaggle/working/nt114_hgt_work
```

Lưu ý: nếu repo private, bạn cần bật Internet trong Kaggle và dùng token GitHub, hoặc đơn giản hơn là upload code bằng Dataset.

### Cách B: Upload Code Thành Kaggle Dataset

Bạn nén/upload repo thành Kaggle Dataset rồi add Dataset đó vào notebook. Notebook sẽ tự tìm repo trong `/kaggle/input` và copy sang `/kaggle/working`.

## Dữ Liệu Graph Vẫn Nên Dùng Kaggle Dataset

Vì `.gitignore` đang bỏ qua `*.npz` và `data/processed/*`, nếu bạn đưa repo lên Kaggle qua GitHub thì file graph sẽ bị thiếu. Do đó file dữ liệu nên được upload vào Kaggle Dataset, tối thiểu gồm:

```text
data/processed/graph_artifact_3tier_t082_k5.npz
data/processed/graph_artifact_3tier_t082_k5.meta.json
```

File quan trọng nhất là:

```text
data/processed/graph_artifact_3tier_t082_k5.npz
```

Notebook sẽ tự tìm file này trong `/kaggle/input` và copy về đúng chỗ nếu nó chưa có trong repo.

## Bật GPU Trên Kaggle

Trong Kaggle Notebook:

```text
Settings -> Accelerator -> GPU T4 x2 hoặc GPU P100/T4
```

Notebook có cell kiểm tra:

```python
assert torch.cuda.is_available()
```

Nếu chưa bật GPU, notebook sẽ dừng sớm.

## Output Sau Khi Train

Notebook ghi:

```text
/kaggle/working/nt114_hgt_work/outputs/
/kaggle/working/hgt_training_results_kaggle.zip
```

Trong file zip có:

```text
outputs/hgt_flow_classifier_t082_k5_l3_d01/
outputs/hgt_paper_variants/
outputs/hgt_kaggle_comparison.csv
outputs/hgt_kaggle_comparison.md
configs/hgt_t082_k5_l3_d01.yaml
configs/hgt_paper_variants/*.yaml
```

Mỗi run có:

```text
training_summary.json
hgt_flow_best.pt
log .txt/.log
```

## Nếu Kaggle Bị Ngắt Giữa Chừng

Notebook có logic skip run đã hoàn thành nếu `training_summary.json` tồn tại. Vì vậy khi chạy lại, các model train xong sẽ không bị train lại.
