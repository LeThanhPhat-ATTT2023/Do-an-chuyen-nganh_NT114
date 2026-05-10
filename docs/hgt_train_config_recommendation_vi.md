# Báo Cáo Trích Xuất Cấu Hình Train HGT Tối Ưu

Nguồn đọc chính: `E:\Tìm kiếm cấu hình huấn luyện HGT.md`

Mục tiêu: chọn cấu hình train HGT phù hợp nhất cho project IDS hiện tại, đồng thời trích các luận điểm từ file báo cáo gốc để giải thích lựa chọn.

## Kết Luận Nhanh

Cấu hình nên dùng ngay là cấu hình đã có trong repo:

```text
configs/hgt_t082_k5_l3_d01.yaml
```

Checkpoint và báo cáo train tương ứng:

```text
outputs/hgt_flow_classifier_t082_k5_l3_d01/hgt_flow_best.pt
outputs/hgt_flow_classifier_t082_k5_l3_d01/training_summary.json
```

Lý do: đây là cấu hình đã được train và có số liệu thực nghiệm tốt nhất trong nhóm thử nghiệm threshold/top-k hiện tại. Theo `docs/hgt_graph_threshold_selection_vi.md`, graph `t082_k5` đạt:

| Metric | Giá trị |
|---|---:|
| Best epoch | 143 |
| Val macro-F1 | 0.3511 |
| Test macro-F1 | 0.3639 |
| Test accuracy | 0.3476 |

## Các Cấu Hình HGT Đọc Được Từ File Gốc

| Nhóm cấu hình | Hidden dim | Layers | Heads | LR/Optimizer | Khi nào dùng |
|---|---:|---:|---:|---|---|
| OAG chuẩn | 256 | 3 | 8 | Adam/AdamW, weight decay rất nhỏ | Đồ thị học thuật cực lớn, cần cân bằng biểu diễn và VRAM |
| OGB-MAG | 512 | 4 | 8 | Adam/AdamW, LayerNorm, RTE | Node classification nhiều lớp trên graph học thuật vừa/lớn |
| GraphStorm | 128 | 2 | 8 | LR cao có warm-up, dropout 0.5, grad clip 0.1 | Triển khai công nghiệp/phân tán, ưu tiên tốc độ và ổn định |
| Locomotion | 128 | 2 | 2 | LR log-uniform 1e-6 đến 1e-2, MSE | Graph vật lý/cảm biến, quan hệ ít loại, tránh dư tham số |
| MolHGT/y sinh | 200-800 | 2-4 | Không cố định | LR 0.0005 hoặc 0.0001, Bayesian tuning | Phân tử/y sinh, đặc trưng phức tạp, cần dò siêu tham số |
| Graph2Seq NLP | 512 | 6 encoder + 6 decoder | 8 | Adam, warm-up + inverse square root decay | Sinh chuỗi từ graph, gần Transformer NLP hơn HGT classifier |

Kết luận từ bảng trên: với project IDS hiện tại, không nên chọn Graph2Seq hoặc MolHGT vì lệch bài toán; cũng không nên áp nguyên OGB-MAG `512/4/8` vì chi phí runtime cao. Cấu hình hợp lý nằm giữa GraphStorm gọn nhẹ và OAG chuẩn: `128-256 hidden`, `3 layers`, `4-8 heads`.

## Cấu Hình Train Được Khuyến Nghị

```yaml
data:
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
  ffn_multiplier: 2

train:
  output_dir: outputs/hgt_flow_classifier_t082_k5_l3_d01
  epochs: 150
  batch_mode: full
  lr: 0.001
  weight_decay: 0.00005
  val_ratio: 0.1
  test_ratio: 0.1
  patience: 30
  class_weight: balanced
  seed: 42
  device: cpu
  monitor: val_macro_f1
  log_every: 1
```

Lệnh train:

```powershell
graphslm-train-hgt --config "configs/hgt_t082_k5_l3_d01.yaml"
```

## Vì Sao Chọn Cấu Hình Này

File báo cáo gốc nêu rõ rằng không có một cấu hình HGT tối ưu duy nhất cho mọi bài toán. Cấu hình phải phụ thuộc vào dạng graph, kích thước graph, loại tác vụ và giới hạn phần cứng.

Các đoạn quan trọng được trích ý:

1. Với mạng tri thức lớn như OAG/OGB-MAG, cấu hình mạnh thường là hidden dimension rộng `256-512`, mạng nông `3-4` lớp, `8` attention heads, kèm LayerNorm và RTE.
2. Với bài toán cảm biến hoặc runtime cần gọn nhẹ, cấu hình nên giảm còn khoảng `2` lớp, ít head hơn, batch nhỏ và learning rate được dò theo log-uniform.
3. Phần tổng kết của file nhấn mạnh chiến lược chung: HGT nên ưu tiên mạng nông nhưng mở rộng chiều ẩn, vì tăng quá nhiều layer dễ gây over-smoothing và over-squashing.
4. File cũng nhấn mạnh `n_heads = 8` là phổ biến trong nghiên cứu, nhưng khi graph vật lý/ứng dụng hẹp hơn thì số head nhỏ hơn có thể tránh dư tham số.

Đối chiếu với project hiện tại:

| Tiêu chí | Đánh giá |
|---|---|
| Graph hiện tại | Graph IDS 3 tầng: flow, packet, technique, tactic |
| Kích thước | 27,541 flow, 86,548 packet, 691 technique, 14 tactic |
| Mục tiêu | Flow classification nhiều lớp |
| Runtime | Fast path cần HGT inference trên subgraph nhỏ |
| Rủi ro nếu tăng layer | Tăng K-hop runtime và dễ over-smoothing |
| Cấu hình hợp lý | `3` layer, hidden `128`, `4` heads, dropout `0.1` |

Vì vậy, cấu hình `t082_k5_l3_d01` là lựa chọn tối ưu hiện tại theo bằng chứng thực nghiệm trong repo.

## Cấu Hình Ứng Viên Nếu Muốn Tối Ưu Tiếp

Nếu có GPU hoặc chấp nhận train chậm hơn, nên thử thêm một cấu hình rộng hơn dựa trên khuyến nghị từ file báo cáo gốc:

```yaml
model:
  hidden_dim: 256
  num_layers: 3
  num_heads: 8
  dropout: 0.1
  ffn_multiplier: 2

train:
  epochs: 200
  lr: 0.0005
  weight_decay: 0.00005
  patience: 40
  monitor: val_macro_f1
```

Lý do thử:

- `hidden_dim = 256` gần với cấu hình OAG chuẩn và tăng khả năng biểu diễn so với baseline `128`.
- `num_heads = 8` khớp với xu hướng HGT/Transformer phổ biến trong file báo cáo gốc.
- Giữ `num_layers = 3` để không làm runtime K-hop phình lên và tránh over-smoothing.
- Giảm `lr` xuống `0.0005` vì mô hình rộng hơn thường nhạy hơn với learning rate.

Tuy nhiên, cấu hình này mới là ứng viên tối ưu tiếp theo, chưa phải cấu hình tốt nhất đã được xác nhận. Cấu hình tốt nhất đã có số liệu hiện tại vẫn là:

```text
configs/hgt_t082_k5_l3_d01.yaml
```

## Thứ Tự Ưu Tiên Khuyến Nghị

1. Dùng cấu hình đã validate: `configs/hgt_t082_k5_l3_d01.yaml`.
2. Nếu muốn cải thiện macro-F1, thử ablation `hidden_dim=256`, `num_heads=8`, `num_layers=3`, `lr=0.0005`.
3. Không tăng `num_layers` lên `4` ngay, vì runtime của project dùng K-hop subgraph theo số layer HGT.
4. Không giảm threshold lên `0.83`, vì graph quá thưa đã làm giảm validation macro-F1 và test accuracy.
5. Không giảm threshold xuống `0.80`, vì graph quá dày đưa thêm cạnh nhiễu vào message passing.
