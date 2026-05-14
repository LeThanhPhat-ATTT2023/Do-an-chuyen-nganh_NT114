# Ghi Chú Lựa Chọn Ngưỡng Graph Cho HGT

## Mục Đích

Tài liệu này tóm tắt kết quả train thử nghiệm HGT với 3 graph có thông số `similarity_threshold` khác nhau. Mục tiêu là giải thích vì sao pipeline hiện tại chọn:

```text
similarity_threshold = 0.82
packet_top_k = 5
flow_top_k = 5
```

## Thiết Lập Thử Nghiệm

Ba lần train được thực hiện với cùng cấu hình model và training. Khác biệt chính nằm ở graph đầu vào:

1. `t080`: threshold = 0.80, top-k = 5
2. `t082`: threshold = 0.82, top-k = 5
3. `t083`: threshold = 0.83, top-k = 5

Trong cả ba lần thử nghiệm, `top_k = 5` được giữ cố định. Nghĩa là mỗi packet hoặc flow chỉ được nối tới tối đa 5 MITRE technique có độ tương đồng cao nhất. Cách làm này giúp hạn chế việc graph phình quá lớn, đồng thời vẫn giữ đủ ứng viên semantic cho quá trình học trên graph.

## Kết Quả Tổng Hợp

| Graph | Threshold | Top-k | Semantic edges | Best epoch | Val macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `t080` | 0.80 | 5 | 314,025 | 150 | 0.3307 | 0.3455 | 0.3309 |
| `t082` | 0.82 | 5 | 107,171 | 143 | **0.3511** | **0.3639** | **0.3476** |
| `t083` | 0.83 | 5 | 55,190 | 136 | 0.3308 | 0.3513 | 0.3284 |

## Phân Tích

### Threshold 0.80

Với threshold `0.80`, graph tạo ra `314,025` semantic edges. Số cạnh này lớn hơn nhiều so với hai cấu hình còn lại. Graph nhiều cạnh có thể giữ được nhiều thông tin liên kết giữa packet/flow và MITRE technique, nhưng đồng thời cũng làm tăng nguy cơ đưa cạnh yếu hoặc cạnh nhiễu vào quá trình message passing.

Kết quả của `t080`:

```text
Val macro-F1  = 0.3307
Test macro-F1 = 0.3455
Test accuracy = 0.3309
```

Điều này cho thấy threshold `0.80` có khả năng quá thoáng, dẫn đến graph nhiều cạnh nhưng chưa tối ưu cho khả năng phân lớp.

### Threshold 0.82

Với threshold `0.82`, graph còn `107,171` semantic edges. Số cạnh giảm mạnh so với `0.80`, nhưng vẫn lớn hơn đáng kể so với `0.83`. Đây là điểm cân bằng tốt giữa việc lọc bớt cạnh nhiễu và giữ lại thông tin semantic có ích.

Kết quả của `t082` là tốt nhất trong ba lần train:

```text
Val macro-F1  = 0.3511
Test macro-F1 = 0.3639
Test accuracy = 0.3476
```

So với `t080`, `t082` tăng cả validation macro-F1, test macro-F1 và test accuracy. Điều này cho thấy việc tăng threshold từ `0.80` lên `0.82` đã giúp loại bớt cạnh nhiễu mà không làm mất quá nhiều thông tin quan trọng.

### Threshold 0.83

Với threshold `0.83`, graph chỉ còn `55,190` semantic edges. Số cạnh giảm gần một nửa so với `0.82`. Graph thưa hơn có thể giảm nhiễu, nhưng nếu lọc quá mạnh thì mô hình sẽ mất các liên kết semantic hữu ích giữa traffic và MITRE technique.

Kết quả của `t083`:

```text
Val macro-F1  = 0.3308
Test macro-F1 = 0.3513
Test accuracy = 0.3284
```

Mặc dù test macro-F1 của `t083` cao hơn `t080`, validation macro-F1 và test accuracy đều thấp hơn `t082`. Điều này cho thấy threshold `0.83` có xu hướng quá chặt, làm graph mất bớt thông tin cần thiết cho khả năng tổng quát hóa.

## Lý Do Chọn Threshold 0.82 Và Top-k 5

`top_k = 5` được giữ cố định vì đây là cơ chế giới hạn số lượng MITRE technique ứng viên cho mỗi packet/flow. Nếu không giới hạn top-k, graph có thể có quá nhiều cạnh semantic, làm tăng nhiễu và chi phí tính toán. Nếu top-k quá nhỏ, graph có thể mất các technique liên quan đứng thứ hai, thứ ba, hoặc các liên kết bổ sung có ý nghĩa.

Trong điều kiện `top_k = 5`, threshold `0.82` là lựa chọn tốt nhất vì:

1. Đạt validation macro-F1 cao nhất: `0.3511`.
2. Đạt test macro-F1 cao nhất: `0.3639`.
3. Đạt test accuracy cao nhất: `0.3476`.
4. Giảm số cạnh semantic từ `314,025` xuống `107,171` so với threshold `0.80`, giúp graph bớt nhiễu.
5. Không lọc quá mạnh như threshold `0.83`, nên vẫn giữ đủ thông tin semantic để HGT học quan hệ giữa flow, packet và MITRE technique.

## Kết Luận

Cấu hình `t082_k5` là baseline hợp lý nhất trong ba thử nghiệm. Threshold `0.82` tạo ra graph có mật độ semantic edge vừa đủ: không quá dày như `0.80`, cũng không quá thưa như `0.83`.

Do đó, pipeline hiện tại chọn:

```text
graph artifact: data/processed/graph_artifact_3tier_t082_k5.npz
similarity_threshold: 0.82
packet_top_k: 5
flow_top_k: 5
hgt config: configs/hgt_t082_k5_l3_d01.yaml
```

Checkpoint HGT tương ứng:

```text
outputs/hgt_flow_classifier_t082_k5_l3_d01/hgt_flow_best.pt
```
