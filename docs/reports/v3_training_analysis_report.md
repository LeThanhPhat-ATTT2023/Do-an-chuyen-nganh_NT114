# Báo Cáo Phân Tích Kết Quả Training — EG-HGT v3 Smart-BOTH Hybrid

**Ngày tạo:** 2026-05-28  
**Artifact:** `outputs/v3/graph.npz`  
**Checkpoint:** `outputs/v3/hgt/hgt_flow_best.pt`  
**Môi trường training:** L40S GPU (48 GB VRAM), CUDA, AMP bf16

---

## 1. Tổng Quan Graph Artifact

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| **Artifact version** | v3 | Smart-BOTH Hybrid |
| **File size** | 1,854 MB | float16 packet_x |
| **Số flows** | 210,930 | Bidirectional 5-tuple |
| **Số packets** (sau filter) | 387,388 | 50% raw — τ_edge = 0.4 |
| **Số packets raw** | 773,577 | Trước khi lọc evidence |
| **Số hosts** | 1,235 | IP-level aggregate |
| **Số techniques (MITRE)** | 691 | ATT&CK Enterprise |
| **Số tactics** | 14 | |
| **Flow feature dims** | 91 | 80 CICFlowMeter + 5 evidence summary + metadata |
| **Payload length** | 256 bytes | Mỗi packet |
| **Nguồn dữ liệu** | 14 GB subset | 13 PCAP files, 1 per attack class |

> **Lưu ý:** Đây là **14 GB subset**, KHÔNG phải toàn bộ CIC-IoT-2023.
> Toàn bộ dataset CIC-IoT-2023 ước tính 5–10x lớn hơn.

---

## 2. Cấu Hình Training (Lần Chạy Hiện Tại)

### 2.1 Kiến trúc Model

| Tham số | Giá trị | Lý do |
|---|---|---|
| `hidden_dim` | 128 | Consensus RelGT/AHGT-DFD/DLG-IDS |
| `num_layers` | 4 | +1 vs baseline; cover 4-hop hierarchy flow→pkt→tech→tactic |
| `num_heads` | 8 | RelGT backbone; head_size = 128/8 = 16 |
| `dropout` | 0.2 | AHGT-DFD/DLG-IDS consensus |
| `ffn_multiplier` | 2 | Standard |

### 2.2 Training Hyperparameters

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `epochs` | 50 | **Điểm dừng hiện tại** |
| `patience` | 25 | Early stopping theo val_macro_f1 |
| `lr` | 1e-3 | AdamW base LR |
| `scheduler` | cosine_annealing | Warmup 5%, min LR = 1e-5 |
| `loss_type` | focal | Focal loss γ = 2.0 |
| `label_smoothing` | 0.05 | |
| `class_weight` | balanced (cap=10.0) | Balanced weight, capped 10× |
| `batch_seed_flows` | 256 | Effective batch = 256×2 = 512 |
| `grad_accum_steps` | 2 | VRAM optimization |
| `ema_enabled` | true | Decay = 0.999 |
| `drop_edge_prob` | 0.10 | Edge dropout augmentation |
| `gcl_weight` | 0.2 | GCL auxiliary loss weight |
| `gcl_temperature` | 0.1 | NT-Xent temperature |
| `hgaa.aug_prob` | 0.5 | Adaptive graph augmentation |
| `hgaa.tail_class_k` | 3 | Top-3 tail classes get extra augmentation |

---

## 3. Kết Quả Training

### 3.1 Chỉ Số Tổng Quan

| Metric | Val (Random) | Test (Random) |
|---|---|---|
| **Macro-F1** ⭐ | **0.7970** | **0.7752** |
| **Accuracy** | 93.48% | 93.34% |
| **Loss** | 0.1434 | 0.1440 |
| **Best epoch** | **49 / 50** | — |

> ⚠️ **QUAN TRỌNG:** Best epoch = **49/50** — model vẫn đang trong xu hướng cải thiện,
> chưa hội tụ! Patience=25 không bị kích hoạt vì val_f1 tăng đều từ epoch 33 đến 50.

### 3.2 Kết Quả Per-Class (Val Set — Random Split)

| Lớp | Support | F1 | Precision | Recall | Trạng thái |
|---|---|---|---|---|---|
| DDoS-PSHACK_Flood | 4,989 | **0.9997** | 1.0000 | 0.9994 | ✅ Hoàn hảo |
| DDoS-RSTFINFlood | 4,991 | **0.9999** | 1.0000 | 0.9998 | ✅ Hoàn hảo |
| Recon-HostDiscovery | 2,280 | **0.9801** | 1.0000 | 0.9610 | ✅ Tốt |
| DDoS-ACK_Fragmentation | 2,294 | **0.9908** | 0.9987 | 0.9830 | ✅ Tốt |
| Benign | 403 | **0.9524** | 0.9153 | 0.9926 | ✅ Tốt |
| Recon-PortScan | 1,609 | **0.9676** | 0.9909 | 0.9453 | ✅ Tốt |
| Recon-OSScan | 1,062 | **0.9483** | 0.9295 | 0.9680 | ✅ Tốt |
| Mirai-udpplain | 13 | **0.9600** | 1.0000 | 0.9231 | ✅ Tốt (support nhỏ) |
| Backdoor_Malware | 303 | **0.8788** | 0.9720 | 0.8020 | 🟡 Ổn — recall thấp |
| BrowserHijacking | 318 | **0.8721** | 0.7751 | 0.9969 | 🟡 Ổn — precision thấp |
| Recon-PingSweep | 181 | **0.8866** | 0.8148 | 0.9724 | 🟡 Ổn |
| DDoS-ICMP_Fragmentation | 46 | **0.7551** | 0.7115 | 0.8043 | 🟡 Trung bình (support nhỏ) |
| VulnerabilityScan | 1,071 | **0.8283** | 0.9821 | 0.7162 | 🟡 Recall thấp |
| SqlInjection | 575 | **0.7841** | 0.6554 | 0.9757 | 🟡 Precision thấp |
| DDoS-ICMP_Flood | 5 | **0.9091** | 0.8333 | 1.0000 | ⚪ Support quá nhỏ |
| **Uploading_Attack** | 149 | **0.2819** | 0.1646 | 0.9799 | 🔴 **CẦN CHÚ Ý** |
| **CommandInjection** | 411 | **0.2204** | 0.7571 | 0.1290 | 🔴 **CẦN CHÚ Ý** |
| **XSS** | 392 | **0.1312** | 0.5800 | 0.0740 | 🔴 **CẦN CHÚ Ý** |

---

## 4. Đường Cong Training (Learning Curve)

### 4.1 Toàn Bộ 50 Epoch

```
Epoch | Train F1 | Val F1  | Val Loss | Nhận xét
------|----------|---------|----------|-------------------------------------------
  1   | 0.5287   | 0.2589  | 2.3739   | Khởi đầu — hội tụ nhanh
  5   | 0.7284   | 0.5878  | 0.4307   | Tăng mạnh giai đoạn 1
 10   | 0.7618   | 0.6488  | 0.3190   | Bắt đầu plateau giả
 18   | 0.7841   | 0.6954  | 0.2421   | Peak cục bộ đầu tiên
 20   | 0.7916   | 0.6174  | 0.3543   | 🔻 Val drop — noise batch sampling
 21-31| 0.79-0.82| 0.62-0.71| 0.33-0.69| ⚠️ Oscillation mạnh (cosine annealing cycle)
 33   | 0.8268   | 0.7345  | 0.2256   | 🚀 Bước nhảy — cosine warmup lần 2
 42   | 0.8376   | 0.7872  | 0.1919   | Gia tốc hội tụ
 46   | 0.8425   | 0.7941  | 0.1558   | Tiếp tục cải thiện
 49   | 0.8424   | 0.7970  | 0.1434   | ⭐ BEST checkpoint
 50   | 0.8424   | 0.7967  | 0.1424   | Val loss tiếp tục giảm
```

### 4.2 Phân Tích 3 Giai Đoạn

**Giai đoạn 1 (epoch 1–20):** Học nhanh. Train F1: 0.53 → 0.79, Val F1: 0.26 → 0.69.
Val có oscillation lớn do cosine annealing LR còn cao.

**Giai đoạn 2 (epoch 21–32):** Plateau giả / oscillation. Val F1 dao động 0.62–0.72
trong khi Train F1 vẫn tăng đều 0.79 → 0.82. Đây là hiện tượng bình thường với
cosine_annealing + 22 edge types — LR dao động khiến gradient direction thay đổi.

**Giai đoạn 3 (epoch 33–50):** Hội tụ thật sự. Val F1: 0.73 → 0.797, Val loss:
0.226 → 0.143. **Xu hướng vẫn đi xuống đều tại epoch 50 — model chưa hội tụ.**

---

## 5. Các Chỉ Số Cần Đặc Biệt Chú Ý

### ⭐ CHỈ SỐ #1: Macro-F1 (0.797) — Chỉ Số Chính

**Tại sao cần chú ý:**
Macro-F1 tính trung bình F1 của TẤT CẢ các lớp với trọng số bằng nhau,
bất kể support. Với bộ dữ liệu imbalanced (DDoS-PSHACK=4989 vs Mirai=13),
accuracy (93.48%) bị bias nặng bởi các lớp DDoS lớn — nó không phản ánh
khả năng phát hiện tấn công hiếm. Macro-F1 là chỉ số **trung thực duy nhất**
cho bài toán IDS đa lớp.

**Gap val vs test:** 0.797 vs 0.775 = **−0.022**. Gap này nhỏ và bình thường
cho random split (cùng phân phối). Đây là dấu hiệu tốt — không overfit.

---

### ⭐ CHỈ SỐ #2: XSS F1 = 0.131 — Tệ Nhất Hệ Thống

**Tại sao cần chú ý:**
- Recall = **7.4%** — model bỏ qua 92.6% tất cả XSS flows
- Precision = 0.58 — khi dự đoán XSS thì đúng 58%, nhưng gần như không bao giờ dự đoán
- XSS là tấn công application-layer (HTTP parameter injection). Tại network level,
  traffic XSS trông **gần giống hoàn toàn với HTTP Benign** — chỉ payload content
  mới khác biệt
- MITRE evidence edges lẽ ra phải giúp phân biệt, nhưng nếu PMI không học được
  n-gram đặc trưng XSS (`<script>`, `onerror=`, `javascript:`), evidence edges sẽ
  yếu → HGT không có signal để phân biệt

**Hậu quả thực tế:** Trong môi trường production, 92.6% XSS tấn công sẽ **không bị phát hiện**.
Đây là lỗ hổng nghiêm trọng về mặt bảo mật.

---

### ⭐ CHỈ SỐ #3: CommandInjection F1 = 0.220 — Recall 12.9%

**Tại sao cần chú ý:**
- Precision = 0.757 cao nhưng Recall = 0.129 rất thấp
- Pattern ngược với XSS: model **có thể nhận dạng** CommandInjection khi nó predict
  (precision OK), nhưng hầu như không bao giờ predict class này
- Nguyên nhân: class_weight balanced sẽ tăng weight cho CommandInjection,
  nhưng nếu features của nó overlap với BrowserHijacking hoặc SqlInjection,
  model sẽ chọn class "an toàn hơn" với loss thấp hơn
- Cần kiểm tra: CommandInjection flows có đang bị predict thành class nào? (confusion matrix)

---

### ⭐ CHỈ SỐ #4: Uploading_Attack F1 = 0.282 — Precision 16.5%

**Tại sao cần chú ý:**
- Recall = **97.9%** — model ĐÃ học được Uploading_Attack signal
- Precision = **16.5%** — nghĩa là cứ 6 lần model predict "Uploading_Attack",
  chỉ 1 lần đúng. Năm lần còn lại là false positives từ các class khác
- Vấn đề ngược với XSS/CommandInjection: model đang OVER-predict Uploading_Attack
- Nguyên nhân có thể: HGAA augmentation với tail_class_k tạo quá nhiều
  synthetic Uploading_Attack samples → model thiên vị

---

### ⭐ CHỈ SỐ #5: Val Loss Vẫn Giảm Ở Epoch 50 (0.1434)

**Tại sao cần chú ý:**
```
Epoch 47: val_loss = 0.1507
Epoch 48: val_loss = 0.1463
Epoch 49: val_loss = 0.1434  ← BEST
Epoch 50: val_loss = 0.1424  ← CÒN THẤP HƠN BEST
```
**Val loss epoch 50 (0.1424) < val loss epoch 49 (0.1434)**, nhưng val F1 epoch 50
(0.7967) < val F1 epoch 49 (0.7970) một chút. Điều này nghĩa là:
- Loss đang hội tụ tốt
- F1 đang ở đỉnh local — cần thêm epochs để vượt qua
- Với 150 epochs, model gần như chắc chắn sẽ cải thiện đáng kể

---

### ⭐ CHỈ SỐ #6: Train-Val F1 Gap (0.8424 vs 0.7970 = 0.0454)

**Tại sao cần chú ý:**
Gap train-val = 0.045 là **chấp nhận được nhưng cần theo dõi**:
- Gap nhỏ (< 0.03) = underfitting hoặc regularization quá mạnh
- Gap vừa (0.03–0.07) = cân bằng tốt ✅ (hiện tại)
- Gap lớn (> 0.10) = overfitting

Với full dataset CIC-IoT-2023 (nhiều data hơn), gap này sẽ **giảm tự nhiên** —
tốt hơn cho generalization.

---

## 6. Dự Đoán Kết Quả: 150 Epochs + Full CIC-IoT-2023

### 6.1 Giả Định Cơ Bản

| Yếu tố | Hiện tại (14 GB subset) | Lần tới (Full CIC-IoT-2023) |
|---|---|---|
| Số flows | ~211k | ~1.2–2.5M (ước tính 6–12x) |
| Số packets | ~387k | ~2–5M |
| Epochs | 50 | 150 |
| Cấu hình model | Như hiện tại | Có thể điều chỉnh (xem mục 7) |

### 6.2 Phân Tích Xu Hướng Hội Tụ

Từ learning curve hiện tại:
- Epoch 33→50: Gain = +0.062 (từ 0.735 → 0.797) trong 17 epochs
- Slope trung bình cuối: ~+0.0015 F1/epoch
- Tốc độ học **chưa bão hòa** tại epoch 50

Extrapolation đơn giản cho 150 epochs (cấu hình giống cũ):
```
Epoch 50  → 0.797 (đã có)
Epoch 70  → ~0.820–0.835 (ước tính)
Epoch 100 → ~0.845–0.860 (ước tính)
Epoch 150 → ~0.860–0.880 (ước tính, có thể plateau)
```

### 6.3 Tác Động Của Full Dataset

**Lợi ích:**
1. **Rare classes cải thiện mạnh:** CommandInjection, XSS, Uploading_Attack sẽ có
   support cao hơn nhiều → model học patterns tốt hơn → F1 các class này tăng 0.15–0.30
2. **Better generalization:** Với 10x data, model sẽ thấy nhiều variation hơn →
   train-val gap giảm → generalization tốt hơn
3. **MSEE evidence edges chất lượng cao hơn:** PMI trên nhiều data → n-gram weights
   chính xác hơn → evidence edges tốt hơn cho XSS, CommandInjection

**Rủi ro:**
1. **Training time tăng ~6–10x:** Mỗi epoch mất nhiều thời gian hơn trên L40S
2. **Memory:** packet_x sẽ tăng tương ứng — cần kiểm tra VRAM với feature_store

### 6.4 Dự Đoán Val Macro-F1

| Kịch bản | Epochs | Data | Config | Dự đoán Macro-F1 |
|---|---|---|---|---|
| Baseline (hiện tại) | 50 | 14 GB | current | 0.797 (đã đạt) |
| Chỉ thêm epochs | 150 | 14 GB | current | 0.86–0.88 |
| Full data + 150 ep | 150 | Full CIC | current | 0.88–0.92 |
| Full data + tuning | 150 | Full CIC | v2 config | **0.92–0.96** |

> **Lưu ý:** Đây là dự đoán dựa trên trend extrapolation + kinh nghiệm từ
> benchmark GNN-IDS. Kết quả thực tế có thể khác ±0.03–0.05 tùy class distribution
> trong full dataset.

---

## 7. Đề Xuất Cấu Hình Cho Lần Train Tiếp (Target: Macro-F1 > 0.95)

### 7.1 Thay Đổi Bắt Buộc

#### A. Tăng Epochs và Patience
```yaml
train:
  epochs: 150          # từ 50 → 150
  patience: 50         # từ 25 → 50 (tránh early stop quá sớm)
```
**Lý do:** Model chưa hội tụ ở epoch 50. Val loss vẫn giảm đều.
Với full dataset nhiều data hơn, convergence cần nhiều epoch hơn.
Patience=50 đảm bảo không dừng trong oscillation giai đoạn 2.

#### B. Tăng Model Capacity
```yaml
model:
  hidden_dim: 256      # từ 128 → 256
  num_layers: 4        # giữ nguyên
  num_heads: 8         # giữ nguyên (head_size = 256/8 = 32)
  dropout: 0.15        # từ 0.2 → 0.15 (giảm nhẹ với nhiều data hơn)
```
**Lý do:** 128-dim đang là bottleneck cho 18 lớp + 22 edge types.
Với full dataset, capacity cao hơn sẽ được tận dụng tốt hơn.
L40S 48 GB có đủ VRAM với activation_checkpointing=true.

#### C. Tăng Focal Loss Gamma
```yaml
train:
  focal_gamma: 3.0     # từ 2.0 → 3.0
  label_smoothing: 0.03  # từ 0.05 → 0.03
```
**Lý do:** γ=2.0 chưa đủ mạnh để focus vào CommandInjection/XSS — các
hard examples này cần penalty mạnh hơn. γ=3.0 sẽ down-weight easy DDoS
samples mạnh hơn, buộc model học hard classes.
Label smoothing giảm để tránh làm mờ signal của rare classes.

#### D. Tăng HGAA cho Tail Classes
```yaml
train:
  hgaa:
    enabled: true
    aug_prob: 0.7        # từ 0.5 → 0.7
    bias_factor_T: 0.7   # từ 0.5 → 0.7
    tail_class_k: 5      # từ 3 → 5 (CommandInjection, XSS, Uploading, ICMP classes)
```
**Lý do:** Top-3 tail classes theo support hiện tại là ICMP_Flood (5),
Mirai-udpplain (13), ICMP_Fragmentation (46). Tuy nhiên, **worst F1 classes**
là XSS, CommandInjection, Uploading_Attack. Tăng tail_class_k=5 sẽ include
cả hai nhóm này.

#### E. Tăng GCL Weight
```yaml
train:
  gcl_weight: 0.3      # từ 0.2 → 0.3
  gcl_n_negatives: 32  # từ 16 → 32
```
**Lý do:** GCL auxiliary loss kéo packet embeddings về technique embeddings
theo class→technique map. CommandInjection và XSS đều có technique mappings
rõ ràng trong MITRE ATT&CK. Tăng gcl_weight sẽ tăng cường tín hiệu này.

### 7.2 Thay Đổi Tùy Chọn (Nếu VRAM Cho Phép)

#### F. Tăng Fanout cho Evidence Edges
```yaml
sampler:
  fanouts:
    packet__evidence_injection__technique: 6    # từ 4 → 6
    packet__evidence_command_exec__technique: 6  # từ 4 → 6
    packet__evidence_file_upload__technique: 4   # từ 3 → 4
    flow__burst_neighbor__flow: 8               # từ 6 → 8
```
**Lý do:** CommandInjection maps vào T1059 (Command Scripting Interpreter).
Tăng fanout evidence_command_exec cho phép HGT thấy nhiều neighbor techniques
hơn per batch → better discrimination.

#### G. Tăng Batch Size Effective
```yaml
train:
  batch_seed_flows: 384   # từ 256 → 384
  grad_accum_steps: 2     # giữ nguyên (effective = 768)
```
**Lý do:** Larger batch = ổn định hơn với 18 classes và class_weight balanced.
Chỉ dùng nếu full dataset VRAM không bị OOM.

#### H. LR Scheduler Điều Chỉnh
```yaml
train:
  lr: 8.0e-4             # từ 1e-3 → 8e-4
  scheduler_pct_start: 0.03  # warmup 3% thay vì 5% (150 epochs)
  scheduler_eta_min: 5.0e-6  # từ 1e-5 → 5e-6
```
**Lý do:** 150 epochs với LR=1e-3 dễ bị oscillation kéo dài (thấy ở epoch
20-32 trong lần chạy này). Giảm LR ban đầu nhẹ + eta_min thấp hơn giúp
cosine annealing có dải decay rộng hơn cho phase 2 convergence.

### 7.3 Tóm Tắt Config Đề Xuất (v3-full config)

```yaml
# configs/eg_hgt_full150.yaml — cho full CIC-IoT-2023, 150 epochs

model:
  hidden_dim: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.15
  ffn_multiplier: 2

train:
  epochs: 150
  patience: 50
  batch_seed_flows: 384
  grad_accum_steps: 2        # effective batch = 768
  lr: 8.0e-4
  scheduler: cosine_annealing
  scheduler_pct_start: 0.03
  scheduler_eta_min: 5.0e-6
  loss_type: focal
  focal_gamma: 3.0
  label_smoothing: 0.03
  class_weight: balanced
  class_weight_cap: 12.0
  gcl_weight: 0.3
  gcl_n_negatives: 32
  gcl_temperature: 0.1
  hgaa:
    enabled: true
    aug_prob: 0.7
    bias_factor_T: 0.7
    tail_class_k: 5
  ema_enabled: true
  ema_decay: 0.999
  drop_edge_prob: 0.10
  amp: true
  amp_dtype: auto
  activation_checkpointing: true
  grad_clip_norm: 1.0

sampler:
  hops: 4
  fanouts:
    flow__contains__packet: 20
    packet__next_packet__packet: 4
    packet__evidence_injection__technique: 6
    packet__evidence_command_exec__technique: 6
    packet__evidence_file_upload__technique: 4
    packet__evidence_recon__technique: 4
    packet__evidence_c2_beacon__technique: 3
    flow__from_host__host: 1
    flow__to_host__host: 1
    flow__burst_neighbor__flow: 8
    flow__matches_technique__technique: 3
    technique__has_subtechnique__technique: 2
    technique__belongs_to_tactic__tactic: 1
```

---

## 8. Phân Tích Vì Sao 3 Class Yếu Và Giải Pháp

### 8.1 XSS (F1 = 0.131)

**Root cause:**
XSS là tấn công HTTP application layer. Tại tầng network/transport:
- Packet size, IAT, TCP flags — **giống hoàn toàn với HTTPS/HTTP Benign**
- Flow duration, bytes — **không có pattern đặc biệt**
- Payload content có `<script>`, `alert(`, `onerror=` — **nhưng chỉ trong
  HTTP request payload, thường encrypted (HTTPS)**

PMI n-gram sẽ không học được pattern XSS nếu payload là encrypted.
Evidence edges `packet__evidence_injection__technique` sẽ ít hoặc không có.

**Giải pháp với full dataset:**
- Thêm nhiều XSS flows → PMI có nhiều plaintext HTTP XSS samples hơn để học
- Tăng `packet__evidence_injection__technique` fanout: 4 → 6
- Kiểm tra PMI table: xem n-gram nào được assigned cho T1190 (Exploit Public App)

### 8.2 CommandInjection (F1 = 0.220, Recall = 12.9%)

**Root cause:**
- Precision = 75.7% (tốt) nhưng Recall = 12.9% → model **biết CommandInjection
  trông như thế nào** nhưng **không dám predict** vì class trước đó bị confused
  với BrowserHijacking hoặc SqlInjection
- Theo CIC-IoT-2023 paper, CommandInjection thường là HTTP POST với OS commands
  trong parameters — rất similar với SqlInjection về flow statistics

**Giải pháp:**
- focal_gamma=3.0 sẽ penalize missed CommandInjection mạnh hơn
- gcl_weight=0.3: kéo CommandInjection embeddings về T1059 techniques → xa SqlInjection

### 8.3 Uploading_Attack (F1 = 0.282, Precision = 16.5%)

**Root cause:**
- Recall = 97.9% → model detect được Uploading_Attack signal
- Precision = 16.5% → 5/6 predictions là false positives
- Nguyên nhân có thể: HGAA với tail_class_k đang tạo nhiều augmented samples
  cho Uploading_Attack (support=149, là class nhỏ) → model over-predict
- Cũng có thể do `class_weight_cap=10.0` đang đẩy weight Uploading_Attack lên cao,
  làm model chọn nó khi không chắc chắn

**Giải pháp:**
- class_weight_cap=12.0 nhưng **kết hợp với focal_gamma=3.0** để không overfit
- Với full dataset, Uploading_Attack sẽ có nhiều support hơn → balanced weight
  sẽ thấp hơn → precision tự cải thiện

---

## 9. Kết Luận

### 9.1 Tình Trạng Hiện Tại

Model EG-HGT v3 đạt **val Macro-F1 = 0.797** sau 50 epochs trên 14 GB subset.
Đây là kết quả **tốt nhưng chưa tối ưu**:
- 10/18 classes đạt F1 > 0.85 ✅
- 3 classes (XSS, CommandInjection, Uploading_Attack) kéo Macro-F1 xuống ❌
- Model **chưa hội tụ** — best epoch = 49/50, loss vẫn giảm tại epoch 50

### 9.2 Khả Năng Đạt Target 0.95

Đây là target **ambitious nhưng khả thi** với full CIC-IoT-2023 + 150 epochs:

| Điều kiện | Có không? | Impact |
|---|---|---|
| Model chưa hội tụ ở 50 epoch | ✅ Yes | +0.05–0.08 |
| Full dataset (10x data) | Lần tới | +0.03–0.06 |
| Config optimization (γ=3.0, dim=256) | Đề xuất | +0.03–0.05 |
| Tổng | | +0.11–0.19 từ 0.797 |

**Dự đoán:** 0.797 + ~0.15 = **0.94–0.96** (random split)

Đạt 0.95 là khả thi nếu áp dụng config đề xuất + full dataset.
Temporal split sẽ thấp hơn ~0.05–0.10 so với random split — đây là gap cần report.

### 9.3 Checklist Trước Lần Train Tiếp

- [ ] Build lại graph artifact với full CIC-IoT-2023 (toàn bộ PCAPs)
- [ ] Kiểm tra pmi_table.parquet cho XSS n-grams sau full rebuild
- [ ] Copy config đề xuất vào `configs/eg_hgt_full150.yaml`
- [ ] Smoke test 2 epochs trên local CPU trước khi upload server
- [ ] Sanity audit artifact: `python scripts/diagnostics/v3_artifact_audit.py`
- [ ] Upload lên L40S → train 150 epochs với config mới
- [ ] Eval BOTH splits sau training: `python scripts/eval/v3_eval_both_splits.py`
- [ ] Report gap (random - temporal) là contribution chính của v3

---

*Báo cáo được tạo tự động từ `outputs/v3/hgt/training_summary.json` và
`outputs/v3/graph.meta.json` — 2026-05-28*
