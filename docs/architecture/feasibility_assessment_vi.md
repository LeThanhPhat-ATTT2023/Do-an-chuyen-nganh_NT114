# Đánh Giá Tính Thực Tiễn và Khả Thi

> **Mục đích tài liệu:** Ghi lại tư duy ban đầu khi chọn hướng đề tài, lập luận vì sao hướng này có giá trị khoa học, và đánh giá tổng thể xem có thể hoàn thành trong điều kiện thực tế của nhóm (máy cá nhân i5/16 GB, thời gian một học kỳ) hay không.

> ⚠️ **CẬP NHẬT (v3_ob, 2026-06):** Đây là tài liệu **giai đoạn đề xuất** — giữ nguyên để ghi lại tư duy ban đầu. Kiến trúc đã thực thi (`v3_ob`) **đã thay đổi** so với mô tả bên dưới ở 3 điểm cốt lõi. Khi trích dẫn cho báo cáo cuối, hãy dùng [system_execution_flows.md](system_execution_flows.md) và `CLAUDE.md` làm nguồn chuẩn:
> 1. **Bỏ hoàn toàn Student 1D-CNN / SecureBERT distillation.** Pipeline v3 dùng **zero learned encoder ngoài chính HGT**. Packet feature là **ordered-byte** (đó là ý nghĩa của `v3_ob`), không còn embedding 768D từ teacher.
> 2. **Bỏ MITRE mapping bằng cosine similarity embedding.** Thay bằng **MSEE** (Multi-Source Evidence Ensemble): PMI counting + L1 multinomial LR + Aho-Corasick procedure matching trên MITRE STIX. Edge mang `family / weight / source / provenance`, có thể audit.
> 3. **Đánh giá theo Smart-BOTH + clean-key (LNL).** Đóng góp chính giờ là **dấu của (noisy − clean) macro-F1** và GAP random↔temporal, không phải MITRE mapping precision như bảng KPI bên dưới.
>
> Phần Slow-Path/SLM XAI vẫn đúng về tinh thần, nhưng nay đọc **graph-text bằng chứng** và có **VG2R verifier** chấm grounding (xem [system_execution_flows.md](system_execution_flows.md)).

---

## 1. Bối Cảnh và Động Lực Nghiên Cứu

### 1.1 Vấn đề thực tế

Các hệ thống phát hiện xâm nhập (IDS) truyền thống — đặc biệt là IDS dựa trên chữ ký — ngày càng tỏ ra không đủ khả năng đối phó với các cuộc tấn công IoT hiện đại. Hai hạn chế cốt lõi:

1. **Thiếu ngữ cảnh:** Mỗi flow hoặc packet được phân tích độc lập, bỏ qua quan hệ phụ thuộc giữa các thiết bị, giao thức và thời điểm trong cùng một chiến dịch tấn công.
2. **Thiếu khả năng giải thích:** Ngay cả khi mô hình học sâu đưa ra nhãn "tấn công", nó không trả lời được câu hỏi: *tại sao*, *kỹ thuật nào*, và *bằng chứng nào* dẫn đến kết luận đó — điều bắt buộc phải có trong môi trường SOC thực tế.

### 1.2 Khoảng trống trong nghiên cứu hiện tại

Khảo sát các công trình gần đây cho thấy:

| Hướng nghiên cứu | Ưu điểm | Khoảng trống |
|---|---|---|
| GNN trên flow graph (GNN4IDS, E-GraphSAGE) | Nắm quan hệ multi-hop | Không gắn MITRE, không XAI thực sự |
| BERT/Transformer trên payload | Hiểu nội dung byte | Quá nặng cho real-time, không có graph context |
| Rule-based MITRE mapping | Giải thích được | Phụ thuộc signature, không generalize |
| XAI post-hoc (SHAP, GradCAM) | Thêm giải thích | Chỉ giải thích feature, không dùng ngôn ngữ tự nhiên |

**Không có công trình nào** kết hợp đồng thời: (i) heterogeneous graph đa tầng, (ii) payload embedding có ngữ nghĩa, (iii) gắn nhãn MITRE ATT&CK tự động, và (iv) sinh báo cáo XAI bằng SLM — đặc biệt trên tập IoT hiện đại như CIC-IoT2023.

### 1.3 Câu hỏi nghiên cứu chính

> *Liệu một hệ thống IDS có thể đồng thời đạt độ chính xác cao, phát hiện real-time (≤50 ms p95), và tự động sinh báo cáo XAI có ngữ cảnh MITRE ATT&CK không, trong giới hạn phần cứng consumer-grade?*

---

## 2. Ý Tưởng Thiết Kế Ban Đầu

### 2.1 Tư duy cốt lõi: "Biết mà không cần giải thích thì vô nghĩa"

Thay vì chỉ tối ưu accuracy như đa số bài báo benchmark, hướng này đặt **giải thích được** lên ngang tầm với **phát hiện được**. Đây là lý do:

- Trong thực tế triển khai, một cảnh báo không có context sẽ bị analyst bỏ qua (alert fatigue).
- Ánh xạ sang MITRE ATT&CK cho phép kết nối phát hiện kỹ thuật với threat intelligence chiến lược.
- SLM sinh báo cáo ngôn ngữ tự nhiên giảm thời gian triage từ phút xuống giây.

### 2.2 Kiến trúc hai đường (Dual-Path)

```
Luồng online:
  Packet → Payload 256B → Student CNN (ONNX) → MITRE Top-k
         ↘                                    ↗
           Hot Graph Buffer → K-hop Subgraph → HGT Classifier
                                                    ↓
                                             [Nhãn + Confidence]
                                                    ↓
                              Benign → trả kết quả ngay (≤50ms)
                              Suspicious → đẩy vào Slow Queue

Luồng async (Slow Path):
  Slow Queue → Context Hydrator → Evidence Builder
             → Evidence Ranker → SLM (Qwen2.5-3B GGUF)
             → Báo cáo XAI: kỹ thuật + bằng chứng + MITRE
```

Tách thành hai đường giải quyết mâu thuẫn cơ bản giữa **tốc độ** và **chiều sâu giải thích**: Fast Path không bị block bởi SLM, Slow Path không bị giới hạn bởi latency yêu cầu.

### 2.3 Ba cải tiến kỹ thuật cốt lõi so với baseline

**Cải tiến 1 — Payload distillation thay vì raw BERT:**
- GNN4IDS dùng payload 1500B → chi phí cao, nhiễu nhiều.
- Đề xuất: cắt về 256B (header + đầu payload), distill SecureBERT → Student 1D-CNN 768D.
- Kết quả: embedding giữ ngữ nghĩa (~cosine similarity ≥ 0.85 với teacher) nhưng inference nhanh hơn 20-40× trên CPU.

**Cải tiến 2 — Heterogeneous graph 3 tầng:**
- Tầng 1 (Packet): node là packet, edge là cùng flow.
- Tầng 2 (Flow): node là flow, edge là cùng src/dst IP hoặc port pattern.
- Tầng 3 (MITRE Tactical): edge nối flow với technique dựa trên cosine similarity embedding ≥ ngưỡng θ.
- Sử dụng HGT (Heterogeneous Graph Transformer) để học message passing đa loại quan hệ.

**Cải tiến 3 — MITRE mapping tự động qua embedding space:**
- Thay vì rule-based, tính cosine similarity giữa flow embedding và 193 MITRE technique embedding 768D.
- Top-k technique được gắn làm soft label và tactical edge trong graph.
- Đây là đóng góp có thể đo lường được: so sánh mapping quality với ground-truth MITRE label thủ công.

---

## 3. Đánh Giá Tính Thực Tiễn

### 3.1 Tính mới về mặt khoa học

| Tiêu chí | Đánh giá | Lý do |
|---|---|---|
| Novelty | Cao | Chưa có paper nào kết hợp đủ 4 thành phần trên tập CIC-IoT2023 |
| Reproducibility | Trung bình-Cao | Dataset public, code sẽ open-source, nhưng SLM inference phụ thuộc hardware |
| Measurability | Cao | F1-macro, AUC, latency p95, MITRE mapping accuracy đều đo được định lượng |
| Impact | Cao | Liên quan trực tiếp IoT security — lĩnh vực đang tăng trưởng nhanh |

### 3.2 Giá trị đóng góp khi bảo vệ

Đề tài có thể trình bày **4 đóng góp cụ thể**:

1. **Phương pháp distillation payload** cho IDS: framework SecureBERT → 1D-CNN với benchmark cosine similarity và inference time.
2. **Heterogeneous graph builder** cho traffic IoT: schema 3 tầng (Packet/Flow/MITRE), chiến lược sliding window, threshold selection.
3. **Pipeline MITRE mapping tự động** qua embedding space: precision/recall so với gold label thủ công.
4. **Kiến trúc Dual-Path IDS** với SLM XAI: ablation study thể hiện đóng góp từng thành phần.

### 3.3 So sánh với GNN4IDS (baseline chính)

| Chỉ số | GNN4IDS (baseline) | Đề xuất | Hướng cải thiện |
|---|---|---|---|
| Payload xử lý | 1500B raw | 256B distilled | Nhanh hơn, ít nhiễu hơn |
| Loại graph | Homogeneous | Heterogeneous 3 tầng | Nắm được tactical context |
| MITRE mapping | Không có | Tự động qua embedding | Thêm chiều giải thích |
| XAI output | Không có | Báo cáo ngôn ngữ tự nhiên | Dùng được trong SOC |
| Latency focus | Offline | ≤50ms p95 online | Gần với deployment thực tế |

---

## 4. Đánh Giá Tính Khả Thi

### 4.1 Phần cứng: Asus Vivobook i5/16 GB RAM

#### Việc làm được tốt

| Bước | Lý do khả thi |
|---|---|
| Trích payload 256B từ PCAP | Xử lý stream, không cần load toàn bộ vào RAM |
| Teacher encoding theo batch nhỏ | Batch 16-32, checkpoint từng chunk, chạy qua đêm |
| Train Student 1D-CNN | Model nhỏ (~2M params), CPU đủ trong 2-4 giờ |
| Build hetero graph 3 tầng | Sliding window, mỗi window ≤50K node; dùng PyG |
| Train HGT mini | 2-3 layer, 4-8 head trên mẫu đã lọc (~100K flow) |
| SLM Qwen2.5-3B GGUF Q4_K_M | ~2 GB RAM; llama.cpp trên CPU, ~1-3 tok/s |

#### Điểm cần quản lý cẩn thận

| Rủi ro | Nguyên nhân | Biện pháp giảm thiểu |
|---|---|---|
| Teacher encoding toàn bộ CIC-IoT2023 | ~1.5M flow × SecureBERT trên CPU = nhiều giờ | Queue + resume từng chunk; chạy offline qua đêm |
| HGT trên đồ thị dày đặc | Sliding window rộng → RAM vượt 12 GB | Hard-cap node/edge/window; test trên 5% data trước |
| SLM latency vượt ngưỡng | Qwen2.5-3B trên CPU chậm | Timeout 10s + fallback template; Slow Path là async |
| CIC-IoT2023 class imbalance | 33 loại tấn công vs. benign | Stratified sampling + focal loss; không over-sample toàn bộ |

### 4.2 Thời gian: 1 học kỳ (~16 tuần)

```
Tuần 1-2:   [DATA]    Phân tích CIC-IoT2023, extractor payload 256B,
                       metadata schema, teacher targets batch đầu tiên.

Tuần 3-4:   [EMBED]   Train & benchmark Student 1D-CNN;
                       xuất ONNX; đo cosine similarity vs. teacher.

Tuần 5-7:   [GRAPH]   Build hetero graph 3 tầng + MITRE cosine mapping;
                       chốt threshold θ; đo MITRE mapping quality.

Tuần 8-10:  [MODEL]   Train HGT; xây fast classifier;
                       đo F1-macro, AUC, latency p95.

Tuần 11-12: [XAI]     Tích hợp SLM Slow Path;
                       evidence builder + report validator.

Tuần 13-14: [EVAL]    Ablation study 4 thành phần;
                       so sánh baseline GNN4IDS.

Tuần 15-16: [WRITE]   Viết báo cáo, chốt demo, chạy tổng duyệt.
```

**Điểm dự phòng:** Nếu tiến độ chậm ở tuần 7-8 (graph/HGT), có thể đơn giản hóa MITRE thành top-1 mapping và dùng HGT 2 tầng thay vì 3 tầng — vẫn đủ để so sánh với baseline.

### 4.3 Kỹ năng yêu cầu và mức độ sẵn có

| Kỹ năng | Yêu cầu | Đánh giá |
|---|---|---|
| Python / PyTorch | Cao | Sẵn có |
| Xử lý PCAP (Scapy/PyShark) | Trung bình | Học được trong 1-2 tuần |
| PyTorch Geometric (hetero graph) | Trung bình-Cao | Cần 1-2 tuần ramp-up; tài liệu tốt |
| Knowledge distillation | Trung bình | Đơn giản với MSE + cosine loss |
| ONNX Runtime | Thấp | Export script có sẵn |
| llama.cpp / GGUF | Thấp-Trung bình | CLI interface đơn giản |
| Thống kê đánh giá (F1, AUC, ablation) | Trung bình | Sklearn + seaborn |

---

## 5. KPI Bảo Vệ Trước Hội Đồng

### 5.1 Hiệu năng phát hiện

| Chỉ số | Ngưỡng tối thiểu | Mục tiêu lý tưởng |
|---|---|---|
| F1-macro (toàn bộ class) | > F1 baseline GNN4IDS | +3-5 pp so với baseline |
| AUC trung bình | ≥ 0.92 | ≥ 0.95 |
| Recall lớp tấn công nguy hiểm | ≥ 0.85 | ≥ 0.90 |

### 5.2 Hiệu năng runtime

| Chỉ số | Ngưỡng tối thiểu | Ghi chú |
|---|---|---|
| Latency Fast Path p95 | ≤ 50 ms | Đo trên local, không phụ thuộc network |
| Latency Fast Path p99 | ≤ 100 ms | Chấp nhận spike ngắn |
| Slow Path end-to-end | ≤ 10 s | Async; không block Fast Path |
| Throughput | ≥ 200 flow/s | Trên Vivobook i5 với ONNX |

### 5.3 Chất lượng XAI và MITRE

| Chỉ số | Ngưỡng | Cách đo |
|---|---|---|
| MITRE mapping precision | ≥ 0.70 | Manual spot-check 200 alert |
| Báo cáo Slow Path hợp lệ (pass validator) | ≥ 85% | Tự động qua report_validator.py |
| Báo cáo có đủ 3 section (kỹ thuật / bằng chứng / confidence) | 100% (hoặc fallback) | Schema check |

### 5.4 Ablation study (bắt buộc để chứng minh đóng góp từng phần)

| Variant | Mục đích |
|---|---|
| Full model | Baseline so sánh |
| Bỏ MITRE tactical edge | Đo đóng góp của MITRE mapping |
| Dùng raw embedding thay vì distilled | Đo đóng góp distillation |
| Dùng homogeneous graph | Đo đóng góp heterogeneous |
| Bỏ Slow Path | Đo đóng góp XAI (định tính) |

---

## 6. Tiêu Chí Dừng / Điều Chỉnh Hướng Sớm

Nếu gặp bất kỳ tình huống nào sau đây trước **tuần 8**, cần đánh giá lại scope:

| Tình huống | Nguyên nhân khả năng | Hành động |
|---|---|---|
| Student CNN cosine similarity với teacher < 0.70 | Tokenization payload byte→text sai; LR quá lớn | Debug tokenization; thử character-level vs. hex encoding |
| HGT không hội tụ sau 20 epoch | Graph quá thưa hoặc class imbalance cực đoan | Giảm scope graph; thử GCN homogeneous trước |
| MITRE mapping precision < 0.50 | Technique embedding không phân biệt được; ngưỡng θ quá thấp | Tăng θ; thêm flow-level context vào query vector |
| RAM vượt 14 GB khi build graph | Window quá rộng | Hard-cap ≤30K node/window; dùng sparse tensor |
| Latency Fast Path p95 > 200 ms | ONNX session overhead; K-hop too large | Giảm K từ 3 xuống 2; giảm HGT layer |

---

## 7. Kết Luận

Hướng đề tài **"Context-rich, Explainable IDS với Heterogeneous Graph + MITRE ATT&CK + SLM Embedding"** được đánh giá là:

- **Có giá trị khoa học rõ ràng:** Lấp đầy khoảng trống chưa có công trình nào làm đủ trên CIC-IoT2023.
- **Khả thi về phần cứng:** Với chiến lược chunking, ONNX, và GGUF quantized, toàn bộ pipeline chạy được trên Vivobook i5/16 GB.
- **Khả thi về thời gian:** 16 tuần đủ nếu ưu tiên đúng: distillation → graph → HGT → Slow Path, không song song hóa quá sớm.
- **Có điểm dự phòng:** Nếu SLM Slow Path không kịp polish, vẫn có Fast Path + MITRE mapping là đóng góp độc lập đủ để bảo vệ.

Rủi ro lớn nhất không phải kỹ thuật mà là **quản lý tiến độ**: các bước offline (teacher encoding, graph build) tốn thời gian thực nhưng không cần can thiệp liên tục — cần lên lịch chạy qua đêm ngay từ tuần 2.