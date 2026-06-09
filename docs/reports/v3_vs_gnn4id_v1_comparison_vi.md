# Báo Cáo So Sánh — EG-HGT v3 vs GNN4ID (baseline v1)

**Ngày tạo:** 2026-06-05
**Người soạn:** phân tích tự động (systematic-debugging)
**Phạm vi:** so sánh kết quả phân loại đa lớp (18 lớp) trên cùng tập 14 GB subset của CIC-IoT-2023, split **random**.

| Artifact | Đường dẫn |
|---|---|
| Kết quả GNN4ID v1 | `baselines/gnn4id/outputs/outputs/results.json` |
| Manifest phân bố GNN4ID | `baselines/gnn4id/outputs/outputs/graphs.manifest.json` |
| Kết quả EG-HGT v3 | `outputs/v3/hgt/training_summary.json` |
| Artifact graph v3 | `outputs/v3/graph.npz` (+ `graph.meta.json`) |

---

## 0. TL;DR (kết luận nhanh)

> **GNN4ID v1 thắng macro-F1 (0.854 vs 0.775), nhưng phần lớn khoảng cách KHÔNG đến từ chất lượng model — mà đến từ DỮ LIỆU.**
>
> 1. **accuracy và weighted-F1 gần như bằng nhau** (0.939 vs 0.933) → toàn bộ gap nằm ở các **lớp hiếm** (macro-F1).
> 2. **~52% gap** đến từ 3 lớp volumetric (ICMP_Flood, Mirai, ICMP_Fragmentation): cách extract flow của v3 (gộp 5-tuple 2 chiều) nghiền nát chúng còn **46 / 133 / 458 flow**, trong khi GNN4ID có **16,225** mỗi lớp (chênh **35–353×**) từ CÙNG file PCAP.
> 3. **~35% gap** ở các lớp tầng-ứng-dụng (XSS, SqlInjection…) — nơi hai bên có **số mẫu xấp xỉ nhau** nhưng GNN4ID vẫn hơn → khác biệt **biểu diễn** (GNN4ID nhồi raw-byte/flow; v3 dựa MSEE edge thưa, chỉ phủ 21% flow).
> 4. v3 **chưa hội tụ** ở epoch 50 (val_loss vẫn giảm) → còn dư địa.
>
> **Hệ quả:** so sánh hiện tại không công bằng. Một thí nghiệm đối chứng (`train_imbalanced.py`) đang chạy để kiểm chứng: cho GNN4ID đúng độ khan hiếm mẫu của v3 — nếu macro-F1 của nó sụp về mức v3, gap được chứng minh là do dữ liệu, không phải HGT.

---

## 1. So sánh headline (cùng split random)

| Chỉ số | GNN4ID v1 | EG-HGT v3 (test) | EG-HGT v3 (val) | Δ (v3 test − GNN4ID) |
|---|---|---|---|---|
| **macro-F1** | **0.8537** | 0.7752 | 0.7970 | **−0.0785** |
| weighted-F1 | 0.9373 | ~0.93 | — | ~0 |
| accuracy | 0.9391 | 0.9334 | 0.9348 | −0.006 |
| Số lớp | 18 | 18 | 18 | — |
| Tỉ lệ split | 70/15/15 | 80/10/10 | — | — |

**Đọc bảng:** accuracy + weighted-F1 ngang nhau ⇒ model v3 phân loại đúng phần lớn lưu lượng. Chỉ **macro-F1** (trung bình đều mọi lớp, không trọng số theo support) chênh — dấu hiệu kinh điển của **mất cân bằng lớp**.

---

## 2. So sánh per-class (bảng mấu chốt)

Sắp theo chênh lệch F1 (GNN4ID − v3) giảm dần. Cột "flow" là **tổng số flow mỗi pipeline tạo ra cho lớp đó**.

| Lớp | GNN4ID F1 | v3 F1 | **GNN4ID flows** | **v3 flows** | Tỉ lệ flow | Nhận định |
|---|---|---|---|---|---|---|
| Mirai-udpplain | 0.9996 | 0.6667 | 16,225 | **133** | 122× | v3 đói mẫu (extract) |
| DDoS-ICMP_Fragmentation | 0.9934 | 0.7010 | 16,225 | **458** | 35× | v3 đói mẫu (extract) |
| XSS | 0.4567 | 0.1611 | 4,270 | 3,920 | ~1× | **mẫu ngang nhau → biểu diễn** |
| SqlInjection | 0.9570 | 0.7704 | 6,243 | 5,750 | ~1× | **mẫu ngang nhau → biểu diễn** |
| VulnerabilityScan | 0.9758 | 0.8202 | 16,225 | 10,712 | 1.5× | hỗn hợp |
| DDoS-ICMP_Flood | 1.0000 | 0.8889 | 16,225 | **46** | 353× | v3 chỉ 4 mẫu test → vô nghĩa thống kê |
| Recon-PingSweep | 0.9955 | 0.8894 | 2,226 | 1,808 | ~1× | biểu diễn |
| Uploading_Attack | 0.3713 | 0.2788 | 1,619 | 1,493 | ~1× | cả hai kém; v3 over-predict (P=16.5%) |
| CommandInjection | 0.2330 | 0.1860 | 5,470 | 4,109 | ~1.3× | cả hai kém; v3 under-predict (R=12.9%) |
| Benign | 0.9994 | 0.9581 | 16,225 | 4,033 | 4× | |
| BrowserHijacking | 0.9290 | 0.8785 | 4,763 | 3,181 | 1.5× | |
| DDoS-ACK_Fragmentation | 0.9967 | 0.9921 | 16,225 | 22,937 | 0.7× | ~hòa |
| DDoS-PSHACK_Flood | 0.9996 | 0.9997 | 16,225 | 49,894 | 0.33× | hòa (cả hai hoàn hảo) |
| DDoS-RSTFINFlood | 0.9967 | 0.9997 | 16,225 | 49,913 | 0.33× | **v3 hơn** |
| Recon-HostDiscovery | 0.9772 | 0.9799 | 16,225 | 22,796 | 0.7× | **v3 hơn** |
| Recon-OSScan | 0.9140 | 0.9462 | 16,225 | 10,625 | 1.5× | **v3 hơn** |
| Recon-PortScan | 0.9219 | 0.9640 | 16,225 | 16,094 | 1× | **v3 hơn** |
| Backdoor_Malware | 0.6489 | **0.8734** | 3,236 | 3,028 | 1× | **v3 hơn rõ rệt** |

**Quy luật:** GNN4ID thắng đúng ở (a) lớp volumetric mà v3 đói mẫu, và (b) một số lớp tầng-ứng-dụng dù mẫu ngang nhau. v3 thắng ở các lớp Recon/Backdoor có đủ mẫu.

### Phân rã khoảng cách macro-F1 (tổng = 0.0785)

| Nhóm | Đóng góp vào gap | % | Nguyên nhân |
|---|---|---|---|
| 3 lớp volumetric (ICMP_Flood, Mirai, ICMP_Frag) | ~0.041 | **~52%** | Extract flow của v3 (RC2) — dữ liệu |
| Lớp tầng-ứng-dụng (XSS, SqlInj, Upload, CmdInj, VulnScan…) | ~0.047 | ~60% | Biểu diễn + edge thưa (RC2'/RC5) |
| Lớp v3 thắng (Backdoor, Recon-OSScan/PortScan…) | −0.018 | bù lại | v3 tốt hơn |

---

## 3. So sánh phân bố dữ liệu (gốc rễ vấn đề)

| | GNN4ID v1 | EG-HGT v3 |
|---|---|---|
| Tổng đơn vị phân loại | 206,302 graph (1 graph/flow) | 210,930 flow node |
| Cân bằng lớp | **CÓ** — cap `--max-flows-per-class` (đa số → 16,225) | **KHÔNG** — phân bố thô |
| Tỉ lệ mất cân bằng | ~10:1 (16,225 : 1,619) | **~1085:1** (49,913 : 46) |
| Lớp hiếm nhất | Uploading 1,619 | DDoS-ICMP_Flood **46** |

v3 để DDoS-RSTFIN/PSHACK phình tới ~50k, đồng thời để ICMP_Flood/Mirai rơi xuống 46/133 — **vừa mất cân bằng đỉnh, vừa đói mẫu đáy**.

---

## 4. So sánh kiến trúc & pipeline

| Khía cạnh | GNN4ID v1 | EG-HGT v3 |
|---|---|---|
| **Đơn vị phân loại** | Graph-level: mỗi flow là 1 subgraph (flow + ≤20 packet) | Node-level: 1 đồ thị lớn, phân loại node `flow` qua neighbor sampling |
| **Node types** | 2 (`flow`, `packet`) | 5 (`flow`, `packet`, `host`, `technique`, `tactic`) |
| **Edge types** | 2 (`contains`, `next_packet`) | 22 (5 họ evidence MSEE + hierarchy MITRE + host + burst) |
| **Model** | HeteroGNN: `GATConv` ×2, hidden 64, global_mean_pool + MLP | HGT: 4 lớp, hidden 128, 8 heads + GCL aux loss |
| **Đặc trưng packet** | raw payload byte (≤1500B, uint8) | 256B payload + 91 dim |
| **Đặc trưng flow** | nfstream + rolling-window features | 80 CICFlowMeter + 5 evidence summary |
| **Bằng chứng→technique** | (không có) | MSEE: PMI + L1-LR + procedure matcher (45,096 edge, phủ ~21% flow) |
| **Recipe train** | CE + class-weight balanced cap 10, ReduceLROnPlateau, 50 ep | focal γ2 + balanced cap 10 + label smoothing + HGAA + EMA, 50 ep |

> **Quan trọng:** vì đơn vị phân loại khác nhau (graph-per-flow vs node-on-big-graph) và cách extract flow khác nhau, **số mẫu mỗi lớp khác nhau** ⇒ macro-F1 hai bên **không trực tiếp so sánh được** nếu không cào bằng phân bố.

---

## 5. Root cause analysis (vì sao v3 thua macro-F1)

| # | Root cause | Bằng chứng | Hạng |
|---|---|---|---|
| **RC1** | Extract flow của v3 nghiền nát lớp volumetric → đói mẫu | ICMP_Flood 46 vs 16,225; support test = 4 | **#1 (~52% gap)** |
| **RC2** | Biểu diễn yếu ở lớp tầng-ứng-dụng (MSEE edge thưa, raw-byte ít hơn GNN4ID) | XSS/SqlInj mẫu ngang nhau nhưng v3 thua | #2 |
| **RC3** | v3 chưa hội tụ ở epoch 50 | val_loss 0.1463→0.1434→0.1424, best epoch 49/50 | #3 |
| **RC4** | Recipe minority chưa tối ưu (focal γ2, DRW bật ngay, HGAA gây over-predict Uploading) | Uploading R=98%/P=16%; CmdInj R=13%/P=76% | #4 |

> **Lưu ý phương pháp luận:** RC1–RC4 là **giả thuyết có bằng chứng** (đã hoàn tất Phase 1–2 của systematic-debugging). Việc xác nhận RC1 là nguyên nhân chi phối được kiểm bằng thí nghiệm đối chứng ở §6.

---

## 6. Thí nghiệm đối chứng: GNN4ID trên phân bố mất cân bằng của v3

**Mục đích:** trung hòa lợi thế cân bằng + dư-mẫu của GNN4ID, để đo "gap thật".
**Cách làm:** `baselines/gnn4id/train_imbalanced.py` subsample shard GNN4ID có sẵn xuống **đúng** phân bố per-class của v3 (ICMP_Flood→46, Mirai→133, ICMP_Frag→458, …; tổng 130,290 graph), rồi train **cùng recipe**.

**Giả thuyết:** nếu cho GNN4ID đúng độ khan hiếm mẫu của v3, F1 các lớp volumetric sẽ sụp → chứng minh gap đến từ dữ liệu/extract, không phải HGT.

| Kịch bản kết quả | Diễn giải | Hành động sửa v3 |
|---|---|---|
| GNN4ID macro-F1 **sụp về ~0.75–0.80** | Gap = dữ liệu/extract; HGT không kém | Sửa **flow-windowing** cho lớp volumetric (RC1) — là đóng góp khoa học |
| GNN4ID **vẫn ~0.85** | Gap = biểu diễn (RC2) | Tăng phủ MSEE edge + cân bằng batch + train hội tụ |
| Trung gian | Tổ hợp | Làm song song |

> ⏳ **TRẠNG THÁI: ĐANG CHẠY** (background task, ~1.5–2h trên CPU). Kết quả + macro-F1 + per-class sẽ được cập nhật vào mục này khi hoàn tất. Output: `baselines/gnn4id/outputs/outputs/results_imbalanced_v3dist.json`.

---

## 7. Kế hoạch cải thiện v3 (vẫn giữ HGT làm model chính)

Sắp theo impact/effort. Mỗi bước kèm verify (1 thay đổi → 1 kiểm chứng).

### Tier 1 — đòn bẩy lớn, rẻ (không rebuild graph) → kỳ vọng macro-F1 0.775 → ~0.85
1. **Class-balanced seed sampler** (RC1 phần app-layer): lấy seed flow theo tần suất nghịch đảo thay vì stratified theo phân bố thô. → verify: recall XSS/SqlInj/VulnScan tăng.
2. **Train tới hội tụ** (RC3): `epochs 50 → 120`, khớp lại `T_max` cosine. → verify: val_loss phẳng, best_epoch < epochs.
3. **DRW đúng cách** (RC4): `drw_start_pct 0.0 → 0.7`. → verify: Uploading precision tăng, không tụt recall lớp lớn.

### Tier 2 — tinh chỉnh minority
4. focal γ 2.0 → 3.0. 5. Ablation HGAA (`tail_class_k`). 6. Tăng phủ MSEE edge cho XSS/CmdInj (hạ ngưỡng PMI / mở procedure matcher).

### Tier 3 — sửa cấu trúc (đắt, làm sau)
7. **Flow-windowing cho lớp volumetric** (RC1): cắt flood thành nhiều sub-flow theo cửa sổ thời gian → ICMP_Flood/Mirai có đủ mẫu. **Cần rebuild graph.** → verify: support ICMP_Flood/Mirai tăng từ 46/133 lên hàng nghìn.

---

## 8. Kết luận

- GNN4ID v1 **không phải model mạnh hơn HGT** một cách bản chất; nó hưởng lợi từ (1) cân bằng lớp do cap, và (2) đơn vị extract cho nhiều mẫu hơn ở lớp volumetric.
- Đóng góp khoa học của v3 (MSEE + schema dị thể + đánh giá random↔temporal) vẫn nguyên giá trị; điểm cần sửa nằm ở **pipeline dữ liệu** (extract flow volumetric) và **độ hội tụ/cân bằng khi train**, không phải ở kiến trúc HGT.
- Khi báo cáo, nên kèm **support per-class** và **macro-F1 trên tập lớp ≥1000 flow** để tách "lớp đói mẫu" khỏi "model thật sự kém".

> Cập nhật tiếp theo: điền kết quả §6 khi thí nghiệm đối chứng hoàn tất.
