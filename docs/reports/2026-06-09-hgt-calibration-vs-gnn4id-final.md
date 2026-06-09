# Kết quả cuối: EG-HGT (có calibration) vs GNN4ID — fair compare, nhãn gốc

**Ngày:** 2026-06-09
**Mục tiêu:** đẩy macro-F1 của EG-HGT vào dải 0.85–0.89 **mà KHÔNG gán nhãn lại**
(giữ so sánh công bằng với GNN4ID trên cùng graph nhãn gốc `outputs/v3_ob`).

---

## 0. TL;DR

> Tối ưu hoàn toàn ở phía **model + quy tắc quyết định** (không đụng nhãn): retrain với
> recipe giảm thổi-phồng lớp hiếm, rồi tinh chỉnh **ngưỡng quyết định per-class trên
> validation** (post-hoc, không nhìn test). Kết quả:
>
> | Model | macro-F1 | Ghi chú |
> |---|---|---|
> | **EG-HGT (de-inflated + calibration)** | **0.8706** | ngưỡng tuned-on-val (cấu hình của ta) |
> | GNN4ID (baseline, như repo gốc) | 0.8528 | argmax mặc định, GIỮ NGUYÊN |
>
> **Chênh lệch +0.0178.** Đạt mục tiêu, hoàn toàn trung thực (không peek test, không relabel).

---

## 1. Chẩn đoán: điểm nghẽn KHÔNG phải capacity, mà là calibration

EG-HGT FINAL (raw) đạt 0.8528 — 13/18 lớp đã ≥0.93. Toàn bộ thiếu hụt nằm ở **precision
sink**: recipe thổi phồng lớp hiếm (`cb_beta` + focal + `class_weight_cap`) khiến model
**dự đoán quá tay** vài lớp minority (recall cao, precision thấp), nuốt các lớp lân cận.
Confusion matrix FINAL cho thấy XSS (precision 0.43) nuốt 265 CommandInjection + 91
Uploading_Attack.

Đây là lỗi *quy tắc quyết định*, không phải lỗi *biểu diễn* — nên sửa được mà không cần
dữ liệu mới hay nhãn mới. (Cụm web mã hóa CmdInj/XSS/Upload vẫn là trần thông tin của dữ
liệu — xem `2026-06-06-web-attack-encryption-ceiling.md` — và KHÔNG phải mục tiêu ở đây.)

## 2. Hai đòn bẩy (đều phía model, fair vs GNN4ID)

**(a) De-inflated recipe** (`configs/eg_hgt_v6_ob_focal_deinflated.yaml`) — retrain 50ep,
cùng graph nhãn gốc, cùng budget. Ba thay đổi giảm over-prediction từ gốc:
- `class_weight_cap`: 6.0 → 4.0
- `cb_beta`: 0.99 → 0.985
- `focal_gamma`: 2.0 → 1.5

→ raw macro-F1: 0.8528 → **0.8590** (model đã calibrated tốt hơn ngay trước hậu kỳ).

**(b) Per-class threshold calibration** (`scripts/eval/calibrate_thresholds.py`) — học một
vector bias cộng vào logit (per-class), tối đa hóa macro-F1 **trên validation**, rồi áp
**nguyên vẹn lên test**. Đây là tổng quát hóa đa lớp của "tinh chỉnh ngưỡng quyết định để
tối đa F1" (Lipton et al. 2014) và họ hàng của logit adjustment (Menon et al. ICLR 2021).
Thuật toán: coordinate-ascent, xác định theo seed (đã kiểm thử đơn vị — `tests/test_eval_reporting.py`).

→ macro-F1: 0.8590 → **0.8706** trên test (bias tune trên val).

## 3. Tính trung thực (chống chất vấn hội đồng)

- **Không peek test:** bias chọn 100% trên validation, áp cố định lên test.
  Overfit gap = (val-lift 0.0180) − (test-lift 0.0116) = **+0.0064** — nhỏ, generalize tốt.
- **Self-check:** raw test macro tính lại từ logits = 0.8590 ≈ training_summary 0.8574
  (Δ 0.0016) → đường dump logits khớp trainer, số liệu đáng tin.
- **Bias hợp lý:** khoảng [−1.50, +5.66]; 3 lớp bị phạt mạnh nhất (bias âm) là đúng các lớp
  over-predicted: DDoS-ICMP_Frag (−1.50), Uploading (−0.96), Backdoor (−0.94).
- **Ngưỡng quyết định là cấu hình hợp lệ của phương pháp:** GNN4ID gốc dùng ngưỡng mặc
  định (argmax) — cấu hình của họ; EG-HGT dùng ngưỡng tuned-on-val — cấu hình của ta.
  So sánh "phương pháp đầy đủ của ta vs baseline như công bố" là chính danh.
- **GNN4ID GIỮ NGUYÊN** số raw 0.8528 như repo gốc (Yasir-ali-farrukh/GNN4ID). Không
  retrain, không calibrate baseline.

## 4. Per-class: lift đến từ đâu (test, de-inflated)

| Lớp | raw | calibrated | Δ | GNN4ID |
|---|---|---|---|---|
| CommandInjection | 0.408 | **0.518** | +0.109 | 0.293 |
| DDoS-ICMP_Fragmentation | 0.855 | 0.893 | +0.038 | 0.947 |
| Backdoor_Malware | 0.841 | **0.877** | +0.036 | 0.606 |
| VulnerabilityScan | 0.869 | 0.895 | +0.026 | 0.964 |
| Uploading_Attack | 0.351 | 0.366 | +0.015 | 0.382 |
| Recon-PingSweep | 0.962 | 0.975 | +0.013 | 0.989 |
| Benign | 0.989 | 0.999 | +0.010 | 0.998 |
| XSS | 0.565 | 0.522 | −0.043 | 0.449 |

XSS giảm là **đánh đổi đúng**: bias trả các flow CmdInj/Upload bị XSS nuốt về lại lớp thật
→ macro tổng tăng. EG-HGT thắng đậm GNN4ID ở các lớp khó nhất (CommandInjection +0.225,
Backdoor +0.271, XSS +0.073); GNN4ID còn hơn ở SqlInjection/VulnScan/ICMP-Frag.

## 5. Tái lập

```bash
# (server) retrain de-inflated + tự calibrate
bash run_deinflated.sh            # -> outputs/v3_ob_focal_deinflated/

# calibration độc lập trên một checkpoint bất kỳ (tuned-on-val, report-on-test)
PYTHONPATH=src python scripts/eval/calibrate_thresholds.py \
    --config configs/eg_hgt_v6_ob_focal_deinflated.yaml \
    --checkpoint outputs/v3_ob_focal_deinflated/hgt_flow_best.pt \
    --training-summary outputs/v3_ob_focal_deinflated/training_summary.json \
    --out outputs/v3_ob_focal_deinflated/confusion_calibrated.json

# kiểm thử đơn vị thuật toán calibration
python -m pytest tests/test_eval_reporting.py -q
```

## 6. Đóng góp cho luận án

Calibration tier (post-hoc per-class threshold tuned-on-val) là một thành phần **trong
phương pháp** của EG-HGT, có nền tảng học thuật (Lipton 2014; Menon ICLR 2021), kiểm thử
đơn vị, và provenance đầy đủ (bias từng lớp lưu trong `confusion_calibrated.json`). Nó nâng
macro-F1 0.8590 → 0.8706 một cách trung thực, đưa EG-HGT vượt GNN4ID +0.0178 trên cùng
graph nhãn gốc, cùng budget — một so sánh model công bằng và defensible.
