# Phương pháp EACS — Evidence-Anchored Candidate-set Self-relabeling (báo cáo chi tiết, 2026-06-17)

Tài liệu này đặc tả đầy đủ **EACS** — bộ điều khiển tự-gán-lại-nhãn (self-relabel)
chống nhiễu nhãn trong khi huấn luyện HGT: động cơ, cơ sở lý thuyết, thuật toán
(công thức từng bước), neo bằng chứng (anchor), hàm loss, tích hợp trainer, cách
đo, và tính phòng thủ khoa học. Mục tiêu: ai đọc xong cũng tái lập được và bảo vệ
được trước hội đồng.

Tài liệu đi kèm:
- [2026-06-13-clean-grading-methodology.md](2026-06-13-clean-grading-methodology.md) — thang **clean** (answer key eval-only) mà EACS được chấm trên đó.
- [2026-06-13-eacs-vs-baselines.md](2026-06-13-eacs-vs-baselines.md) — bảng kết quả fair-comparison.
- `docs/superpowers/specs/2026-06-11-eacs-noise-robust-design.md` — design doc kỹ thuật (đã APPROVED).

---

## 0. TL;DR

- **EACS** train trên **nhãn nhiễu gốc** (per-pcap), một model, một lần chạy online,
  **không gán lại nhãn thủ công**. Model tự phát hiện và tự sửa nhiễu nhãn ở 4 lớp
  web-attack, tốt dần mỗi epoch.
- Mỗi flow nghi ngờ (label ∈ web-attack, **không** có bằng chứng MITRE khớp) chỉ
  được di chuyển trong **tập ứng viên 2 phần `{nhãn gốc, Benign}`** — không bao giờ
  sang lớp thứ ba. Flow neo (anchor — tấn công thật có bằng chứng) và mọi lớp khác
  (≈97% dữ liệu) **bất biến**.
- Đây là một thể hiện của **học với nhãn nhiễu (LNL)** theo giao thức chuẩn
  *train-on-noisy / evaluate-on-clean*, ghép với **disambiguation theo tập nhãn ứng
  viên (partial-label)** và **soft-relabel** kiểu ICGNN, **giới hạn 2 chiều**.
- Bằng chứng đóng góp: **dấu của (noisy − clean) đảo chiều** — chỉ EACS đạt âm
  (−0.135), giảm **~97% báo động giả** web-attack ở cùng recall (xem report kết quả).

---

## 1. Vấn đề — nhiễu nhãn bất đối xứng và vì sao cơ chế cũ thất bại

### 1.1. Nguồn nhiễu

CIC-IoT-2023 gán nhãn **theo từng capture** (Neto et al., 2023): toàn bộ traffic
trong một thí nghiệm tấn công bị dán *một* nhãn. Với DDoS/Recon/Mirai điều này đúng;
với 4 lớp web-attack (CommandInjection / XSS / SqlInjection / Uploading_Attack —
thực thi qua DVWA) thì ~95% capture là traffic nền IoT→cloud vô tội bị dán nhầm là
tấn công. Đây là **nhiễu nhãn bất đối xứng, phụ thuộc-instance**: chỉ tồn tại ở chiều
"attack-class thực ra là Benign", tập trung ở 4 lớp web. Chi tiết: [clean-grading-methodology §1](2026-06-13-clean-grading-methodology.md).

### 1.2. Vì sao cơ chế tiền nhiệm (EPC/EM soft-relabel) hỏng — đo được

Lần chạy EPC/EM (`outputs/nr_full.log`, 50 ep) đạt val_macro_f1 0.852 so với baseline
0.857 cùng epoch — và các lớp nó định cứu lại **tệ hơn** (CommandInjection
0.402→0.324, Uploading 0.343→0.318). Hai nguyên nhân cấu trúc:

1. **Tập gradable bị đảo ngược.** Soft-relabel bị giới hạn vào flow CÓ bằng chứng
   MITRE. Nhưng nhiễu là flow nền **không** có bằng chứng → ứng viên nhiễu thật giữ
   `beta=1` (tin nhãn tuyệt đối, không bao giờ relabel), trong khi flow có bằng chứng
   (đa số là **tấn công thật, nhãn sạch**) lại là thứ duy nhất bị EM phán xử. EM
   2-thành-phần luôn tách input thành 2 cụm kể cả khi phân phối là đơn-đỉnh-sạch, nên
   một phần tấn công thật bị soft-relabel về phía dự đoán (bị nền chi phối) của model.
   Cơ chế xói mòn đúng tín hiệu nó định bảo vệ.
2. **Mù metric.** Nhãn val/test là **cùng** nhãn nhiễu per-pcap như train. Một model
   học đúng "flow nền này là Benign" bị **phạt** trên val. Đo trên nhãn nhiễu, một bộ
   học chống-nhiễu hoàn hảo lại điểm *thấp hơn* một bộ học overfit-nhiễu. Giao thức
   LNL chuẩn là **train-on-noisy / evaluate-on-clean** (Northcutt et al., 2021;
   NoisyGL, 2024; BeGIN, 2025) — không có nó, không cơ chế nào thể hiện được gain.

> Hệ quả thiết kế: (a) đặt đúng **suspect = flow nền không bằng chứng**, (b) bổ sung
> thang **clean** eval-only, (c) thay EM mong manh bằng **tập ứng viên 2-chiều** an
> toàn hơn.

---

## 2. Nền tảng — giao thức LNL và họ phương pháp kế thừa

EACS đứng trên ba dòng nghiên cứu, **mỗi dòng được dùng có chọn lọc**:

| Dòng | Vai trò trong EACS | Tham chiếu |
|---|---|---|
| **LNL clean-test protocol** | train nhãn nhiễu, *grade* trên answer key sạch (eval-only) | Northcutt et al. 2021; NoisyGL 2024; BeGIN 2025 |
| **Soft-relabel / pseudo-target** | target = β·onehot(y) + (1−β)·q | ICGNN (soft-relabel template); Mean Teacher (EMA target) |
| **Partial-label disambiguation** | hạn chế relabel vào **tập ứng viên** `{y, Benign}` | PiCO / partial-label learning |

EACS = **giao điểm có kiểm soát** của ba dòng: soft-relabel kiểu ICGNN nhưng
*giới hạn vào tập ứng viên 2 phần* kiểu partial-label, *neo* bằng bằng chứng MITRE,
và *chấm* theo giao thức LNL clean-test. Phần "Out of scope" cố ý loại
**co-teaching / mạng thứ hai** (đơn-model, đơn-run) để giữ chi phí và tính tái lập.

---

## 3. Cơ chế EACS

Module: [noise_consensus.py](../../src/graphslm_ids/offline/training/noise_consensus.py)
(`EACSController`, `build_eacs_controller`, `neighbor_consensus`, `EMAConsensusBuffer`).

### 3.1. Ba nhóm flow (precompute từ graph + config)

| Nhóm | Định nghĩa | Cách xử lý |
|---|---|---|
| **Anchor** | label ∈ `suspect_classes` **VÀ** có bằng chứng khớp (mask procedure) | `β=1` luôn — dạy model mẫu tấn công thật |
| **Suspect** | label ∈ `suspect_classes` **VÀ** không có bằng chứng | tập ứng viên `{y, Benign}`; model tự disambiguate online |
| **Untouched** | mọi lớp khác | `β=1` luôn — nhãn per-pcap vốn đúng (DDoS/Recon/Mirai) |

`suspect_classes` mặc định = 4 lớp web mà nghiên cứu cô lập đã chứng minh bị nhiễu
(`flow_attack_labeler.WEB_ATTACK_CLASSES`). Mã dựng nhóm
([build_eacs_controller](../../src/graphslm_ids/offline/training/noise_consensus.py#L552)):

```python
suspect_mask = in_suspect_class & ~has_matching_ev
anchor_mask  = in_suspect_class &  has_matching_ev
benign_class_id = label_mapping["Benign"]
```

### 3.2. Công thức soft target (sau warmup)

Với mỗi epoch sau `warmup_epochs` (=5), cho flow suspect trong batch
([soft_targets](../../src/graphslm_ids/offline/training/noise_consensus.py#L502)):

```text
p        = softmax(class_logits)                 # (S, C) niềm tin của model epoch này
p_y, p_b = p[y], p[Benign]
β_raw    = p_y / (p_y + p_b)                      # disambiguation 2 chiều
cons     = neighbor_consensus(p, flow–flow edges) # support của y trong neighbor
β_raw    = β_raw^λ · cons^(1−λ)                    # λ = lambda_disambig = 0.7 (geometric blend)
β        = EMA(β_raw)                              # buffer per-flow, decay 0.9, init 1.0
target   = β·onehot(y) + (1−β)·onehot(Benign)
```

- **Warmup** (`epoch ≤ warmup_epochs`): trả về `onehot(y)` cho mọi flow → học cấu
  trúc cơ bản từ tất cả nhãn trước khi can thiệp.
- **Non-suspect**: luôn `onehot(y)` (`targets = where(suspect, targets, onehot)`).
- `neighbor_consensus` ([dòng 175](../../src/graphslm_ids/offline/training/noise_consensus.py#L175)):
  `consensus_i = mean_{j ∈ N(i)} p[j, y_i]` — neighbor (cạnh flow–flow `burst_neighbor`)
  có ủng hộ chính nhãn `y` của flow i không. Đây là tín hiệu SUPPORTING, làm mượt β.
- `λ=0.7` ưu tiên niềm tin của model (evidence) hơn consensus; geometric blend giữ
  β ∈ [0,1].

### 3.3. Neo bằng chứng — anchor mask v2 (điểm mấu chốt)

Định nghĩa "có bằng chứng khớp" quyết định toàn bộ chất lượng. Hai phiên bản:

| | Quy tắc anchor | #anchored | anchor precision | noise-AUC |
|---|---|---|---|---|
| **v1** | MSEE evidence weight > 0 | 6.439 | 0.16 | 0.609 |
| **v2** | HTTP-request + MITRE procedure literal | 1.272 | **0.952** | **0.760** |

v1 neo nhầm 5.406 flow nền vào nhãn tấn công cứng (precision 16%), dìm ~1k tấn công
thật theo tỉ lệ 5:1. v2 neo trên **so khớp procedure-literal precision cao** (MSEE
nguồn 2, train-legitimate) → phục hồi 95% anchor precision, nâng mọi số downstream.
Mask v2 dựng bằng [extract_eacs_anchor_mask.py](../../scripts/tools/extract_eacs_anchor_mask.py)
và nạp qua `noise_robust.anchor_mask_npy`; nó **override** anchor mặc định
(evidence-weight>0) trong `build_eacs_controller`.

### 3.4. Vòng tự cải thiện + chặn confirmation bias

Warmup dạy cấu trúc cơ bản. **Anchor** (tấn công thật có bằng chứng) tiếp tục dạy
mẫu tấn công thật với trọng số đầy đủ. Khi model học được, `p[Benign]` của flow nền
suspect tăng → β giảm → soft target dịch về Benign → gradient mâu thuẫn trên lớp
tấn công co lại → mẫu tấn công sạch hơn → suspect tách nhanh hơn.

Sụp đổ do **confirmation bias** (Arazo et al., 2020) bị **chặn cấu trúc**: một
suspect chỉ di chuyển giữa **nhãn của chính nó và Benign**, không bao giờ sang lớp
thứ ba; anchor + untouched (≈97% flow) là ground truth bất biến. Đây chính là lý do
giới hạn tập ứng viên 2-chiều (partial-label) thay vì self-labeling mở.

---

## 4. Hàm loss

Giữ **nguyên công thức focal/CE của baseline**, chỉ thay hard label bằng soft target:

```text
p_t  = Σ_c target_c · p_c          # focal factor tính trên soft target
loss = focal(p_t; γ) với alpha-weight per-class + label smoothing
```

- Khi `β=1` (warmup, anchor, untouched) công thức **rút gọn đúng bằng** focal
  baseline bit-for-bit ([soft-target focal](../../src/graphslm_ids/offline/training/noise_consensus.py#L415)).
  Đây là điểm sửa quan trọng so với EPC/EM (vốn dùng `p_t` của *hard* label kể cả
  khi target là soft → khuếch đại loss sai chiều).
- Tham số (config `eg_hgt_v6_ob_eacs_v2.yaml`): `focal_gamma=1.5`,
  `label_smoothing=0.05`, `class_weight=balanced` theo **effective number**
  (Cui et al., 2019: `class_weight_method=cb`, `cb_beta=0.985`, `cap=4.0`).
- Đã **bỏ** (so với EPC/EM): family head + `0.3·family_supervision_loss`, EM
  2-thành-phần per-batch, và EPC. Tập ứng viên thay tất cả bằng một primitive an toàn.

---

## 5. Tích hợp trainer + cấu hình

Module: [train_hgt_flow_classifier.py](../../src/graphslm_ids/offline/training/train_hgt_flow_classifier.py)

- `noise_robust.mode: eacs` chọn `EACSController`; đường EPC/EM đã bị xóa (đo được là
  có hại) — mọi mode khác raise lỗi ([dòng 1934](../../src/graphslm_ids/offline/training/train_hgt_flow_classifier.py#L1934)).
- Config-gated, mặc định OFF → run baseline **byte-identical**.
- `train.clean_eval_labels` (path): khi set, mỗi val pass tính thêm
  `val_macro_f1_clean` (cùng prediction, answer key sạch); `train.monitor` có thể
  chọn nó để checkpoint.

```yaml
train:
  monitor: val_macro_f1_clean
  clean_eval_labels: outputs/v3_ob/clean_eval_labels.npy   # eval-only
  loss_type: focal
  focal_gamma: 1.5
  label_smoothing: 0.05
  class_weight_method: cb
  cb_beta: 0.985
  noise_robust:
    enabled: true
    mode: eacs
    warmup_epochs: 5
    ema_decay: 0.9
    lambda_disambig: 0.7
    anchor_mask_npy: outputs/v3_ob/eacs_anchor_mask.npy
    suspect_classes: [CommandInjection, XSS, SqlInjection, Uploading_Attack]
```

---

## 6. Đo lường (không bay mù)

- **Per-epoch:** `[eacs] epoch=K suspects_seen=N mean_beta=… relabeled(beta<0.5)=M per_class={…}`.
- **Cuối train:** ROC-AUC của `(1−β_final)` so với oracle nhiễu
  (`clean_eval_labels != flow_y`) trên các flow suspect **đã thấy khi train** →
  `outputs/<run>/eacs_noise_detection.json`. Đây là số "model tự khám phá ra nhiễu".
- **Hai đường cong** `val_macro_f1` (nhãn gốc — fair so GNN4ID) và
  `val_macro_f1_clean` (answer key) trong history dump.
- **Tiêu chí thành công:** `val_macro_f1_clean ≥ 0.90` (chính), noise-detection
  ROC-AUC ≥ 0.75 (phụ). Kết quả thực: VAL clean 0.8856 raw → 0.9269 calibrated;
  AUC 0.760. Xem [eacs-vs-baselines](2026-06-13-eacs-vs-baselines.md).

---

## 7. Tính trung thực & phòng thủ

- **Answer key clean KHÔNG bao giờ là input của train** — chỉ là khóa chấm, đúng
  giao thức LNL chuẩn (train noisy / eval clean).
- **Hai nguồn khác nhau:** bằng chứng suspect-set (MSEE: PMI + L1-LR + procedure
  match) dùng *trong* train; oracle eval (chữ ký HTTP thủ công trong
  `flow_attack_labeler`) chỉ *eval-only*. Phần chồng lấn được công bố trong luận văn.
- **Metric nhãn gốc vẫn được báo cáo** → so sánh GNN4ID giữ nguyên công bằng.
- **Soft relabel bị giới hạn** vào `{nhãn gốc, Benign}` cho đúng 4 lớp đã liệt kê,
  có nghiên cứu cô lập §3d hậu thuẫn — không self-labeling mở.

---

## 8. Tái lập

```bash
# (1) Answer key clean (eval-only) + (2) anchor mask procedure (cần pcap, CPU)
python scripts/tools/extract_clean_eval_labels.py \
  --graph-meta outputs/v3_ob/graph.meta.json --raw-root data/raw \
  --out-npy outputs/v3_ob/clean_eval_labels.npy --out-audit outputs/v3_ob/clean_eval_labels.audit.json
python scripts/tools/extract_eacs_anchor_mask.py \
  --graph-meta outputs/v3_ob/graph.meta.json --raw-root data/raw \
  --out-npy outputs/v3_ob/eacs_anchor_mask.npy --out-audit /tmp/anchor.json

# (3) Train HGT + EACS v2
python -m graphslm_ids.offline.training.train_hgt_flow_classifier \
  --config configs/eg_hgt_v6_ob_eacs_v2.yaml --device cuda

# (4) Chấm noisy + clean (+ calibration trên clean val)
python scripts/eval/calibrate_thresholds.py \
  --config configs/eg_hgt_v6_ob_eacs_v2.yaml \
  --checkpoint outputs/v3_ob_eacs_v2/hgt_flow_best.pt \
  --training-summary outputs/v3_ob_eacs_v2/training_summary.json \
  --clean-labels outputs/v3_ob/clean_eval_labels.npy \
  --out outputs/v3_ob_eacs_v2/confusion_calibrated.json --device cuda
```

Unit test khóa bất biến: mask suspect/anchor trên toy artifact; toán β 2-chiều
(gồm biên); soft-target == focal baseline khi β=1 (exact); consensus blend.

---

## 9. Giới hạn đã biết (trung thực)

- **Support test nhỏ** ở 3/4 lớp web sau khi lọc (XSS n=5, CmdInj n=7, Upload n=2):
  một FP làm F1 dao động 0.1–0.3 — nhiễu *của metric*, không phải model (recall gộp
  0.991). Re-split 60/20/20 theo attack-key sẽ ổn định.
- **Chồng lấn nguồn** suspect-evidence (MSEE) và oracle (chữ ký HTTP) — đã công bố;
  oracle eval-only nên không rò vào train.
- **Đơn model / đơn run** (cố ý, YAGNI): không co-teaching, không retrain trên
  artifact tự-làm-sạch (two-stage), không graph rewiring.
- **Chỉ 4 lớp web** được coi là suspect; nếu lớp non-web cũng nhiễu per-pcap, EACS
  không bắt — nhưng bằng chứng tách cụm cho thấy chúng sạch.

---

## 10. Tài liệu tham khảo (các bài báo được dùng)

> Ghi chú trung thực: các mã định danh arXiv dưới đây được **chép đúng từ mục
> References của design doc nội bộ** (`2026-06-11-eacs-noise-robust-design.md §9`)
> và `clean-grading-methodology.md`. Khi đưa vào luận văn, hãy đối chiếu lại tên
> tác giả/năm/venue chính thức của từng mục trước khi trích.

### 10.1. Học với nhãn nhiễu & giao thức clean-test (LNL)

1. **Friends and Foes in Learning from Noisy Labels** — arXiv:2103.. (Giao
   thức train-noisy / evaluate-clean mà EACS tuân theo.)15055
2. **NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label
   Noise** — NeurIPS 2024, arXiv:2406.04299.
3. **BeGIN: Instance-dependent Graph Label Noise Benchmark** — arXiv:2506.12468
   (2025).
4. **SilentSentinel: graph-based sample selection & purification for NIDS label
   noise** — *Scientific Reports*, 2026 (s41598-026-45988-y). (Nhiễu nhãn trong IDS.)

### 10.2. Soft-relabel & disambiguation theo tập ứng viên (partial-label)

5. **ICGNN soft-relabel** — arXiv:2601.17469. (Template `β·onehot(y) + (1−β)·q`;
   EACS giới hạn `q` vào tập ứng viên 2 phần.)
6. **PiCO: Contrastive Label Disambiguation for Partial Label Learning** —
   Wang et al., ICLR 2022 (dòng partial-label / candidate-set restriction).
7. **Mean Teacher** (Tarvainen & Valpola) — NeurIPS 2017. (EMA của target/buffer β.)
8. **Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning** —
   Arazo et al., IJCNN 2020. (Cơ sở cho lập luận chặn confirmation bias.)

### 10.3. Hàm loss & hiệu chỉnh quyết định (kế thừa trong recipe)

9. **Focal Loss for Dense Object Detection** — Lin et al., ICCV 2017
   (`loss_type=focal`, `γ=1.5`).
10. **Class-Balanced Loss Based on Effective Number of Samples** — Cui et al.,
    CVPR 2019 (`class_weight_method=cb`, `cb_beta=0.985`).
11. **Long-tail learning via Logit Adjustment** — Menon et al., ICLR 2021
    (additive logit bias dùng ở bước calibration; xem clean-grading §5.3).
12. Lipton et al., 2014 — additive logit bias / label-shift threshold (ghi theo
    clean-grading §5.3).

### 10.4. Dataset

13. **CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT
    Environment** — Neto, E.C.P. et al., *Sensors* 2023, 23(13), 5941.
    <https://www.mdpi.com/1424-8220/23/13/5941> (per-capture labeling — nguồn nhiễu).

---

## Phụ lục — file liên quan

| File | Vai trò |
|---|---|
| [noise_consensus.py](../../src/graphslm_ids/offline/training/noise_consensus.py) | `EACSController`, công thức β, neighbor consensus, EMA buffer |
| [train_hgt_flow_classifier.py](../../src/graphslm_ids/offline/training/train_hgt_flow_classifier.py) | tích hợp `noise_robust.mode=eacs`, soft-target focal, diagnostics |
| [extract_eacs_anchor_mask.py](../../scripts/tools/extract_eacs_anchor_mask.py) | dựng anchor mask procedure-literal (v2, 95.2%) |
| [extract_clean_eval_labels.py](../../scripts/tools/extract_clean_eval_labels.py) | dựng answer key clean (eval-only) |
| `configs/eg_hgt_v6_ob_eacs_v2.yaml` | config EACS v2 chính |
| `docs/superpowers/specs/2026-06-11-eacs-noise-robust-design.md` | design doc kỹ thuật gốc |
