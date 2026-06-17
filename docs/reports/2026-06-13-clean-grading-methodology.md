# Phương pháp gán lại nhãn sạch & chấm điểm "clean" — báo cáo chi tiết (2026-06-13)

Tài liệu này đặc tả đầy đủ thang đo **clean**: tại sao cần nó, dữ liệu gốc nó
dựa vào, thuật toán làm sạch (gán lại nhãn) theo chữ ký, cách dựng answer key
căn theo graph, và cách chấm điểm trên TEST/VAL. Mục tiêu: ai đọc xong cũng
tái lập được đúng từng bước, và bảo vệ được tính hợp lệ khoa học của giao thức.

---

## 0. TL;DR

- **clean = đáp án chuẩn (answer key) đã lọc nhiễu nhãn**, dùng để *đánh giá*
  theo giao thức **Learning-with-Noisy-Labels (LNL): train trên nhãn nhiễu,
  evaluate trên nhãn sạch**.
- **clean là eval-only**: nó KHÔNG bao giờ đi vào loss, gradient, hay việc chọn
  checkpoint của model. Mọi model (EACS, HGT de-inflated, GNN4ID, XGBoost) đều
  train trên nhãn noisy y hệt, chỉ khác lúc *chấm*.
- **Làm sạch dựa trên chữ ký HTTP**, chỉ áp dụng cho 4 lớp web-attack
  (CommandInjection / XSS / SqlInjection / Uploading_Attack); 14 lớp còn lại
  giữ nguyên.
- Trên artifact `v3_ob` (18 lớp, 211.851 flow): **14.276 flow** đổi nhãn so với
  noisy — đây chính là lượng nhiễu nhãn của CIC-IoT-2023 được định lượng.

---

## 1. Vì sao cần "clean" — nguồn gốc của nhiễu nhãn

CIC-IoT-2023 gán nhãn theo **từng file pcap** (`infer_label_from_path`): mọi flow
trong `CommandInjection.pcap` được gán nhãn `CommandInjection`, v.v. Điều này
ĐÚNG với các lớp DDoS / Recon / Mirai / Benign — cả capture đúng là một chiến
dịch tấn công duy nhất.

Nhưng với **4 lớp web-attack**, cách gán này SAI nặng. Các pcap đó được thu trên
một testbed có web app DVWA, và:

- Tấn công thật là **HTTP request plaintext** tới web app (ví dụ `GET /sqli?id=1' OR '1'='1`).
- ~95% lưu lượng còn lại là **traffic nền IoT→cloud bình thường** (telemetry,
  keep-alive, TLS tới dịch vụ đám mây) — vô tội nhưng bị gán nhãn là tấn công.

Hệ quả (xem `docs/reports/2026-06-06-web-attack-encryption-ceiling.md`):
độ tách cụm web giảm còn ~0.51. Khi cô lập đúng flow tấn công, độ tách lên ~0.95.
Tức là nhãn — không phải mô hình — mới là trần thực sự của 4 lớp này.

> **Đây là nhiễu nhãn (label noise) bất đối xứng**: nhiễu chỉ tồn tại ở chiều
> "attack-class thực ra là Benign", tập trung ở 4 lớp web.

### 1.1. Chính tác giả CIC-IoT-2023 xác nhận nguồn nhiễu

Nhiễu này KHÔNG phải suy đoán của chúng tôi — nó đến **thẳng từ phương pháp gán
nhãn mà nhóm tác giả mô tả trong paper gốc** (Neto et al., *Sensors* 2023):

> *"For each attack, a different experiment is performed targeting all applicable
> devices … for each attack executed, the **entire traffic captured is labeled as
> belonging to that particular attack**."* — §3.3.2 (Methodology / Labeling)

Đây chính xác là **per-capture labeling**: toàn bộ traffic bắt trong một thí
nghiệm tấn công được dán *một* nhãn tấn công, kể cả lưu lượng nền không liên
quan chạy đồng thời. Với DDoS/Recon/Mirai điều này vô hại (cả thí nghiệm đúng là
tấn công); với web-attack thì sinh nhiễu, vì:

- 4 lớp web đều được thực thi bằng **DVWA (Damn Vulnerable Web Application)** —
  Bảng 2 của paper ghi rõ tool tấn công web-based là DVWA. Tấn công thật là HTTP
  request tới web app dễ tổn thương; phần còn lại của capture là traffic nền của
  testbed 105 thiết bị (§3.2).
- Định nghĩa 4 lớp của tác giả (§3.3.4 — *Web-based attacks*) đều là tấn công
  **tầng ứng dụng HTTP**, khẳng định dấu hiệu tấn công nằm trong nội dung HTTP
  (đúng cơ sở để chữ ký của ta hoạt động):
  - **SQL Injection** — *"an attack that targets web applications by injecting
    malicious SQL code into the application's input fields."*
  - **Command Injection** — *"an attack that targets web applications by injecting
    malicious commands into an input field with the ultimate goal of gaining
    unauthorized access to a system."*
  - **XSS** — *"allows an attacker to inject malicious code (e.g., a script) into
    a web page."*
  - **Uploading Attack** — *"targets a web application by exploiting
    vulnerabilities in the application's file upload functionality."*

Tóm lại: tác giả gán nhãn **theo thí nghiệm/capture**, không theo từng flow; và
attack thật là HTTP-tới-DVWA. Hai sự thật đó cộng lại = nhiễu nhãn ở 4 lớp web,
và cũng chính là lý do chữ ký HTTP (§3) tách được attack thật khỏi nền.

**Nguồn (open-access):**
- Neto, E.C.P.; Dadkhah, S.; Ferreira, R.; Shoeleh, F.; Ghorbani, A.A. et al.
  *"CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT
  Environment."* **Sensors 2023, 23(13), 5941.**
  - MDPI (HTML/PDF): <https://www.mdpi.com/1424-8220/23/13/5941>
  - PMC (toàn văn miễn phí): <https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/>
  - Preprint: <https://www.preprints.org/manuscript/202305.0443>
  - Trang dataset chính thức (UNB CIC): <https://www.unb.ca/cic/datasets/iotdataset-2023.html>

> Lưu ý trung thực: bản thân paper **không** thừa nhận đây là "nhiễu" — họ coi
> per-capture labeling là thiết kế. Việc *định lượng* nó thành nhiễu nhãn và lọc
> bằng chữ ký HTTP là **đóng góp của đề tài này**, không phải tuyên bố của tác giả gốc.

---

## 2. Dữ liệu mà thang clean dựa vào

| Thành phần | Nguồn | Vai trò |
|---|---|---|
| pcap thô | `data/raw/14gb/<class>/*.pcap` | Để chạy lại bộ cô lập theo chữ ký |
| `graph.meta.json` | artifact đã build (`outputs/v3_ob/`) | Cung cấp `flow_id_order` + `label_mapping` để căn nhãn |
| `splits.json` (qua training summary) | artifact | `val_idx` / `test_idx` — clean dùng **đúng cùng split** với noisy |
| Chữ ký HTTP + STIX procedure | code + MITRE | Quy tắc quyết định flow nào là attack thật |

Điểm mấu chốt: clean **không phải** một tập dữ liệu khác. Nó là **cùng những
flow đó, cùng split đó**, chỉ thay **vector nhãn** từ noisy → đã lọc. Vì vậy so
sánh noisy vs clean là so sánh công bằng tuyệt đối (same model, same data,
same split — chỉ khác answer key).

---

## 3. Thuật toán làm sạch (gán lại nhãn) — theo chữ ký

Module: [flow_attack_labeler.py](../../src/graphslm_ids/offline/preprocessing/flow_attack_labeler.py)

### 3.1. Phạm vi — chỉ 4 lớp web

```python
WEB_ATTACK_CLASSES = {"CommandInjection", "XSS", "SqlInjection", "Uploading_Attack"}
```

Hàm `label_pcap_flows` trả về no-op cho mọi lớp khác
([dòng 87-88](../../src/graphslm_ids/offline/preprocessing/flow_attack_labeler.py#L87-L88)):
lớp ngoài tập này giữ nguyên `original_label`. Đây là quyết định thiết kế quan
trọng — ta KHÔNG đụng tới các lớp mà nhãn per-pcap vốn đúng, để tránh tự tạo ra
nhiễu mới.

### 3.2. Khóa flow chuẩn hóa (canonical key)

Để map giữa pcap và graph, dùng khóa 5-tuple **không phương hướng** (bidirectional),
bỏ nhãn và segment ([dòng 60-66](../../src/graphslm_ids/offline/preprocessing/flow_attack_labeler.py#L60-L66)):

```python
def _canon_key(src_ip, sport, dst_ip, dport, proto):
    a, b = f"{src_ip}:{sport}", f"{dst_ip}:{dport}"
    lo, hi = (a, b) if a <= b else (b, a)   # sắp xếp để A→B và B→A trùng khóa
    return f"{lo}|{hi}|{proto}"             # proto 0 = TCP
```

Khóa này khớp đúng core `lo|hi|proto` mà `flows.assign_flows` sinh ra, nên một
flow trong pcap ánh xạ 1-1 sang một flow node trong graph.

### 3.3. Quy tắc chữ ký — 2 tầng

Khớp substring trên HTTP request đã lowercase (request-line + headers + body),
[dòng 38-57](../../src/graphslm_ids/offline/preprocessing/flow_attack_labeler.py#L38-L57):

- **Tầng A — endpoint DVWA** (testbed-specific, precision rất cao):
  `/exec`, `/xss`, `/sqli`, `/upload`.
- **Tầng B — mẫu payload tấn công** (tổng quát hóa cho web app bất kỳ):
  - CommandInjection: `;cat`, `;ls`, `|sh`, `$(`, `` ` ``, `/bin/`, `/etc/passwd`, `whoami`, `ping -c`, `uname`…
  - XSS: `<script`, `%3cscript`, `onerror=`, `onload=`, `javascript:`, `alert(`, `<svg`, `<iframe`, `document.cookie`…
  - SqlInjection: `union select`, `or 1=1`, `' or '`, `%27or`, `information_schema`, `sleep(`, `concat(`, `--`, `/*`…
  - Uploading_Attack: `multipart/form-data`, `content-disposition`, `filename=`, `.php`, `.jsp`, `.asp`…

> Quyết định dựa trên **nội dung HTTP**, KHÔNG hardcode IP nạn nhân → quy tắc
> tổng quát hóa ngoài testbed này, đúng tinh thần "evidence-grounded".

### 3.4. Logic gán lại — "ANY packet matches ⇒ cả flow là attack"

`label_pcap_flows` quét pcap 2 vòng
([dòng 90-137](../../src/graphslm_ids/offline/preprocessing/flow_attack_labeler.py#L90-L137)):

1. Với mỗi packet TCP, lọc packet có HTTP method ở đầu payload
   (`_HTTP_METHOD` regex), kiểm tra `request_matches`.
2. Một flow vào tập `attack_flows` nếu **bất kỳ packet nào** của nó khớp chữ ký.
3. Sinh mapping: `key → original_label nếu key ∈ attack_flows, ngược lại → Benign`.

```python
mapping = {k: (original_label if k in attack_flows else BENIGN_LABEL)
           for k in all_flows}
```

Audit mỗi pcap: `{total_tcp_flows, true_attack_flows, relabeled_benign}`.

> Có một biến thể song song `relabel_packets_df` ([dòng 140-205](../../src/graphslm_ids/offline/preprocessing/flow_attack_labeler.py#L140-L205))
> làm cùng logic nhưng trên packets DataFrame (dùng trong pipeline preprocessing
> để làm sạch ngay từ đầu nếu muốn). Cùng chữ ký, cùng quy tắc "ANY packet".

---

## 4. Dựng answer key căn theo graph

Script: [extract_clean_eval_labels.py](../../scripts/tools/extract_clean_eval_labels.py)

Mục tiêu: biến tập attack-key (theo pcap) thành **vector nhãn `int64` căn chỉ
y hệt `flow_y`** của artifact.

### 4.1. Parse flow_id

`flow_id_order` trong meta có dạng `Label|lo|hi|proto#seg.dir`. Hai helper:

```python
label_of_flow_id("CmdInj|a:1|b:2|6#2.1")          -> "CmdInj"
canonical_key_of_flow_id("CmdInj|a:1|b:2|6#2.1")  -> "a:1|b:2|6"   # bỏ #seg.dir
```

Việc bỏ hậu tố `#seg.dir` đảm bảo nhiều segment của cùng 5-tuple **chia sẻ một
khóa** (test `test_segment_suffix_does_not_leak_into_key`).

### 4.2. Vòng gán nhãn

[`clean_labels_from_attack_keys`](../../scripts/tools/extract_clean_eval_labels.py#L48-L73):

```text
với mỗi flow i trong flow_id_order:
    name = lớp của flow
    nếu name KHÔNG thuộc 4 web-attack:   giữ nguyên nhãn        # pass-through
    nếu canonical_key ∈ attack_keys[name]: giữ nhãn attack       # tấn công thật
    ngược lại:                            -> Benign + ghi audit  # nền bị demote
```

Output:
- `clean_eval_labels.npy` — vector nhãn sạch (`int64`, dài = số flow).
- `clean_eval_labels.audit.json` — `{n_flows, n_demoted_to_benign,
  demoted_per_class, attack_keys_per_class}`.

Tính bất biến được khóa bằng unit test [test_clean_eval_labels.py](../../tests/test_clean_eval_labels.py):
giữ nhãn khi có key, demote khi không, lớp non-web bất biến, dtype int64.

---

## 5. Chấm điểm trên clean

Script: [calibrate_thresholds.py](../../scripts/eval/calibrate_thresholds.py) (cờ `--clean-labels`)

### 5.1. Luồng

1. Load `clean_eval_labels.npy`, cắt theo `val_idx`/`test_idx` — **cùng split**
   với chấm noisy ([dòng 236-238](../../scripts/eval/calibrate_thresholds.py#L236-L238)).
2. Lấy logits model trên VAL/TEST (`_dump_logits`, tái lập đúng F1 của trainer
   — có self-check).
3. **Chấm raw**: `argmax(logits)` so với clean key → `_grade`.
4. **Calibration trên clean val**: tune một vector bias cộng theo lớp
   `b` (length C) tối đa hóa macro-F1 **chỉ trên VAL clean**, rồi áp y nguyên
   sang TEST (`apply_bias`) — không peeking test.
5. Báo cáo cả raw lẫn calibrated cho TEST.

### 5.2. Hai metric

Hàm `_grade` ([dòng 244-262](../../scripts/eval/calibrate_thresholds.py#L244-L262)) trả về:

- **macro-F1** — trung bình F1 trên các lớp có support > 0 (`er._macro_f1`).
  Đây là số "headline" trong bảng so sánh.
- **web_binary** — metric quyết định: gộp 4 lớp web thành "có phải web-attack
  thật không?", tính TP/FP/FN/precision/recall/F1 trên clean key:

```python
true_bin = isin(true, web_ids)   # 4 lớp web theo answer key sạch
pred_bin = isin(pred, web_ids)   # model dự đoán là 1 trong 4 lớp web
```

Metric này là nơi giá trị thực sự lộ ra: mọi model đều *tìm thấy* tấn công
(recall ≈ 1.0); cái khác nhau là **số báo động giả (FP)**. EACS giảm FP ~97% ở
cùng recall so với XGBoost / HGT de-inflated.

### 5.3. Calibration là decision-rule, KHÔNG phải relabel

Quan trọng cho tính hợp lệ: calibration chỉ học **ngưỡng quyết định per-class**
(additive logit bias — Lipton 2014 / logit adjustment Menon ICLR 2021), tune
trên VAL áp sang TEST. Nó không retrain, không đổi nhãn train. Cùng giao thức
calibration này áp được cho mọi baseline emit logits ⇒ so sánh apples-to-apples.

---

## 6. Đọc kết quả — dấu của (noisy − clean) là đóng góp

| Dấu | Nghĩa | Model điển hình |
|---|---|---|
| **Dương** (noisy > clean) | Model điểm cao hơn khi chấm bằng nhãn SAI ⇒ **đã nhớ nhiễu nhãn** (campaign fingerprint) | XGBoost +0.237, GNN4ID +0.129, HGT de-inflated +0.130 |
| **Âm** (clean > noisy) | Model điểm cao hơn trên sự thật ⇒ **đã lọc nhiễu** | HGT + EACS v2 −0.135 |

Cùng backbone HGT, cùng data, cùng split — biến duy nhất thay đổi là bộ điều
khiển tự-relabel EACS. **Sự đảo dấu chính là đóng góp của đề tài.** XGBoost minh
họa rõ nhất: macro-F1 noisy = **1.0000** (trông như SOTA hoàn hảo) nhưng là kẻ
nhớ nhiễu tệ nhất — clean chỉ 0.7626, web precision sạch 0.070.

---

## 7. Tái lập

```bash
# (1) Dựng answer key sạch từ pcap + meta (chạy local, cần pcap)
python scripts/tools/extract_clean_eval_labels.py \
  --graph-meta outputs/v3_ob/graph.meta.json \
  --raw-root  data/raw/14gb \
  --out-npy   outputs/v3_ob/clean_eval_labels.npy \
  --out-audit outputs/v3_ob/clean_eval_labels.audit.json

# (2) Chấm một checkpoint trên cả noisy + clean (+ calibration trên clean val)
python scripts/eval/calibrate_thresholds.py \
  --config            configs/eg_hgt_v6_ob_eacs_v2.yaml \
  --checkpoint        outputs/v3_ob_eacs_v2/hgt_flow_best.pt \
  --training-summary  outputs/v3_ob_eacs_v2/training_summary.json \
  --clean-labels      outputs/v3_ob/clean_eval_labels.npy \
  --out               outputs/v3_ob_eacs_v2/confusion_calibrated.json --device cuda

# Baseline GNN4ID chấm trên cùng answer key sạch
python baselines/gnn4id/regrade_clean.py --device cuda
```

Kiểm tra bất biến: `python -m pytest tests/test_clean_eval_labels.py -q`

---

## 8. Giới hạn đã biết (trung thực)

- **Support nhỏ ở test**: sau khi lọc, XSS (n=5), CommandInjection (n=7),
  Uploading_Attack (n=2) có support test một chữ số — một FP làm F1 dao động
  0.1–0.3. Đây là nhiễu *của metric*, không phải của model (đối chiếu recall
  gộp 0.991). Một re-split 60/20/20 theo attack-key sẽ cho các lớp này đủ support.
- **Chữ ký có thể bỏ sót** biến thể tấn công mã hóa/obfuscated; nhưng vì quy tắc
  precision cao (95.2% anchor precision ở EACS v2), sai số nghiêng về phía
  *bảo thủ* (giữ ít attack hơn), không thổi phồng kết quả EACS.
- clean **không dùng** cho lớp non-web; nếu các lớp đó cũng có nhiễu per-pcap,
  thang này không bắt được — nhưng bằng chứng tách cụm cho thấy chúng sạch.

---

## Phụ lục — file liên quan

| File | Vai trò |
|---|---|
| [flow_attack_labeler.py](../../src/graphslm_ids/offline/preprocessing/flow_attack_labeler.py) | Cô lập attack-flow theo chữ ký HTTP |
| [extract_clean_eval_labels.py](../../scripts/tools/extract_clean_eval_labels.py) | Dựng answer key căn theo graph |
| [calibrate_thresholds.py](../../scripts/eval/calibrate_thresholds.py) | Chấm noisy + clean, web-binary, calibration |
| [test_clean_eval_labels.py](../../tests/test_clean_eval_labels.py) | Khóa tính bất biến của logic gán nhãn |
| [2026-06-13-eacs-vs-baselines.md](2026-06-13-eacs-vs-baselines.md) | Bảng kết quả fair-comparison dùng thang này |
| [2026-06-06-web-attack-encryption-ceiling.md](2026-06-06-web-attack-encryption-ceiling.md) | Chẩn đoán gốc của nhiễu nhãn web |
