# Phương pháp đánh giá Smart-BOTH — gap random vs temporal (báo cáo chi tiết, 2026-06-17)

Tài liệu này đặc tả đầy đủ **giao thức đánh giá Smart-BOTH** — đóng góp **lớp 3**
trong khung novelty (xem CLAUDE.md): chạy *cùng một kiến trúc HGT* trên **hai cách
chia split** (random-stratified và temporal) rồi báo cáo **khoảng cách (gap) F1**
giữa chúng như một *finding* độc lập. Mục tiêu: vì sao gap là phát hiện chứ không
phải chỉ là con số phụ, đo nó thế nào, và diễn giải dấu/độ lớn ra sao.

Tài liệu đi kèm:
- [2026-06-13-eacs-vs-baselines.md](2026-06-13-eacs-vs-baselines.md) — bảng fair-comparison (random split) làm động cơ định lượng.
- [2026-06-13-clean-grading-methodology.md](2026-06-13-clean-grading-methodology.md) — trục đo *thứ hai* (noisy−clean), bổ trợ cho gap này.
- [2026-05-24-v3-smart-both-design.md](../superpowers/specs/2026-05-24-v3-smart-both-design.md) — design doc tổng.

> **Trạng thái thực nghiệm (trung thực).** Giao thức + công cụ đã **implement đầy đủ**
> ([split.py](../../src/graphslm_ids/offline/preprocessing/split.py),
> [v3_eval_both_splits.py](../../scripts/eval/v3_eval_both_splits.py)). Tính đến
> 2026-06-17, bộ kết quả fair-comparison đã công bố (`results/2026-06-13/`) **mới
> chấm trên random split**; **lần chạy temporal-split cho model v3_ob cuối chưa được
> thực thi**, nên các ô số temporal trong báo cáo này để **TBD** chứ không điền số
> bịa. Đây là bước thực nghiệm kế tiếp; tài liệu này khoá *phương pháp* để khi chạy
> xong chỉ việc điền số.

---

## 0. TL;DR

- **Hai split, một model.** Cùng đồ thị v3_ob, cùng kiến trúc/siêu tham số HGT, chỉ
  đổi **cách chia train/val/test**:
  - **random** — stratified theo lớp, xáo trộn bằng RNG seed 42 (giống prior work
    E-GraphSAGE/TFE-GNN). **Thổi phồng** F1 vì flow test cùng burst PCAP với flow train.
  - **temporal** — cắt **theo thời gian** per-class trên `flow_start_ts`: 80% sớm
    nhất → train, 10% kế → val, 10% muộn nhất → test. **Thực tế triển khai**: không
    flow test nào có "hậu duệ" nằm trong train.
- **Gap = `macro_f1(random) − macro_f1(temporal)`.** Gap **nhỏ** ⇒ model học mẫu *nội
  tại của tấn công*; gap **lớn** ⇒ model dựa vào **dấu vân chiến dịch (campaign
  fingerprint)** rò từ train sang test — chính là dạng overfit mà random split che giấu.
- **Bổ trợ trục noisy−clean.** Hai trục đo cùng kể một câu chuyện ghi-nhớ: random split
  để lộ rò *thời gian/burst*; nhãn noisy để lộ ghi-nhớ *nhiễu nhãn*. Một model tốt phải
  hẹp ở **cả hai** gap.
- **Động cơ định lượng đã có:** XGBoost đạt macro-F1 **1.0000** trên random split (ghi
  nhớ hoàn hảo dấu vân chiến dịch) — bằng chứng mạnh nhất rằng random split *một mình*
  là vô nghĩa, và vì sao cần temporal split.

---

## 1. Vì sao cần hai split — vấn đề rò dấu vân chiến dịch

CIC-IoT-2023 ghi traffic theo **chiến dịch (capture/PCAP)**: mỗi lớp tấn công là một
phiên thí nghiệm liên tục. Một **random split** rút ngẫu nhiên flow test từ *cùng* các
burst PCAP có trong train. Hệ quả: hai flow gần nhau về thời gian/cấu hình (cùng IP,
cùng nhịp, cùng MTU…) bị tách một cái vào train, một cái vào test → model chỉ cần học
"**flow này thuộc PCAP nào**" là đoán đúng nhãn test, **không cần** học bản chất tấn công.

Đây là lý do điểm random split bị **thổi phồng** và **không** đo được khả năng tổng quát
hoá sang traffic *tương lai/chưa thấy*. Bằng chứng mạnh nhất trong dự án
([eacs-vs-baselines §headline](2026-06-13-eacs-vs-baselines.md)):

> XGBoost (chỉ 79 đặc trưng flow, cây boosting) đạt **macro-F1 random = 1.0000** —
> hoàn hảo mọi lớp. Một model "SOTA hoàn hảo" trên giấy, nhưng thực chất chỉ **ghi nhớ
> flow đến từ PCAP nào**. Khả năng ghi nhớ tỉ lệ với dung lượng: XGBoost (1.000) > graph
> models (~0.85) > EACS (0.72). Random split *một mình* không phân biệt được "học bản
> chất" với "ghi nhớ chiến dịch".

**Temporal split** phá rò này: huấn luyện trên quá khứ, chấm trên tương lai. Nếu model
chỉ ghi nhớ dấu vân chiến dịch, điểm temporal **sụt mạnh** so với random → **gap lớn**.
Nếu model học mẫu tấn công nội tại, điểm temporal **giữ gần** random → **gap nhỏ**. Bản
thân **độ lớn của gap chính là thước đo mức rò** — đó là *finding*, không chỉ là số phụ.

---

## 2. Hai giao thức chia split (định nghĩa chính xác)

Module: [split.py](../../src/graphslm_ids/offline/preprocessing/split.py)
(`make_splits`, `_stratified_random`, `_stratified_temporal`). Cả hai dựng từ **cùng tập
flow**, ghi ra `splits.json` với hai khoá `random` và `temporal` (mỗi khoá có
`train/val/test`). Tỉ lệ mặc định: **train 0.80 / val 0.10 / test 0.10**.

### 2.1. `random` — stratified, seed 42

[\_stratified_random](../../src/graphslm_ids/offline/preprocessing/split.py#L32):

```python
for cls in np.unique(labels):
    cls_ids = ids[labels == cls].copy()
    rng.shuffle(cls_ids)                 # RNG seed 42, per-class
    n_train = round(n * train_frac);  n_val = round(n * val_frac)
    n_train = min(n_train, n - 2)        # đảm bảo val & test mỗi cái ≥ 1 flow
    n_val   = max(1, min(n_val, n - n_train - 1))
    train += cls_ids[:n_train]; val += cls_ids[n_train:n_train+n_val]; test += cls_ids[n_train+n_val:]
```

- **Phân tầng theo lớp** (giữ tỉ lệ lớp ở mọi split) → so sánh công bằng với prior work
  vốn dùng cùng kiểu split.
- **Xác định** nhờ seed 42.

### 2.2. `temporal` — cắt thời gian per-class, seed-free

[\_stratified_temporal](../../src/graphslm_ids/offline/preprocessing/split.py#L72):

```python
for cls in np.unique(labels):
    order = np.argsort(cls_ts, kind="stable")   # sắp theo flow_start_ts tăng dần
    cls_ids_sorted = cls_ids[order]              # NaN ts bị đẩy về CUỐI -> rơi vào test
    # cùng công thức cắt n_train/n_val, nhưng KHÔNG xáo trộn
    train += sorted[:n_train]; val += sorted[n_train:n_train+n_val]; test += sorted[n_train+n_val:]
```

- **Cắt theo `flow_start_ts`**: 80% **sớm nhất** → train, 10% kế → val, 10% **muộn
  nhất** → test. Per-class để mọi lớp đều có mặt ở cả ba split.
- **Không seed** (xác định bởi thứ tự thời gian; ties phá theo index gốc, `kind="stable"`).
- **NaN timestamp** bị coi là "mới nhất" → đẩy vào test, tránh làm bẩn train.
- **Ý nghĩa triển khai:** mọi flow test xảy ra **sau** mọi flow train của cùng lớp →
  mô phỏng đúng cảnh "model huấn luyện hôm nay, gặp tấn công ngày mai".

### 2.3. Quy ước chung

- **Lớp < 3 flow** bị **ghim vào train** ở *cả hai* giao thức (chia ra sẽ tạo bucket
  val/test rỗng làm hỏng metric stratified).
- `splits.json` lưu **flow_id dạng chuỗi**; loader eval ánh xạ về chỉ số nút qua
  `flow_id_order` của artifact.

---

## 3. Cách đo gap (cùng một thước, không lệch định nghĩa)

Script: [v3_eval_both_splits.py](../../scripts/eval/v3_eval_both_splits.py). Nó nạp
**một** artifact, dựng **một** backend, rồi chấm **hai checkpoint** (random & temporal)
trên test-set tương ứng bằng **chính hàm `evaluate_neighbor_sampling` của trainer** —
nên metric eval **trùng khít** định nghĩa lúc train, không có "trôi định nghĩa" thầm lặng.

```python
gap = {
    "macro_f1": random_res["macro_f1"] - temporal_res["macro_f1"],
    "accuracy": random_res["accuracy"] - temporal_res["accuracy"],
}
# ghi ra JSON dưới khoá "gap_random_minus_temporal"
```

- **Đầu ra:** một JSON gồm cả hai split + `gap_random_minus_temporal`, kèm **per-class
  F1** cho từng split (cho thấy lớp nào sụt nhiều nhất khi sang temporal).
- **Cùng pipeline sampler/feature-stats** cho cả hai split (chuẩn hoá flow feature từ
  manifest, fanout, luôn-gồm tactic/technique) → khác biệt duy nhất là **tập test**.

### Phụ trợ tính phòng thủ (đã có sẵn trong script)

Mỗi split JSON còn ghi: **confusion matrix** + thứ tự nhãn, **per-class support**,
**bootstrap 95% CI** (seed 42, 1000 resample) cho macro-F1 và từng lớp, và
**feature_flags** (ordered-byte / attack-isolation đọc từ config checkpoint) + `--tag`.
Nhờ đó hai lần chạy khác cờ có thể **diff như một ablation**, và gap đi kèm CI để biết
nó có **vượt nhiễu thống kê** không.

---

## 4. Diễn giải gap — bảng quyết định

| Dấu/độ lớn gap (random − temporal) | Diễn giải | Kết luận |
|---|---|---|
| **≈ 0** (CI chồng lấn) | điểm temporal ≈ random | model học **mẫu tấn công nội tại**, tổng quát hoá theo thời gian — *kết quả mong muốn* |
| **dương, nhỏ** | sụt nhẹ khi sang tương lai | có chút phụ thuộc chiến dịch nhưng phần lớn là tín hiệu thật |
| **dương, lớn** | sụt mạnh khi sang tương lai | model dựa **dấu vân chiến dịch**; điểm random bị thổi phồng — *cảnh báo overfit* |
| **âm** | temporal > random | hiếm; thường do test temporal dễ hơn (mất cân bằng lớp theo thời gian) — soi per-class trước khi kết luận |

- **Đọc kèm per-class:** một gap tổng có thể do *một vài lớp* sụt mạnh (thường là lớp
  có chiến dịch tập trung thời gian). Bảng per-class F1 (random/temporal) trong JSON chỉ
  ra chính xác lớp nào.
- **Đọc kèm CI:** chỉ coi gap là *finding* khi CI macro-F1 hai split **không chồng lấn**;
  nếu chồng lấn, gap nằm trong nhiễu — vẫn là một kết luận (model ổn định theo thời gian).

---

## 5. Quan hệ với trục noisy−clean (hai trục bổ trợ)

Dự án có **hai** gap, đo hai dạng ghi-nhớ khác nhau — không trùng:

| Trục | Cố định gì, đổi gì | Bắt dạng ghi nhớ |
|---|---|---|
| **Smart-BOTH** (báo cáo này) | cùng nhãn, cùng model; đổi **cách chia thời gian** | rò **dấu vân chiến dịch / thời gian** |
| **noisy−clean** ([clean-grading](2026-06-13-clean-grading-methodology.md)) | cùng split, cùng model; đổi **thang nhãn chấm** | ghi nhớ **nhiễu nhãn per-pcap** |

Một model lý tưởng phải **hẹp ở cả hai**: gap random−temporal nhỏ *và* gap noisy−clean
âm. Hai trục cùng phản bác cách đánh giá một-chiều của prior work (chỉ random split, chỉ
nhãn noisy) — đó là toàn bộ luận điểm "Smart-BOTH".

---

## 6. Tái lập

```bat
:: (1) splits.json đã chứa cả random + temporal (sinh khi build graph)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.preprocessing.cli ^
  --raw-root data/raw --out-npz outputs/v3_ob/graph.npz ... ^
  --temporal-train-frac 0.80 --temporal-val-frac 0.10

:: (2) Train HGT HAI LẦN — cùng config, train trên split random rồi split temporal
::     (trainer chọn split theo config; checkpoint ra hai thư mục khác nhau)

:: (3) Đo gap random vs temporal (cùng một thước)
D:\v\nt114\Scripts\python.exe scripts/eval/v3_eval_both_splits.py ^
  --checkpoint-random   outputs/v3_ob_eacs_v2/hgt_flow_best.pt ^
  --checkpoint-temporal outputs/v3_ob_eacs_v2_temporal/hgt_flow_best.pt ^
  --graph outputs/v3_ob/graph.npz --graph-meta outputs/v3_ob/graph.meta.json ^
  --splits outputs/v3_ob/splits.json --out outputs/v3_ob/both_splits.json ^
  --tag v3_ob-eacs-v2
```

Đầu ra in bảng `metric | random | temporal | gap` + per-class F1 và CI 95%.

---

## 7. Bảng kết quả (điền khi chạy xong temporal)

| Model | macro-F1 random | macro-F1 temporal | gap (random − temporal) |
|---|---|---|---|
| XGBoost (tabular) | 1.0000 | **TBD** | **TBD** |
| GNN4ID | 0.8588 | **TBD** | **TBD** |
| HGT de-inflated (no EACS) | 0.8520 | **TBD** | **TBD** |
| **HGT + EACS v2** | 0.7228 | **TBD** | **TBD** |

> Cột random lấy từ [eacs-vs-baselines](2026-06-13-eacs-vs-baselines.md) (noisy-label
> TEST). Cột temporal/gap **chưa chạy** cho v3_ob cuối — điền sau khi thực thi §6 bước 2–3.
> Giả thuyết kiểm chứng: XGBoost gap **lớn nhất** (ghi nhớ chiến dịch mạnh nhất),
> EACS gap **nhỏ nhất** (học mẫu nội tại) — song song với thứ hạng noisy−clean.

---

## 8. Giới hạn đã biết (trung thực)

- **Chưa có số temporal** cho model cuối (đang chờ chạy) — đây là bước thực nghiệm kế tiếp.
- **Cắt thời gian per-class** dùng `flow_start_ts`; với lớp có nhiều chiến dịch tách rời
  theo thời gian, test temporal có thể lệch phân phối — phải soi per-class khi diễn giải.
- **Support test nhỏ** ở vài lớp web (XSS/CmdInj/Upload) khiến F1 dao động mạnh; bootstrap
  CI trong JSON là để định lượng đúng độ dao động này.
- **Hai checkpoint phải cùng feature build**; script tự đối chiếu `feature_flags` và đánh
  dấu `_mismatch` nếu chúng khác nhau (tránh so sánh táo-cam thầm lặng).

---

## Phụ lục — file liên quan

| File | Vai trò |
|---|---|
| [split.py](../../src/graphslm_ids/offline/preprocessing/split.py) | dựng random + temporal split (`make_splits`) |
| [v3_eval_both_splits.py](../../scripts/eval/v3_eval_both_splits.py) | chấm hai split bằng cùng thước, ghi gap + CI + confusion |
| [2026-06-13-eacs-vs-baselines.md](2026-06-13-eacs-vs-baselines.md) | cột random + động cơ XGBoost-1.0000 |
| [2026-06-13-clean-grading-methodology.md](2026-06-13-clean-grading-methodology.md) | trục noisy−clean bổ trợ |
| [2026-05-24-v3-smart-both-design.md](../superpowers/specs/2026-05-24-v3-smart-both-design.md) | design doc tổng |
