# Kế hoạch train sơ bộ trên laptop — FL-HGT testbed

**Ngày lập:** 2026-10-18
**Nhánh:** `feat/fl-poisoning-testbed`
**Spec:** `docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md`
**Máy chạy:** laptop Intel Core i5-8350U (4 nhân / 8 luồng), RAM 16 GB, **không có GPU dùng được**
(Intel UHD 620), torch 2.14.1 bản CPU trong `D:\v\nt114`.

## 0. Mục tiêu

Có **kết quả thật đầu tiên** cho buổi báo cáo tiếp theo với giảng viên khi chưa có server GPU, và
**đo được thời gian thật** để xin tài nguyên GPU.

**Sản phẩm cần có khi xong plan này:**

| # | Sản phẩm | Phục vụ |
|---|---|---|
| 1 | Bảng chia 20 client × 18 lớp **với số flow chính thức** cho 3 mức lệch (CSV + biểu đồ nhiệt) | Yêu cầu GV #3 |
| 2 | **CH1 với MLP đầy đủ:** centralized / local-only / FL × mức lệch 0% / 50% / 80% (9 run, dữ liệu đầy đủ) | Yêu cầu GV #1, #6 |
| 3 | **Cặp HGT cấu hình A:** centralized và FL ở mức lệch 80% (thêm 0% nếu kịp), dữ liệu thu gọn | Yêu cầu GV #1, #6 |
| 4 | Số local epoch đã chọn cho HGT (pilot rút gọn) | Chốt cấu hình |
| 5 | Thời gian mỗi round, RAM đỉnh trên laptop → ước lượng ngân sách GPU cho P0/P1/P2 | Xin GPU |

**Không làm trong plan này:** HGT cấu hình B (đo được khoảng 4,5 giây/bước, khoảng 50 phút/epoch trên
CPU → không khả thi), ma trận tấn công/phòng thủ P0/P1/P2, 3 seed. CNN chỉ là tùy chọn (mục 8).

Mọi kết quả từ plan này được ghi là **sơ bộ** (1 seed; HGT chạy dữ liệu thu gọn).

## 1. Số đo tham chiếu (benchmark CPU ngày 18/10)

Đo forward + backward + AdamW trên batch giả lập cùng kích thước dữ liệu thật; **chưa gồm** lấy mẫu đồ
thị, nạp dữ liệu và đánh giá, nên thời gian thật sẽ chậm hơn.

| Mô hình | Tham số | 1 bước (batch 256) | 1 epoch trên khoảng 168.500 flow train |
|---|---|---|---|
| HGT A (64 / 2 layer / 2-hop) | 0,94 triệu | 0,35 s | khoảng 4 phút |
| HGT B (128 / 4 layer / 4-hop) | 6,43 triệu | 4,5 s | khoảng 50 phút |
| MLP | rất nhỏ | 4 ms | vài giây |
| 1D-CNN (8 packet × 256 byte) | nhỏ | khoảng 1 s | khoảng 10,5 phút |

1 run 20 round ≈ 20 epoch trên phần dữ liệu đang dùng. Ước lượng mỗi run (đã cộng chi phí phụ):
MLP vài phút; HGT A thu gọn khoảng 30–60 phút; HGT A đầy đủ khoảng 2–4 giờ; CNN thu gọn khoảng 1 giờ.

## 2. Điều kiện tiên quyết

### 2.1 Dữ liệu (đang thiếu — chặn toàn bộ plan)

Laptop hiện **không có** `outputs/v3_ob/`, `data/raw/` trống, `data/mitre/` trống. Cần chép từ máy bạn
cùng nhóm (hoặc máy đã build artifact trước đây) vào đúng đường dẫn trong repo:

| File | Dùng cho | Bắt buộc |
|---|---|---|
| `outputs/v3_ob/graph.npz` | Mọi mô hình | Có |
| `outputs/v3_ob/graph.meta.json` | Mọi mô hình (tên đặc trưng, nhãn, thời gian flow) | Có |
| `outputs/v3_ob/splits.json` | Split train/val/test | Có |
| `outputs/v3_ob/clean_eval_labels.npy` | Chấm nhãn clean (chỉ số chính) | Có |
| `data/mitre/` (đặc biệt `class_technique_map.csv`) | HGT: `fl_base.yaml` bật GCL | Có cho HGT |
| `outputs/v3_ob/eacs_anchor_mask.npy` | Chỉ khi bật EACS (P2) | Không |

Không có cách lấy file → phương án dự phòng là dựng lại graph từ 18 file PCAP bằng
`graphslm_ids.offline.preprocessing.cli` (lệnh trong `CLAUDE.md`). Trên laptop 4 nhân sẽ mất nhiều giờ
và RAM rất sát; chỉ làm khi bắt buộc.

### 2.2 Môi trường

```powershell
cd D:\DACN\Project\Do-an-chuyen-nganh_NT114
$PY = "D:\v\nt114\Scripts\python.exe"

# Gói graphslm_ids chưa được cài vào venv -> cài dạng editable (một lần)
& $PY -m pip install -e .
& $PY -c "import graphslm_ids; print(graphslm_ids.__file__)"   # phải trỏ vào ...\NT114\src\graphslm_ids

# Code đúng nhánh, mới nhất, test xanh
git switch feat/fl-poisoning-testbed
git pull origin feat/fl-poisoning-testbed
& $PY -m pytest tests/federated -q        # kỳ vọng: 347 passed
```

Nếu không muốn cài editable: đặt `$env:PYTHONPATH = "src"` trong mỗi phiên PowerShell trước khi chạy
`python -m graphslm_ids...` (các script trong `scripts/fl/` tự thêm `src/`).

### 2.3 Chuẩn bị laptop

- Cắm sạc; chế độ nguồn **Best performance**.
- Tắt chế độ ngủ khi cắm sạc trong thời gian chạy đêm: `powercfg /change standby-timeout-ac 0`
  (đặt lại sau khi xong).
- **Đóng trình duyệt, VS Code và các ứng dụng nặng** khi chạy HGT (graph cần vài GB RAM).
- Kê máy thoáng; CPU 15W chạy lâu sẽ giảm xung.
- Mọi lệnh dưới đây chạy từ thư mục repo, luôn thêm `--device cpu` (`fl_base.yaml` đặt `device: cuda`).

## 3. Bước 1 — Kiểm tra dữ liệu và RAM (khoảng 15 phút)

```powershell
Get-ChildItem outputs\v3_ob | Select-Object Name, @{n="MB";e={[math]::Round($_.Length/1MB)}}
Test-Path data\mitre\class_technique_map.csv
```

- Ghi lại kích thước `graph.npz`. Phần lớn là ma trận đặc trưng packet (2.579 chiều/packet, float16;
  `fl_base.yaml` giữ nguyên float16 nhờ `packet_dtype: preserve`).
- Nếu kích thước file gần bằng hoặc lớn hơn RAM trống sau khi đóng ứng dụng → dừng, xem mục 10 (OOM).

## 4. Bước 2 — Xuất bảng chia chính thức (khoảng 15 phút)

```powershell
foreach ($p in 0.0, 0.5, 0.8) {
  $tag = [int]($p * 100)
  & $PY scripts/fl/export_partition_table.py --config configs/fl/fl_base.yaml `
    --set partition.dominance=$p `
    --out-csv outputs/fl/partition_p$tag.csv --out-png outputs/fl/partition_p$tag.png
}
```

Kiểm tra trong kết quả:

- Tổng mỗi lớp khớp số flow train; mỗi client ≥ 50 flow (`min_flows_per_client`).
- Trường `ordered_by` (trong `partition.json` của các run sau): `flow_start_ts` = flow được chọn theo
  thời gian thật; `flow_id` = metadata không có thời gian bắt đầu flow → ghi chú điều này khi báo cáo.
- Lớp nhỏ (DDoS-ICMP_Flood, Mirai-udpplain) ở mức 1%: mỗi client chỉ còn vài flow — ghi nhận.

Sản phẩm #1: 3 file CSV + 3 biểu đồ nhiệt → đưa vào tài liệu gửi giảng viên.

## 5. Bước 3 — Chạy thử trên dữ liệu thật (khoảng 1 giờ)

Mục đích: xác nhận nạp dữ liệu đúng, đo thời gian một round và RAM đỉnh trên laptop.

```powershell
# MLP, 2 round, dữ liệu đầy đủ
& $PY -m graphslm_ids.offline.federated.cli --config configs/fl/fl_base.yaml --device cpu `
  --rounds 2 --output-dir outputs/fl/laptop_smoke_mlp --set model.name=mlp --set output.save_checkpoints=false

# HGT A, 2 round, dữ liệu thu gọn
& $PY -m graphslm_ids.offline.federated.cli --config configs/fl/fl_base.yaml --device cpu `
  --rounds 2 --output-dir outputs/fl/laptop_smoke_hgt --set model.name=hgt `
  --set partition.max_flows_per_client=2000 --set output.save_checkpoints=false
```

Ghi lại từ `outputs/fl/laptop_smoke_*/fl_summary.json` và `rounds.jsonl`:

| Chỉ số | MLP | HGT A |
|---|---|---|
| Thời gian train client / round | | |
| Thời gian đánh giá / round | | |
| RAM đỉnh (peak RSS) | | |
| `partition.json` → `params.cap_scale`, tổng số flow train | | |

**Lưu ý về tầng thu gọn:** `max_flows_per_client` co **cả bảng** theo một hệ số (`cap_scale`) để giữ
tỉ lệ lớp. Vì client lớn nhất ở mức lệch 80% lớn hơn nhiều so với ở 0%, cùng một giá trị 2000 sẽ cho
**tổng dữ liệu ở mức 80% ít hơn rõ rệt** so với mức 0%. Khi so sánh giữa các mức lệch, chọn
`max_flows_per_client` riêng cho từng mức sao cho tổng flow train xấp xỉ nhau (đọc tổng thật từ
`partition.json`).

**Quyết định sau bước này:** nếu một round HGT A thu gọn > 5 phút, giảm `max_flows_per_client` hoặc
đặt `eval.every: 2` cho các grid HGT ở bước 5–6.

## 6. Bước 4 — CH1 với MLP, dữ liệu đầy đủ (khoảng 1–2 giờ)

Tạo grid `configs/fl/matrix_laptop_ch1_mlp.yaml` (cùng định dạng với `matrix_ch1.yaml`):

```yaml
# Laptop, CH1 với MLP (9 run): centralized / local-only / FL x 3 mức lệch, dữ liệu đầy đủ.
base_config: configs/fl/fl_base.yaml
output_root: outputs/fl/laptop_ch1_mlp
name_template: "{mode}__mlp__{partition}"
overrides:
  model.name: mlp
  output.save_checkpoints: false
axes:
  mode:
    centralized: {federated.mode: centralized}
    local_only: {federated.mode: local_only}
    fl: {federated.mode: fl}
  partition:
    p0: {partition.dominance: 0.0}
    p50: {partition.dominance: 0.5}
    p80: {partition.dominance: 0.8}
```

```powershell
& $PY scripts/fl/run_matrix.py --grid configs/fl/matrix_laptop_ch1_mlp.yaml --device cpu --dry-run
& $PY scripts/fl/run_matrix.py --grid configs/fl/matrix_laptop_ch1_mlp.yaml --device cpu
& $PY scripts/fl/summarize_matrix.py --root outputs/fl/laptop_ch1_mlp
```

`run_matrix.py` nạp dữ liệu một lần cho cả grid và bỏ qua ô đã có `fl_summary.json` → dừng giữa chừng
thì chạy lại lệnh là tiếp tục.

Sản phẩm #2: bảng macro-F1 (nhãn clean và nhãn gốc), recall theo họ tấn công, FPR cho 9 ô.

## 7. Bước 5–6 — HGT cấu hình A (chạy qua đêm)

### Bước 5: pilot rút gọn — chọn số local epoch (4 run, khoảng 2–4 giờ)

Tạo `configs/fl/matrix_laptop_pilot.yaml`:

```yaml
# Laptop pilot: chỉ cấu hình A, 1 hoặc 2 local epoch x mức lệch 0% / 80%, dữ liệu thu gọn, không tấn công.
base_config: configs/fl/fl_base.yaml
output_root: outputs/fl/laptop_pilot
name_template: "A__e{local_epochs}__{partition}"
overrides:
  model.name: hgt
  output.save_checkpoints: false
axes:
  local_epochs:
    "1": {client.local_epochs: 1}
    "2": {client.local_epochs: 2}
  partition:
    p0: {partition.dominance: 0.0, partition.max_flows_per_client: 2000}
    p80: {partition.dominance: 0.8, partition.max_flows_per_client: 2000}
```

(Thay 2000 bằng giá trị chọn ở bước 3 để tổng flow của p0 và p80 xấp xỉ nhau.)

```powershell
& $PY scripts/fl/run_matrix.py --grid configs/fl/matrix_laptop_pilot.yaml --device cpu
& $PY scripts/fl/summarize_matrix.py --root outputs/fl/laptop_pilot
```

**Quy tắc chọn:** dùng 1 local epoch, trừ khi FL với 1 epoch chưa hội tụ trong 20 round (macro-F1 clean
trên val còn tăng rõ ở các round cuối) còn 2 epoch thì hội tụ. Ghi lại thời gian mỗi run.

Sản phẩm #4: số local epoch đã chọn.

### Bước 6: cặp centralized vs FL (2–4 run, khoảng 2–4 giờ)

Tạo `configs/fl/matrix_laptop_ch1_hgt.yaml`:

```yaml
# Laptop, CH1 với HGT A: centralized vs FL, dữ liệu thu gọn, lưu checkpoint (dùng lại cho XAI).
base_config: configs/fl/fl_base.yaml
output_root: outputs/fl/laptop_ch1_hgt
name_template: "{mode}__hgt__{partition}"
overrides:
  model.name: hgt
  client.local_epochs: 1          # thay bằng giá trị chọn ở bước 5
  output.save_checkpoints: true
axes:
  mode:
    centralized: {federated.mode: centralized}
    fl: {federated.mode: fl}
  partition:
    p80: {partition.dominance: 0.8, partition.max_flows_per_client: 2000}
    p0: {partition.dominance: 0.0, partition.max_flows_per_client: 2000}
```

```powershell
# Ưu tiên mức lệch 80% trước; p0 chạy sau nếu còn thời gian
& $PY scripts/fl/run_matrix.py --grid configs/fl/matrix_laptop_ch1_hgt.yaml --device cpu --only "*p80"
& $PY scripts/fl/run_matrix.py --grid configs/fl/matrix_laptop_ch1_hgt.yaml --device cpu --only "*p0"
& $PY scripts/fl/summarize_matrix.py --root outputs/fl/laptop_ch1_hgt
```

Hai run FL ở bước 5 (cùng local epoch, cùng `max_flows_per_client`) trùng cấu hình với ô `fl` ở đây —
nếu trùng hoàn toàn có thể dùng lại kết quả pilot thay vì chạy lại.

Sản phẩm #3: macro-F1 của HGT A centralized vs FL (sơ bộ, dữ liệu thu gọn).

## 8. Bước 7 (tùy chọn) — CNN

Chỉ làm khi bước 4–6 đã xong và còn thời gian. CNN tốn hơn HGT A (khoảng 1 giây/bước). Chạy cặp
centralized vs FL ở mức lệch 80%, dữ liệu thu gọn, theo mẫu grid ở bước 6 với `model.name: cnn`. Muốn
nhanh hơn: `model.cnn.num_packets: 4` (ghi rõ nếu đổi).

## 9. Bước 8 — Tổng hợp cho buổi báo cáo

1. **Bảng CH1 sơ bộ:** MLP (9 ô, dữ liệu đầy đủ) + HGT A (centralized vs FL ở p80, có thể p0, dữ liệu
   thu gọn). Ghi rõ: 1 seed; HGT dữ liệu thu gọn với tổng số flow train cụ thể; chấm trên tập test
   toàn cục; chỉ số chính là macro-F1 **round cuối** theo nhãn clean.
2. **Bảng chia** (3 CSV + biểu đồ nhiệt).
3. **Ngân sách GPU:** từ thời gian thật của bước 3–6, tính số giờ cho từng grid P0/P1/P2 trên laptop và
   so với nhu cầu → căn cứ xin quyền dùng GPU của lab (máy metis đã dùng trong đồ án) hoặc dùng
   Kaggle/Colab.
4. **Không** trình bày kết quả smoke test trên dữ liệu giả lập như kết quả thí nghiệm.

## 10. Lịch dự kiến

| Thời điểm | Việc |
|---|---|
| Ngày 1 (khi đã có dữ liệu) | Mục 2–5: môi trường, kiểm tra dữ liệu, bảng chia, chạy thử |
| Ngày 1 (chiều tối) | Bước 4: CH1 MLP (9 run) |
| Đêm 1 | Bước 5: pilot HGT A (4 run) |
| Ngày 2 | Đọc pilot, chốt local epoch; chạy bước 6 ở p80 |
| Đêm 2 | Bước 6 ở p0; (tùy chọn) CNN |
| Ngày 3 | Bước 8: tổng hợp bảng, ước lượng ngân sách GPU, chuẩn bị báo cáo |

## 11. Xử lý sự cố

| Tình huống | Cách xử lý |
|---|---|
| `ModuleNotFoundError: graphslm_ids` | Chạy `pip install -e .` (mục 2.2) hoặc đặt `$env:PYTHONPATH = "src"` |
| Lỗi thiếu `class_technique_map.csv` | Chép `data/mitre/` (mục 2.1); không có thì thêm `--set client.gcl.enabled=false` và ghi rõ trong báo cáo |
| Lỗi `cuda` / không thấy GPU | Thiếu `--device cpu` |
| Hết RAM (máy đơ, swap nặng) | Đóng hết ứng dụng; giảm `max_flows_per_client`; giảm `eval.max_val_flows` (mặc định 5000); nếu vẫn không đủ thì chỉ chạy MLP trên laptop, HGT chuyển sang GPU |
| `ValueError` khi chia client (client dưới 50 flow) | `max_flows_per_client` quá nhỏ ở tầng thu gọn → tăng giá trị |
| Một round HGT quá chậm (> 10 phút) | Giảm `max_flows_per_client`; đặt `eval.every: 2`; kiểm tra máy có đang giảm xung / chạy pin không |
| Máy ngủ hoặc tắt giữa chừng | Chạy lại đúng lệnh `run_matrix.py`: ô đã xong được bỏ qua |

## 12. Tiêu chí hoàn thành

- [ ] `pytest tests/federated` xanh trên laptop sau khi pull.
- [ ] 3 bảng chia chính thức (CSV + PNG) đã xuất.
- [ ] 9 ô CH1 MLP có `fl_summary.json`; đã chạy `summarize_matrix.py`.
- [ ] Pilot HGT A xong; số local epoch đã chốt.
- [ ] Ít nhất cặp HGT A centralized vs FL ở p80 có kết quả, checkpoint đã lưu.
- [ ] Bảng thời gian/round và RAM đỉnh thật trên laptop; ước lượng ngân sách GPU cho P0/P1/P2.
- [ ] Mọi số liệu trong báo cáo được ghi "sơ bộ" kèm điều kiện chạy.
