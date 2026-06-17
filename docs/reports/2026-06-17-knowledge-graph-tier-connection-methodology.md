# Phương pháp nối tầng tri thức của đồ thị HGT — Technique ↔ Tactic (báo cáo chi tiết, 2026-06-17)

Tài liệu này đặc tả đầy đủ **cách hai tầng tri thức trên của đồ thị dị-cấu
(heterogeneous graph) được dựng và nối với nhau**:

- **Tầng 2 — technique** (kỹ thuật MITRE ATT&CK), và
- **Tầng 3 — tactic** (chiến thuật MITRE ATT&CK),

cùng cạnh `has_subtechnique` cấu trúc nội-tầng-2 và cạnh `belongs_to_tactic`
(technique → tactic) nối tầng 2 lên tầng 3. Mục tiêu: ai đọc xong cũng tái lập
được, hiểu vì sao HGT khai thác được, và bảo vệ được trước hội đồng.

Phần nối **tầng 1 (flow/packet/host) → tầng 2 (technique)** — tức Multi-Source
Evidence Ensemble (MSEE) — đã được tài liệu riêng và **không** lặp lại ở đây:
- `docs/superpowers/specs/2026-05-24-v3-smart-both-design.md §4` — MSEE: PMI + L1-LR + procedure match.

Tài liệu đi kèm:
- [2026-05-24-v3-smart-both-design.md](../superpowers/specs/2026-05-24-v3-smart-both-design.md) — design doc tổng (schema §3, MSEE §4).
- [2026-06-17-eacs-methodology.md](2026-06-17-eacs-methodology.md) — bộ điều khiển chống nhiễu nhãn EACS (tầng dữ liệu).

---

## 0. TL;DR

- Đồ thị có **3 tầng**: (1) mặt dữ liệu quan sát `flow / packet / host`, (2) tri
  thức **technique** (691 nút), (3) tri thức **tactic** (14 nút).
- Hai tầng trên là **tri thức tĩnh MITRE ATT&CK**, dựng **xác định (deterministic)**
  từ CSV/STIX trong `data/mitre/`, **không có encoder học** — đúng nguyên tắc
  "zero learned encoders besides HGT itself".
- Hai loại cạnh nối tầng trên:
  - `has_subtechnique` (technique → technique): **475 cạnh**, parse quan hệ cha→con
    từ định danh `T1190 → T1190.001`.
  - `belongs_to_tactic` / `technique_tactic` (technique → tactic): **887 cạnh**,
    đọc từ ánh xạ MITRE `(technique_id, tactic_shortname)`.
- Loader thêm **cạnh đảo (`rev_*`)** nên thông điệp chạy **hai chiều**:
  `packet → technique → tactic` (gộp ngữ cảnh lên) và `tactic → technique → packet`
  (lan toả prior chiến thuật xuống). Cả hai cạnh nằm trong `metadata` của HGT nên
  **thực sự tham gia message passing**, không phải trang trí.
- Vai trò khoa học: đây là **lớp 2 (typed heterogeneous schema)** trong khung novelty
  — PMI cấp PRIOR, **HGT attention REFINES**, GCL SUPERVISES; tầng tactic giữ đồ thị
  tri thức nguyên vẹn để tín hiệu định tuyến được qua phả hệ kỹ thuật.

Số liệu trích từ artifact thực `outputs/v3/graph.meta.json` (211K flows).

---

## 1. Ba tầng của đồ thị — định nghĩa và đặc trưng nút

Module dựng: [graph_builder.py](../../src/graphslm_ids/offline/preprocessing/graph_builder.py)
(docstring schema dòng 13–36). Module nạp cho HGT:
[hetero_graph_artifact.py](../../src/graphslm_ids/offline/training/hetero_graph_artifact.py).

| Tầng | Loại nút | #nút (v3) | Đặc trưng đầu vào | Nguồn |
|---|---|---|---|---|
| **1 — mặt dữ liệu** | `flow` | 210,930 | ~80 đặc trưng CICFlowMeter + 5-d tóm tắt evidence | trích từ pcap |
| | `packet` | 387,388 | 2,323-d payload features (float16) | trích từ pcap |
| | `host` | 1,235 | 4-d: out_deg, in_deg, #dst_port, #dst_host | gộp từ flow |
| **2 — technique** | `technique` | **691** (toàn bộ MITRE) | **768-d SecureBERT embedding** | `mitre_technique_embeddings.npy` |
| **3 — tactic** | `tactic` | **14** (toàn bộ MITRE) | **không có đặc trưng nội dung** (placeholder rỗng, kích thước = số tactic) | `mitre_tactics` |

### 1.1. Tầng 2 — nút technique (691)

Đọc trong [graph_builder.py §8](../../src/graphslm_ids/offline/preprocessing/graph_builder.py#L695):

```python
techniques_df = pd.read_csv(mitre_techniques_csv)
technique_x   = np.load(mitre_technique_embeddings_npy)   # (691, 768) SecureBERT
assert technique_x.shape[0] == len(techniques_df)
technique_id_to_idx = {tid: i for i, tid in enumerate(techniques_df["technique_id"])}
```

- **Giữ toàn bộ 691 technique** dù chỉ ~12–20 cái thực nhận bằng chứng packet. Lý do
  (design doc §3): phả hệ + cạnh tactic giữ **đồ thị tri thức nguyên vẹn**; HGT có thể
  định tuyến tín hiệu qua `T1190 → has_subtechnique → T1190.001`. Đây là "Knowledge-only
  Layer" — không nhận bằng chứng trực tiếp nhưng vẫn có nghĩa ngữ nghĩa và là cầu nối
  lên tactic.
- Đặc trưng là **SecureBERT embedding của mô tả technique** — đây là embedding *văn bản
  tri thức MITRE*, **không** phải embedding payload (khác hẳn lỗi v1: cosine
  SecureBERT(payload) vs SecureBERT(MITRE) — xem CLAUDE.md "Diagnosis history").

### 1.2. Tầng 3 — nút tactic (14)

[graph_builder.py §9](../../src/graphslm_ids/offline/preprocessing/graph_builder.py#L705):

```python
tactic_x = np.arange(n_tactics, dtype=np.int64)[:, None]   # cột id thuần
```

14 tactic (theo `tactic_to_idx`, sắp xếp alphabet):

```
collection(0) command-and-control(1) credential-access(2) defense-evasion(3)
discovery(4) execution(5) exfiltration(6) impact(7) initial-access(8)
lateral-movement(9) persistence(10) privilege-escalation(11)
reconnaissance(12) resource-development(13)
```

> **Trung thực về đặc trưng tactic:** trên đĩa tactic chỉ mang **cột id**. Khi nạp,
> [load_v3_artifact](../../src/graphslm_ids/offline/training/hetero_graph_artifact.py#L325)
> thay bằng `_empty_tactic_features(num_tactics)` — **placeholder rỗng**. Nghĩa là nút
> tactic **không mang thông tin nội dung đầu vào**; biểu diễn của nó do HGT **học hoàn
> toàn từ vị trí trong đồ thị** (những technique nào trỏ vào nó). Đây là lựa chọn có
> chủ đích: tactic là nhãn phân loại, không phải thực thể có nội dung văn bản cần encode.

---

## 2. Cạnh nội-tầng-2 — `has_subtechnique` (technique → technique)

Phả hệ ATT&CK: một technique cha (`T1190`) có nhiều sub-technique
(`T1190.001`, `T1190.002`, …). Cạnh này dệt 691 nút technique rời rạc thành **rừng phả
hệ**, để bằng chứng rơi vào một sub-technique vẫn lan được lên technique cha (và ngược
lại qua cạnh đảo).

Hàm dựng — [\_build_has_subtechnique_edges](../../src/graphslm_ids/offline/preprocessing/graph_builder.py#L377):

```python
for tid in techniques_df["technique_id"]:
    if "." not in tid:              # bỏ technique gốc (không có cha)
        continue
    parent = tid.split(".", 1)[0]   # "T1190.001" -> "T1190"
    if parent in technique_id_to_idx and tid in technique_id_to_idx:
        src.append(technique_id_to_idx[parent])   # cha
        dst.append(technique_id_to_idx[tid])      # con
# Spec hướng: parent --[has_subtechnique]--> sub-technique
```

- **Phát hiện quan hệ thuần cú pháp**: dấu `.` trong định danh là dấu hiệu sub-technique;
  prefix trước `.` là technique cha. Không cần bảng tra phụ.
- **Hướng cạnh:** cha → con. Cạnh đảo `rev_has_subtechnique` (loader tự thêm) cho phép
  con → cha.
- **Số cạnh thực (v3):** **475** (`n_has_subtechnique_edges`). Chỉ những cặp mà *cả*
  cha lẫn con đều nằm trong danh sách 691 mới tạo cạnh.
- **Đặc trưng cạnh:** không (cạnh thuần cấu trúc, `attr=None`).

---

## 3. Cạnh nối tầng 2 → tầng 3 — `belongs_to_tactic` (technique → tactic)

Đây là **cạnh nối tầng** chính của báo cáo. Mỗi technique thuộc một hoặc nhiều tactic
(một technique có thể phục vụ nhiều mục tiêu chiến thuật) → đồ thị **đa-cạnh**
technique→tactic.

Hàm dựng — [\_load_mitre_tactics](../../src/graphslm_ids/offline/preprocessing/graph_builder.py#L944):

```python
edges_df = pd.read_csv(technique_tactic_csv)          # cột: technique_id, tactic_shortname
required = {"technique_id", "tactic_shortname"}        # kiểm tra schema, raise nếu thiếu
tactics  = sorted({str(t) for t in edges_df["tactic_shortname"]})
tactic_to_idx = {t: i for i, t in enumerate(tactics)}  # 14 tactic -> id 0..13

for _, row in edges_df.iterrows():
    tid, tac = str(row["technique_id"]), str(row["tactic_shortname"])
    if tid not in technique_id_to_idx:                 # technique không có trong 691 -> bỏ
        continue
    src.append(technique_id_to_idx[tid])               # technique
    dst.append(tactic_to_idx[tac])                     # tactic
edge_attr = np.ones((n_edges, 1), dtype=np.float32)    # trọng số = 1.0
```

- **Nguồn:** ánh xạ chính thức `(technique_id → tactic_shortname)` của MITRE trong
  `data/mitre/`. Quan hệ là **kiến thức tĩnh**, không suy ra từ dữ liệu traffic.
- **Hướng cạnh:** technique → tactic. Cạnh đảo `rev_belongs_to_tactic` cho phép
  tactic → technique (prior chiến thuật lan xuống).
- **Số cạnh thực (v3):** **887** (`n_technique_tactic_edges`). Lớn hơn 691 vì quan hệ
  nhiều-nhiều (một technique nhiều tactic).
- **Đặc trưng cạnh:** trọng số hằng `1.0` (1-d) — quan hệ có/không, không có cường độ.
- **Kích thước tầng tactic suy từ cạnh:** nếu `num_tactics` vắng trong metadata, loader
  lấy `max(dst)+1` từ chính cạnh này
  ([dòng 317](../../src/graphslm_ids/offline/training/hetero_graph_artifact.py#L317)).

---

## 4. Hai cạnh này vào HGT thế nào (message passing thật, không trang trí)

[load_v3_artifact](../../src/graphslm_ids/offline/training/hetero_graph_artifact.py#L261)
khai báo cả hai trong `edge_specs`, nên chúng nằm trong `metadata` mà HGT dựng ma trận
attention theo từng loại quan hệ:

```python
(("technique", "has_subtechnique", "technique"), "has_subtechnique_edge_index", None),
(("technique", "belongs_to_tactic", "tactic"),
     "technique_tactic_edge_index", "technique_tactic_edge_attr"),
```

- **Cạnh đảo tự sinh:** loader thêm `rev_*` cho mọi loại cạnh → đường truyền **hai
  chiều**. Một bó bằng chứng packet→technique do MSEE tạo ra có thể: đi **lên**
  `technique → tactic` để gộp ngữ cảnh chiến thuật; đi **ngang** `technique ↔
  sub-technique` theo phả hệ; rồi **xuống** lại flow qua cạnh đảo. Nhờ đó nút flow
  thu được biểu diễn giàu ngữ cảnh MITRE dù bản thân nó chỉ nối trực tiếp tới host và
  technique.
- **HGT attention REFINES prior:** PMI/CSV chỉ cấp *cấu trúc và trọng số tiên nghiệm*;
  ma trận attention đa-đầu của HGT học **mức độ tin** từng loại quan hệ — kể cả
  `has_subtechnique` và `belongs_to_tactic` — theo tín hiệu phân loại flow.
- **Đối lập v1:** v1 gộp mọi cạnh technique vào *một* loại `matches_technique` → HGT
  thoái hoá thành đồng-cấu. Việc tách `has_subtechnique` và `belongs_to_tactic` thành
  **loại quan hệ riêng** chính là điều khiến schema "dị-cấu thực sự".

---

## 5. Vì sao nối hai tầng trên — biện minh thiết kế

1. **Giữ đồ thị tri thức nguyên vẹn.** ~671/691 technique không nhận bằng chứng trực
   tiếp; nếu bỏ chúng + tactic, đồ thị mất phần "tri thức". Phả hệ + tactic biến chúng
   thành **đường định tuyến** cho tín hiệu lan toả (design doc §3).
2. **Trừu tượng hoá theo cấp.** Bằng chứng thường khớp ở mức sub-technique cụ thể;
   tactic cho HGT một **mức gộp ổn định hơn** (14 lớp) ít thưa hơn 691 technique.
3. **Tính kiểm toán (SOC-audit).** Mỗi quyết định flow truy ngược được tới technique →
   tactic cụ thể — chuỗi `flow → technique → tactic` là lời giải thích theo khung MITRE,
   thứ XG-NID (kể chuyện hậu kỳ) và PacketCLIP (encoder đơn nguồn) không có.
4. **Zero learned encoder.** Cả hai cạnh dựng bằng đọc CSV + so khớp chuỗi định danh —
   xác định, seed-independent. HGT là model **duy nhất** học.

---

## 6. Số liệu chốt (artifact thực `outputs/v3/graph.meta.json`)

| Đại lượng | Giá trị |
|---|---|
| `num_techniques` | 691 |
| `num_tactics` | 14 |
| `n_has_subtechnique_edges` (technique→technique) | **475** |
| `n_technique_tactic_edges` (technique→tactic) | **887** |
| `n_flow_technique_edges` (tham chiếu, tầng 1→2) | 45,096 |
| technique feature dim | 768 (SecureBERT) |
| tactic feature | placeholder rỗng (id trên đĩa) |

> **Đính chính số liệu:** design doc §3 (dòng 84) viết `has_subtechnique` "~887 cạnh,
> khớp với technique_tactic". Số thực: `has_subtechnique = 475`, `technique_tactic =
> 887` — **hai con số khác nhau**. Khi đưa vào luận văn, dùng số ở bảng trên.

---

## 7. Tái lập

Cả hai loại cạnh được dựng trong cùng lệnh build graph (CPU local):

```bat
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.preprocessing.cli ^
  --raw-root data/raw --out-npz outputs/v3_ob/graph.npz ^
  --out-meta outputs/v3_ob/graph.meta.json ...
```

Kiểm tra nhanh số cạnh nối tầng từ artifact đã build:

```python
import json
m = json.load(open("outputs/v3_ob/graph.meta.json"))
print(m["n_has_subtechnique_edges"], m["n_technique_tactic_edges"])  # 475 887
print(m["num_techniques"], m["num_tactics"])                          # 691 14
```

Unit test liên quan: kiểm parse phả hệ `T1190.001 → T1190` và schema CSV
`(technique_id, tactic_shortname)` trong `tests/` (preprocessing).

---

## Phụ lục — file liên quan

| File | Vai trò |
|---|---|
| [graph_builder.py](../../src/graphslm_ids/offline/preprocessing/graph_builder.py) | `_build_has_subtechnique_edges` (§2), `_load_mitre_tactics` (§3), nạp technique/tactic node |
| [hetero_graph_artifact.py](../../src/graphslm_ids/offline/training/hetero_graph_artifact.py) | khai báo cạnh vào `metadata` HGT, cạnh đảo, placeholder tactic |
| `data/mitre/mitre_techniques.csv` | danh sách 691 technique (nguồn phả hệ) |
| `data/mitre/` (technique↔tactic CSV) | ánh xạ `(technique_id, tactic_shortname)` (nguồn cạnh tactic) |
| `mitre_technique_embeddings.npy` | SecureBERT 768-d cho tầng 2 |
| [2026-05-24-v3-smart-both-design.md](../superpowers/specs/2026-05-24-v3-smart-both-design.md) | schema §3 + MSEE §4 (nối tầng 1→2) |
