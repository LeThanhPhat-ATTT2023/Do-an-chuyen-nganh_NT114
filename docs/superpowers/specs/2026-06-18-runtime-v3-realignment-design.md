# Runtime v3 realignment — online MSEE edge-assignment + schema alignment — design

**Ngày:** 2026-06-18
**Trạng thái:** Design (chưa triển khai). Spec để chuyển sang `writing-plans`.
**Mục tiêu:** Đưa **fast-path runtime** (packet → HGT → alert → SLM/VG²R) về **đúng schema
đồ thị v3** mà model EACS đã train, để mô phỏng end-to-end chạy đúng — thay vì rơi vào
schema cũ thời student-CNN.

> **Bối cảnh người dùng:** "tôi đã có mô hình chuẩn HGT (tự lọc nhiễu, EACS), muốn mô phỏng
> end-to-end: 1 packet vào HGT phân loại, nếu là tấn công thì qua SLM đọc XAI. Fix cho
> hoàn chỉnh trước rồi mới chạy."

---

## 0. TL;DR

- **Luồng đúng đã có sẵn** trong code: [run_runtime_pipeline.py](../../src/graphslm_ids/runtime/pipeline/run_runtime_pipeline.py)
  → `FastPathPipeline.on_packet` → `SubgraphBuilder` → `HGTRuntime.infer` → `PolicyEngine`
  → nếu alert: `SlowPathWorker` (**đã là VG²R**).
- **Hợp đồng model/checkpoint CÒN KHỚP**: train và runtime dùng chung class
  `HeteroGraphTransformer`; checkpoint lưu đủ key (`model_state_dict, config,
  node_input_dims, edge_types, num_classes, num_tactics, label_names, flow_feature_stats`)
  và `HGTRuntime` đọc đúng các key đó. → checkpoint EACS **load vào runtime được**.
- **4 chỗ TRÔI cần vá** (fast-path dựng từ schema cũ, train đã tiến hoá sang v3_ob/EACS):
  - **A1** — Gán technique-edge ONLINE chưa nối (`mitre_topk = []`, TODO).
  - **A2** — **Tên edge-type runtime KHÔNG khớp v3** (lỗi nặng nhất): runtime phát
    `matches_technique`/`belongs_to_tactic`; train v3 dùng 5 loại `evidence_<family>` +
    `flow_technique` + `technique_tactic` + `has_subtechnique` + `burst_neighbor`. Model
    **bỏ qua** edge tên lạ ⇒ runtime mất toàn bộ bằng chứng technique.
  - **B** — `packet_feature` mặc định `"semantic"` nhưng model là **v3_ob = ordered-byte**.
  - **C** — Không có config runtime nào (`configs/pipeline.example.yaml` không tồn tại).
  - **D** — Không có test integration fast-path → drift không bị bắt.
- **Nguyên tắc fix:** *tái dùng* đúng các hàm MSEE offline (không reimplement); *làm
  SubgraphBuilder schema-driven* (đọc edge-type từ checkpoint/meta thay vì hardcode).
- **Trung thực:** schema + edge-assigner **test được bằng unit test không cần artifact**;
  **chạy end-to-end thật** vẫn cần `graph.npz` + checkpoint + `pmi_table.parquet` + STIX ở
  local (hiện chưa có).

---

## 1. Hợp đồng train ↔ runtime (đã audit — phần CÒN ĐÚNG)

| Điểm nối | Train | Runtime | Khớp |
|---|---|---|---|
| Model class | `HeteroGraphTransformer` ([models/hgt.py](../../src/graphslm_ids/models/hgt.py)) | cùng ([hgt_runtime.py:48](../../src/graphslm_ids/runtime/fast_path/hgt_runtime.py#L48)) | ✅ |
| Checkpoint keys | [train:2888-2904](../../src/graphslm_ids/offline/training/train_hgt_flow_classifier.py#L2888) | [hgt_runtime:38-59](../../src/graphslm_ids/runtime/fast_path/hgt_runtime.py#L38) | ✅ |
| Slow path | — | VG²R (serialize→verify→repair) | ✅ |

→ **Không sửa gì** ở model/checkpoint/slow-path. Chỉ vá fast-path subgraph.

## 2. Schema edge-type v3 (nguồn sự thật — graph_builder offline)

Tên edge-type **chính tắc** mà model train trên đó ([graph_builder npz keys](../../src/graphslm_ids/offline/preprocessing/graph_builder.py#L855)):

| Quan hệ | Edge type v3 | attr |
|---|---|---|
| flow → packet | `flow_contains_packet` (containment) | — |
| packet → packet | `next_packet` | delta_t |
| flow → flow | `burst_neighbor` | share_src, share_dst |
| packet → technique | `evidence_injection`, `evidence_command_exec`, `evidence_file_upload`, `evidence_recon`, `evidence_c2_beacon` (**routed by family**) | weight |
| flow → technique | `flow_technique` | weight |
| technique → technique | `has_subtechnique` | — |
| technique → tactic | `technique_tactic` | 1.0 |

Reverse edges thêm dạng `rev_<relation>` ([hetero_graph_artifact:153](../../src/graphslm_ids/offline/training/hetero_graph_artifact.py#L153)).

> Tên chính xác của từng edge-type **được lưu trong `checkpoint["edge_types"]`** — runtime
> đã đọc nó (`HGTRuntime.edge_types`). Thiết kế sẽ **lấy danh sách này làm chuẩn** thay vì
> SubgraphBuilder hardcode.

---

## 3. Thiết kế

### A1 — `RuntimeEdgeAssigner` (PMI + procedure online, faithful)

File mới: `src/graphslm_ids/runtime/fast_path/edge_assigner.py`.

**Tái dùng nguyên các hàm offline** ([ensemble.py](../../src/graphslm_ids/offline/preprocessing/ensemble.py),
[procedure_matcher.py](../../src/graphslm_ids/offline/preprocessing/procedure_matcher.py)) —
không reimplement:

```python
class RuntimeEdgeAssigner:
    def __init__(self, pmi_table_path, stix_json_path, technique_family_map, tau_edge=0.4):
        self._pmi_lookup = build_pmi_lookup_from_table(pd.read_parquet(pmi_table_path))
        self._proc = ProcedureMatcher(Path(stix_json_path))
        self._family = dict(technique_family_map)
        self._tau = float(tau_edge)

    def assign_packet(self, payload: bytes,
                      flow_consensus: dict[str, float] | None = None
                      ) -> list[tuple[str, str, float]]:
        if not payload:
            return []
        pmi_hits  = lookup_pmi_per_packet(payload, self._pmi_lookup)
        proc_hits = self._proc.weight_per_technique(payload)
        return aggregate_evidence(pmi_hits, proc_hits, flow_consensus or {},
                                  self._family, tau_edge=self._tau)
```

- Trả về `(technique_id, family, weight)` — **giống hệt offline per-packet**
  ([graph_builder:128-130](../../src/graphslm_ids/offline/preprocessing/graph_builder.py#L128)).
- `flow_consensus`: tính 1 lần / flow bằng `signatures.match_flow_signatures` /
  `flow_consensus` trên flow-features. **Tùy chọn** (chỉ là voter boost ×1.2); v1 có thể
  để `{}` rồi nối sau.
- Artifact nạp 1 lần lúc khởi tạo pipeline: `pmi_table.parquet`, `enterprise-attack.json`
  (STIX), `technique_family_map` (từ `class_technique_map.csv` / cột family của pmi_table).

**Nối vào pipeline:** [runtime_pipeline.py:140-145](../../src/graphslm_ids/runtime/pipeline/runtime_pipeline.py#L140)
thay `mitre_topk = []` bằng `mitre_topk = assigner.assign_packet(payload_bytes, flow_consensus)`.
`payload_bytes` lấy từ `extracted` (online payload extractor đã có raw bytes / hex).

### A2 — SubgraphBuilder realign sang schema v3 (schema-driven)

Sửa [subgraph_builder.py](../../src/graphslm_ids/runtime/fast_path/subgraph_builder.py)
`build()` để phát **đúng tên edge-type v3**:

- **Nhận `canonical_edge_types`** (từ `HGTRuntime.edge_types` / `graph.meta.json`) khi khởi
  tạo, và chỉ phát các edge-type có trong đó (đảm bảo khớp model).
- **packet → technique:** route mỗi `(tech, family, weight)` vào
  `("packet", f"evidence_{family}", "technique")` — 5 loại typed. (Thay cho
  `matches_technique` đơn).
- **flow → technique:** `("flow", "flow_technique", "technique")` (thay `matches_technique`).
- **technique → tactic:** `("technique", "technique_tactic", "tactic")` (thay
  `belongs_to_tactic`).
- **flow → packet:** dùng tên containment v3 (`flow_contains_packet`) thay `contains`.
- **technique → technique:** `has_subtechnique` từ static map (parse `T1190.001 → T1190`
  từ techniques CSV, dùng lại logic [graph_builder._build_has_subtechnique_edges](../../src/graphslm_ids/offline/preprocessing/graph_builder.py#L377)).
- **flow → flow `burst_neighbor`:** runtime 1 flow seed ⇒ không có neighbor ⇒ **để rỗng**
  (model chịu được edge-type rỗng). Ghi rõ là giới hạn.
- **reverse edges:** nếu `add_reverse_edges`, thêm `rev_<relation>` cho mọi quan hệ (khớp
  cách trainer dựng).
- `mitre_topk` trên packet entry đổi format `(tech, score)` → `(tech, family, weight)`;
  cập nhật [hot_graph_buffer](../../src/graphslm_ids/runtime/fast_path/hot_graph_buffer.py)
  + [graph_store](../../src/graphslm_ids/runtime/pipeline/graph_store.py) cho khớp.

### B — packet_feature ordered-byte

- `SubgraphBuilder._packet_features` phải dựng **đúng ordered-byte feature** như v3_ob,
  bằng cách **tái dùng** [payload_features.py](../../src/graphslm_ids/offline/preprocessing/payload_features.py)
  (hàm dựng packet feature offline) thay vì path "semantic".
- Đổi default `packet_feature` và đảm bảo `pipeline_config` + `graph_store` không hardcode
  `"semantic"` ([graph_store.py:624](../../src/graphslm_ids/runtime/pipeline/graph_store.py#L624)).

### C — Config runtime

Tạo `configs/pipeline.example.yaml` (v3_ob): trỏ checkpoint, `graph.meta.json`,
`pmi_table.parquet`, STIX, MITRE CSVs; đặt `hgt.packet_feature: ordered_byte`,
`hgt.add_reverse_edges: true`, slm = Ollama qwen2.5:3b, verifier config. Đặt
`run_runtime_pipeline.py --config` mặc định trỏ file này.

### D — Test integration

`tests/runtime/pipeline/test_fastpath_end_to_end.py`: dựng `FastPathPipeline` với
**model HGT nhỏ synthetic** (`HGTRuntime.from_model`) + buffer in-memory + `RuntimeEdgeAssigner`
giả lập (PMI lookup + procedure nhỏ), bơm vài packet qua `on_packet`, assert: (1) ra
alert khi nhãn tấn công > threshold; (2) subgraph có đúng edge-type v3; (3) slow path
sinh được VG²R report. **Không cần artifact thật.**

---

## 4. Artifact cần lúc CHẠY THẬT (không cần cho unit test)

| Artifact | Dùng cho | Vị trí |
|---|---|---|
| `hgt_flow_best.pt` (checkpoint EACS) | HGTRuntime | server / cần copy local |
| `graph.meta.json` | edge-type schema, protocol/tactic mapping | outputs/v3_ob/ |
| `pmi_table.parquet` | RuntimeEdgeAssigner | outputs/v3_ob/ |
| `enterprise-attack.json` (STIX) | ProcedureMatcher | data/mitre/ |
| MITRE techniques/tactic CSV | technique metadata, has_subtechnique | data/mitre/ |
| PCAP | nguồn packet replay | data/raw/ |

## 5. Bản đồ file

| File | Hành động |
|---|---|
| `src/graphslm_ids/runtime/fast_path/edge_assigner.py` | **mới** — RuntimeEdgeAssigner |
| `src/graphslm_ids/runtime/fast_path/subgraph_builder.py` | sửa — schema v3 + ordered-byte + reverse |
| `src/graphslm_ids/runtime/fast_path/hot_graph_buffer.py` | sửa — mitre_topk format (tech,family,weight) |
| `src/graphslm_ids/runtime/pipeline/runtime_pipeline.py` | sửa — nối assigner, bỏ `mitre_topk=[]` |
| `src/graphslm_ids/runtime/pipeline/pipeline_config.py` | sửa — packet_feature default, artifact paths |
| `src/graphslm_ids/runtime/pipeline/graph_store.py` | sửa — bỏ hardcode "semantic" |
| `configs/pipeline.example.yaml` | **mới** — config v3_ob |
| `tests/runtime/fast_path/test_edge_assigner.py` | **mới** |
| `tests/runtime/fast_path/test_subgraph_schema_v3.py` | **mới** |
| `tests/runtime/pipeline/test_fastpath_end_to_end.py` | **mới** |

## 6. Kiểm thử

- **Unit (không cần artifact):** edge_assigner (PMI+proc nhỏ → đúng (tech,family,weight));
  subgraph phát đúng tên edge-type v3 + reverse + route family; ordered-byte feature khớp
  offline trên payload mẫu.
- **Integration (synthetic model):** on_packet→alert→VG²R.
- **End-to-end thật (cần artifact):** `run_runtime_pipeline.py --input <pcap>` trên một
  mẫu nhỏ; xác nhận alert + báo cáo VG²R có mục MITRE/path không rỗng.

## 7. Giới hạn (trung thực)

- **`burst_neighbor` rỗng ở runtime** (1 flow seed, không có hàng xóm corpus) → model thiếu
  tín hiệu homophily flow-flow so với train. Chấp nhận; ghi rõ.
- **`flow_consensus` v1 có thể để `{}`** (chỉ mất boost ×1.2) — nối signature online sau.
- **Chạy thật phụ thuộc artifact** chưa có local (checkpoint/graph.meta/pmi_table/STIX).
- **packet_feature ordered-byte** phải khớp **bit-by-bit** cách offline dựng, nếu lệch →
  model nhận sai input; test so khớp trên payload mẫu là bắt buộc.

---

## Phụ lục — quan hệ với VG²R

Spec này **không** đụng slow-path (VG²R đã xong). Nó chỉ chữa fast-path để subgraph runtime
khớp model v3 → khi alert đi vào VG²R, `EvidenceBuilder` sẽ có **node technique/tactic thật**
(thay vì rỗng) ⇒ báo cáo XAI có mục MITRE/graph-path đầy đủ.
