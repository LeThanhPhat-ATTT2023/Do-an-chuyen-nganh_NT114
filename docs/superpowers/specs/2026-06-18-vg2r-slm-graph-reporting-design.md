# VG²R — Verifiable Graph-Grounded Reporting (SLM đọc đồ thị HGT) — design

**Ngày:** 2026-06-18
**Trạng thái:** Design (chưa triển khai). Đây là spec để chuyển sang `writing-plans`.
**Đóng góp:** Một lớp novelty mới cho GraphSLM-IDS — SLM đọc **trực tiếp** subgraph
giải thích của HGT (thay cho lớp EvidenceBundle-prose hiện tại) và sinh **báo cáo XAI
được-máy-kiểm-chứng** (không ảo giác có bảo đảm), kèm **thang đánh giá kép** (fidelity
giải thích GNN × faithfulness NLG).

> **Ràng buộc kiến trúc (người dùng chốt):** khi triển khai, VG²R **thay thế hẳn**
> đường "SLM đọc graph" hiện tại trong `runtime/slow_path` (serialize + prompt + hậu-kiểm),
> **không** chạy song song. Chỉ giữ lại phần *trích xuất bằng chứng* từ `subgraph_snapshot`
> (`evidence_builder.py`) làm nguồn dữ liệu typed.

---

## 0. TL;DR

- **Vấn đề.** Hiện SLM (Ollama `qwen2.5:3b`) **không** đọc đồ thị HGT — nó đọc một
  `EvidenceBundle` (JSON-prose) đã bị dẹt hoá, lossy, và **không có cơ chế đối soát** câu
  chữ với đồ thị thật ⇒ rủi ro ảo giác, không audit được cho SOC.
- **Mục tiêu.** Báo cáo XAI **chuẩn và chính xác nhất** dựa trên SLM chạy local.
- **Lựa chọn kiến trúc.** *Verifiable Graph-Grounded Reporting* (VG²R): SLM đọc **bản
  serialize không-mất-mát của subgraph HGT** (mọi thực thể có "tay cầm" trích dẫn được),
  rồi một **verifier đối soát từng claim với đồ thị thật** và sửa/loại claim không bám
  bằng chứng. Faithfulness được bảo đảm **by-construction**.
- **Vì sao KHÔNG dùng lớp chiếu (soft-prompt/GraphToken).** Soft token là vector mờ đục,
  *không trích dẫn được* node/edge cụ thể ⇒ không audit ⇒ hại đúng mục tiêu "chính xác".
  GraphToken mạnh ở *graph-reasoning QA*, không tối ưu cho *báo cáo an ninh kiểm chứng được*.
  Nó được giữ làm **ablation tùy chọn** (App 2) để vẫn tuyên bố thêm novelty LLM4Graph,
  **không** phải xương sống.
- **Thang đánh giá kép.** Trục A = fidelity giải thích so với HGT (Fid+/Fid−/sparsity,
  GraphFramEx + robust-fidelity). Trục B = faithfulness văn bản (CGR/HR/NumAcc/FCS/Coverage/
  Plausibility/Safety). Gộp thành composite `F*` + tương quan LLM-judge.
- **Local + tái lập.** Toàn bộ xương sống chạy trên Ollama `qwen2.5:3b` sẵn có; **zero mô
  hình train mới**; deterministic (seed 42); chấm trên cả random + temporal split.

---

## 1. Bối cảnh & động cơ

### 1.1. Hiện trạng (cái bị thay)

Đường slow-path hiện tại (`src/graphslm_ids/runtime/slow_path/`):

```
HGT alert → SlowPathJob(subgraph_snapshot, hgt_attention, predicted_label, confidence)
  → ContextHydrator.hydrate()      (lấy context từ hot/cold store)
  → EvidenceBuilder.build()        (→ EvidenceBundle: typed dataclasses)
  → EvidenceRanker.rank_and_truncate()
  → ReportGenerator.generate()     (nhồi EvidenceBundle JSON vào prompt → Ollama)
  → ReportValidator.validate()     (kiểm tra mỗi claim có [evidence_id])
```

Hạn chế cốt lõi:
1. **SLM không đọc đồ thị** — nó đọc JSON-prose đã dẹt; cấu trúc đồ thị (node typed,
   edge có kiểu/trọng số/provenance, attention, counterfactual) bị mất nhiều.
2. **Không có vòng đối soát ngữ nghĩa** — `ReportValidator` chỉ kiểm tra *hình thức*
   (có chuỗi `[E_xxx]` không), **không** kiểm tra claim có *đúng giá trị/đúng đường*
   trong đồ thị hay không ⇒ ảo giác định lượng/định tính lọt lưới.
3. **TODO chưa nối** — `mitre_topk` runtime đang rỗng (xem `runtime_pipeline.py`), nên
   bằng chứng technique online còn yếu.

### 1.2. Nguyên tắc dự án phải tôn trọng

- **"Zero learned encoders besides HGT itself."** VG²R xương sống **không train mô hình
  mới**. (Verifier NLI dùng cross-encoder *pretrained sẵn*, không train; nó không tham
  gia quyết định phân loại của HGT — HGT vẫn ra nhãn một mình.)
- **Deterministic given seed 42** cho mọi bước ngoài SLM. GraphSerializer deterministic;
  SLM đặt `temperature` thấp + seed cố định để tái lập tối đa.
- **Eval BOTH random + temporal split.** Rubric chấm trên cả hai (nhất quán Smart-BOTH).

### 1.3. Định vị novelty (khung Q1)

| | XG-NID | PacketCLIP | **VG²R (đề xuất)** |
|---|---|---|---|
| Knowledge graph | Không | Không | **Có (MSEE typed, có provenance)** |
| SLM đọc gì | narrative hậu kỳ | encoder học | **serialize subgraph kiểm-chứng-được** |
| Kiểm chứng đầu ra | Không | Không | **máy kiểm chứng 3 tầng + repair** |
| Bảo đảm faithfulness | Không | Không | **by-construction** |
| Thang đo | rời rạc | rời rạc | **rubric kép (fidelity × faithfulness)** |

Hai đóng góp chính: **(i)** vòng *generate→verify-against-graph→repair* cho IDS reporting;
**(ii)** rubric kép hợp nhất hai dòng tài liệu (GNN-explanation fidelity × NLG faithfulness)
— chưa tài liệu IDS nào ghép.

---

## 2. Cơ sở khoa học (2024–2025)

### 2.1. Cách cho LLM/SLM "đọc" đồ thị

- **Taxonomy LLM4Graph** — HKUDS, *A Survey of LLMs for Graphs*, **KDD 2024**
  ([arXiv:2405.08011](https://arxiv.org/abs/2405.08011)): 4 paradigm — *GNN-as-Prefix*,
  *LLM-as-Prefix*, *LLM-Graph Integration*, *LLM-Only*. VG²R xương sống = **LLM-Only với
  graph-as-text**; App 2 (ablation) = *GNN-as-Prefix* (soft token).
- **Talk like a Graph** — Fatemi, Halcrow, Perozzi, **ICLR 2024**
  ([PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/bf72f65f30eedf5d48da6980ee02b589-Paper-Conference.pdf),
  [GraphQA repo](https://github.com/google-research/talk-like-a-graph)): *cách mã hoá đồ
  thị thành text quyết định lớn tới khả năng LLM suy luận*. → GraphSerializer phải chọn
  encoding có chủ đích, không serialize tuỳ tiện.
- **GraphToken / Let Your Graph Do the Talking** — Perozzi et al., 2024
  ([arXiv:2402.05862](https://arxiv.org/pdf/2402.05862)): GNN encode subgraph → *soft
  token* cho LLM (LLM đóng băng, chỉ train encoder). → cơ sở cho App 2 ablation.
- **GraphGPT** — Tang et al., SIGIR 2024: graph instruction tuning, projector + (tuỳ chọn)
  LoRA. → cơ sở cho biến thể App 2 có LoRA.

### 2.2. GNN→LLM giải thích trong an ninh

- *From Nodes to Narratives: Explaining GNNs with LLMs and Graph Context*, 2025
  ([arXiv:2508.07117](https://arxiv.org/html/2508.07117v1)).
- *NIDS thế hệ mới với LLM + GNN + XAI*, **Springer JIIS 2025**
  ([DOI](https://link.springer.com/article/10.1007/s10844-025-00964-2)).
- *ContextualGraph-LLM* (darknet, GNN+LLM multi-label), Expert Systems w/ Apps 2025.

### 2.3. Thang đánh giá

- **GraphFramEx** — Amara et al. ([arXiv:2206.09677](https://arxiv.org/abs/2206.09677)):
  **Fid+ / Fid− / sparsity**, phân loại explanation *necessary / sufficient*,
  *characterization score*.
- **Robust Fidelity** — ICLR 2024 ([arXiv:2310.01820](https://arxiv.org/pdf/2310.01820)):
  vá thiên lệch subgraph-OOD của Fid+/Fid−.
- **F-Fidelity** — 2024 ([arXiv:2410.02970](https://arxiv.org/html/2410.02970v2)): khung
  faithfulness robust cho XAI.
- **BAGEL** ([arXiv:2206.13983](https://arxiv.org/pdf/2206.13983)): 4 khía cạnh —
  *faithfulness, sparsity, correctness, plausibility*.
- **Faithfulness vs factuality / hallucination** — review
  ([arXiv:2501.00269](https://arxiv.org/abs/2501.00269)): phân biệt *faithfulness* (bám
  ngữ cảnh) vs *factuality* (đúng thế giới thực); **LLM-as-judge** + **NLI cross-encoder
  (HHEM)** là metric tương quan cao nhất với người.

---

## 3. Kiến trúc VG²R

### 3.1. Nguồn sự thật

Subgraph giải thích per-alert mà HGT thực sự dùng — đã có dưới dạng `subgraph_snapshot`:
node `flow`/`packet`/`technique`/`tactic`, edge có kiểu + trọng số, **attention HGT**,
**counterfactual drop**, logits/confidence. VG²R **không** tạo nguồn mới; nó thay lớp
*trình bày*: EvidenceBundle-prose → graph-text kiểm-chứng-được.

### 3.2. Module (3 mới, 2 tái dùng)

| Module | Trạng thái | Vai trò |
|---|---|---|
| **SubgraphExtractor** | tái dùng (`SubgraphBuilder` + `evidence_builder`) | lấy k-hop subgraph + attention + cf-drop quanh flow bị cảnh báo; xuất EvidenceBundle typed |
| **GraphSerializer** ⭐ | **mới** (`graph_serializer.py`) | EvidenceBundle → graph-text tối ưu cho LLM (Talk-like-a-Graph). Deterministic |
| **GroundedGenerator** | tái dùng + viết lại prompt (`report_generator.py`) | Ollama `qwen2.5:3b`; system prompt ép "chỉ dùng graph-text, mọi claim cite tay cầm node/edge" |
| **GraphVerifier** ⭐ | **mới** (`graph_verifier.py`) | 3 tầng symbolic ▸ path ▸ NLI → nhãn từng claim + faithfulness score |
| **RepairLoop + Fallback** | mới + tái dùng `fallback_template` | claim không grounded → re-prompt có đánh dấu (≤N) → vẫn fail thì template |

### 3.3. GraphSerializer — đặc tả encoding

Đầu ra là một khối Markdown/text **deterministic** gồm 4 phần; mọi thực thể có **tay
cầm trích dẫn ổn định** (`[F0]`, `[P3]`, `[T1190]`, `[TA0001]`):

1. **Decision header** — nhãn HGT dự đoán, confidence, top-k thay thế, threshold.
2. **Node table** (theo kiểu) — mỗi node: handle + thuộc tính. Packet kèm `attn`,
   `cf_drop`, `payload_preview` (ASCII/hex đã cắt). Technique kèm `cosine`, `mapping_type`.
3. **Edge list** — quan hệ có kiểu + trọng số + **provenance** (PMI / procedure / consensus).
4. **Salience block** — top-k node attention-cao + cf-drop (để SLM ưu tiên đúng bằng chứng).

Quy ước:
- **Thứ tự ổn định**: node sắp theo `(−attention, handle)`; edge theo `(src_handle, type, dst_handle)`.
- **Không mất mát ở mức cần audit**: mọi số được cite trong báo cáo phải truy được về node/edge.
- **Encoding A/B-able**: hàm encode tách riêng để chạy ablation A3 (biến thể Talk-like-a-Graph).

Ví dụ (rút gọn):
```
## ALERT A_001 — HGT decision
pred=SqlInjection conf=0.88 ; top2=Benign 0.07 ; threshold=0.70

## NODES
flow [F0]  proto=TCP dport=80 dur=2.1s pkts=14
pkt  [P3]  attn=0.82 cf_drop=0.31 payload_ascii="GET /?q=1' OR 1=1--"
tech [T1190] cosine=0.71 mapping=pmi+procedure name="Exploit Public-Facing App"
tactic [TA0001] name="Initial Access"

## EDGES
[F0] -contains-> [P3]                w=1.00
[P3] -matches_technique-> [T1190]     w=0.71 src=PMI+procedure
[T1190] -belongs_to-> [TA0001]        w=1.00

## SALIENCE (top-k by HGT attention)
1) [P3] attn=0.82 cf_drop=0.31
```

### 3.4. GraphVerifier — 3 tầng (lõi novelty)

Verifier nhận `(report, graph_text, subgraph)` → trả về danh sách claim kèm nhãn
`{supported, unsupported, contradicted}` + faithfulness record.

1. **Symbolic citation check** (deterministic, *cổng cứng*):
   - Mọi tay cầm được cite (`[P3]`, IP, port, packet_id, technique_id) **phải tồn tại**
     trong subgraph.
   - Mọi **số định lượng** (conf, attention, cf_drop, port…) **phải khớp** giá trị graph
     trong dung sai `τ` (mặc định: `|Δ| ≤ 0.01` cho xác suất/attention; khớp tuyệt đối cho
     IP/port/ID). → quyết định `NumAcc`.
2. **Path / provenance check**:
   - Mọi đường flow→packet→technique→tactic được khẳng định **phải là path thật** (mọi
     edge tồn tại trong subgraph). Sai một cạnh ⇒ `contradicted`.
3. **NLI entailment check** (cho claim *định tính*):
   - Mỗi câu phải được graph-text **entail** bởi một cross-encoder NLI/HHEM *pretrained*.
   - Ngưỡng entailment `θ` (mặc định 0.5) tách `supported` vs `unsupported`.
   - NLI **chỉ** áp cho claim không định lượng (định lượng đã do tầng symbolic xử).

**Model NLI mặc định:** một cross-encoder NLI gọn chạy CPU/L40S (ví dụ họ
`cross-encoder/nli-*` hoặc HHEM-2.1). *Cấu hình hoá* để đổi model; ghi rõ phiên bản trong
faithfulness record để tái lập.

### 3.5. RepairLoop + Fallback

- Claim `unsupported`/`contradicted` → **re-prompt** SLM, *đánh dấu* các câu lỗi
  ("các câu sau không bám bằng chứng; sửa hoặc loại"), tối đa `N` vòng (mặc định 2).
- Vẫn fail → **drop** câu lỗi hoặc rơi về `fallback_template` (đảm bảo luôn có báo cáo
  grounded). → đây là chỗ tạo **bảo đảm faithfulness by-construction**.
- Mọi vòng repair ghi vào faithfulness record (tier cuối, số câu bị sửa/loại).

### 3.6. Luồng dữ liệu

```
HGT alert
  → subgraph_snapshot (sẵn có: node/edge + attention + cf_drop + logits)
  → EvidenceBuilder (tái dùng) → EvidenceBundle typed
  → GraphSerializer  → graph-text (mỗi thực thể có handle trích dẫn)
  → SLM (grounded generate, Ollama qwen2.5:3b, temp thấp, seed cố định)
  → GraphVerifier   (symbolic ▸ path ▸ NLI) → nhãn claim + faithfulness record
      ├─ pass     → emit report + faithfulness record
      └─ fail     → RepairLoop (re-prompt ≤N) → vẫn fail → template fallback
  → persist (report, bundle, faithfulness record, fallback_tier)
```

---

## 4. Thang đánh giá kép (rubric chi tiết)

Ký hiệu: `E_cited` = tập node/edge báo cáo trích dẫn; `f(·)_c` = xác suất HGT cho lớp dự
đoán `c`; `G` = subgraph đầy đủ.

### 4.1. Trục A — Fidelity giải thích so với HGT *(GNN-explainability)*

Tận dụng đúng cỗ máy mask-rồi-chạy-lại-HGT đã có trong `evidence_builder` (counterfactual).

| Metric | Công thức / cách đo | Tốt khi | Nguồn |
|---|---|---|---|
| **Fid+ (necessity)** | `f(G)_c − f(G∖E_cited)_c` — bỏ bằng chứng được cite, conf tụt nhiều | **cao** | GraphFramEx |
| **Fid− (sufficiency)** | `f(G)_c − f(E_cited)_c` — chỉ giữ bằng chứng được cite, conf đổi ít | **thấp** | GraphFramEx |
| **Characterization** | harmonic-mean có trọng số của `Fid+` và `1−Fid−` | **cao** | GraphFramEx |
| **Sparsity** | `|E_cited| / |G|` | **thấp** (ở fidelity bằng nhau) | GraphFramEx |
| **Robust-Fidelity** | bản OOD-robust của Fid+/Fid− (báo cáo kèm bản thô) | — | Robust-Fid ICLR'24, F-Fidelity |

### 4.2. Trục B — Faithfulness & chất lượng văn bản *(NLG-faithfulness)*

| Metric | Cách đo | Tốt khi | Nguồn |
|---|---|---|---|
| **CGR** (Citation Grounding Rate) | % claim có ≥1 trích dẫn graph được verifier xác nhận | ≈ **1.0** (by-construction) | grounding/RAG |
| **HR** (Hallucination Rate) | % claim bị gán `unsupported`/`contradicted` | **thấp** | faithfulness vs factuality |
| **NumAcc** | % claim định lượng khớp graph trong dung sai `τ` | **cao** | symbolic (tự định nghĩa) |
| **FCS** (Factual Consistency) | entailment-prob trung bình (NLI/HHEM) | **cao** | HHEM-style |
| **Coverage** | recall của top-k node attention-cao của HGT được nhắc tới | **cao** | nối Trục A |
| **Plausibility** | technique được cite khớp class→technique MITRE map + clean-key; chấm phụ LLM-judge | **cao** | BAGEL |
| **Safety** | không lộ bước khai thác/offensive (rule validator sẵn có) | **pass** | giữ rule hiện tại |

### 4.3. Gộp điểm + giao thức chấm

- **Composite** `F* = HMean(CGR, 1−HR, NumAcc, FCS, Characterization)` — một số xếp hạng.
- **LLM-as-judge phụ trợ**: judge mạnh (Claude/GPT-4) chấm plausibility/clarity 1–5, rồi
  **tương quan Pearson/Spearman** với metric symbolic để *validate* metric rẻ tiền khớp
  đánh giá người.
- **Bootstrap 95% CI (seed 42, 1000 resample)** cho mọi số — đồng nhất phong cách dự án.
- **Chấm trên cả random + temporal split** (nhất quán Smart-BOTH).
- **Human spot-check** một mẫu nhỏ cho plausibility (BAGEL).

### 4.4. Baseline & ablation

- **B0**: EvidenceBundle-prose→SLM **hiện tại** (cái bị thay) — đo mức cải thiện.
- **B1**: narrative hậu kỳ kiểu **XG-NID** (không verify). **B2**: template-only.
- **Ablation**: (A1) bỏ repair-loop; (A2) bỏ tầng NLI; (A3) đổi encoding graph (biến thể
  Talk-like-a-Graph); (A4) nhánh GraphToken soft-prompt (App 2) — so LLM4Graph.

---

## 5. Phạm vi build & phân pha

### 5.1. Thay thế (đúng ràng buộc người dùng)

- **`report_generator.py`**: prompt viết lại để đọc **graph-text** thay JSON-prose. *Thay hẳn.*
- **`graph_serializer.py`** (mới): EvidenceBundle → graph-text deterministic.
- **`graph_verifier.py`** (mới): 3 tầng symbolic ▸ path ▸ NLI + nhãn claim.
- **`slow_path_worker.py`**: chèn serializer trước generator, verifier+repair sau; nhánh
  prose cũ **bị gỡ**.
- **Giữ** `evidence_builder.py` làm nguồn typed (không phí công đã có).
- **`scripts/eval/vg2r_report_eval.py`** (mới): rubric kép + CI + tương quan LLM-judge.

### 5.2. Phân pha (cho `writing-plans`)

1. **GraphSerializer** + test (golden serialization, determinism).
2. **GroundedGenerator** đổi prompt sang graph-text + thay đường trong worker.
3. **GraphVerifier** (symbolic→path→NLI) + test (chèn ảo giác giả phải bị bắt).
4. **RepairLoop + fallback** tích hợp.
5. **Eval harness**: Trục A (Fid+/Fid−/sparsity/robust) + Trục B (CGR/HR/NumAcc/FCS/Cov/
   Plaus/Safe) + composite + CI + tương quan judge; chạy cả hai split.
6. *(Tùy chọn)* nhánh **GraphToken App 2** cho ablation LLM4Graph (cần đổi engine HF/vLLM
   + dữ liệu train projector; tách spec riêng nếu làm).

### 5.3. Kiểm thử

- **Unit**: serializer deterministic (golden file); verifier bắt được báo cáo gài sai
  `packet_id`/sai số/đường không tồn tại; dung sai số; path-check; NLI mock.
- **Integration**: end-to-end vài alert thật (CPU smoke), đảm bảo có faithfulness record,
  repair-loop có chặn vòng, fallback hoạt động.
- **Tái lập**: seed 42, cả hai split; ghi phiên bản model NLI + SLM trong record.

### 5.4. Bản đồ file (dự kiến)

| File | Loại | Vai trò |
|---|---|---|
| `src/graphslm_ids/runtime/slow_path/graph_serializer.py` | mới | EvidenceBundle → graph-text |
| `src/graphslm_ids/runtime/slow_path/graph_verifier.py` | mới | verifier 3 tầng + faithfulness record |
| `src/graphslm_ids/runtime/slow_path/report_generator.py` | sửa | prompt graph-text (thay prose) |
| `src/graphslm_ids/runtime/slow_path/slow_path_worker.py` | sửa | nối serializer + verifier + repair, gỡ nhánh cũ |
| `scripts/eval/vg2r_report_eval.py` | mới | rubric kép + CI + judge |
| `tests/runtime/slow_path/test_graph_serializer.py` | mới | golden + determinism |
| `tests/runtime/slow_path/test_graph_verifier.py` | mới | bắt ảo giác gài sẵn |
| `tests/runtime/slow_path/test_vg2r_end_to_end.py` | mới | smoke E2E + repair/fallback |

---

## 6. Quyết định thiết kế (chốt)

- **Engine local:** giữ **Ollama `qwen2.5:3b`** cho xương sống (graph-as-text không cần
  inject embedding). Chỉ App 2 (ablation) mới cần HF/vLLM.
- **Thay vì song song:** đường prose cũ bị **gỡ hẳn** khi VG²R vào.
- **Dung sai NumAcc `τ`:** `0.01` cho xác suất/attention; tuyệt đối cho ID/IP/port. *Cấu hình hoá.*
- **Số vòng repair `N`:** mặc định **2**. *Cấu hình hoá.*
- **Model NLI:** cross-encoder pretrained gọn (cấu hình hoá; ghi version trong record).
- **Determinism:** GraphSerializer + verifier deterministic; SLM temp thấp + seed cố định.

---

## 7. Giới hạn (trung thực)

- **SLM 3b** fluency thấp hơn model lớn → giảm thiểu bằng repair-loop; có thể distill sau
  (không thuộc xương sống).
- **NLI verifier tự nó có thể sai** → tầng **symbolic làm cổng cứng**; NLI chỉ lo claim
  định tính, và lỗi NLI được định lượng qua tương quan với LLM-judge/human.
- **Fid+/Fid− có caveat OOD** (subgraph sau khi mask có thể lệch phân phối) → báo cáo kèm
  **bản robust** (Robust-Fid / F-Fidelity).
- **Plausibility** cần một phần human/LLM-judge → có yếu tố chủ quan; nêu rõ + báo cáo CI.
- **App 2 (GraphToken)** nếu làm sẽ phá tạm nguyên tắc zero-encoder và cần đổi engine +
  dữ liệu train → tách spec riêng, coi là extension.

---

## 8. Tài liệu tham khảo

1. HKUDS. *A Survey of Large Language Models for Graphs.* KDD 2024. [arXiv:2405.08011](https://arxiv.org/abs/2405.08011)
2. Fatemi, Halcrow, Perozzi. *Talk like a Graph: Encoding Graphs for LLMs.* ICLR 2024.
3. Perozzi et al. *Let Your Graph Do the Talking (GraphToken).* 2024. [arXiv:2402.05862](https://arxiv.org/pdf/2402.05862)
4. Tang et al. *GraphGPT: Graph Instruction Tuning.* SIGIR 2024.
5. *From Nodes to Narratives: Explaining GNNs with LLMs and Graph Context.* 2025. [arXiv:2508.07117](https://arxiv.org/html/2508.07117v1)
6. *Leveraging LLMs, GNNs, and XAI for next-gen NIDS.* Springer JIIS 2025. [DOI](https://link.springer.com/article/10.1007/s10844-025-00964-2)
7. Amara et al. *GraphFramEx: Systematic Evaluation of Explainability for GNNs.* [arXiv:2206.09677](https://arxiv.org/abs/2206.09677)
8. *Towards Robust Fidelity for Evaluating Explainability of GNNs.* ICLR 2024. [arXiv:2310.01820](https://arxiv.org/pdf/2310.01820)
9. *F-Fidelity: A Robust Framework for Faithfulness Evaluation of XAI.* 2024. [arXiv:2410.02970](https://arxiv.org/html/2410.02970v2)
10. *BAGEL: A Benchmark for Assessing GNN Explanations.* [arXiv:2206.13983](https://arxiv.org/pdf/2206.13983)
11. *A review of faithfulness metrics for hallucination assessment in LLMs.* 2025. [arXiv:2501.00269](https://arxiv.org/abs/2501.00269)

---

## Phụ lục — quan hệ với các lớp novelty hiện có

VG²R là **lớp 5** bổ sung cho 4 lớp trong `CLAUDE.md` (MSEE / typed schema / Smart-BOTH /
EACS+clean-key). Nó *tiêu thụ* đầu ra của MSEE (graph có provenance) và *giải thích* quyết
định HGT — không thay đổi pipeline phân loại, chỉ thêm tầng báo cáo kiểm-chứng-được.
