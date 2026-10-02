# FL-HGT poisoning testbed — thiết kế

**Ngày:** 2026-10-02
**Nhánh:** `feat/fl-poisoning-testbed`
**Đầu vào:** `Tong_hop_4_bai_bao_Anti_Poisoning_FL_HGT_IDS.docx` (SecureDyn-FL, BPFLH,
multi-signal fingerprinting, dual watermark) + đề cương FL đã chốt
(`docs/superpowers/specs/2026-09-10-fl-poisoning-proposal-design.md`).

## 1. Mục tiêu và phạm vi

Xây dựng bộ mô phỏng Federated Learning cho HGT flow classifier của GraphSLM IDS, đủ để
chạy toàn bộ ma trận thực nghiệm chiều **client → server**:

- Phân vùng phần train thành K client mô phỏng (IID, Dirichlet non-IID).
- Mỗi client giữ graph con riêng; sampling không vượt biên client.
- Local HGT training tại client (focal + CB, GCL, EACS), chỉ gửi `Δ_k = w_k − W_t`.
- Năm model-poisoning attack: sign-flip, ALIE, Min-Max, Min-Sum, ScaleSign.
- Robust aggregation: FedAvg, Median, Trimmed Mean, Multi-Krum, SignGuard, MSGuard, DnC,
  norm clipping, relation-aware masking (đặc thù HGT).
- Ba bộ sàng lọc chỉ dựa trên delta: TemporalAudit (lấy ý tưởng SecureDyn-FL),
  MultiSignal (bản gọn của OptiGradTrust), BrayCurtis (BPFLH, bản plaintext).
- Baseline cùng trainer: `centralized`, `local_only`, `fl`.
- Matrix runner + summarizer cho 4 điều kiện (có/không attacker × FedAvg/defense).

**Ngoài phạm vi đợt này:** thí nghiệm MITM chiều server → client (xác thực kênh/round,
watermark) — spec riêng sau; secure aggregation/DP; data/backdoor poisoning; sửa trainer
tập trung `train_hgt_flow_classifier.py` (giữ nguyên để không ảnh hưởng kết quả EACS v2).

Train thật chạy ở máy khác (L40S). Máy local chỉ chạy unit test + smoke test CPU trên
graph tổng hợp nhỏ (`outputs/v3_ob` không có ở local).

## 2. Quyết định đã chốt

| Vấn đề | Quyết định |
|---|---|
| Khung FL | Tự viết, không dùng Flower: cần graph con, sampler HGT, attack omniscient cần mọi delta benign của round |
| Trainer | Module FL gọn, tái dụng building block của trainer cũ; bỏ HGAA, TEW, DDP, OOM-retry, EMA |
| Quy mô | Code hỗ trợ cả tầng gọn (`fl_hgt_screen.yaml`) và tầng đầy đủ (`fl_hgt_full.yaml`); runner chạy toàn ma trận, resume được |
| Graph client | Giữ global node ID; giữ cạnh khi mọi đầu flow/packet thuộc client; host/technique/tactic dùng chung |
| Đặc trưng host | Tính lại `host_x` từ flow của client (bản gốc tổng hợp trên toàn bộ flow → rò rỉ chéo client) |
| Chuẩn hóa flow | Round 0: client gửi moment `(n, Σx, Σx²)` của `flow_x`; server ra mean/std chung |
| Đánh giá | Experiment-only: model global chấm trên val/test bằng graph toàn cục (cùng giao thức centralized). Báo cáo cả round cuối và best-val |
| Biết f | Krum/DnC/Trimmed Mean dùng `assumed_malicious_ratio` của defense, **không** lấy từ attack |

## 3. Kiến trúc

Package mới `src/graphslm_ids/offline/federated/`. Mỗi file một trách nhiệm:

```
federated/
  __init__.py
  config.py          # DEFAULT_FL_CONFIG, load_fl_config, --set overrides (dotted keys)
  partition.py       # IID / Dirichlet, subsample, JSON
  client_graph.py    # packet_owner, build_client_artifact, recompute_host_features
  params.py          # ParamLayout: flatten/unflatten, nhóm tham số HGT, active groups
  local_trainer.py   # ClientState, LocalTrainer.train(...) -> LocalUpdate
  attacks.py         # NoAttack, SignFlip, ALIE, MinMax, MinSum, ScaleSign
  aggregators.py     # clip_deltas, FedAvg, Median, TrimmedMean, MultiKrum, SignGuard, MSGuard, DnC
  screening.py       # NoScreen, TemporalAudit, MultiSignal, BrayCurtis
  server.py          # ClientMessage, FLServer (integrity → clip → screen → aggregate)
  metrics.py         # metric phụ (FPR, family recall, detection rates, convergence)
  simulation.py      # run_experiment(): fl / local_only / centralized
  cli.py             # python -m graphslm_ids.offline.federated.cli
scripts/fl/run_matrix.py
scripts/fl/summarize_matrix.py
configs/fl/fl_hgt_screen.yaml, fl_hgt_full.yaml, matrix_screen.yaml, matrix_full.yaml
tests/federated/...
```

Ranh giới quan trọng: `attacks.py`, `aggregators.py`, `screening.py`, `server.py` chỉ làm
việc trên tensor `(N, D)` float32 và `ParamLayout` — **không import HGT/graph**. Nhờ vậy
test nhanh trên vector ngẫu nhiên và có thể tái dụng cho model khác.

### 3.1 Partition (`partition.py`)

- Đầu vào: `train_ids` (global flow id của split đã chọn `random`/`temporal`), `labels`.
- `iid`: hoán vị theo seed, chia đều.
- `dirichlet(alpha)`: với mỗi lớp c, `p ~ Dir(alpha·1_K)`, chia flow lớp c theo `p`;
  thử lại (tối đa 100 lần, seed tăng dần) cho tới khi mọi client có
  ≥ `min_flows_per_client`. Hết lượt → `ValueError`.
- `max_flows_per_client`: subsample phân tầng theo lớp trong từng client (tầng gọn).
- `ClientPartition` lưu `client_flow_ids: list[np.ndarray]` (sorted), `scheme`, `alpha`,
  `seed`; `to_json/from_json`; `class_histogram(labels, C) -> (K, C)`.
- Bất biến: các client rời nhau, hợp lại ⊆ train, không chứa val/test.

### 3.2 Graph con (`client_graph.py`)

- `packet_owner(artifact) -> int64[num_packets]`: flow sở hữu mỗi packet, từ
  `flow__contains__packet` (−1 nếu không có).
- `build_client_artifact(artifact, client_flow_ids, packet_owner, recompute_host=True)`:
  - `owned[flow] = isin(flow, client)`, `owned[packet] = owned_flow[packet_owner]`,
    các node type khác: luôn đúng.
  - Với **mọi** edge key (kể cả `rev_*`): giữ cạnh khi
    `owned[src_type][src] & owned[dst_type][dst]`; lọc `edge_attr` cùng mask.
  - `node_features`: dùng chung mảng của artifact gốc (không copy), trừ `host` (tính lại).
  - `flow_y`, `metadata`: dùng chung (global id ⇒ EACS mask, clean labels vẫn đúng).
- `recompute_host_features`: out-degree, in-degree, số host đích khác nhau tính từ cạnh
  `from_host`/`to_host` của flow client; số port đích khác nhau từ cột `dst_port` của
  `flow_x` (tên lấy trong `metadata.flow_feature_names`; thiếu → cột đó = 0, log cảnh báo).
- Hệ quả: sampler của client chỉ thấy flow/packet của client; technique/tactic là tri
  thức MITRE công khai được replicate.

### 3.3 Tham số (`params.py`)

- `ParamLayout.from_model(model)`: mọi entry float của `state_dict` (HGT không có buffer),
  thứ tự cố định; `numel`, `offsets`.
- `flatten(state_dict) -> Tensor[D]` (float32, CPU), `unflatten(vec) -> state_dict`.
- Nhóm tham số (`group_of(name)`):
  `rel:<src__rel__dst>` cho `layers.*.relation_{key,value,prior}.*`;
  `node:<type>` cho `input_projection.<type>` và `layers.*.{query,key,value,out,norm_attn,norm_ffn,ffn}.<type>`;
  `tactic_embedding`; `classifier`; còn lại `other`.
- `group_index: dict[str, LongTensor]` (chỉ số coordinate), `group_norms(deltas) -> (N, G)`,
  `active_groups(deltas) -> bool (N, G)` (norm nhóm > 0). Relation không xuất hiện ở
  client ⇒ grad None ⇒ AdamW bỏ qua ⇒ delta nhóm đó đúng bằng 0.

### 3.4 Local trainer (`local_trainer.py`)

- `ClientState` (bền qua round): `client_id`, `flow_ids`, `backend`
  (`InMemoryNeighborBackend` trên graph con), `sampler`, `class_weights`,
  `eacs` (`EACSController` riêng, dựng từ cùng anchor mask), `optimizer_state | None`,
  `rounds_trained`.
- `LocalTrainer(model, edge_types, device, cfg, gcl_map)`; `train(state, global_state_dict,
  round_idx, lr) -> LocalUpdate(delta: Tensor[D], num_samples, metrics, seconds, steps)`.
- Vòng train: `make_neighbor_loader(state.flow_ids, state.sampler, cfg_dict, shuffle=True,
  class_labels=...)` → `to_torch_batch` → `_drop_edges` → forward (dùng `encode` +
  `classifier` khi GCL bật) → loss (EACS: `soft_targets` + `soft_target_focal_loss`,
  consensus qua `burst_neighbor`; không EACS: `_compute_train_loss`) + `gcl_weight·gcl`
  (ánh xạ technique global→local như trainer cũ) → backward → clip grad → AdamW step.
  Bỏ batch có loss không hữu hạn.
- `local_epochs` epoch, dừng sớm ở `max_local_steps` nếu đặt.
- AdamW: `optimizer_state: reset` (mặc định FL) hoặc `persist` (centralized/local-only;
  lưu state trên CPU).
- LR theo round: `cosine_rounds`: `lr_r = lr_min + ½(lr − lr_min)(1 + cos(π(r−1)/R))`,
  hoặc `constant`.
- EACS warm-up tính theo **round toàn cục** (`epoch = round_idx`), vì độ trưởng thành
  của model global mới là điều warm-up cần.
- Class weight CB tính từ nhãn **cục bộ** của client (`class_weights_from_backend`).
- RNG: sampler seed = `seed + 1000·round + client_id`; `torch.manual_seed` cùng công thức.

### 3.5 Attack (`attacks.py`)

Giao diện: `Attack.craft(ctx) -> Tensor (M, D)`; `ctx` gồm `honest` (delta trung thực của M
attacker tham gia round), `benign` (delta của client lành tham gia), `num_participants`,
`rng`. `knowledge`: `omniscient` (tập tham chiếu = benign, mặc định, kịch bản xấu nhất)
hoặc `partial` (tập tham chiếu = honest của attacker). Không có benign ⇒ dùng honest.
Attack bắt đầu từ `start_round`.

| Attack | Định nghĩa (nguồn) | Mặc định |
|---|---|---|
| `sign_flip` | `Δ_m = −s·Δ_m^honest`, mỗi attacker độc lập | `s = 1.0` |
| `alie` | `s = ⌊n/2+1⌋ − m`; `z = Φ⁻¹((n−s)/n)`; mọi attacker gửi `μ + z·σ` (μ, σ theo coordinate trên tập tham chiếu) — Baruch et al. 2019, Alg. 3; n = số client tham gia | `z` tính theo công thức gốc; cho phép ghi đè |
| `min_max` | `Δ_m = μ + γ·p`, γ lớn nhất sao cho `max_i ‖Δ_m − Δ_i‖ ≤ max_{i,j} ‖Δ_i − Δ_j‖`; tìm γ theo NDSS'21 Alg. 1 | `p = −std` (tùy chọn `unit`: `−μ/‖μ‖`, `sign`: `−sign(μ)`), `γ_init = 10`, `τ = 1e−5` |
| `min_sum` | như trên với `Σ_i ‖Δ_m − Δ_i‖² ≤ max_i Σ_j ‖Δ_i − Δ_j‖²` (Eq. 7, dùng max như bài báo; code chính thức dùng min — ghi chú) | như Min-Max |
| `scale_sign` | Yang et al. TIFS 2025 §V: `ḡ` = mean honest của attacker; `φ_r = γ` nếu `ḡ_r > 0`, `1/γ` nếu `< 0`, `0` nếu `= 0`; γ ∈ [1, γ_init] lớn nhất thỏa ràng buộc Min-Max trên tập tham chiếu (oracle NDSS — **sửa** Alg. 2 in sai vì J được kiểm trên g* cũ); sau đó chỉnh số coordinate dương/âm về `(P̄, N̄)` chọn kiểu Krum trên đếm dấu của attacker (Eq. 10), theo quy tắc đặt 0 / đặt τ của Alg. 4 | `γ_init = 10`, `τ = 1e−3`, `constraint = min_max` |

Mọi attacker trong ALIE/Min-Max/Min-Sum/ScaleSign gửi cùng vector (collusion theo
định nghĩa gốc). `attacker_trains: auto` — attacker chỉ train trung thực khi attack cần
`honest` (sign_flip, scale_sign, hoặc `knowledge: partial`).

### 3.6 Aggregator (`aggregators.py`)

Giao diện: `Aggregator.aggregate(deltas (N, D), weights (N,), ctx) -> AggResult(delta (D,),
selected: bool (N,) | None, scores: dict, info: dict)`.

| Tên | Định nghĩa | Mặc định |
|---|---|---|
| `fedavg` | `Σ w_i Δ_i / Σ w_i` | `weighting: samples` |
| `median` | median theo coordinate (N chẵn: trung bình hai giá trị giữa) | |
| `trimmed_mean` | bỏ k lớn nhất và k nhỏ nhất mỗi coordinate, `k = ⌊β·N⌋`, cần `k < N/2` | `β = assumed_malicious_ratio` |
| `multi_krum` | điểm Krum `s(i) = Σ` bình phương khoảng cách tới `N−f−2` láng giềng gần nhất; chọn lặp m vector (Blanchard 2017), trung bình | `f = ⌊assumed·N⌋`, `m = N − f` |
| `signguard` | S1 = `{0.1 ≤ ‖Δ_i‖/med ≤ 3.0}`; đặc trưng = tỷ lệ dương/0/âm trên một khối liên tục 10% coordinate (vị trí ngẫu nhiên), chia max theo client; MeanShift(`estimate_bandwidth(q=0.5)`, `bin_seeding`, `cluster_all=False`), S2 = cụm lớn nhất; trung bình `Δ_i·min(1, med/‖Δ_i‖)` trên S1∩S2 (code chính thức SignGuard); tùy chọn `variant: sim` thêm cosine tới aggregate round trước | `L=0.1, R=3.0, ratio=0.1` |
| `msguard` | Yang et al. Alg. 8: lọc norm như SignGuard; điểm cosine (median cosine tới client khác, scale về [0, 0.2]); điểm outlier phổ (top/bottom ζ/2 coordinate của mean, vector kỳ dị chính, scale [0, 0.2]); MeanShift trên `(P, Z, N, cos, outlier)`; trung bình clipped trên giao. Bài báo không cho tham số mặc định → dùng quy ước SignGuard, `ζ = 10000` (ghi rõ là dựng lại) | |
| `dnc` | NDSS'21 Alg. 2: `b` chiều ngẫu nhiên, tâm hóa, vector kỳ dị chính, điểm `⟨Δ_i − μ, v⟩²`, giữ `N − c·f` thấp nhất, giao qua `niters` | `niters=1, c=1, b=10000` |

Tiền xử lý: `clip.mode: none | median | fixed` — co mỗi `Δ_i` về `factor·median_norm`
hoặc `value`. `relation_aware: true` (chỉ cho `fedavg`/`median`/`trimmed_mean`): với mỗi
nhóm tham số, chỉ tổng hợp trên client có nhóm hoạt động; nhóm không ai hoạt động → 0.
Dùng cho aggregator khác ⇒ `ValueError`.

### 3.7 Sàng lọc (`screening.py`)

Giao diện: `Screener.screen(deltas, client_ids, round_idx, layout, prev_aggregate) ->
ScreenResult(trust: float (N,) ∈ [0,1], decision: list["accept"|"downweight"|"reject"],
features: dict[str, ndarray])`. Server lấy trọng số cuối `= trust · w_i`, bỏ client
`reject`, rồi gọi aggregator trên phần còn lại. Screener giữ state qua round.

**Tham chiếu dùng chung:** `ref = prev_aggregate` (aggregate round trước); round đầu dùng
median theo coordinate của round hiện tại.

**TemporalAudit** (lấy ý tưởng SecureDyn-FL; bài báo không công bố tập đặc trưng, ngưỡng
số, α, và auditor của họ học có giám sát trên gradient độc hại có nhãn — đây là bản
**dựng lại không giám sát**, phải ghi rõ khi viết luận văn):

- Đặc trưng mỗi client: `log(‖Δ_k‖ / median_norm)`, `cos(Δ_k, ref)`, tỷ lệ đồng thuận dấu
  với `ref`, median cosine tới client khác, độ lệch nhóm relation
  `max_g |log(‖Δ_k[g]‖ / median_j ‖Δ_j[g]‖)|` trên nhóm hoạt động (tín hiệu đặc thù HGT).
- Mô hình quần thể: tâm/scale robust (median, 1.4826·MAD, sàn 1e−6) trên client của round,
  làm mượt qua round `θ_t = α·θ_{t−1} + (1−α)·θ_new` (Eq. 24), `α = 0.7`; θ_new chỉ lấy từ
  client được accept. Khoảng cách Mahalanobis đường chéo `MD_k`.
- Tín hiệu thời gian: `ΔMD_k = |MD_k(t) − MD_k(t−1)|` (Eq. 25) và
  `cos(sketch_k^t, h_k)` với `h_k` là EMA (0.5) của count-sketch 4096 chiều các delta đã
  accept của client k (bộ nhớ O(K·4096)).
- Ngưỡng: `T_MD = k_md · σ_MD` (Eq. 26; `σ_MD` = std robust của MD trên client accept,
  làm mượt EMA), `k_md = 3`; ΔMD > median + 3·MAD lịch sử; norm > R·EMA(median norm), R = 3.
- Quyết định: không vượt ngưỡng nào → `accept` (trust 1); vượt đúng một ngưỡng với mức
  < 2× ngưỡng → `downweight` (trust 0.5); còn lại → `reject` (0).
- Warm-up `warmup_rounds = 2`: chỉ dùng MD cắt ngang + norm, bỏ ΔMD và tín hiệu sketch.

**MultiSignal** (bản gọn của OptiGradTrust — Karami et al.; bỏ VAE, Shapley, DDQN vì cần dữ
liệu server/học thêm):

- Tín hiệu: `cos(Δ_k, ref)` (f2), cosine trung bình tới client khác (f3), `log‖Δ_k‖` (f4),
  tỷ lệ đồng thuận dấu với `ref` (f5); tùy chọn tín hiệu độ lệch nhóm relation như trên.
- Chuẩn hóa robust z (median/MAD) trong round. Nghi ngờ mỗi tín hiệu:
  `clip(−z/3, 0, 1)` cho tín hiệu kiểu tương đồng, `clip(|z|/3, 0, 1)` cho norm và độ lệch
  nhóm. Trọng số đều (giống ablation rule-based của bài báo).
- `trust_now = 1 − mean(nghi ngờ)`; lịch sử `trust_k = 0.8·trust_now + 0.2·trust_k^{t−1}`
  (β = 0.8 như EMA Shapley của bài báo).
- `trust < 0.5` → `reject` (ngưỡng 0.5 của bài báo); `0.5 ≤ trust < 0.8` → `downweight`
  (trọng số = trust); còn lại `accept`.

**BrayCurtis** (BPFLH Eq. 1, 16, 17; bỏ CKKS vì chỉ mô phỏng):

- `variant: abs` (đúng bài báo): `BC_ij = Σ_k ||g_i[k]| − |g_j[k]|| / Σ_k (|g_i[k]| + |g_j[k]|)`.
  `variant: signed` (mở rộng của ta): áp dụng trên `[Δ⁺, Δ⁻]` để nhạy dấu — bản `abs` mù
  sign-flip (BC(g, −g) = 0), cần ablation.
- `C_i = mean_{j≠i} BC_ij`; `θ = median(C) + m·std(C)`, `m = 0.5`; `C_i > θ` → `reject`.
- Danh tiếng: khởi tạo 1.0, mỗi lần bị loại −0.25; `< 0` → loại vĩnh viễn.
- Tính L1 cặp bằng `torch.cdist(p=1)` theo khối coordinate để giới hạn bộ nhớ.

### 3.8 Server (`server.py`)

`ClientMessage(client_id, round_idx, delta, num_samples)`. `FLServer.step(messages)`:

1. Kiểm tra toàn vẹn: đúng round, đúng `D`, toàn bộ hữu hạn → nếu sai: `invalid`, loại.
2. `clip` → `screen` → loại `reject` → `aggregate` (trọng số `trust·w`).
3. `W_{t+1} = W_t + η·Δ_agg` (`server_lr`, mặc định 1.0). Ghi `prev_aggregate`.
4. Trả `RoundReport`: id client, quyết định (gộp `selected` của aggregator lọc như Krum/
   SignGuard/MSGuard/DnC: không được chọn ⇒ `reject`), trust, đặc trưng, norm, thời gian.

Server không bao giờ nhận cờ "là attacker"; simulation tự đối chiếu sau khi server trả kết
quả để tính chỉ số phát hiện.

### 3.9 Simulation (`simulation.py`)

`run_experiment(cfg, *, out_dir, context=None) -> dict`. `context` (artifact, backend toàn
cục, split, clean labels, cache client) cho phép matrix runner nạp artifact **một lần**.

Chế độ:

- `fl`: vòng round như dưới.
- `centralized`: 1 client chứa toàn bộ train (đã subsample nếu có), `optimizer_state:
  persist`, mỗi round = 1 epoch, không attack, FedAvg.
- `local_only`: mỗi client tự train `rounds × local_epochs` từ cùng khởi tạo, không tổng hợp;
  chấm từng model trên val/test toàn cục; báo cáo mean/std/min/max.

Một round `fl`:

1. Chọn client tham gia (`participation`, seed theo round). Attacker cố định từ đầu:
   `round(malicious_ratio·K)` client chọn theo seed.
2. Client lành (và attacker nếu cần) train tuần tự trên một instance model dùng chung.
3. Từ `start_round`, attacker thay delta bằng `attack.craft`.
4. `server.step` → model global mới.
5. Đối chiếu ground truth → TP/FP/FN/TN phát hiện, loại nhầm client lành.
6. Mỗi `eval.every` round: chấm val (subsample cố định `max_val_flows` nếu đặt) bằng
   `evaluate_neighbor_sampling` + clean key; giữ best theo `eval.monitor`.
7. Kết thúc: chấm test cho model round cuối và best-val; lưu checkpoint.

Round 0 trước vòng lặp: thu moment flow từ client → `flow_feature_stats` chung.

### 3.10 Chỉ số (`metrics.py` + summary)

- Phát hiện IDS: macro-F1, weighted-F1, accuracy (nhãn noisy và clean); recall theo họ
  tấn công (`eval.families`, mặc định web/ddos/recon/mirai/malware) = tổng TP / tổng support;
  FPR Benign → tấn công = `1 − recall(Benign)`; recall nhị phân tấn công (suy ra từ
  precision/recall của Benign).
- Poisoning (theo round và cộng dồn): TPR phát hiện update độc hại, FPR = tỷ lệ loại nhầm
  client lành, tỷ lệ downweight; "malicious acceptance rate" = 1 − TPR (ASR của bài
  ScaleSign).
- Hội tụ: round đầu tiên đạt ≥ 95% val macro-F1 tốt nhất của run.
- Chi phí: thời gian train client/round, thời gian server/round, peak RSS (psutil), peak
  VRAM, bytes lên/xuống = `4·D·|S_t|`.
- Attack impact = F1(không attack, cùng defense, partition, seed) − F1(attack) — tính ở
  summarizer.

### 3.11 Đầu ra một run

`outputs/fl/<run>/`: `config.resolved.yaml`, `partition.json`, `rounds.jsonl`,
`fl_summary.json`, `global_final.pt`, `global_best.pt`. Checkpoint có các key runtime cần:
`model_state_dict, config, node_input_dims, edge_types, num_classes, num_tactics,
label_names, flow_feature_stats, epoch, val_metrics, test_metrics`.

### 3.12 Matrix

`scripts/fl/run_matrix.py --grid configs/fl/matrix_screen.yaml [--device cuda] [--dry-run]
[--only <glob>]`: grid = `base_config` + `axes` (mỗi giá trị là dict override dotted-key)
+ quy tắc: `attack=none` ⇒ `malicious_ratio=0` và chỉ một ô. Chạy tuần tự trong cùng
process, sắp theo partition để tái dụng cache client; bỏ qua ô đã có `fl_summary.json`.
`scripts/fl/summarize_matrix.py --root <dir>` → `matrix_results.csv` + bảng markdown
(macro-F1 noisy/clean, attack impact, TPR/FPR phát hiện, loại nhầm benign).

## 4. Xử lý lỗi

- Config sai (tên attack/aggregator/screener lạ, `k ≥ N/2` cho trimmed mean, Krum thiếu
  `N > 2f + 2`) → `ValueError` khi dựng, trước khi train.
- Dirichlet không đạt `min_flows_per_client` sau 100 lần → `ValueError` gợi ý tăng α.
- Delta không hữu hạn/sai kích thước/sai round → `invalid`, bị loại, ghi log; round không có
  delta hợp lệ → giữ nguyên `W_t`.
- MeanShift không tìm được cụm (bandwidth 0) → giữ toàn bộ S1, ghi `info.fallback`.
- Clean labels/anchor mask sai số flow → `ValueError` (giống trainer cũ).

## 5. Kiểm thử

Mọi test chạy CPU, dưới `tests/federated/`:

- Fixture `tiny_v3_artifact()`: khoảng 60 flow / 3–4 lớp (có `Benign` và một lớp suspect),
  packet, host, `burst_neighbor`, `evidence_*`, `matches_technique`, technique, tactic.
- `partition`: xác định theo seed, rời nhau, phủ đủ, Dirichlet α nhỏ lệch hơn α lớn,
  subsample giữ lớp.
- `client_graph`: không cạnh nào chạm flow/packet ngoài client (cả `rev_*`); cạnh MITRE
  giữ nguyên; host feature tính lại đúng trên ví dụ tay.
- `params`: flatten/unflatten khứ hồi; nhóm relation đúng tên; nhóm không hoạt động = 0.
- `attacks`: hình dạng; sign-flip; z của ALIE khớp ví dụ (n=50, m=24 → z ≈ 1.75); Min-Max /
  Min-Sum thỏa ràng buộc và γ gần cực đại; ScaleSign giữ đúng số dương/âm mục tiêu và thỏa
  ràng buộc.
- `aggregators`: FedAvg khớp trung bình có trọng số; median/trimmed bền với một outlier lớn;
  Krum loại outlier; SignGuard/MSGuard/DnC loại cụm bị lật dấu; relation-aware bỏ qua
  client không hoạt động.
- `screening`: bắt outlier rõ (scale ×50, sign-flip); FPR thấp trên benign Gaussian IID;
  TemporalAudit bắt client đột ngột đổi hành vi sau warm-up; BrayCurtis `abs` không bắt
  sign-flip còn `signed` thì bắt (chính là ablation).
- `server`: cập nhật W đúng; từ chối NaN/sai round.
- Smoke: `run_experiment` 3 client × 2 round × 1 attacker sign-flip + trimmed mean trên
  fixture; tạo đủ file đầu ra; checkpoint dựng lại được model; các chế độ `centralized`,
  `local_only` chạy được.
- Chạy lại toàn bộ `tests/` (baseline: 340 passed, 3 skipped).

## 6. Giới hạn ghi vào luận văn

- Đặc trưng flow dạng cửa sổ (`fan_dst_ip_5s`, …) được xem là đầu ra cảm biến cục bộ.
- `n_k` và moment flow do client tự khai báo có thể bị gian lận; ngoài phạm vi.
- Screening từng client mâu thuẫn với secure aggregation; không giả định cả hai cùng lúc.
- TemporalAudit, MSGuard là bản dựng lại khi bài báo thiếu chi tiết; mọi hằng số là lựa
  chọn của ta và phải báo cáo.
- Đánh giá dùng graph toàn cục (experiment-only), không phải một bước của giao thức FL.
- Kết luận chỉ đúng trên các attack, partition và tỷ lệ độc hại đã thử.
