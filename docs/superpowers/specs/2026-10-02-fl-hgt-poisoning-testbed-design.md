# FL-HGT poisoning testbed — thiết kế

**Ngày:** 2026-10-02 · **Cập nhật:** 2026-10-15 (điều chỉnh theo yêu cầu của giảng viên)
**Nhánh:** `feat/fl-poisoning-testbed`
**Đầu vào:** `Tong_hop_4_bai_bao_Anti_Poisoning_FL_HGT_IDS.docx` (SecureDyn-FL, BPFLH,
multi-signal fingerprinting, dual watermark) + đề cương FL đã chốt
(`docs/superpowers/specs/2026-09-10-fl-poisoning-proposal-design.md`) + yêu cầu của giảng viên
(mục 0).

## 0. Yêu cầu của giảng viên và câu hỏi nghiên cứu

Yêu cầu của giảng viên cho giai đoạn này:

1. Cần một model baseline.
2. Thiết kế giải pháp phòng thủ (defend).
3. Chọn tỉ lệ chia lớp có kiểm soát (tỉ lệ và số liệu tường minh, không ngẫu nhiên).
4. 20 round, 20 client.
5. Hai mô hình baseline (kiến trúc khác HGT).
6. Chạy và so sánh có FL và không FL.

Mọi thí nghiệm phải phục vụ ít nhất một câu hỏi nghiên cứu:

| Câu hỏi | Nội dung | Yêu cầu |
|---|---|---|
| CH1 | FL với HGT có đạt hiệu năng gần huấn luyện tập trung không, khi dữ liệu chia đều và chia lệch? | 1, 6 |
| CH2 | Poisoning ảnh hưởng HGT khác gì so với mô hình đơn giản hơn (MLP, 1D-CNN)? | 5 |
| CH3 | Các phòng thủ hiện có còn hiệu quả trên HGT không, đặc biệt khi dữ liệu lệch (có loại nhầm client lành không)? | 3 |
| CH4 | Giải pháp đề xuất có tốt hơn không, và mất hiệu lực ở ngưỡng tỉ lệ attacker nào? | 2 |

## 1. Mục tiêu và phạm vi

Xây dựng bộ mô phỏng Federated Learning cho bài toán phân loại flow của GraphSLM IDS, đủ để
chạy toàn bộ ma trận thực nghiệm chiều **client → server**:

- Phân vùng phần train thành **20 client** theo **bảng tỉ lệ tường minh** (3 mức lệch: 0%, 50%,
  80%); chạy **20 round**.
- **Ba mô hình**, cùng dữ liệu, cùng giao thức FL:
  - **HGT** (mô hình chính): mỗi client giữ graph con riêng; sampling không vượt biên client.
  - **MLP** trên 79 đặc trưng flow (`flow_x`): đại diện mô hình phẳng.
  - **1D-CNN** trên chuỗi byte có thứ tự của các packet trong flow: đại diện mô hình học trên
    nội dung packet.
- Local training tại client, chỉ gửi `Δ_k = w_k − W_t`.
- Năm model-poisoning attack: sign-flip, ALIE, Min-Max, Min-Sum, ScaleSign (**đã code**).
- Sáu phòng thủ so sánh: FedAvg, Median, Trimmed Mean, Multi-Krum, SignGuard, DnC; MSGuard
  là tùy chọn (chỉ làm nếu còn thời gian).
- **Giải pháp phòng thủ đề xuất** `RAGuard` (tên tạm): sàng lọc theo từng nhóm quan hệ của HGT
  + lịch sử theo round + tổng hợp bền vững theo nhóm (mục 3.7).
- Baseline cùng trainer: `centralized`, `local_only`, `fl`.
- Matrix runner + summarizer; chỉ số phát hiện tính được ngoại tuyến trên bản cập nhật đã lưu.

**Ngoài phạm vi đợt này:**

- XAI: **giữ trong phạm vi khóa luận, làm ở giai đoạn sau** khi xong phần FL cốt lõi. Đợt này
  chỉ cần lưu checkpoint mô hình chung của các run quan trọng để giai đoạn XAI dùng lại.
- Thí nghiệm MITM chiều server → client (xác thực kênh/round, watermark): hướng phát triển.
- Secure aggregation/DP; data/backdoor poisoning.
- GNN4ID và XGBoost **không** dùng làm baseline FL (GNN4ID dùng pipeline mẫu riêng, không cùng
  flow với v3; XGBoost là mô hình cây, không có bản cập nhật tham số để tổng hợp). Kết quả
  XGBoost tập trung của đồ án chỉ trích dẫn làm bối cảnh.
- Sửa trainer tập trung `train_hgt_flow_classifier.py` (giữ nguyên để không ảnh hưởng kết quả
  EACS v2).

Train thật chạy ở máy khác (L40S). Máy local chỉ chạy unit test + smoke test CPU trên graph
tổng hợp nhỏ (`outputs/v3_ob` không có ở local).

## 2. Quyết định đã chốt

| Vấn đề | Quyết định |
|---|---|
| Khung FL | Tự viết, không dùng Flower: cần graph con, sampler HGT, attack omniscient cần mọi delta benign của round |
| Quy mô | 20 client × 20 round; 100% client tham gia mỗi round |
| Phân vùng | Bảng tỉ lệ tường minh theo nhóm gateway (mục 3.1); số flow xác định, chọn flow theo thứ tự thời gian; IID = mức lệch 0%. Dirichlet giữ trong code nhưng không dùng trong ma trận chính |
| Mô hình | HGT + 2 baseline MLP, 1D-CNN. Một `ModelAdapter` chung để local trainer/simulation không phụ thuộc kiến trúc |
| Trainer | Module FL gọn, tái dụng building block của trainer cũ; bỏ HGAA, TEW, DDP, OOM-retry, EMA |
| Cấu hình HGT | Chọn qua pilot giữa A (64 chiều / 2 layer / 2-hop) và B (128 / 4 / 4-hop); số layer bằng số hop; chế độ `centralized` chạy lại cùng cấu hình (không dùng số 0.858 của đồ án trong bảng so sánh) |
| Local epoch | 1 hoặc 2, chọn qua pilot (FL không tấn công phải hội tụ trong 20 round) rồi cố định |
| EACS | Tắt trong ma trận chính cho mọi mô hình (MLP/CNN không có EACS); thêm thí nghiệm riêng "HGT trên FL có EACS" |
| PMI | Giả định "tri thức chia sẻ": bảng PMI xây một lần và phân phối tới client như bộ luật IDS; ghi rõ hạn chế; ablation bỏ nguồn PMI nếu pipeline cho phép |
| Split | Ma trận dùng split ngẫu nhiên; CH1 chạy cả split ngẫu nhiên và theo thời gian |
| Nhãn | Train trên nhãn gốc (per-pcap); chấm chính bằng nhãn clean, báo cáo kèm nhãn gốc |
| Graph client | Giữ global node ID; giữ cạnh khi mọi đầu flow/packet thuộc client; host/technique/tactic dùng chung |
| Đặc trưng host | Tính lại `host_x` từ flow của client (bản gốc tổng hợp trên toàn bộ flow → rò rỉ chéo client) |
| Chuẩn hóa flow | Round 0: client gửi moment `(n, Σx, Σx²)` của `flow_x`; server ra mean/std chung (dùng cho HGT và MLP) |
| Đánh giá | Experiment-only: model global chấm trên val/test toàn cục (cùng giao thức centralized). **Round cuối là chỉ số chính**; best-val báo cáo kèm |
| Biết f | Krum/DnC/Trimmed Mean dùng `assumed_malicious_ratio = 0.2` của defense, **cố định** trong mọi thí nghiệm, không lấy từ attack |
| Attacker | 20% (4/20) trong ma trận chính; vị trí cố định theo seed; thêm kịch bản cả nhóm attacker nằm trong một nhóm gateway; tấn công từ round 1, thêm kịch bản bắt đầu muộn |
| Seed | 1 seed cho ma trận; 3 seed cho các kết quả chính |

## 3. Kiến trúc

Package `src/graphslm_ids/offline/federated/`. Mỗi file một trách nhiệm:

```
federated/
  __init__.py
  config.py          # DEFAULT_FL_CONFIG, load_fl_config, --set overrides (dotted keys)      [đã code]
  records.py         # to_jsonable, write_json, append_jsonl                                 [đã code]
  linalg.py          # pairwise dist/cosine, median/trimmed mean theo chunk, robust z, SVD   [đã code]
  partition.py       # IID / Dirichlet [đã code]; bảng tường minh `explicit` [mới]
  client_graph.py    # packet_owner, build_client_artifact, recompute_host_features         [đã code]
  params.py          # ParamLayout: flatten/unflatten, nhóm tham số, active groups           [đã code]
  attacks.py         # NoAttack, SignFlip, ALIE, MinMax, MinSum, ScaleSign                   [đã code]
  models.py          # FlowMLP, ByteCNN1D + ModelAdapter (HGTAdapter, MLPAdapter, CNNAdapter) [mới]
  local_trainer.py   # ClientState, LocalTrainer.train(...) -> LocalUpdate (qua ModelAdapter)
  aggregators.py     # clip_deltas, FedAvg, Median, TrimmedMean, MultiKrum, SignGuard, DnC, (MSGuard)
  defense.py         # RAGuard (giải pháp đề xuất) + các cờ ablation                          [mới]
  server.py          # ClientMessage, FLServer (integrity → [defense] → aggregate)
  metrics.py         # FPR, family recall, detection rates, convergence
  offline_eval.py    # chấm lại chỉ số phát hiện trên bản cập nhật đã lưu                     [mới]
  simulation.py      # run_experiment(): fl / local_only / centralized
  cli.py             # python -m graphslm_ids.offline.federated.cli
  matrix.py          # expand_grid, run_grid, summarize (logic cho scripts/fl/*)
scripts/fl/run_matrix.py
scripts/fl/summarize_matrix.py
scripts/fl/export_partition_table.py   # xuất bảng client × lớp (CSV) + biểu đồ nhiệt       [mới]
configs/fl/...                          # cấu hình 3 mô hình, 3 mức lệch, ma trận P0/P1/P2
tests/federated/...
```

Ranh giới quan trọng: `attacks.py`, `aggregators.py`, `defense.py`, `server.py` chỉ làm việc trên
tensor `(N, D)` float32 và `ParamLayout` — **không import model/graph**. Nhờ vậy dùng chung cho
HGT, MLP và CNN, và test nhanh trên vector ngẫu nhiên.

### 3.1 Partition (`partition.py`)

**IID / Dirichlet (đã code):**

- Đầu vào: `train_ids` (global flow id của split đã chọn `random`/`temporal`), `labels`.
- `iid`: hoán vị theo seed, chia đều.
- `dirichlet(alpha)`: với mỗi lớp c, `p ~ Dir(alpha·1_K)`; thử lại tối đa 100 lần (stream độc lập
  `default_rng([seed, attempt])`) tới khi mọi client có ≥ `min_flows_per_client`; hết lượt →
  `ValueError`. Giữ trong code, **không dùng trong ma trận chính**.
- `max_flows_per_client`: subsample phân tầng theo lớp trong từng client (largest remainder).
- `ClientPartition` lưu `client_flow_ids: list[np.ndarray]` (sorted), `scheme`, `alpha`, `seed`;
  `to_json/from_json`; `class_histogram(labels, C) -> (K, C)`.
- Bất biến: các client rời nhau, hợp lại ⊆ train, không chứa val/test.

**Bảng tường minh `explicit` (mới, phương án chính):**

- Họ lớp (dựa trên `DEFAULT_FAMILIES` trong `config.py`, gộp `mirai` + `malware` thành **Khác**): **DDoS** (5 lớp), **Recon**
  (5 lớp), **Web** (CommandInjection, XSS, SqlInjection, Uploading_Attack, BrowserHijacking),
  **Khác** (Backdoor_Malware, Mirai-udpplain), **Benign**.
- Nhóm gateway (mặc định, cấu hình được):

  | Nhóm | Client (1-based) | Họ chủ đạo |
  |---|---|---|
  | G1 | 1 – 5 | DDoS |
  | G2 | 6 – 10 | Recon |
  | G3 | 11 – 14 | Web |
  | G4 | 15 – 16 | Khác |
  | G5 | 17 – 20 | Không có (gateway tổng hợp) |

- Tỉ lệ client k nhận của lớp c, với mức lệch `p` (`partition.dominance`):
  - c thuộc họ F, k thuộc nhóm của F: `f[k, c] = p / |nhóm F| + (1 − p) / K`.
  - c thuộc họ F, k ngoài nhóm của F: `f[k, c] = (1 − p) / K`.
  - c = Benign: `f[k, c] = 1 / K` (luôn chia đều).
- Với nhóm 5/5/4/2/4 và K = 20, mọi tỉ lệ là số hữu hạn:

  | Họ | p = 80%: trong / ngoài nhóm | p = 50%: trong / ngoài nhóm | p = 0% |
  |---|---|---|---|
  | DDoS, Recon | 17% / 1% | 12,5% / 2,5% | 5% |
  | Web | 21% / 1% | 15% / 2,5% | 5% |
  | Khác | 41% / 1% | 27,5% / 2,5% | 5% |
  | Benign | 5% | 5% | 5% |

- **Số lượng (không ngẫu nhiên):** `n[k, c] = floor(f[k, c] · N_c)`; phần dư giao cho client có phần
  lẻ lớn nhất, hòa thì client có chỉ số nhỏ hơn.
- **Chọn flow (không ngẫu nhiên):** sắp flow train của lớp c theo thời gian bắt đầu flow (nếu
  metadata có), nếu không theo flow id; cắt thành các đoạn liên tiếp dài `n[1, c], n[2, c], …`,
  giao lần lượt cho client 1, 2, … Mỗi gateway thấy một khoảng thời gian liên tục của traffic.
- `max_flows_per_client` (tầng gọn): co bảng theo tỉ lệ bằng largest remainder, lấy **phần đầu** của
  mỗi đoạn — vẫn xác định.
- Kiểm tra khi dựng: tổng `f[·, c] = 1` cho mọi lớp; nhóm không giao nhau và phủ đủ K client; mỗi
  client ≥ `min_flows_per_client`; sai → `ValueError` trước khi train.
- `partition.json` lưu bảng số lượng `(K, C)` và tham số; `scripts/fl/export_partition_table.py`
  xuất CSV + biểu đồ nhiệt client × lớp cho luận văn.
- Cùng bảng tỉ lệ dùng cho cả ba mô hình (chung `train_ids`).

### 3.2 Dữ liệu cục bộ theo mô hình

**HGT — graph con (`client_graph.py`, đã code):**

- `packet_owner(artifact) -> int64[num_packets]`: flow sở hữu mỗi packet, từ
  `flow__contains__packet` (−1 nếu không có; packet có > 1 cạnh `contains`, hoặc flow có > 1 cạnh
  `from_host`/`to_host`, ⇒ `ValueError`).
- `build_client_artifact(artifact, client_flow_ids, packet_owner, recompute_host=True)`: giữ cạnh
  khi `owned[src_type][src] & owned[dst_type][dst]` (mọi edge key, kể cả `rev_*`); `node_features`
  dùng chung mảng gốc trừ `host` (tính lại); `flow_y`, `metadata` dùng chung (global id ⇒ EACS
  mask, clean labels vẫn đúng).
- `recompute_host_features`: out/in-degree, số host đích khác nhau, số port đích khác nhau tính từ
  flow của client.

**MLP:** chỉ dùng `flow_x[client_flow_ids]` (79 đặc trưng), chuẩn hóa bằng mean/std liên kết ở
round 0. Không cần graph.

**1D-CNN:** với mỗi flow, lấy **khối ordered-byte** (`obyte_000 … obyte_255`, tức 256 byte payload
đầu, đã chuẩn hóa `byte/255`; xác định cột qua `metadata.packet_feature_names`) của **K packet đầu**
theo thứ tự trong flow (cạnh `contains`, thứ tự theo `next_packet`); thiếu packet thì đệm 0. Đầu vào
`(1, K·256)`; mặc định K = 8. Chỉ dùng packet của flow thuộc client.

### 3.3 Tham số (`params.py`, đã code)

- `ParamLayout.from_model(model)`: mọi entry float của `state_dict`, thứ tự cố định; `numel`,
  `offsets`; entry không phải float ⇒ `ValueError`.
- `flatten(state_dict) -> Tensor[D]` (float32, CPU), `unflatten(vec) -> state_dict`.
- Nhóm tham số HGT (`group_of(name)`): `rel:<src__rel__dst>` cho `layers.*.relation_{key,value,prior}.*`;
  `node:<type>` cho `input_projection.<type>` và `layers.*.{query,key,value,out,norm_attn,norm_ffn,ffn}.<type>`;
  `tactic_embedding`; `classifier`; còn lại `other`.
- `groups`, `group_coords(names)`, `group_norms(deltas) -> (N, G)`, `active_groups(deltas) -> bool (N, G)`.
  Relation không xuất hiện ở client ⇒ grad None ⇒ AdamW bỏ qua ⇒ delta nhóm đó đúng bằng 0.
- **MLP/CNN** không có tên khớp các mẫu trên ⇒ toàn bộ rơi vào `other` (và `classifier` nếu đặt tên
  lớp cuối như vậy) ⇒ phần "theo nhóm quan hệ" của `RAGuard` suy biến thành kiểm tra toàn bộ vector.
- **Ràng buộc kiến trúc:** model không được có buffer không phải float. CNN **không dùng
  BatchNorm** (có `num_batches_tracked` kiểu int và running stats không phải tham số học);
  dùng GroupNorm hoặc không chuẩn hóa.

### 3.4 Mô hình và local trainer (`models.py`, `local_trainer.py`)

**Hai mô hình baseline:**

| Mô hình | Kiến trúc mặc định | Ghi chú |
|---|---|---|
| `FlowMLP` | 79 → 256 → 128 → 64 → C; ReLU; dropout 0.2 | Mô hình phẳng |
| `ByteCNN1D` | 3 khối [Conv1d(k=7) → GroupNorm → ReLU → MaxPool], kênh 32/64/128; global max pool; FC → C | Kiến trúc tham khảo DFNet trong `scripts/diagnostics/web_df_cnn_ceiling.py`, thay BatchNorm bằng GroupNorm |

**`ModelAdapter`** (giao diện chung):

- `build_model(cfg, context) -> nn.Module`
- `make_loader(state, cfg, shuffle) -> Iterable[batch]`
- `forward_loss(model, batch, class_weights, cfg) -> (loss, logits, labels)`
- `predict(model, loader) -> (y_true, y_pred)` cho đánh giá

`HGTAdapter` dùng lại building block của trainer cũ (`make_neighbor_loader`, `to_torch_batch`,
`_drop_edges`, `encode` + `classifier` khi GCL bật, `_compute_train_loss`; EACS qua `soft_targets` +
`soft_target_focal_loss` nếu bật). `MLPAdapter`, `CNNAdapter` dùng `DataLoader` thường trên flow
của client.

**Cấu hình huấn luyện chung cho cả ba mô hình** (để so sánh công bằng): focal loss
(`focal_gamma = 1.5`, `label_smoothing = 0.05`), class weight CB tính từ nhãn **cục bộ** của client,
AdamW, clip grad, LR theo round.

**LocalTrainer:**

- `ClientState` (bền qua round): `client_id`, `flow_ids`, dữ liệu cục bộ (backend/sampler cho HGT,
  tensor cho MLP/CNN), `class_weights`, `eacs` (chỉ HGT, nếu bật), `optimizer_state | None`,
  `rounds_trained`.
- `train(state, global_state_dict, round_idx, lr) -> LocalUpdate(delta: Tensor[D], num_samples,
  metrics, seconds, steps)`; bỏ batch có loss không hữu hạn; `local_epochs` epoch, dừng sớm ở
  `max_local_steps` nếu đặt.
- AdamW: `optimizer_state: reset` (mặc định FL) hoặc `persist` (centralized/local-only).
- LR theo round: `cosine_rounds` hoặc `constant`.
- EACS warm-up tính theo round toàn cục.
- RNG: seed = `seed + 100003·round + 1009·client_id`.

### 3.5 Attack (`attacks.py`, đã code)

Giao diện: `Attack.craft(ctx, num_malicious) -> Tensor (M, D)`; `ctx` (`AttackContext`) gồm `honest`
(delta trung thực của M attacker), `benign` (delta client lành), `num_participants`, `knowledge`
(`omniscient` mặc định — tập tham chiếu là benign; `partial` — tập tham chiếu là honest của
attacker). Mọi attack tất định. Attack bắt đầu từ `start_round`. `Attack.info` ghi chẩn đoán:
`gamma`, `saturated`, `z` (ALIE), `degenerate` (Min-Max/Min-Sum), `feasible` + `target_pos/target_neg`
(ScaleSign).

| Attack | Định nghĩa (nguồn) | Mặc định |
|---|---|---|
| `sign_flip` | `Δ_m = −s·Δ_m^honest`, mỗi attacker độc lập | `s = 1.0` |
| `alie` | `s = ⌊n/2+1⌋ − m`; `z = Φ⁻¹((n−s)/n)`; mọi attacker gửi `μ + z·σ` — Baruch et al. 2019, Alg. 3 | `z` theo công thức gốc; cho phép ghi đè |
| `min_max` | `Δ_m = μ + γ·p`, γ lớn nhất sao cho `max_i ‖Δ_m − Δ_i‖ ≤ max_{i,j} ‖Δ_i − Δ_j‖`; NDSS'21 Alg. 1 | `p = −std` (tùy chọn `unit`, `sign`, rescale về ‖σ‖), `γ_init = 10`, `τ = 1e−5` |
| `min_sum` | như trên với `Σ_i ‖Δ_m − Δ_i‖² ≤ max_i Σ_j ‖Δ_i − Δ_j‖²` (Eq. 7, dùng max như bài báo; code chính thức dùng min — ghi chú) | như Min-Max |
| `scale_sign` | Yang et al. TIFS 2025 §V: nhân γ / (1/γ) theo dấu của mean honest; γ lớn nhất thỏa ràng buộc Min-Max (**sửa** Alg. 2 in sai vì J được kiểm trên g* cũ); chỉnh số tọa độ dương/âm về mục tiêu kiểu Krum (Eq. 10, Alg. 4) | `γ_init = 10`, `τ = 1e−3`, `constraint = min_max` |

ALIE/Min-Max/Min-Sum/ScaleSign: mọi attacker gửi cùng vector (collusion theo định nghĩa gốc).
Attacker chỉ train trung thực khi attack cần `honest` (sign_flip, scale_sign, hoặc `partial`).
`σ` là std quần thể (`unbiased=False`).

Attack làm việc trên vector `(N, D)` nên dùng nguyên cho MLP và CNN.

### 3.6 Aggregator (`aggregators.py`)

Giao diện: `Aggregator.aggregate(deltas (N, D), weights (N,), ctx) -> AggResult(delta (D,),
selected: bool (N,) | None, scores: dict, info: dict)`.

| Tên | Định nghĩa | Mặc định | Vai trò |
|---|---|---|---|
| `fedavg` | `Σ w_i Δ_i / Σ w_i` | `weighting: samples` | Mốc không phòng thủ |
| `median` | median theo coordinate | | Thống kê; mục tiêu của Min-Max/Min-Sum |
| `trimmed_mean` | bỏ k lớn nhất và k nhỏ nhất mỗi coordinate, `k = ⌊β·N⌋`, cần `k < N/2` | `β = assumed_malicious_ratio` | Thống kê; mục tiêu của Min-Max/Min-Sum |
| `multi_krum` | điểm Krum `s(i)` = tổng bình phương khoảng cách tới `N−f−2` láng giềng gần nhất; chọn lặp m vector, trung bình (Blanchard 2017) | `f = ⌊assumed·N⌋`, `m = N − f` | Khoảng cách |
| `signguard` | S1 = `{0.1 ≤ ‖Δ_i‖/med ≤ 3.0}`; đặc trưng tỉ lệ dương/0/âm trên khối 10% coordinate; MeanShift, S2 = cụm lớn nhất; trung bình clipped trên S1∩S2 (code chính thức SignGuard) | `L=0.1, R=3.0, ratio=0.1` | Dấu; mục tiêu của ScaleSign |
| `dnc` | NDSS'21 Alg. 2: `b` chiều ngẫu nhiên, tâm hóa, vector kỳ dị chính, giữ `N − c·f` điểm thấp nhất | `niters=1, c=1, b=10000` (b ≤ D) | Phổ; đi cặp với Min-Max/Min-Sum |
| `msguard` *(tùy chọn)* | Yang et al. Alg. 8; bài báo không cho tham số mặc định → dựng lại theo quy ước SignGuard, `ζ = 10000` (ghi rõ là dựng lại) | | Đi cặp với ScaleSign |

Các phòng thủ so sánh **giữ đúng định nghĩa gốc**: không áp thêm clipping hay relation-aware.
`clip_deltas` và tổng hợp theo nhóm chỉ dùng bên trong `RAGuard`.

Với 20 client: Multi-Krum cần `N > 2f + 2` (thỏa tới 40% attacker); Trimmed Mean cần `k < N/2`.

### 3.7 Giải pháp đề xuất `RAGuard` (`defense.py`)

**Ý tưởng:** khi dữ liệu lệch, client không có một loại quan hệ (ví dụ client chỉ có traffic DDoS
hầu như không có cạnh bằng chứng tấn công web) sẽ có delta gần 0 ở nhóm tham số tương ứng, nên
delta của nó trông "khác số đông" dù trung thực. Các phòng thủ so sánh toàn bộ vector dễ loại nhầm.
`RAGuard` **so sánh riêng từng nhóm tham số, chỉ giữa các client có nhóm đó hoạt động**.

**Bước 0 — chuẩn bị:** server đã kiểm tra toàn vẹn (mục 3.8); co mỗi `Δ_k` về
`clip_factor · median_norm` (mặc định 1.0).

**Bước 1 — tín hiệu theo nhóm:** với mỗi nhóm g (theo `ParamLayout`) và tập client hoạt động
`A_g` (bỏ qua nhóm có `|A_g| < 3`):

- độ lớn: `log(‖Δ_k[g]‖ / median_{j∈A_g} ‖Δ_j[g]‖)`;
- hướng: `cos(Δ_k[g], ref[g])`, `ref` = aggregate round trước (round 1: median theo coordinate);
- dấu: tỉ lệ đồng thuận dấu với `ref[g]`.

Chuẩn hóa robust z (median/MAD) trong `A_g`; nghi ngờ mỗi tín hiệu có vùng chết 1σ
(`s = clip((|z| − 1)/(z_max − 1), 0, 1)` cho độ lớn; `clip((−z − 1)/(z_max − 1), 0, 1)` cho hướng/dấu;
`z_max = 3`); gộp trong nhóm bằng noisy-OR `u_kg = 1 − Π(1 − s)`. Nghi ngờ của client
`U_k` = trung bình có trọng số (theo số tham số của nhóm) của `u_kg` trên các nhóm hoạt động
(ablation: `max`). MLP/CNN: chỉ có một nhóm ⇒ kiểm tra toàn bộ vector.

**Bước 2 — lịch sử:** `trust_k^t = 0.8·(1 − U_k^t) + 0.2·trust_k^{t−1}` (EMA như MultiSignal);
đánh dấu thay đổi đột ngột khi `|U_k^t − U_k^{t−1}|` > median + 3·MAD lịch sử của client (ý tưởng
ΔMD của SecureDyn-FL) ⇒ hạ trust về `min(trust, 0.5)`. Warm-up 2 round: chỉ dùng tín hiệu cắt
ngang.

**Bước 3 — quyết định và tổng hợp:** `trust < 0.5` → `reject`; `0.5 ≤ trust < 0.8` → `downweight`
(trọng số = trust); còn lại `accept`. Tổng hợp **trimmed mean có trọng số theo từng nhóm** trên
client không bị loại và có nhóm hoạt động (cắt `β = assumed_malicious_ratio` theo khối lượng trọng
số mỗi đầu); nhóm không ai hoạt động → 0.

**Giả thuyết cần kiểm chứng:**

- GT1: không tấn công, dữ liệu lệch ⇒ `RAGuard` loại nhầm client lành ít hơn các phòng thủ so
  sánh toàn bộ vector.
- GT2: có tấn công ⇒ macro-F1 bằng hoặc tốt hơn phòng thủ so sánh mạnh nhất.

**Cờ ablation:** `relation_aware: false` (chỉ toàn bộ vector), `temporal: false`,
`downweight: false` (chỉ nhận/loại), `aggregator: median` (thay trimmed mean), `client_reduce: max`.

**Chọn hằng số:** chỉ trên các run **không tấn công** (IID và lệch 80%), mục tiêu giữ tỉ lệ loại nhầm
client lành thấp; không tinh chỉnh trên run có tấn công để tránh khớp quá mức. Mọi hằng số là lựa
chọn của nhóm và phải báo cáo.

**Nguồn ý tưởng:** SecureDyn-FL (lịch sử theo round), multi-signal gradient fingerprinting (tín hiệu
độ lớn/hướng/dấu, noisy-OR thay tổ hợp tuyến tính), Yin et al. 2018 (trimmed mean). Phần so sánh
theo nhóm quan hệ là đóng góp của nhóm. TemporalAudit, MultiSignal của bản thiết kế trước trở
thành thành phần của `RAGuard`; BrayCurtis bị bỏ.

**Hạn chế cần nêu:** 5 attack hiện có không biết phòng thủ; kịch bản attacker thích ứng (ví dụ
Min-Max ràng buộc theo từng nhóm) là khảo sát bổ sung có điều kiện.

### 3.8 Server (`server.py`)

`ClientMessage(client_id, round_idx, delta, num_samples)`. `FLServer.step(messages)`:

1. Kiểm tra toàn vẹn: đúng round, đúng `D`, toàn bộ hữu hạn → nếu sai: `invalid`, loại.
2. Nếu defense là `RAGuard`: `defense.screen_and_aggregate(...)`; nếu không: `aggregator.aggregate`
   với trọng số theo số mẫu.
3. `W_{t+1} = W_t + η·Δ_agg` (`server_lr`, mặc định 1.0). Ghi `prev_aggregate`.
4. Trả `RoundReport`: id client, quyết định (aggregator lọc như Krum/SignGuard/DnC: không được chọn
   ⇒ `reject`), trust, đặc trưng, norm, thời gian.
5. Tùy chọn `save_deltas: true`: lưu `(N, D)` của round (float16) để chấm phát hiện ngoại tuyến.

Server không bao giờ nhận cờ "là attacker"; simulation tự đối chiếu sau khi server trả kết quả.

### 3.9 Simulation (`simulation.py`)

`run_experiment(cfg, *, out_dir, context=None) -> dict`. `context` (dữ liệu, split, clean labels,
cache client) cho phép matrix runner nạp dữ liệu **một lần**. `model.name ∈ {hgt, mlp, cnn}`
chọn `ModelAdapter`.

Chế độ:

- `fl`: vòng round như dưới.
- `centralized`: 1 client chứa **hợp các flow của các client FL** (cùng bảng chia ⇒ cùng dữ liệu
  train), `optimizer_state: persist`, mỗi round = 1 epoch, không attack, FedAvg.
- `local_only`: mỗi client tự train `rounds × local_epochs` từ cùng khởi tạo, không tổng hợp; báo
  cáo mean/std/min/max trên val/test toàn cục.

Một round `fl`:

1. Chọn client tham gia (`participation`, mặc định 1.0). Attacker cố định từ đầu theo
   `attack.placement`: `random` (`round(malicious_ratio·K)` client chọn theo seed) hoặc
   `group:<Gi>` (attacker lấy từ một nhóm gateway, theo thứ tự client).
2. Client lành (và attacker nếu cần) train tuần tự trên một instance model dùng chung.
3. Từ `start_round`, attacker thay delta bằng `attack.craft`.
4. `server.step` → model global mới.
5. Đối chiếu ground truth → TP/FP/FN/TN phát hiện, loại nhầm client lành.
6. Mỗi `eval.every` round: chấm val trên nhãn gốc và clean key; giữ best theo `eval.monitor`.
7. Kết thúc: chấm test cho model **round cuối** (chỉ số chính) và best-val; lưu checkpoint
   (`output.save_checkpoints`, bật cho các run quan trọng — dùng lại ở giai đoạn XAI).

Round 0 trước vòng lặp: thu moment flow từ client → `flow_feature_stats` chung.

### 3.10 Chỉ số (`metrics.py`, `offline_eval.py`)

- Phát hiện IDS: macro-F1, weighted-F1, accuracy (nhãn gốc và clean); recall theo họ tấn công;
  FPR Benign → tấn công = `1 − recall(Benign)`.
- Poisoning (theo round và cộng dồn): TPR phát hiện update độc hại, tỉ lệ loại nhầm client lành,
  tỉ lệ downweight; malicious acceptance rate = 1 − TPR.
- **Ngoại tuyến:** `offline_eval.py` nạp các `(N, D)` đã lưu từ một run FedAvg có attacker và cho mọi
  phòng thủ chấm lại để tính TPR/tỉ lệ loại nhầm mà không cần train lại. Macro-F1 khi bị tấn công
  vẫn cần chạy đầy đủ từng phòng thủ.
- Hội tụ: round đầu tiên đạt ≥ 95% val macro-F1 tốt nhất của run.
- Chi phí: thời gian train client/round, thời gian server/round, peak RSS, peak VRAM, bytes lên/xuống
  = `4·D·|S_t|`.
- Attack impact = F1(không attack, cùng mô hình, defense, bảng chia, seed) − F1(attack) — tính ở
  summarizer; so sánh giữa mô hình dùng mức suy giảm tương đối.

### 3.11 Đầu ra một run

`outputs/fl/<run>/`: `config.resolved.yaml`, `partition.json` (kèm bảng số lượng), `rounds.jsonl`,
`fl_summary.json`, `global_final.pt`, `global_best.pt`, (tùy chọn) `deltas/round_XXX.pt`.
Checkpoint HGT có các key runtime cần: `model_state_dict, config, node_input_dims, edge_types,
num_classes, num_tactics, label_names, flow_feature_stats, epoch, val_metrics, test_metrics`.

### 3.12 Ma trận thí nghiệm

`scripts/fl/run_matrix.py --grid <yaml> [--device cuda] [--dry-run] [--only <glob>]`: grid =
`base_config` + `axes` + quy tắc `attack=none` ⇒ `malicious_ratio=0`; chạy tuần tự, tái dụng cache
client, bỏ qua ô đã có `fl_summary.json`. `scripts/fl/summarize_matrix.py --root <dir>` →
`matrix_results.csv` + bảng markdown.

Ma trận thu gọn theo ưu tiên (số run là ước lượng; chốt sau pilot):

| Nhóm | Cấu hình | Số run | Ưu tiên |
|---|---|---|---|
| CH1 | `centralized` / `local_only` / `fl` × 3 mức lệch × 3 mô hình; thêm split theo thời gian cho HGT | ~30 | P0 |
| CH3 + CH4 (HGT) | 2 mức lệch (0%, 80%) × (không attack + 5 attack) × 7 phòng thủ (6 so sánh + `RAGuard`) | ~84 | P0 |
| Ablation `RAGuard` | 4 biến thể × 2 attack + không attack | ~12 | P0 |
| Quét tỉ lệ attacker | 3 phòng thủ × 2 attack × {10%, 30%, 40%} | ~18 | P1 |
| Attacker cùng nhóm | 3 phòng thủ × 2 attack, `placement: group:G2` | ~6 | P1 |
| CH2 (MLP, CNN) | 2 mô hình × 2 mức lệch × 6 tình huống × 4 phòng thủ (FedAvg, 2 so sánh mạnh nhất, `RAGuard`) | ~96 | P1 |
| 3 seed | các kết quả chính | ~20 | P1 |
| FL có EACS (HGT) | không attack, lệch 80% | ~4 | P2 |
| Bắt đầu muộn, `partial`, ablation PMI | | ~10–15 | P2 |

Một round ≈ 1 epoch trên toàn bộ train ⇒ 1 run 20 round ≈ 20 epoch. Vượt ngân sách GPU thì cắt P2
trước, rồi giảm P1; không cắt P0. Ma trận dùng tầng gọn (`max_flows_per_client`), kết quả chính
dùng dữ liệu đầy đủ.

## 4. Xử lý lỗi

- Config sai (tên model/attack/aggregator/defense lạ, tham số lạ, mode/monitor lạ) → `ValueError`
  khi dựng, trước khi train. `k ≥ N/2` cho trimmed mean hoặc Krum thiếu `N > 2f + 2` → `ValueError`
  lúc aggregate.
- Bảng tường minh: tổng tỉ lệ một lớp ≠ 1, nhóm giao nhau/không phủ đủ, client dưới
  `min_flows_per_client` → `ValueError`.
- Dirichlet không đạt `min_flows_per_client` sau 100 lần → `ValueError` gợi ý tăng α.
- Model có buffer không phải float (ví dụ BatchNorm) → `ValueError` từ `ParamLayout`.
- Delta không hữu hạn/sai kích thước/sai round → `invalid`, bị loại, ghi log; round không có delta
  hợp lệ → giữ nguyên `W_t`.
- MeanShift không tìm được cụm → giữ toàn bộ S1, ghi `info.fallback`.
- Clean labels/anchor mask sai số flow → `ValueError`.

## 5. Kiểm thử

Mọi test chạy CPU, dưới `tests/federated/`:

- Đã có (task 1–6): `config/records`, `linalg`, `partition` (IID/Dirichlet), `client_graph`,
  `params`, `attacks`.
- `partition` (explicit): bảng tỉ lệ khớp công thức; tổng mỗi lớp = 1; số lượng xác định (chạy lại
  ra cùng kết quả, không phụ thuộc seed); flow giao theo đoạn thời gian liên tục; `p = 0` cho kết
  quả giống chia đều; tầng gọn giữ tỉ lệ.
- `models`: `FlowMLP`, `ByteCNN1D` forward đúng shape; `ParamLayout` dựng được (không BatchNorm);
  adapter MLP/CNN train một bước trên dữ liệu nhỏ.
- `aggregators`: FedAvg khớp trung bình có trọng số; median/trimmed bền với outlier; Krum loại
  outlier; SignGuard/DnC loại cụm bị lật dấu.
- `defense` (`RAGuard`): bắt outlier rõ (scale ×50, sign-flip); **không loại client lành chỉ vì
  một nhóm quan hệ bằng 0** (kịch bản non-IID tổng hợp); bắt client đổi hành vi đột ngột sau
  warm-up; từng cờ ablation hoạt động.
- `offline_eval`: chấm lại trên delta đã lưu cho cùng quyết định như chạy trực tiếp.
- `server`: cập nhật W đúng; từ chối NaN/sai round.
- Smoke: `run_experiment` 3 client × 2 round × 1 attacker sign-flip cho **cả ba mô hình**; tạo đủ
  file đầu ra; checkpoint dựng lại được model; `centralized`, `local_only` chạy được.
- Chạy lại toàn bộ `tests/`.

## 6. Giới hạn ghi vào luận văn

- Đặc trưng flow dạng cửa sổ (`fan_dst_ip_5s`, …) được xem là đầu ra cảm biến cục bộ.
- Bảng PMI dùng chung là giả định; thông tin thống kê của tập train đi vào cạnh bằng chứng ở mọi
  client.
- `n_k` và moment flow do client tự khai báo có thể bị gian lận; ngoài phạm vi.
- Screening từng client mâu thuẫn với secure aggregation; không giả định cả hai cùng lúc.
- `RAGuard` dùng ý tưởng từ SecureDyn-FL và multi-signal nhưng là thiết kế của nhóm; MSGuard (nếu
  làm) là bản dựng lại; mọi hằng số là lựa chọn của nhóm và phải báo cáo.
- Lớp nhỏ và tấn công web thật (theo nhãn clean) rất ít; ở mức 1% nhiều client chỉ có 1–3 flow
  hoặc không có mẫu của lớp đó.
- MLP/CNN không có nhóm quan hệ ⇒ `RAGuard` suy biến; đây là đối chứng chủ đích.
- Đánh giá dùng dữ liệu toàn cục (experiment-only), không phải một bước của giao thức FL.
- Kết luận chỉ đúng trên các attack, bảng chia và tỉ lệ độc hại đã thử.

## 7. Trạng thái và thứ tự triển khai

**Đã xong (task 1–6 của plan cũ):** config/records, linalg, partition IID/Dirichlet, client_graph,
params, attacks (114 hàm test; cần xác nhận pass trên máy có torch).

**Thứ tự mới** (đưa vòng FL tối giản lên trước để chạy CH1 sớm):

| # | Việc | Phục vụ |
|---|---|---|
| T7 | `server.py` tối giản (integrity + FedAvg) + `metrics.py` | CH1 |
| T8 | `models.py`: `ModelAdapter`, `HGTAdapter`; `local_trainer.py` dùng adapter | CH1 |
| T9 | `simulation.py` + `cli.py` (3 chế độ) → pilot cấu hình HGT, đo thời gian | CH1 |
| T10 | Bảng tường minh trong `partition.py` + `export_partition_table.py` | Yêu cầu 3 |
| T11 | `FlowMLP`, `ByteCNN1D`, `MLPAdapter`, `CNNAdapter` | CH2 |
| T12 | `aggregators.py`: median, trimmed mean, Multi-Krum, SignGuard, DnC (MSGuard tùy chọn) | CH3 |
| T13 | `defense.py`: `RAGuard` + ablation | CH4 |
| T14 | Lưu delta + `offline_eval.py` | CH3, CH4 |
| T15 | `matrix.py`, scripts, `configs/fl/` theo P0/P1/P2 | Tất cả |
| T16 | Tài liệu + kiểm thử toàn bộ | |

Plan chi tiết `docs/superpowers/plans/2026-10-02-fl-hgt-poisoning-testbed.md` cần cập nhật theo
thứ tự này trước khi code tiếp.

## 8. Việc còn chờ

- **Giảng viên duyệt:** bảng chia (nhóm 5/5/4/2/4, mức lệch 0/50/80%); hướng `RAGuard`; danh sách
  phòng thủ so sánh (có làm MSGuard không); sửa đề cương (20 client, 2 baseline MLP/CNN, XAI làm sau,
  MITM sang hướng phát triển).
- **Số liệu từ server:** số flow train từng lớp cho split ngẫu nhiên và theo thời gian
  (`outputs/v3_ob/splits.json`); metadata có thời gian bắt đầu flow hay không (quy tắc chọn flow).
- **Pilot:** cấu hình HGT A/B, số local epoch, thời gian một run 20 × 20 → chốt ngân sách và kích
  thước ma trận.
