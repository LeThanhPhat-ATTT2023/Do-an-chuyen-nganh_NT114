# GraphSLM IDS — NT114 (Đồ án chuyên ngành)

Hệ thống phát hiện xâm nhập trên đồ thị dị thể (heterogeneous graph) cho lưu lượng
kiểu CIC-IoT-2023. Luồng end-to-end:

> Raw PCAP → tri thức đồ thị **có bằng chứng** (flow / packet / host / technique /
> tactic) → bộ phân loại flow **HGT** huấn luyện kèm **GCL auxiliary loss** và bộ
> điều khiển tự-relabel **EACS** → đánh giá theo giao thức **Smart-BOTH** (random +
> temporal) và **clean-key (LNL)**.

Ý tưởng cốt lõi: **không có encoder học nào ngoài chính HGT**. Mọi cạnh
packet→technique được sinh bằng một **ensemble thống kê đa nguồn (MSEE)** —
PMI + hồi quy logistic đa lớp chính quy L1 + so khớp chuỗi procedure của MITRE
(Aho-Corasick) — mỗi cạnh mang provenance ở mức token, audit được cho SOC. HGT là
mô hình duy nhất được train.

> ⚠️ **Lưu ý lịch sử:** phiên bản v1 (SecureBERT teacher → student 1D-CNN →
> cạnh `matches_technique` bằng cosine embedding → runtime fast/slow path + SLM
> XAI) đã bị **loại bỏ** vì cosine giữa SecureBERT(payload-hex) và SecureBERT(MITRE)
> là vô nghĩa về ngữ nghĩa. README này mô tả pipeline hiện tại; mã runtime cũ vẫn
> còn trong `src/graphslm_ids/runtime/` nhưng **không** thuộc phạm vi đồ án hiện tại.

## Mục lục

- [Đóng góp học thuật](#đóng-góp-học-thuật)
- [Kết quả hiện tại](#kết-quả-hiện-tại)
- [Dataset](#dataset)
- [Kiến trúc đồ thị](#kiến-trúc-đồ-thị)
- [Pipeline tiền xử lý — Smart-BOTH Hybrid](#pipeline-tiền-xử-lý--smart-both-hybrid)
- [Huấn luyện HGT + EACS](#huấn-luyện-hgt--eacs)
- [Đánh giá: noisy / clean và both-splits](#đánh-giá-noisy--clean-và-both-splits)
- [Baselines](#baselines)
- [Môi trường](#môi-trường)
- [Tái lập đầu-cuối](#tái-lập-đầu-cuối)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Kiểm thử](#kiểm-thử)
- [Giới hạn đã biết](#giới-hạn-đã-biết)
- [Tài liệu liên quan](#tài-liệu-liên-quan)

## Đóng góp học thuật

Bốn lớp đóng góp (framing Q1):

1. **MSEE — Multi-Source Evidence Ensemble** cho cạnh packet→technique: thay cả
   luật chữ ký thủ công lẫn cosine embedding (vô nghĩa) bằng ensemble thống kê
   3 nguồn — sinh ứng viên PMI, tinh chỉnh bằng logistic L1 đa lớp, và so khớp
   procedure literal của MITRE. **Không encoder học**, mỗi cạnh audit được.
2. **Schema dị thể có kiểu**: 5 loại cạnh bằng chứng theo họ tấn công
   (`injection / command_exec / file_upload / recon / c2_beacon`), cạnh đồng cảm
   flow–flow (`burst_neighbor`), tầng host, tầng technique↔tactic. PMI cung cấp
   trọng số PRIOR; attention đa-đầu của HGT TINH CHỈNH; GCL auxiliary loss GIÁM SÁT.
3. **Giao thức Smart-BOTH**: báo cáo **cả** split random-stratified (khớp prior
   work) **và** split temporal (sát triển khai). Khoảng cách (gap) giữa hai split
   chính là một phát hiện — gap nhỏ nghĩa là mô hình bắt được mẫu nội tại của tấn
   công thay vì dấu vân tay của chiến dịch trong dataset.
4. **EACS + chấm điểm clean (LNL)**: định lượng và **lọc** nhiễu nhãn của
   CIC-IoT-2023 (per-capture labeling), rồi chứng minh bằng **dấu của (noisy −
   clean)** — xem dưới.

Đây là điều mà XG-NID (narrative LLM hậu kỳ, không có đồ thị tri thức) và
PacketCLIP (encoder contrastive học, đơn nguồn) về mặt cấu trúc không làm được.

## Kết quả hiện tại

Cả 4 mô hình train trên **cùng** đồ thị `v3_ob` (18 lớp, **211.851 flow**, nhãn
gốc per-pcap), **cùng** split random, chấm trên **hai** bộ nhãn:

- **noisy** — nhãn per-pcap của CIC-IoT-2023 (cái prior work báo cáo).
- **clean** — answer key cô lập theo chữ ký (giao thức LNL): một flow web-attack
  giữ nhãn iff 5-tuple của nó mang chữ ký tấn công HTTP, ngược lại bị hạ về
  Benign. 14.276 / 211.851 flow lệch nhãn so với noisy. **Clean là eval-only** —
  không bao giờ vào loss/gradient/chọn checkpoint.

### Bảng headline (TEST, macro-F1)

| Model | noisy | clean (raw) | clean (calibrated) | noisy − clean |
|---|---|---|---|---|
| XGBoost (đặc trưng flow, tabular) | **1.0000** | 0.7626 | — | **+0.237** |
| GNN4ID (retrain trên phân phối v3) | 0.8588 | 0.7294 | — | +0.129 |
| HGT de-inflated (không EACS) | 0.8520 | 0.7224 | 0.7753 | +0.130 |
| **HGT + EACS v2** | 0.7228 | **0.8582** | 0.8518 | **−0.135** |

EACS v2 trên VAL clean: **0.8856 raw → 0.9269 calibrated**.

**Dấu của (noisy − clean) là đóng góp.** Dương ⇒ model điểm cao hơn khi chấm bằng
nhãn SAI ⇒ **đã nhớ nhiễu nhãn** (XGBoost +0.237 là kẻ nhớ tệ nhất, dù noisy =
1.0000 trông như SOTA). Âm ⇒ model điểm cao hơn trên sự thật ⇒ **đã lọc nhiễu**
(chỉ EACS). Cùng backbone HGT, cùng data, cùng split — biến duy nhất thay đổi là
bộ điều khiển tự-relabel EACS.

### Metric quyết định — phát hiện web-attack nhị phân (TEST)

| Model | TP | FP | FN | recall | precision | F1 |
|---|---|---|---|---|---|---|
| XGBoost | 107 | **1426** | 0 | 1.000 | 0.070 | 0.130 |
| HGT de-inflated | 106 | **1667** | 1 | 0.991 | 0.060 | 0.113 |
| **HGT + EACS v2** | 106 | **42** | 1 | 0.991 | **0.716** | **0.831** |

Mọi model đều *tìm thấy* tấn công (recall ≈ 1.0); khác biệt là **báo động giả**.
EACS giảm **~97 % FP ở cùng recall** — giá trị SOC thực sự cảm nhận được.

> Chi tiết: [docs/reports/2026-06-13-eacs-vs-baselines.md](docs/reports/2026-06-13-eacs-vs-baselines.md)
> và [docs/reports/2026-06-13-clean-grading-methodology.md](docs/reports/2026-06-13-clean-grading-methodology.md).
> Artifact số: [results/2026-06-13/](results/2026-06-13/).

## Dataset

CIC-IoT-2023 style, **18 lớp**, một PCAP/lớp trong `data/raw/<class>/*.pcap`.
Nhãn gán per-pcap qua `infer_label_from_path`.

```text
Backdoor_Malware          DDoS-ICMP_Flood           Recon-OSScan
Benign                    DDoS-ICMP_Fragmentation   Recon-PingSweep
BrowserHijacking          DDoS-PSHACK_Flood         Recon-PortScan
CommandInjection          DDoS-RSTFINFlood          SqlInjection
DDoS-ACK_Fragmentation    Mirai-udpplain            Uploading_Attack
                          Recon-HostDiscovery       VulnerabilityScan  XSS
```

`packet_x` (mảng đặc trưng packet thống trị) lưu trên đĩa dạng **float16** để giảm
nửa dung lượng; loader upcast lên float32 mặc định. Artifact build ra
`outputs/v3_ob/graph.npz` (+ `graph.meta.json`, `splits.json`, `pmi_table.parquet`) —
gitignored.

> **Nguồn nhiễu nhãn (đã định lượng):** tác giả gốc (Neto et al., *Sensors* 2023,
> §3.3.2) gán nhãn **theo từng capture** — toàn bộ traffic trong một thí nghiệm
> tấn công bị dán *một* nhãn. Với DDoS/Recon/Mirai điều này vô hại; với 4 lớp
> web-attack (thực thi qua DVWA) thì traffic nền IoT→cloud bị dán nhầm là tấn
> công. Đó là cơ sở của thang **clean**.

## Kiến trúc đồ thị

Node: `flow`, `packet`, `host`, `technique`, `tactic`. Các loại cạnh chính (xem
`sampler.fanouts` trong config):

| Cạnh | Ý nghĩa |
|---|---|
| `flow → contains → packet` | flow chứa packet |
| `packet → next_packet → packet` | trình tự packet trong flow |
| `packet → evidence_{injection,command_exec,file_upload,recon,c2_beacon} → technique` | 5 cạnh bằng chứng có kiểu (MSEE) |
| `flow → matches_technique → technique` | bằng chứng mức flow |
| `flow → from_host / to_host → host` | tầng host |
| `flow → burst_neighbor → flow` | đồng cảm flow–flow (homophily) |
| `technique → has_subtechnique → technique` | phân cấp MITRE |
| `technique → belongs_to_tactic → tactic` | phân cấp MITRE |

## Pipeline tiền xử lý — Smart-BOTH Hybrid

Một package phẳng: `src/graphslm_ids/offline/preprocessing/`. Chạy đầu-cuối trên
CPU local. Các stage 3–11 **tất định** với seed 42.

| Module | Vai trò |
|---|---|
| `extractor.py` | parse pcap → metadata mỗi packet (TẤT CẢ packet + TCP flags + IP len + hướng) |
| `flows.py` | flow 5-tuple hai chiều + ~79 đặc trưng CICFlowMeter |
| `split.py` | split temporal AND random stratified (từ cùng packets) |
| `tokenizer.py` | byte n-gram + token HTTP/text tất định (không train) |
| `pmi_learner.py` | sinh ứng viên PMI + tinh chỉnh LR đa lớp L1 trên subsample TRAIN |
| `procedure_matcher.py` | Aho-Corasick trên procedure literal của MITRE STIX |
| `flow_consensus.py` | boost hành vi `signatures.match_flow_signatures` |
| `ensemble.py` | gộp 3 nguồn (PMI + procedure + flow consensus) → trọng số cạnh cuối |
| `edge_writers.py` | ghi cạnh streaming memmap (out-of-core) |
| `graph_builder.py` | lắp artifact: node + cạnh bằng chứng có kiểu + cạnh phân cấp |
| `flow_attack_labeler.py` | cô lập attack-flow theo chữ ký HTTP (dựng answer key clean) |
| `cli.py` | orchestrator một-lệnh (CPU local) |

### Nguyên tắc thiết kế (không lệch)

- **Zero learned encoder ngoài HGT.** PMI = đếm + LR L1 lồi. Procedure matcher =
  so khớp chuỗi Aho-Corasick.
- Tiền xử lý chạy CPU local; chỉ HGT train trên server (L40S 48 GB).
- Eval CẢ random + temporal. Gap là đóng góp.

## Huấn luyện HGT + EACS

`training/train_hgt_flow_classifier.py` đọc `data.artifact_version: v3`, thêm
**GCL auxiliary loss** (positive pair từ class→technique map) và bộ điều khiển
nhiễu **EACS**.

**EACS (Evidence-Anchored Candidate-set Self-relabeling):** flow nghi ngờ
(nhãn web-attack, không có bằng chứng MITRE khớp) nhận soft target do model điều
khiển trong tập `{nhãn gốc, Benign}`; **anchor** (tấn công thật có bằng chứng) và
mọi lớp khác giữ hard label. Mask anchor neo trên **procedure-literal khớp**
(MSEE nguồn 2, train-legitimate) — đạt anchor precision **95,2 %**, noise-detection
ROC-AUC **0,760** (so với v1 anchor MSEE-scalar: 16 % / 0,609).

Config hiện tại (xem `configs/`):

| Config | Vai trò |
|---|---|
| `eg_hgt_v6_ob_eacs_v2.yaml` | **EACS v2** (procedure-anchor) — mô hình chính |
| `eg_hgt_v6_ob_focal_deinflated.yaml` | HGT de-inflated (control nhớ-nhiễu) |
| `eg_hgt_v6_ob_focal_deinflated_clean.yaml` | de-inflated + chấm clean |
| `eg_hgt_v6_ob_noiserobust.yaml` | biến thể noise-robust khác |

Khối quan trọng trong `eg_hgt_v6_ob_eacs_v2.yaml`:

```yaml
data:
  artifact_version: v3
  graph_npz: outputs/v3_ob/graph.npz
  split_protocol: random            # đổi sang temporal cho Smart-BOTH
train:
  output_dir: outputs/v3_ob_eacs_v2
  monitor: val_macro_f1_clean
  clean_eval_labels: outputs/v3_ob/clean_eval_labels.npy
  loss_type: focal
  gcl_enabled: true
  noise_robust:
    enabled: true
    mode: eacs
    anchor_mask_npy: outputs/v3_ob/eacs_anchor_mask.npy
    suspect_classes: [CommandInjection, XSS, SqlInjection, Uploading_Attack]
feature_store: { enabled: true, cache_fraction: 0.6, model_reserve_gb: 4.0 }
```

**Tiered feature store** (`training/feature_store.py`): phân tầng bộ nhớ
GPU hot cache → CPU RAM → disk memmap, tự co giãn theo VRAM đo được. **GPU
neighbor sampling** (`training/gpu_sampling.py`) tùy chọn, đã test numpy-parity.
Cả hai config-gated, OFF mặc định.

## Đánh giá: noisy / clean và both-splits

`scripts/eval/calibrate_thresholds.py` — chấm một checkpoint trên **cả** noisy và
clean, tính metric **web-binary**, và calibrate per-class additive logit bias
(tune trên VAL clean, áp sang TEST — không peeking test). Calibration là
decision-rule, KHÔNG retrain/relabel ⇒ apples-to-apples cho mọi baseline emit logits.

`scripts/eval/v3_eval_both_splits.py` — chạy cùng model trên random + temporal,
báo cáo GAP (giao thức Smart-BOTH).

## Baselines

| Baseline | Vị trí | Ghi chú |
|---|---|---|
| **XGBoost** | `baselines/xgboost/train_xgboost.py` | tabular 79 đặc trưng flow; noisy 1.0000 / clean 0.7626 |
| **GNN4ID** | `baselines/gnn4id/` (`train_imbalanced.py`, `regrade_clean.py`) | retrain trên phân phối v3, chấm cả noisy + clean |

## Môi trường

- **venv riêng:** `D:\v\nt114` (Windows). Interpreter:
  `D:\v\nt114\Scripts\python.exe` (Python 3.13).
- Đã cài: numpy 2.4, pandas 3.0, torch 2.11+cpu, scikit-learn 1.8, dpkt, scapy,
  pytest 9.
- Máy local: 16 CPU, ~16 GB RAM, torch CPU-only. HGT train đầy đủ chạy trên
  L40S (48 GB) từ xa.
- Luôn chạy script bằng interpreter trên, ví dụ:
  `D:\v\nt114\Scripts\python.exe -m pytest tests/ -q`.

## Tái lập đầu-cuối

```bat
:: 1) Build đồ thị (CPU local)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.preprocessing.cli ^
  --raw-root data/raw ^
  --out-npz outputs/v3_ob/graph.npz ^
  --out-meta outputs/v3_ob/graph.meta.json ^
  --pmi-table outputs/v3_ob/pmi_table.parquet ^
  --pmi-meta outputs/v3_ob/pmi.meta.json ^
  --splits-json outputs/v3_ob/splits.json ^
  --pmi-subsample-per-class 25000 ^
  --temporal-train-frac 0.80 --temporal-val-frac 0.10

:: 2) Audit artifact TRƯỚC khi train
D:\v\nt114\Scripts\python.exe scripts/diagnostics/v3_artifact_audit.py outputs/v3_ob/graph.npz

:: 3) Dựng answer key clean (cần pcap)
D:\v\nt114\Scripts\python.exe scripts/tools/extract_clean_eval_labels.py ^
  --graph-meta outputs/v3_ob/graph.meta.json --raw-root data/raw ^
  --out-npy outputs/v3_ob/clean_eval_labels.npy ^
  --out-audit outputs/v3_ob/clean_eval_labels.audit.json

:: 4) Dựng mask anchor EACS
D:\v\nt114\Scripts\python.exe scripts/tools/extract_eacs_anchor_mask.py ^
  --graph-meta outputs/v3_ob/graph.meta.json --raw-root data/raw ^
  --out-npy outputs/v3_ob/eacs_anchor_mask.npy --out-audit outputs/v3_ob/anchor.audit.json

:: 5) Train HGT + EACS v2 (server CUDA)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.training.train_hgt_flow_classifier ^
  --config configs/eg_hgt_v6_ob_eacs_v2.yaml --device cuda

:: 6) Chấm noisy + clean (+ calibration trên clean val)
D:\v\nt114\Scripts\python.exe scripts/eval/calibrate_thresholds.py ^
  --config configs/eg_hgt_v6_ob_eacs_v2.yaml ^
  --checkpoint outputs/v3_ob_eacs_v2/hgt_flow_best.pt ^
  --training-summary outputs/v3_ob_eacs_v2/training_summary.json ^
  --clean-labels outputs/v3_ob/clean_eval_labels.npy ^
  --out outputs/v3_ob_eacs_v2/confusion_calibrated.json --device cuda

:: 7) Eval both splits (Smart-BOTH gap)
D:\v\nt114\Scripts\python.exe scripts/eval/v3_eval_both_splits.py ^
  --checkpoint-random outputs/v3_ob_eacs_v2/hgt_flow_best.pt ^
  --checkpoint-temporal outputs/v3_ob_eacs_v2_temporal/hgt_flow_best.pt ^
  --graph outputs/v3_ob/graph.npz
```

> Smoke test trên CPU: thêm `--device cpu --epochs 2`. Downcast fp32→fp16 không
> rebuild: `scripts/tools/downcast_packet_x.py`.

## Cấu trúc thư mục

```text
.
├── configs/                         # eg_hgt_v6_ob_eacs_v2.yaml + họ v6_ob, hgt_paper_variants/
├── baselines/
│   ├── gnn4id/                      # train_imbalanced.py, regrade_clean.py
│   └── xgboost/                     # train_xgboost.py
├── data/
│   ├── raw/<class>/*.pcap           # 18 PCAP attack class
│   └── mitre/                       # MITRE CSV + STIX enterprise-attack.json
├── docs/
│   ├── reports/                     # eacs-vs-baselines, clean-grading-methodology, ...
│   └── superpowers/specs|plans/     # design docs (smart-both, tiered store, eacs)
├── outputs/v3_ob/                   # graph.npz, splits.json, clean_eval_labels.npy (gitignored)
├── results/2026-06-13/              # JSON kết quả fair-comparison
├── scripts/
│   ├── eval/                        # calibrate_thresholds, v3_eval_both_splits, eval_reporting
│   ├── tools/                       # extract_clean_eval_labels, extract_eacs_anchor_mask, downcast_packet_x
│   └── diagnostics/                 # v3_artifact_audit, label_audit, web_*_separability, ...
├── src/graphslm_ids/
│   ├── models/                      # hgt.py
│   ├── offline/preprocessing/       # extractor, flows, split, tokenizer, pmi_learner, ...
│   ├── offline/training/            # train_hgt_flow_classifier, feature_store, gpu_sampling, ...
│   └── runtime/                     # [LEGACY v1 — ngoài phạm vi hiện tại]
└── tests/
```

## Kiểm thử

```bat
D:\v\nt114\Scripts\python.exe -m pytest tests/ -q
```

Một số test khóa bất biến quan trọng:

```bat
D:\v\nt114\Scripts\python.exe -m pytest tests/test_clean_eval_labels.py -q
```

## Giới hạn đã biết

- **Split temporal chưa chạy đủ.** Bảng kết quả hiện tại là **random-split**;
  phần gap Smart-BOTH (đóng góp #3) cần chạy temporal cho EACS + ≥1 baseline rồi
  báo cáo gap. Đây là lỗ hổng bằng chứng lớn nhất hiện tại.
- **Support test nhỏ ở 3/4 lớp web** sau khi lọc: XSS (n=5), CommandInjection
  (n=7), Uploading_Attack (n=2) — một FP làm F1 dao động 0.1–0.3 (nhiễu *của
  metric*, không phải model; recall gộp vẫn 0.991). Một re-split 60/20/20 theo
  attack-key sẽ cho các lớp này đủ support.
- **Một dataset (CIC-IoT-2023).** Chưa có cross-dataset để tuyên bố tổng quát hóa.
- Chữ ký clean có thể bỏ sót biến thể mã hóa/obfuscated; vì luật precision cao
  (95,2 %), sai số nghiêng *bảo thủ*, không thổi phồng EACS.

## Tài liệu liên quan

| Tài liệu | Nội dung |
|---|---|
| [docs/reports/2026-06-13-eacs-vs-baselines.md](docs/reports/2026-06-13-eacs-vs-baselines.md) | Bảng fair-comparison + metric web-binary |
| [docs/reports/2026-06-13-clean-grading-methodology.md](docs/reports/2026-06-13-clean-grading-methodology.md) | Đặc tả thang clean (LNL) đầy đủ, tái lập được |
| [docs/reports/2026-06-06-web-attack-encryption-ceiling.md](docs/reports/2026-06-06-web-attack-encryption-ceiling.md) | Chẩn đoán gốc của nhiễu nhãn web |
| `docs/superpowers/specs/2026-05-24-v3-smart-both-design.md` | Thiết kế pipeline Smart-BOTH |
| `docs/superpowers/specs/2026-05-25-tiered-feature-store-design.md` | Thiết kế tiered feature store |
| `docs/superpowers/specs/2026-06-11-eacs-noise-robust-design.md` | Thiết kế EACS |

## License

Xem file `LICENSE`.
