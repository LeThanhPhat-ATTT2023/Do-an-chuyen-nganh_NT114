# Thiết Kế Scalable Training Cho HGT Trên Graph Lớn

Tài liệu này đề xuất kiến trúc training HGT khi dataset không còn vừa RAM. Áp
dụng khi pipeline phát triển từ mức 27K flows hiện tại lên hàng chục triệu flow
và graph artifact đạt cỡ ~400GB.

Mục tiêu không phải đổi model. Mục tiêu là thay đường **storage → loading →
sampling → training loop** sao cho HGT vẫn học được trên đồ thị lớn mà không
load toàn bộ vào RAM.

## 1. Phân tích vấn đề hiện tại

### 1.1 Đường code đang chạy

Tham chiếu code:

```text
src/graphslm_ids/offline_path/training/train_hgt_flow_classifier.py
src/graphslm_ids/offline_path/training/hetero_graph_artifact.py
src/graphslm_ids/models/hgt.py
```

Luồng hiện tại:

```text
np.load(graph_artifact_3tier_t082_k5.npz)
  -> dict[str, np.ndarray] toàn bộ vào RAM
  -> torch.from_numpy(...).to(device) toàn bộ
  -> mỗi epoch: model(node_features, edge_index) trên FULL graph
  -> loss trên train_idx (mask)
```

Trong `train_hgt_flow_classifier.py` có dòng:

```python
if str(config["train"]["batch_mode"]).lower() != "full":
    raise ValueError("Only full batch HGT training is supported in this script.")
```

Đây là điểm nghẽn cứng cần phá bỏ.

### 1.2 Quy mô hiện tại và tương lai

```text
Hiện tại:
  num_flows      = 27_541
  num_packets    = 86_548
  num_techniques = 691
  num_tactics    = 14
  graph_npz      = 223 MB
  PCAP raw       = 2.4 GB

Mục tiêu:
  num_flows      ~ 10M - 100M
  num_packets    ~ 30M - 1B
  graph_npz      ~ 100 - 400 GB
  PCAP raw       ~ vài TB
```

### 1.3 Bốn cấp độ "phình" cần xử lý riêng

Không thể fix một chỗ là xong. Bốn lớp đều phải đổi:

```text
Storage layer:
  NPZ monolith -> sharded, streamable, partial-readable.

Loading layer:
  np.load full -> memory-mapped, lazy, chỉ đọc node/edge cần thiết.

Computation layer:
  forward full graph -> forward subgraph nhỏ quanh seed flow.

Training loop:
  full-batch GD -> mini-batch SGD với neighbor sampling.
```

Nếu chỉ fix một lớp (ví dụ chỉ memmap nhưng vẫn full-batch forward) thì vẫn OOM
ở GPU.

## 2. Kiến trúc tổng thể đề xuất

```text
PCAP shards (theo thời gian / theo file)
  -> Sharded Preprocessor (streaming, không hold toàn graph)
  -> On-Disk Graph Store
       node feature shards (numpy memmap)
       edge CSR shards (numpy memmap)
       knowledge nodes: technique, tactic (load full vào RAM)
       manifest.json (index toàn cục)
  -> NeighborLoader (per-batch K-hop sampling, K = num_layers)
  -> HGT mini-batch training
       AMP + activation checkpointing (đã có)
       gradient accumulation
  -> Checkpoint + validation theo seed-flow batch
```

Bốn thành phần chính:

```text
1. Sharded On-Disk Graph Store
2. Sharded Preprocessing Pipeline
3. Heterogeneous Neighbor Sampler
4. Mini-Batch Training Loop
```

## 3. Phương pháp 1: Sharded On-Disk Graph Store

### 3.1 Mục tiêu

Thay vì một file `graph_artifact_3tier_t082_k5.npz` chứa hết, tách graph thành
các file độc lập, đọc được một phần qua memory-map.

### 3.2 Layout đề xuất

```text
data/processed/graph_store_v1/
  manifest.json
  nodes/
    flow/
      features.f32           # memmap [num_flows, flow_dim]
      labels.i64             # memmap [num_flows]
      shard_index.i64        # [num_flows] -> shard_id
    packet/
      features.f32           # memmap [num_packets, 768]
      shard_index.i64
    technique/
      features.f32           # full load, ~691 x 768 = 2 MB
    tactic/
      ids.i64                # full load, 14 phần tử
  edges/
    flow__contains__packet/
      indptr.i64             # CSR row pointer, len = num_flows + 1
      indices.i64            # CSR column indices, len = num_edges
      attr.f32               # optional edge weight
    packet__next_packet__packet/
      indptr.i64
      indices.i64
    packet__matches_technique__technique/
      indptr.i64
      indices.i64
      attr.f32               # cosine score
    flow__matches_technique__technique/
      indptr.i64
      indices.i64
      attr.f32
    technique__belongs_to_tactic__tactic/
      indptr.i64
      indices.i64
    rev_*/                   # reverse edges nếu add_reverse_edges
  splits/
    train_flow_ids.i64
    val_flow_ids.i64
    test_flow_ids.i64
```

### 3.3 Vì sao dùng CSR thay vì edge_index dạng (2, E)

Edge index dạng `(2, E)` phù hợp full-batch nhưng tệ cho sampling. Mỗi lần lấy
neighbor của một node phải scan toàn bộ E. Khi `E ~ 10^9`, scan là không khả
thi.

CSR cho phép:

```text
neighbors_of(node_id) = indices[indptr[node_id] : indptr[node_id + 1]]
```

Đây là O(deg) thay vì O(E). Build một lần ở preprocessing, dùng nhiều lần.

### 3.4 Vì sao tách static knowledge

```text
technique: 691 nodes x 768 dim x 4B = ~2 MB
tactic: 14 nodes
```

Nhỏ. Load full vào RAM một lần, không cần memmap. Đây là khác biệt quan trọng
so với flow/packet (tăng vô hạn theo dataset).

### 3.5 manifest.json

Index toàn cục, không lưu data, chỉ metadata:

```json
{
  "version": "v1",
  "created_at_utc": "...",
  "node_counts": {"flow": ..., "packet": ..., "technique": 691, "tactic": 14},
  "feature_dims": {"flow": 6, "packet": 768, "technique": 768},
  "edge_counts": {"flow__contains__packet": ..., "...": ...},
  "label_mapping": {...},
  "num_classes": 9,
  "tactic_shortname_to_idx": {...},
  "technique_id_to_idx": {...},
  "shards": {
    "flow": [{"shard_id": 0, "range": [0, 1000000]}, ...],
    "packet": [...]
  },
  "flow_feature_stats": {"mean": [...], "std": [...]}
}
```

`flow_feature_stats` quan trọng: chuẩn hóa flow phải dùng stats cố định, không
recompute từ subset (vì train_idx ở mini-batch chỉ thấy một phần).

### 3.6 API tối thiểu

```text
class GraphStore:
  load_manifest() -> dict
  get_flow_features(flow_ids: int64[]) -> float32[N, flow_dim]
  get_packet_features(packet_ids: int64[]) -> float32[N, 768]
  get_technique_features() -> float32[691, 768]   # in-RAM
  get_tactic_index() -> int64[14]                  # in-RAM
  get_flow_labels(flow_ids: int64[]) -> int64[N]
  out_neighbors(edge_type, src_ids: int64[]) -> tuple(indptr_local, indices_local, attr_local)
  in_neighbors(edge_type, dst_ids: int64[]) -> tuple(indptr_local, indices_local, attr_local)
```

Tất cả dùng numpy advanced indexing trên memmap. Linux/Windows page cache lo
caching, không cần tự cache.

## 4. Phương pháp 2: Sharded Preprocessing Pipeline

### 4.1 Vấn đề ở preprocessing

Code hiện tại trong `build_three_tier_graph_artifact.py` build toàn graph trong
RAM rồi `np.savez`. Khi PCAP raw vài TB, bước build cũng OOM.

### 4.2 Pipeline streaming

```text
PCAP file_i
  -> stream packets từng batch (ví dụ 10_000 packet/lần)
  -> flow tracker (5-tuple + timeout) cập nhật state on-the-fly
  -> flow đóng (timeout / cờ FIN) -> flush ra shard
  -> packet feature tính song song -> ghi append vào packet shard
  -> append edges vào edge shard tạm dạng (src, dst) plain
  -> đóng shard khi đủ kích thước (ví dụ 1M flows / shard)
```

### 4.3 Hai pha rõ ràng

```text
Pha 1: Stream -> Raw shards (per-PCAP, append-only)
  output:
    nodes_flow_raw/shard_*.parquet
    nodes_packet_raw/shard_*.npy
    edges_*_raw/shard_*.npy   # edge_list dạng (src, dst, attr)

Pha 2: Compact -> CSR Graph Store
  - merge raw shards
  - đánh global node id (flow_id, packet_id) ổn định
  - sort edges theo src
  - build indptr / indices / attr ra final layout (mục 3.2)
  - tính flow_feature_stats trên train shard
  - viết manifest.json
```

Pha 1 không cần hold toàn graph. Pha 2 chạy một lần, có thể dùng external sort
(ví dụ `numpy.memmap` + chunked sort) nếu edge list quá lớn.

### 4.4 Tránh tái nhận diện node

```text
flow_id global = hash(5-tuple, time_bucket) -> int64
hoặc:
flow_id global = atomic counter trong pha 1
```

Khuyến nghị atomic counter + bảng tra `(5-tuple, time_bucket) -> flow_id`. Hash
có nguy cơ va chạm khi N lớn.

### 4.5 Split deterministic

Split train/val/test phải tính trên flow_id global, không trên local index của
shard. Lý do: nếu split theo shard, val có thể trùng phân phối tấn công với
train do PCAP grouping.

```text
Khuyến nghị:
  Stratified split theo nhãn, deterministic theo seed.
  Lưu split ra file splits/*.i64.
  Mọi run sau đọc lại file split, không random lại.
```

## 5. Phương pháp 3: Heterogeneous Neighbor Sampler

### 5.1 Tại sao cần sampler

Mini-batch training trên graph khác với image. Không thể "lấy 1024 flow rồi
forward". Phải kèm hàng xóm K-hop của flow đó để HGT có context, vì
`num_layers = K` nghĩa là model expect K hop neighbors.

### 5.2 K = HGT num_layers

Đây là nguyên tắc đã chốt trong `streaming_hgt_runtime_v3_vi.md`. Áp dụng
nguyên xi cho training:

```text
HGT num_layers = 3
=> sampler lấy K-hop = 3 quanh seed flows.
```

### 5.3 Logic sampling

```text
input:
  seed_flow_ids: int64[B]       # batch_size flows
  fanouts_per_relation: dict
  K = 3

frontier = {"flow": set(seed_flow_ids)}
output_subgraph = empty hetero data

for hop in 1..K:
  next_frontier = {}
  for edge_type (src, rel, dst):
    src_nodes = frontier.get(src, set())
    if not src_nodes: continue
    fanout = fanouts_per_relation[edge_type]
    sampled_dst, sampled_edges = sample_neighbors(
      src_nodes, edge_type, fanout
    )
    output_subgraph.add_edges(edge_type, sampled_edges)
    next_frontier[dst] |= sampled_dst
  frontier = merge(frontier, next_frontier)

reindex local:
  global_flow_id   -> local_flow_id    (theo thứ tự xuất hiện)
  global_packet_id -> local_packet_id
  global_tech_id   -> local_tech_id
  tactic           -> giữ GLOBAL index (xem mục 5.5)

load features:
  flow_features    = store.get_flow_features(local_to_global["flow"])
  packet_features  = store.get_packet_features(local_to_global["packet"])
  tech_features    = store.get_technique_features()   # full, in-RAM
  tactic_features  = store.get_tactic_index()         # full

output:
  node_features dict (đúng schema HGT model hiện tại)
  edge_index dict (đã reindex)
  edge_weight dict (cosine cho matches_technique)
  seed_mask: bool[num_flow] đánh dấu flow nào là seed
  labels: int64[num_seed]
```

### 5.4 Per-relation fanout

Không dùng fanout đồng nhất. Mỗi relation có hub khác nhau:

```yaml
sampler:
  fanouts:
    flow__contains__packet: 20            # tới đa 20 packet/flow
    packet__next_packet__packet: 4        # 4 hàng xóm liền kề
    packet__matches_technique__technique: 5   # top-5 (đã giới hạn ở build)
    flow__matches_technique__technique: 5
    technique__belongs_to_tactic__tactic: 1   # 1 tactic / technique (đa số)
    rev_*: same as forward
```

### 5.5 Xử lý đặc biệt cho tactic

Trong `models/hgt.py`:

```python
self.tactic_embedding = nn.Embedding(max(num_tactics, 1), hidden_dim)
tactic_index = torch.arange(tactic_count, ...)
x_dict["tactic"] = self.tactic_embedding(tactic_index.clamp_max(...))
```

Tactic dùng embedding theo index. Nếu sampler chỉ đưa subset tactic và đánh
lại index, embedding sẽ sai cho ID đã train.

```text
Quy tắc bắt buộc:
  Luôn đưa toàn bộ 14 tactic node vào mọi subgraph.
  Giữ nguyên global tactic_to_idx order.
  Edge technique -> tactic chỉ giữ với technique đã sampled.
```

14 tactic là constant nhỏ, chi phí gần như không.

### 5.6 Chặn hub explosion

Một số node có degree rất cao:

```text
Hub thường gặp:
  Tactic phổ biến (defense-evasion, discovery): nối ~100 technique
  Technique phổ biến: có thể nối hàng triệu flow ở dataset lớn
```

Giải pháp:

```text
1. Hạn chế reverse expansion từ hub:
   technique -> tất cả flow: KHÔNG mở rộng mặc định.
   Chỉ giữ nếu fanout config cho phép, và phải clip top-k theo cosine.

2. Per-node degree cap:
   Ở pha preprocess, cap tối đa degree đầu vào / đầu ra cho mỗi node.
   Ví dụ packet -> max 5 technique cosine cao nhất (đã có top_k = 5).
```

### 5.7 Đầu ra: tương thích với HGT model hiện tại

Quan trọng: HGT model trong `src/graphslm_ids/models/hgt.py` không cần đổi.
Nó nhận:

```text
node_features: dict[str, Tensor]
edge_index_dict: dict[EdgeKey, Tensor[2, E]]
edge_weight_dict: optional dict[EdgeKey, Tensor[E]]
```

Sampler chỉ cần xuất đúng schema này với local index, không phải HeteroData
PyG. Giữ HGT model nguyên trạng.

## 6. Phương pháp 4: Mini-Batch Training Loop

### 6.1 Thay block full-batch hiện tại

Trong `train_hgt_flow_classifier.py`, đoạn cần thay:

```python
# Ngày xưa:
logits = model(node_features, edge_index, edge_weight_dict=edge_weight)
loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=weight)
```

Đoạn mới:

```text
for batch in train_loader:
    batch = batch.to(device)
    with autocast(...):
        logits = model(batch.node_features, batch.edge_index, batch.edge_weight)
        seed_logits = logits[batch.seed_mask]
        loss = F.cross_entropy(seed_logits, batch.seed_labels, weight=weight)
    scaler.scale(loss).backward()
    if (step + 1) % grad_accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### 6.2 DataLoader

```text
PyTorch DataLoader:
  Dataset: list of seed_flow_id (chỉ index, không load feature ở đây)
  collate_fn: gọi sampler + load features từ GraphStore
  num_workers: 4-8 (mỗi worker open memmap riêng, OS chia page cache)
  pin_memory: True nếu dùng CUDA
  prefetch_factor: 2-4
```

`num_workers > 0` rất quan trọng: sampling là CPU-bound còn forward là
GPU-bound. Pipeline song song để GPU không idle chờ sampler.

### 6.3 Gradient accumulation

Khi mini-batch quá nhỏ làm gradient nhiễu, dùng gradient accumulation:

```yaml
train:
  batch_seed_flows: 256
  grad_accum_steps: 4         # effective batch = 1024
```

### 6.4 Validation trong mini-batch

Validation cũng phải mini-batch (val set có thể vẫn lớn):

```text
val_loader = DataLoader(seed = val_flow_ids, sampler same as train, shuffle=False)

for batch in val_loader:
  with no_grad, autocast:
    logits = model(...)
    accumulate per-class TP/FP/FN
compute macro-F1 ở cuối epoch.
```

Không recompute logits cho train set ở mỗi epoch như code hiện tại — đó là
luxury của full-batch.

### 6.5 AMP + activation checkpointing

Đã có sẵn trong `train_hgt_flow_classifier.py` và `models/hgt.py`. Giữ nguyên.
Mini-batch + AMP + checkpointing là combo chuẩn để fit graph lớn vào GPU.

### 6.6 Class weight với dataset lớn

`class_weights` hiện tính từ `labels[train_idx]` toàn cục. Khi train_idx có
hàng triệu flow, vẫn chỉ cần một lần `np.bincount` trên file
`splits/train_flow_ids.i64` → labels memmap. Lưu vào manifest, không tính lại
mỗi epoch.

## 7. Quản lý memory

### 7.1 Phân tích budget RAM

Với cấu hình 32GB RAM, target dataset 400GB:

```text
RAM budget khoảng:
  In-RAM (luôn giữ):
    technique features: ~2 MB
    tactic index: vài KB
    manifest: vài MB
    optimizer state HGT: ~vài chục MB
    model weights: vài chục MB
  Memmap (không tính vào RAM):
    flow features, packet features, edges (OS lo page cache)
  Active per-batch:
    subgraph 3-hop quanh 256 seed flows ~ 100 MB
    feature tensor + grad ~ vài GB tùy hidden_dim

Tổng: vài GB active RAM, kể cả khi disk artifact 400GB.
```

### 7.2 GPU budget

```text
HGT hidden_dim=128, num_layers=3 với subgraph mini-batch:
  forward activation: ~500 MB - 2 GB
  AMP fp16: giảm ~40%
  activation checkpointing: giảm thêm ~30-50% (đánh đổi compute)
=> Có thể fit vào GPU 8GB.
```

### 7.3 Monitor

Log mỗi epoch:

```text
avg_subgraph_nodes (per type)
avg_subgraph_edges (per relation)
peak_gpu_memory
sampler_time_per_batch
forward_time_per_batch
```

Nếu sampler_time > forward_time, tăng `num_workers`.

## 8. Roadmap thực thi

### Phase 0: hiện trạng

```text
Dataset: 27K flows, 223 MB NPZ.
Code: full-batch hoạt động ổn.
Hành động: KHÔNG xóa code full-batch. Giữ làm baseline so sánh.
```

### Phase 1: Drop-in graph store

```text
- Viết script convert NPZ -> graph_store_v1 layout (mục 3.2).
- Viết class GraphStore (mục 3.6) với memmap.
- Sanity check: load qua GraphStore -> reconstruct NPZ-equivalent dict
  -> chạy train_hgt_flow_classifier.py không đổi -> kết quả giống Phase 0.
- Mục đích: kiểm chứng store đúng, chưa thay đổi training loop.
```

### Phase 2: Sampler + mini-batch training

```text
- Viết HeteroNeighborSampler (mục 5).
- Viết DataLoader collate_fn.
- Thêm batch_mode = "neighbor_sampling" vào config.
- Train song song với baseline, so macro-F1 trên cùng split.
- Yêu cầu: macro-F1 không tụt quá ~1-2 điểm so với full-batch.
```

### Phase 3: Sharded preprocessing

```text
- Refactor build_three_tier_graph_artifact.py thành streaming pipeline (mục 4).
- Test trên PCAP set nhỏ (đảm bảo output graph_store giống Phase 1 convert).
- Scale dần lên dataset lớn hơn.
```

### Phase 4: tối ưu nâng cao (tùy chọn)

```text
- Distributed training (DDP, mỗi rank 1 GPU + memmap riêng).
- LMDB hoặc HDF5 thay numpy memmap nếu cần random access nhanh hơn.
- Caching tactic/technique features trên GPU permanent (không đi qua memmap mỗi batch).
- Warm sampler: pre-sample subgraph cho 1-2 epoch tới và đẩy vào queue.
```

## 9. Cấu hình mẫu

```yaml
data:
  graph_store: data/processed/graph_store_v1
  packet_feature: semantic
  add_reverse_edges: true
  use_semantic_edge_weights: true

model:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  ffn_multiplier: 2

train:
  batch_mode: neighbor_sampling
  batch_seed_flows: 256
  grad_accum_steps: 4
  epochs: 20
  lr: 0.001
  weight_decay: 0.00005
  patience: 5
  class_weight: balanced
  amp: true
  activation_checkpointing: true
  device: cuda
  monitor: val_macro_f1

sampler:
  hops: 3
  fanouts:
    flow__contains__packet: 20
    packet__next_packet__packet: 4
    packet__matches_technique__technique: 5
    flow__matches_technique__technique: 5
    technique__belongs_to_tactic__tactic: 1
  always_include_all_tactics: true
  reverse_fanouts:
    rev_flow__contains__packet: 1
    rev_packet__matches_technique__technique: 0
    rev_flow__matches_technique__technique: 0
    rev_technique__belongs_to_tactic__tactic: 0

dataloader:
  num_workers: 6
  prefetch_factor: 4
  pin_memory: true
```

Lưu ý:

```text
epochs giảm từ 150 -> 20:
  Mini-batch SGD có nhiều update/epoch hơn full-batch GD.
  Một epoch mini-batch xấp xỉ vài chục - vài trăm full-batch epoch về số gradient steps.
```

## 10. Đánh giá và kiểm thử

### 10.1 Kiểm thử correctness

```text
Test 1: GraphStore round-trip
  NPZ -> GraphStore -> reconstruct dict -> so sánh từng array.
  Yêu cầu: bit-exact với NPZ gốc trừ kiểu dữ liệu.

Test 2: Sampler correctness
  Sample subgraph K = num_layers quanh một seed flow.
  Verify: tất cả packet con của seed_flow đều xuất hiện trong subgraph (nếu
  fanout >= deg_actual).
  Verify: tactic index trong subgraph trùng global index.

Test 3: Equivalence baseline
  Trên dataset 27K hiện tại, train full-batch và mini-batch trên cùng split.
  Macro-F1 không lệch quá 2 điểm.
```

### 10.2 Kiểm thử scale

```text
Test 4: Memory scale
  Sinh dataset giả 1M flows / 5M packets (random features).
  Build graph_store. Đo RSS process trong khi train.
  Yêu cầu: RSS < 4 GB suốt training.

Test 5: Throughput
  Đo flows/giây processed trong training.
  Mục tiêu phase 2: > 5000 flows/giây trên CPU sampler + GPU forward.
```

### 10.3 Kiểm thử regression

```text
Mỗi commit đổi sampler/store:
  Chạy lại Test 1 + Test 3 trên dataset hiện tại 27K.
  Block merge nếu macro-F1 tụt > 2 điểm.
```

## 11. Cách viết trong báo cáo

```text
Khi dataset huấn luyện vượt mức RAM khả dụng, việc giữ toàn bộ graph trong RAM
như cấu hình full-batch ban đầu không còn khả thi. Hệ thống đề xuất kiến trúc
training scalable gồm bốn lớp: (1) On-Disk Graph Store dưới dạng numpy memmap
và CSR cho phép đọc một phần graph mà không nạp toàn bộ; (2) Sharded
Preprocessing Pipeline xây graph theo streaming, tránh OOM ở bước build; (3)
Heterogeneous Neighbor Sampler lấy K-hop subgraph quanh seed flow với K bằng
số layer của HGT, kèm fanout riêng cho mỗi relation và quy tắc giữ toàn bộ
tactic node theo global index để bảo toàn tactic embedding; (4) Mini-Batch
Training Loop kết hợp AMP và activation checkpointing để fit subgraph mini-batch
vào GPU. Static knowledge node (MITRE technique, tactic) được giữ thường trực
trong RAM do kích thước nhỏ, trong khi flow và packet feature được map từ disk
theo nhu cầu mỗi batch. Nhờ tách biệt rõ storage, sampling và training, hệ
thống có khả năng huấn luyện trên graph cỡ vài trăm GB mà không cần thay đổi
mô hình HGT.
```

## 12. Liên hệ với runtime hiện có

Tài liệu `streaming_hgt_runtime_v3_vi.md` đã chốt cách runtime online:
Incremental RTEC + Affected Subgraph. Bản thiết kế này dùng cùng tư tưởng
affected-set nhưng cho training:

```text
Online runtime:
  Persistent Store + Affected Subgraph -> Incremental RTEC -> HGT inference

Offline training:
  On-Disk Graph Store (memmap) -> K-hop subgraph -> HGT training
```

Hai con đường chia sẻ:

```text
- Cùng K-hop sampler logic (có thể tái sử dụng implementation).
- Cùng quy tắc tactic global index.
- Cùng schema node_features / edge_index / edge_weight cho HGT.
```

Khác biệt:

```text
- Runtime: nguồn dữ liệu là Hot Buffer (RAM), TTL evict.
- Training: nguồn là Graph Store (disk), không evict, có labels.
```

Khuyến nghị triển khai một module `subgraph_builder` chung, nhận một backend
abstract (`HotGraphBuffer` hoặc `GraphStore`) đều xuất ra cùng schema. Tránh
viết lại logic K-hop hai chỗ.
