# Chiến Lược Thống Nhất Xử Lý Graph Phình To Cho Runtime Và Training

Tài liệu này hợp nhất hai bài toán đã được trình bày riêng trước đó:

```text
streaming_hgt_runtime_v3_vi.md
  -> Runtime: PCAP stream thật, graph không được phình vô hạn trong RAM.
     Dùng NeutronRT incremental RTEC + RelGT centroids.

scalable_hgt_training_design_vi.md
  -> Training: dataset có thể lên ~400GB, không load hết được.
```

Hai bài toán này cùng một bản chất: graph dòng mạng tăng vô hạn theo thời gian.
Không có lúc nào toàn bộ graph được nạp vào RAM, dù là online hay offline.
Tài liệu này chốt một kiến trúc duy nhất phục vụ cả hai con đường.

## 1. Khẳng định lại vấn đề

### 1.1 Runtime PCAP thật

```text
Packet stream chảy 24/7:
  num_flows tăng theo thời gian, không có biên trên.
  num_packets tăng còn nhanh hơn.

RAM hữu hạn:
  Không thể giữ graph từ T = 0 đến hiện tại trong RAM.

Hot Graph Buffer (đã chốt ở doc cũ):
  Giải quyết bằng TTL + max_events.
  Nhưng: evict = mất dữ liệu vĩnh viễn nếu không có store khác.
```

### 1.2 Training offline với dataset lớn

```text
Dataset 400 GB:
  Không thể np.load -> RAM.
  Không thể forward full graph trong một bước.

scalable_hgt_training_design_vi.md đã đề xuất On-Disk Graph Store + Neighbor
Sampler.
```

### 1.3 Điểm chung

```text
Cả hai đều phải:
  Lưu graph trên disk dạng có thể đọc một phần.
  Chỉ giữ trong RAM phần đang xử lý ngay (hot subset).
  Có quy tắc retention rõ ràng để không phình disk.
```

Câu hỏi đặt ra: tại sao phải có hai store khác nhau? Có thể dùng **một** Graph
Store duy nhất, runtime ghi vào, training đọc ra, slow path tra cứu ra cùng.

## 2. Nguyên tắc thiết kế

### 2.1 Một nguồn sự thật duy nhất

```text
Persistent Graph Store on disk:
  Là source of truth duy nhất cho cả runtime và training.
  Append-only theo thời gian.
  Có retention policy.

Hot Graph Buffer trong RAM:
  Chỉ là write-through cache cho fast path inference.
  Có thể mất, có thể rebuild lại từ store.
  Không bao giờ là source of truth.
```

Lý do:

```text
Nếu hot buffer là duy nhất:
  - Slow path không hydrate được context cũ -> XAI report kém chất lượng.
  - Training không có dữ liệu lịch sử -> không học được attack pattern dài hạn.
  - Crash mất hết.

Nếu store là duy nhất nhưng không có hot buffer:
  - Mỗi inference phải đọc disk -> latency tăng -> mất tính chất real-time.
```

Giải pháp: hai lớp, mỗi lớp một vai trò rõ ràng.

### 2.2 Schema chung cho cả hai con đường

Schema graph (node type, edge type, feature dim) phải GIỐNG NHAU giữa runtime
và training. Lý do: HGT model train xong phải inference được trên đúng cấu
trúc đó.

```text
Node types:   flow, packet, technique, tactic
Edge types:   contains, next_packet, matches_technique, belongs_to_tactic, rev_*
Feature dims: flow=6, packet=768, technique=768, tactic=embedding by index
```

### 2.3 Tách static knowledge khỏi dynamic data

Đây là điểm chốt từ hai doc trước, lặp lại để nhấn mạnh:

```text
Static knowledge (load full vào RAM, không evict):
  - 691 MITRE technique embeddings (~2 MB)
  - 14 tactic indices

Dynamic data (memmap on disk + hot cache RAM):
  - flow features
  - packet features
  - flow/packet/technique edges
```

Static knowledge là chung cho mọi run, mọi thời điểm. Không có lý do gì để nó
phải đi qua disk.

## 3. Kiến trúc thống nhất

### 3.1 Sơ đồ tổng

```text
                  +----------------------+
                  |  PCAP Source         |
                  |  (file hoặc NIC)     |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |  Common Preprocessor |
                  |  flow_tracker        |
                  |  payload_extractor   |
                  |  student_cnn         |
                  |  mitre_matcher       |
                  +----+--------------+--+
                       |              |
            (write-through)    (write-through)
                       |              |
                       v              v
              +----------------+  +-----------------+
              |  Hot Graph     |  |  Persistent     |
              |  Buffer (RAM)  |  |  Graph Store    |
              |  TTL evict     |  |  (disk, append) |
              +-------+--------+  +--------+--------+
                      |                    |
        +-------------+----+    +----------+--------+
        |                  |    |                   |
        v                  v    v                   v
   +---------+      +-------------+         +---------------+
   | Fast    |      | Slow Path   |         | Offline       |
   | Path    |      | Hydrate     |         | Training      |
   | (HGT    |      | (SLM XAI)   |         | (HGT trainer) |
   |  online)|      +-------------+         +---------------+
   +---------+
```

### 3.2 Bốn người dùng chính của Graph Store

```text
1. Common Preprocessor: ghi (writer).
2. Fast Path: đọc qua Hot Buffer; cold lookup qua store khi miss.
3. Slow Path: đọc trực tiếp từ store (luôn dùng cold path).
4. Offline Training: đọc các shard đã sealed của store.
```

## 4. Persistent Graph Store - Layout dùng chung

### 4.1 Kế thừa từ doc training

Layout giữ nguyên đề xuất ở `scalable_hgt_training_design_vi.md` mục 3.2,
thêm một thay đổi quan trọng: **shard theo thời gian**.

```text
data/graph_store_v1/
  manifest.json
  nodes/
    flow/
      shards/
        shard_2026-05-10T00.f32     # 1 shard / giờ (hoặc / ngày)
        shard_2026-05-10T01.f32
        ...
    packet/
      shards/
        ...
    technique/
      features.f32                  # static, không shard
    tactic/
      ids.i64                       # static
  edges/
    flow__contains__packet/
      shards/
        shard_2026-05-10T00/
          indptr.i64
          indices.i64
    ...
  splits/                           # chỉ tồn tại sau khi seal đủ data
    train_flow_ids.i64
    val_flow_ids.i64
    test_flow_ids.i64
  state/
    current_shard.json              # shard đang được runtime ghi
    seal_log.jsonl                  # log đóng shard
```

### 4.2 Vì sao shard theo thời gian

```text
Runtime ghi append vào current shard.
Đủ thời gian (1 giờ) hoặc đủ kích thước (1M flows): seal shard, mở shard mới.

Lợi ích:
  - Training chỉ đọc shard đã sealed -> không race với writer.
  - Retention dễ: xóa shard cũ là xóa cả block.
  - Fast path lookup miss có thể skip đi xa nếu biết flow_id quá cũ.
```

### 4.3 Append-only invariant

Trong một shard:

```text
flow_id chỉ tăng (atomic counter).
packet_id chỉ tăng.
Edges chỉ append, không update, không delete.

Khi cần "sửa" (ví dụ flow tracker đóng flow muộn):
  - Tạo update record mới ở shard sau.
  - Không sửa shard cũ.
```

Không bao giờ sửa shard đã sealed. Đây là invariant cho phép training đọc
song song mà không lo dữ liệu thay đổi giữa epoch.

### 4.4 Retention policy

```yaml
graph_store:
  retention:
    hot_window_seconds: 300         # phải có trong Hot Buffer
    warm_window_days: 7             # giữ trên SSD nhanh
    cold_window_days: 90            # archive ra HDD chậm hoặc nén
    drop_after_days: 365            # xóa hẳn (tùy compliance)
```

Training thường lấy data trong warm + cold window. Hot không quan trọng cho
training (mới một phút, chưa label hết).

## 5. Hot Graph Buffer - Vai trò mới

### 5.1 Trước đây và bây giờ

```text
Cách tiếp cận cũ:
  Hot Buffer là source duy nhất cho runtime.
  Evict = mất hẳn dữ liệu (chỉ còn cold_store JSONL phụ trợ).

Doc này:
  Hot Buffer là CACHE LAYER trên Persistent Graph Store.
  Evict = chỉ rời RAM, dữ liệu vẫn còn trên disk.
  Slow path KHÔNG đọc Hot Buffer nữa, đọc thẳng store.
```

### 5.2 Write-through pattern

```text
on_packet(packet):
  flow_id = flow_tracker.update(packet)
  payload = extract_payload_256(packet)
  emb = student_cnn(payload)
  topk = cosine_topk(emb, technique_emb, k=5)

  # Bước 1: ghi xuống Persistent Store TRƯỚC.
  store.append_packet(packet_id, flow_id, emb, topk, timestamp)

  # Bước 2: cập nhật Hot Buffer SAU.
  hot_buffer.add(packet_id, flow_id, emb, topk, timestamp)

  # Bước 3: build K-hop từ Hot Buffer (fall back store nếu miss).
  subgraph = build_khop(seed_flow_id=flow_id, hops=K, hot=hot_buffer, cold=store)

  logits = hgt(subgraph)
  return policy(logits)
```

Quan trọng:

```text
Ghi store TRƯỚC, hot buffer SAU.
=> Nếu crash giữa chừng, dữ liệu vẫn an toàn trên disk.
=> Hot buffer rebuild được khi service restart bằng cách replay last N seconds.
```

Nếu lo I/O disk chậm cho fast path: dùng append batching (gom 1000 packet
ghi 1 lần) hoặc dùng WAL (write-ahead log) dạng memory-mapped append-only,
seal định kỳ thành shard.

### 5.3 Build K-hop với fallback

```text
build_khop(seed_flow_id, hops, hot, cold):
  frontier = {"flow": {seed_flow_id}}
  subgraph = empty
  for h in 1..hops:
    next_frontier = {}
    for edge_type:
      for src in frontier[src_type]:
        # Thử lấy neighbor từ Hot trước
        nbrs = hot.out_neighbors(edge_type, src)
        if nbrs is None:           # miss: src đã evict khỏi hot
          nbrs = cold.out_neighbors(edge_type, src)
        subgraph.add(edge_type, src, nbrs)
        next_frontier[dst_type] |= set(nbrs)
    frontier = merge(frontier, next_frontier)
  return subgraph
```

Hot miss không phải lỗi. Đó là trường hợp bình thường khi flow đã cũ. Cold
lookup trên SSD đủ nhanh cho subgraph nhỏ (vài trăm node).

### 5.4 Khi nào cần Hot Buffer

Hot Buffer chỉ tồn tại để giảm latency của fast path. Nếu store nhanh đủ
(SSD NVMe + memmap warm cache), có thể bỏ hot buffer hoàn toàn và đọc thẳng
store. Đây là một biến thể đơn giản hơn cho hệ thống nhỏ.

## 6. Slow Path - Luôn dùng store

### 6.1 Thay đổi so với doc cũ

```text
Doc cũ:
  Slow path đọc Hot Buffer trước; nếu data đã evict thì không hydrate được
  hoặc fall back JSONL cold_store nhỏ.

Doc này:
  Slow path luôn đọc thẳng Persistent Graph Store.
  Không phụ thuộc Hot Buffer.
  JSONL cold_store cũ KHÔNG cần nữa - đã được Persistent Store thay thế.
```

### 6.2 Vì sao bỏ JSONL cold_store

```text
JSONL cold_store ngày xưa được thiết kế để bù cho Hot Buffer evict.
Khi đã có Persistent Graph Store đầy đủ, nó trở nên thừa.
=> Đơn giản hóa hệ thống bằng cách bỏ một lớp.
```

### 6.3 Hydrate context cho XAI

```text
worker_slow_path:
  while True:
    flow_id = queue.get()
    context = build_khop(seed_flow_id=flow_id, hops=K_slow, hot=None, cold=store)
    payload_texts = store.get_payload_text_for_packets(context.packet_ids)
    technique_meta = store.get_technique_metadata(context.technique_ids)
    prompt = format_xai_prompt(context, payload_texts, technique_meta)
    report = slm_inference(prompt)
    save_report(report)
```

K_slow có thể lớn hơn K_fast (ví dụ K_slow=4 thay vì K_fast=3) vì slow path
không bị giới hạn latency.

## 7. Offline Training - Đọc shard đã sealed

### 7.1 Nguyên tắc

```text
Training:
  - Chỉ đọc shard đã sealed (state.current_shard không tham gia).
  - Đọc qua memmap như đã thiết kế ở scalable_hgt_training_design_vi.md.
  - NeighborLoader với K = HGT num_layers.
  - Không cần sao chép data từ runtime sang training.
```

### 7.2 Đồng bộ schema

Vì runtime ghi và training đọc cùng store, schema phải đồng bộ qua manifest
versioning:

```text
manifest.json:
  version: "v1"
  schema:
    flow_feature_names: [...]
    packet_feature_kind: "semantic" | "payload"
    add_reverse_edges: true
    similarity_threshold: 0.82
    flow_top_k: 5
    packet_top_k: 5
```

Khi đổi schema (ví dụ thêm feature flow mới):

```text
- Tạo store v2 song song.
- Runtime ghi vào v2.
- Training đọc v2 sau khi có đủ data.
- Giữ v1 cho retention period rồi xóa.
```

Không bao giờ "migrate in place".

### 7.3 Retraining periodic

```text
Cron hằng ngày / hằng tuần:
  - Snapshot các shard mới sealed kể từ training trước.
  - Build splits trên cumulative data (giữ split deterministic theo seed).
  - Resume training HGT từ checkpoint cũ trên data mở rộng.
  - Validate macro-F1 trước khi promote model mới.
  - Hot-swap student_cnn / hgt weights vào runtime.
```

Đây là vòng feedback dài hạn: runtime sinh data, training tiêu thụ data, mô
hình tốt hơn quay lại runtime.

## 8. Quản lý disk

### 8.1 Tăng trưởng

Ước lượng cho dataset thực:

```text
flow record    ~ 50 byte
packet record  ~ 768 * 4 + 100 ~ 3.2 KB
edge record    ~ 16 byte

10K packets/giây liên tục:
  ~ 32 MB/giây ~ 2.7 TB/ngày

Phải có retention. Không có chuyện giữ raw mọi packet mãi mãi.
```

### 8.2 Compaction định kỳ

```text
Daily job:
  - Sealed shards trong warm window: re-encode sang format nén (Parquet zstd).
  - Sealed shards trong cold window: gộp thành block lớn, nén mạnh hơn.
  - Sealed shards quá retention: xóa.

Training reader hỗ trợ cả hai format (memmap raw hoặc decode Parquet on-the-fly).
```

### 8.3 Cap cứng để chống đầy disk

```text
graph_store:
  disk_quota_gb: 1000
  on_quota_exceeded: "drop_oldest"
```

Khi disk gần đầy, drop shard cũ nhất trước khi nhận shard mới. Phải có alert.

## 9. Tóm tắt thay đổi so với hai doc trước

### 9.1 Với streaming_hgt_runtime_v3_vi.md

```text
GIỮ NGUYÊN:
  - Hot Graph Buffer với TTL.
  - Affected-set K-hop builder.
  - Phân biệt runtime nodes vs static knowledge nodes.
  - K = HGT num_layers.

THAY ĐỔI:
  - Hot Buffer không còn là source of truth.
  - Bỏ JSONL cold_store nhỏ; thay bằng Persistent Graph Store đầy đủ.
  - Slow path đọc thẳng Persistent Store, không qua Hot Buffer.
  - Thêm pattern write-through: ghi store trước, hot buffer sau.
```

### 9.2 Với scalable_hgt_training_design_vi.md

```text
GIỮ NGUYÊN:
  - Layout sharded On-Disk Graph Store.
  - Heterogeneous Neighbor Sampler.
  - Mini-batch training với AMP + activation checkpointing.

THAY ĐỔI:
  - Store không còn là kết quả của một preprocessing pass tách rời.
  - Store được xây dựng dần (incremental) bởi runtime preprocessor.
  - Training đọc shard đã sealed, không re-build artifact.
  - Có shard theo thời gian thay vì shard tự do.
```

## 10. Roadmap thực thi thống nhất

### Phase A: Persistent Graph Store offline only

```text
- Implement Graph Store layout + writer + reader (mục 4).
- Convert NPZ hiện tại sang store v1.
- Đảm bảo training (đã có) đọc store ra đúng kết quả như NPZ.
- Phase này không động đến runtime.
```

### Phase B: Mini-batch training trên store

```text
- Implement NeighborLoader (đã đề xuất ở scalable_hgt_training_design_vi.md).
- So sánh full-batch vs mini-batch trên dataset hiện tại.
- Kéo dữ liệu test lên 5-10x để verify scale.
```

### Phase C: Runtime ghi vào store

```text
- Refactor runtime preprocessor: ghi write-through Hot Buffer + Store.
- Sealing scheduler theo thời gian / kích thước.
- Verify slow path đọc được context lịch sử qua store.
- Bỏ JSONL cold_store nhỏ.
```

### Phase D: Continuous training loop

```text
- Cron retraining đọc shard mới + validate + hot-swap.
- Retention + compaction.
- Alert khi disk/RAM gần ngưỡng.
```

### Phase E: tối ưu nâng cao

```text
- Distributed runtime ghi nhiều process / nhiều node.
- WAL hoặc Kafka làm bộ đệm bền giữa preprocessor và store writer.
- DDP training nhiều GPU đọc cùng store.
```

## 11. Cấu hình thống nhất mẫu

```yaml
graph_store:
  root: data/graph_store_v1
  shard_seal:
    by_time_seconds: 3600
    by_size_flows: 1000000
  retention:
    hot_window_seconds: 300
    warm_window_days: 7
    cold_window_days: 90
    drop_after_days: 365
  disk_quota_gb: 500
  on_quota_exceeded: drop_oldest

runtime:
  hot_graph:
    ttl_seconds: 60
    max_events: 100000
    max_packets_per_flow: 64
    max_techniques_per_node: 5
  subgraph:
    hops: 3
    use_reverse_edges: true
    include_all_tactics_in_global_order: true
  slow_path:
    enabled: true
    hydrate_hops: 4
    queue_max_size: 1000
    cold_source: graph_store          # thay JSONL cũ

training:
  source: graph_store
  read_sealed_only: true
  batch_mode: neighbor_sampling
  batch_seed_flows: 256
  grad_accum_steps: 4
  sampler:
    hops: 3
    fanouts:
      flow__contains__packet: 20
      packet__next_packet__packet: 4
      packet__matches_technique__technique: 5
      flow__matches_technique__technique: 5
      technique__belongs_to_tactic__tactic: 1
    always_include_all_tactics: true
```

## 12. Cách viết trong báo cáo

```text
Cả runtime online và offline training đều đối mặt với cùng một vấn đề: graph
dòng mạng tăng vô hạn theo thời gian, không thể giữ trọn vẹn trong RAM. Hệ
thống đề xuất một Persistent Graph Store dạng sharded on-disk, append-only,
shard theo thời gian, đóng vai trò nguồn sự thật duy nhất cho toàn bộ pipeline.
Runtime ghi đồng thời vào Persistent Graph Store và một Hot Graph Buffer trong
RAM theo pattern write-through, trong đó Hot Buffer chỉ là cache giảm latency
cho fast path inference, không phải nơi giữ dữ liệu duy nhất. Slow Path khi
hydrate context cho SLM đọc thẳng từ Persistent Store, đảm bảo bất kỳ flow
suspicious nào cũng có đầy đủ context lịch sử dù đã rời khỏi Hot Buffer. Offline
training đọc các shard đã sealed của cùng Persistent Store thông qua memory
map và neighbor sampler K-hop, với K bằng số layer của HGT. Nhờ kiến trúc
thống nhất này, không lúc nào hệ thống cần load toàn bộ graph vào RAM, đồng
thời tránh được việc duy trì hai store song song dễ lệch dữ liệu. Static
knowledge gồm 691 MITRE technique embedding và 14 tactic vẫn được giữ thường
trực trong RAM ở mọi tầng vì kích thước nhỏ.
```

## 13. Bảng quan hệ ba tài liệu

```text
streaming_hgt_runtime_v3_vi.md
  -> Incremental RTEC engine (NeutronRT-style operator decomposition).
  -> Affected subgraph propagation thay full recomputation.
  -> RelGT centroids cho global context.
  -> HetSGFormer + ILLE two-tier offline + online.
  -> Slow path RAG + LoRA SLM.

scalable_hgt_training_design_vi.md
  -> On-Disk Graph Store layout.
  -> Sharded preprocessing pipeline.
  -> Heterogeneous Neighbor Sampler.
  -> Mini-batch training loop.

unified_graph_growth_strategy_vi.md (tài liệu này)
  -> Hợp nhất hai bài toán.
  -> Persistent Graph Store là source of truth duy nhất.
  -> Hot Buffer = cache, không phải store.
  -> Slow path đọc thẳng store.
  -> Training đọc shard sealed.
  -> Schema và retention thống nhất.
```

Đọc theo thứ tự: doc này trước (kiến trúc tổng), rồi hai doc còn lại cho chi
tiết từng nhánh.
