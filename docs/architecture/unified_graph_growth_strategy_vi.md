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

## 10. Quyết định: giữ kỹ thuật cũ, thay mới, hay kết hợp?

Phần này trả lời thẳng ba câu hỏi thiết kế quan trọng nhất khi đối
chiếu kiến trúc cũ (mục 1-9) với các kỹ thuật chống phình to công bố
trong giai đoạn 2024-2026:

```text
Q1. Có giữ kỹ thuật cũ không?
Q2. Có thể thay hoàn toàn bằng kỹ thuật 2024-2026 không?
Q3. Nếu phải kết hợp, có nặng / phức tạp quá không?
```

### 10.1 Trả lời nhanh

```text
Q1. CÓ, bắt buộc giữ kỹ thuật cũ (Phase A-D).
Q2. KHÔNG. Kỹ thuật 2024-2026 không thay thế được kiến trúc cũ.
Q3. CÓ NGUY CƠ. Bật tất cả cùng lúc sẽ nặng và khó debug.
    Phải triển khai theo TẦNG, mỗi tầng giải một triệu chứng cụ thể.
```

Lý do gốc: **hai nhóm kỹ thuật giải hai bài toán hoàn toàn khác nhau.**

```text
Kỹ thuật cũ  (Phase A-D)  = NỀN MÓNG dữ liệu.
  Trả lời các câu: "Dữ liệu sống ở đâu? Khi nào hết hạn? Đọc lại
  bằng cách nào? Schema thế nào?"

Kỹ thuật mới (2024-2026)  = TỐI ƯU tính toán/bộ nhớ.
  Trả lời các câu: "Cạnh nào đáng giữ? Quên gì sớm? Học liên tục
  không bị forget? Train mô hình rộng trên VRAM nhỏ thế nào?"
```

Bỏ nền móng -> không có chỗ lưu dữ liệu, không có gì cho tối ưu
chạy lên trên. Bật toàn bộ tối ưu cùng lúc -> nhiều mảnh logic, mỗi
mảnh có hyperparameter riêng, một bug ở SIGC có thể bị che bởi HERO,
debug rất khó.

### 10.2 Bảng quyết định: layer-by-layer

Bảng so sánh trực tiếp **từng tầng kiến trúc**, đối chiếu kỹ thuật cũ
với kỹ thuật mới, và đưa ra quyết định cuối.

| Tầng kiến trúc | Kỹ thuật cũ (Phase A-D) | Kỹ thuật 2024-2026 | Quyết định | Diễn giải |
|---|---|---|---|---|
| Storage backend | Persistent Graph Store on disk + sharding theo thời gian + retention 4-mức | RapidStore (multi-version, decoupled R/W) | **Giữ cũ, RapidStore chỉ là drop-in khi cần** | RapidStore là engine cao cấp hơn nhưng vẫn cần layout sharded. Chỉ thay khi đo được Read/Write amplification > 2x. |
| RAM cache lifecycle | Hot Buffer TTL + `max_events` (passive evict) | BGML selective forgetting (active forget request) | **Kết hợp: TTL bắt buộc, BGML là add-on** | TTL bảo đảm cận trên dung lượng, BGML chỉ giúp drop sớm khi đã xác minh sạch. Không kỹ thuật nào thay được kỹ thuật còn lại. |
| Data ingestion filter | (Không có ở doc cũ) | SIGC Local Contribution Score + HGSampling-budget per node-type | **Thêm SIGC, HGSampling-budget chỉ khi cần** | SIGC rất rẻ (1 phép tính tỷ lệ độ), lợi ích lớn: giảm số cạnh ghi xuống store. HGSampling-budget chỉ cần khi quan sát loại nút thiểu số (technique, tactic) bị nhấn chìm. |
| Subgraph builder | K-hop với fanout cố định mỗi edge type | DLG-IDS Top-N edges per seed + localized temporal attention | **Thay fanout cố định bằng DLG-IDS Top-N** | Bản chất giống nhau (đều là chiến lược lấy mẫu lân cận), DLG-IDS chỉ thông minh hơn: chọn cạnh theo trọng số cosine/score thay vì lấy ngẫu nhiên K cạnh. Latency giảm ~50% theo paper. |
| Time encoding | Timestamp tuyệt đối lưu cùng shard | RTE (Relative Temporal Encoding, sinusoid hàm delta_t) | **Cộng thêm, không thay** | Timestamp vẫn cần cho retention và compaction; RTE chỉ thêm vector embedding khi đọc cross-shard cho HGT. Bỏ qua được nếu chưa thấy lợi ở thực nghiệm. |
| Training data loop | Cumulative retrain trên toàn bộ shard sealed | HERO continual learning (DiSCo sampling + knowledge distillation) | **Thay khi data > vài tháng** | HERO thực sự thay được retrain full. Cứ tiếp tục cumulative thì sau 6-12 tháng sẽ vỡ. Tuy nhiên HERO có thêm hyperparameter (replay size, distillation weight), nên chỉ bật khi data growth thực sự yêu cầu. |
| Optimizer memory | AdamW + AMP + activation checkpointing | GaLore (gradient low-rank projection) hoặc ZeRO | **Tùy chọn cho cấu hình rộng** | Cấu hình baseline (hidden 128) không cần GaLore. Khi train GATransformer (hidden 256 / 6 layers) trên T4 15GB và OOM, mới cần GaLore rank=128. |
| Schema / static knowledge | Tách static MITRE/tactic khỏi dynamic data | (Không có kỹ thuật mới tương ứng) | **Giữ cũ** | Đây vẫn là tối ưu rẻ và bắt buộc cho kiến trúc IDS này. |

### 10.3 Kiến trúc tinh gọn tối ưu (Lean Optimal)

Đây là cấu hình **ưu việt nhất cho project IDS hiện tại** theo các
tiêu chí: thấp về độ phức tạp, cao về tỷ lệ lợi ích / chi phí, có thể
triển khai trong 1-2 tuần.

```text
+----------------------------------------------------------+
|  TIER 1 - NỀN MÓNG (bắt buộc, giữ nguyên kỹ thuật cũ)    |
+----------------------------------------------------------+
| 1. Persistent Graph Store on disk, sharding theo thời    |
|    gian, retention 4-mức (mục 4).                        |
| 2. Hot Graph Buffer write-through, TTL + max_events      |
|    (mục 5).                                              |
| 3. Slow Path đọc thẳng Persistent Store, không qua Hot   |
|    Buffer (mục 6).                                       |
| 4. Training đọc shard đã sealed qua memmap + Neighbor    |
|    Sampler K-hop (mục 7).                                |
| 5. Static knowledge MITRE/tactic luôn nằm trong RAM      |
|    (mục 2.3).                                            |
+----------------------------------------------------------+

+----------------------------------------------------------+
|  TIER 2 - TỐI ƯU RẺ (thêm ngay, lợi ích / chi phí cao)   |
+----------------------------------------------------------+
| 6. SIGC ở Common Preprocessor: chỉ thêm 1 score = (out/  |
|    in_degree_ratio) * decay(hop_distance). Cạnh dưới     |
|    ngưỡng bị bỏ trước khi ghi store.                     |
| 7. DLG-IDS Top-N edges ở subgraph builder: thay fanout   |
|    cố định bằng chọn N cạnh có trọng số semantic cao     |
|    nhất cho mỗi seed.                                    |
+----------------------------------------------------------+

+----------------------------------------------------------+
|  TIER 3 - TỐI ƯU TÙY CHỌN (chỉ bật khi có triệu chứng)   |
+----------------------------------------------------------+
| 8. BGML selective forgetting -> Triệu chứng: Hot Buffer  |
|    churn > 80%, tỷ lệ evict / insert vượt 0.5.           |
| 9. HERO continual learning -> Triệu chứng: test macro-F1 |
|    giảm > 3% sau retraining full hoặc data lịch sử > 6   |
|    tháng.                                                |
|10. GaLore low-rank optimizer -> Triệu chứng: OOM khi     |
|    train hidden_dim >= 256 trên GPU 15GB.                |
|11. RapidStore multi-version backend -> Triệu chứng: I/O  |
|    contention giữa writer runtime và reader training.    |
|12. RTE cross-shard -> Triệu chứng: XAI cần lý giải chuỗi |
|    tấn công dài (> 6h) và cosine similarity chuỗi không  |
|    đủ phân biệt.                                         |
+----------------------------------------------------------+
```

Mặc định triển khai chỉ gồm **Tier 1 + Tier 2**. Tier 3 nằm trong
backlog, mỗi mục có entry trong runbook với chỉ số đo lường cụ thể
để biết khi nào cần kích hoạt.

### 10.4 So sánh chi phí ba phương án

Đánh giá định tính ba lựa chọn triển khai:

| Tiêu chí | A. Chỉ giữ kỹ thuật cũ (Phase A-D) | B. Lean Optimal (Tier 1 + 2) | C. Full Stack (Tier 1 + 2 + 3) |
|---|---|---|---|
| Số module logic | ~5 | ~7 | ~12 |
| Số hyperparameter cần tune | ~8 | ~12 | ~25 |
| Effort triển khai | 1-2 tuần (đã thiết kế) | +1 tuần | +2-3 tuần thêm |
| Effort vận hành | Tháp | Trung bình | Cao (cần dashboard cho từng kỹ thuật) |
| Disk growth | Đúng theo retention | Giảm 30-50% (SIGC lọc cạnh trước khi ghi) | Như Lean Optimal (Tier 3 không động vào disk) |
| Fast path latency | OK | Giảm 30-50% (DLG-IDS Top-N) | Giảm thêm 5-10% (BGML giảm cache lookup miss) |
| Catastrophic Forgetting | Có rủi ro nếu retraining đè trực tiếp | Vẫn có rủi ro | Được giải quyết (HERO) |
| Khả năng debug | Dễ | Dễ | Khó (nhiều tầng tương tác) |
| Phù hợp với project hiện tại | Đủ tốt nhưng để lại tiềm năng | **Khuyến nghị** | Quá sức cho 1 đồ án |

Phương án B (**Lean Optimal**) là điểm cân bằng tốt nhất giữa lợi ích
ká»¹ thuật và độ phức tạp vận hành. Nó giữ toàn bộ ưu điểm của kiến trúc
cũ (một nguồn sự thật, write-through, sealed-shard training) và chỉ
thêm hai kỹ thuật mới (SIGC, DLG-IDS Top-N) - cả hai đều rẻ, có công
thức rõ ràng, không phá vỡ luồng dữ liệu.

### 10.5 Cấu hình tham khảo cho Lean Optimal

```yaml
# Phụ lục cho mục 12 (Cấu hình thống nhất mẫu). Chỉ bao gồm các trường
# mới của Tier 2; các trường Tier 1 đã có sẵn ở mục 12.

preprocessor:
  sigc:
    enabled: true
    score_formula: "in_out_ratio * exp(-alpha * hop_distance)"
    alpha: 0.4
    min_score_to_keep_edge: 0.15
    apply_to_edge_types:
      - flow__contains__packet
      - packet__next_packet__packet
      - packet__matches_technique__technique

subgraph_builder:
  dlg_ids_top_n:
    enabled: true
    top_n_per_seed:
      flow__contains__packet: 24
      packet__next_packet__packet: 4
      packet__matches_technique__technique: 5
      flow__matches_technique__technique: 5
    sort_by: semantic_edge_weight    # đã có trong manifest
    fallback_to_random_when_tie: true

# Tier 3 (default tắt). Bật khi có triệu chứng ở mục 10.3.
hot_graph:
  bgml_selective_forgetting:
    enabled: false
training:
  hero_continual_learning:
    enabled: false
  galore:
    enabled: false
```

Tinh thần: ngoại trừ hai block `sigc` và `dlg_ids_top_n` được bật mặc
định, mọi kỹ thuật 2024-2026 khác đều `enabled: false` và chỉ kích
hoạt khi runbook xác nhận triệu chứng đã xuất hiện.

### 10.6 Bảng nguồn kỹ thuật 2024-2026

Tham chiếu nhanh để khi báo cáo / bảo vệ đồ án có thể trích dẫn.

| Kỹ thuật | Nguồn / Bài báo | Bản chất tóm gọn |
|---|---|---|
| SIGC | Structural Importance Graph Compression literature, 2025 | Score độ-bậc * suy giảm theo khoảng cách hop, loại cạnh dưới ngưỡng |
| DLG-IDS | DLG-IDS for ICS, 2026 | Sparse topology + localized temporal attention, giảm 53% latency |
| BGML | Graph Memory Learning, 2024-2025 | Quên có chọn lọc, lấy cảm hứng từ tháp khớp thần kinh |
| HERO + DiSCo | HEterogeneous continual gRaph learning via meta-knOwledge distillation, 2025 | Replay subgraph + chưng cất tri thức để chống Catastrophic Forgetting |
| GaLore | Gradient Low-Rank Projection, 2024-2025 | Chiếu gradient vào không gian low-rank, giảm 65% VRAM optimizer |
| RapidStore | Dynamic graph storage systems, 2025 | Multi-version + decoupled R/W cho concurrent queries |
| RTE | Heterogeneous Graph Transformer (Hu et al.) + dynamic graph variants | Sinusoid encoding của delta_t cho cross-shard temporal attention |

## 11. Roadmap thực thi thống nhất

Roadmap chia hai mảng: **Bắt buộc** (Tier 1 + Tier 2 trong mục 10.3)
và **Tùy chọn theo triệu chứng** (Tier 3). Mặc định chỉ làm Bắt buộc;
Tùy chọn chỉ kích hoạt khi runbook xác nhận triệu chứng.

### Bắt buộc

#### Phase A: Persistent Graph Store offline only

```text
- Implement Graph Store layout + writer + reader (mục 4).
- Convert NPZ hiện tại sang store v1.
- Đảm bảo training (đã có) đọc store ra đúng kết quả như NPZ.
- Phase này không động đến runtime.
```

#### Phase B: Mini-batch training trên store

```text
- Implement NeighborLoader (đã đề xuất ở scalable_hgt_training_design_vi.md).
- So sánh full-batch vs mini-batch trên dataset hiện tại.
- Kéo dữ liệu test lên 5-10x để verify scale.
```

#### Phase C: Runtime ghi vào store

```text
- Refactor runtime preprocessor: ghi write-through Hot Buffer + Store.
- Sealing scheduler theo thời gian / kích thước.
- Verify slow path đọc được context lịch sử qua store.
- Bỏ JSONL cold_store nhỏ.
```

#### Phase D: Continuous training loop (đơn giản)

```text
- Cron retraining đọc shard mới + validate + hot-swap.
- Retention + compaction.
- Alert khi disk/RAM gần ngưỡng.
- KHÔNG bật HERO ở phase này. Retrain cumulative đơn giản đủ tốt
  trong giai đoạn đầu, HERO chỉ kích hoạt ở Phase G nếu cần.
```

#### Phase E: Tier 2 (SIGC + DLG-IDS Top-N)

```text
- Thêm Local Contribution Score vào Common Preprocessor.
  Drop cạnh dưới ngưỡng TRƯỚC khi ghi store.
  -> Tiết kiệm disk + giảm số cạnh phải xử lý ở mọi tầng sau.
- Thay fanout cố định trong subgraph_builder bằng DLG-IDS Top-N
  per seed flow.
  -> Mục tiêu đo: -30% inference latency, macro-F1 không giảm
     quá 1%.
- Hai thay đổi này độc lập, có thể release từng cái với A/B test.
```

### Tùy chọn (chỉ làm khi có triệu chứng)

#### Phase F: BGML selective forgetting (nếu Hot Buffer churn cao)

```text
Triệu chứng kích hoạt:
  - Tỷ lệ evict/insert ở Hot Buffer > 0.5 trong 24h.
  - Hoặc CPU bị nghẽn vì traverse các cạnh nháp đã quá hạn.

Hành động:
  - Thêm Forgetting Request queue song song với write-through.
  - Verified-benign events (whitelist signature) -> drop khỏi Hot
    Buffer ngay, không chờ TTL.
  - Giữ shadow log nhỏ để đo tỷ lệ false-forgetting.
```

#### Phase G: HERO continual learning (nếu Catastrophic Forgetting xuất hiện)

```text
Triệu chứng kích hoạt:
  - Test macro-F1 trên các attack family cũ giảm > 3% sau retraining.
  - Hoặc data lịch sử đã > 6 tháng và cumulative retrain bắt đầu
    quá dài / quá tốn VRAM.

Hành động:
  - Implement DiSCo sampler: trích Top-K subgraph đại diện theo
    meta-path từ shard sealed.
  - Thêm knowledge distillation loss (teacher = checkpoint trước).
  - Cron retraining: replay subgraphs + dữ liệu mới -> validate
    bằng cách so accuracy trên cả task cũ và task mới trước khi
    hot-swap.
```

#### Phase H: Tối ưu phần cứng & phân tán

```text
Triệu chứng kích hoạt:
  - OOM khi train hidden_dim >= 256 trên GPU 15GB -> bật GaLore
    rank=128, update_proj_gap=200.
  - I/O contention giữa runtime writer và training reader -> đánh
    giá RapidStore-style multi-version graph store.
  - Nhu cầu xử lý nhiều node/process -> Distributed runtime, WAL
    hoặc Kafka làm bộ đệm bền, DDP training nhiều GPU đọc cùng store.
```

## 12. Cấu hình thống nhất mẫu

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

## 13. Cách viết trong báo cáo

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

Để khống chế thêm hiện tượng Graph Bloating theo cập nhật nghiên cứu 2024-2026,
hệ thống áp dụng phương án Lean Optimal: giữ nguyên kiến trúc cũ (Persistent
Graph Store + Hot Buffer write-through + sealed-shard training) làm nền móng
bắt buộc, và chỉ thêm hai tối ưu rẻ. Thứ nhất, Structural Importance Graph
Compression (SIGC) đặt ở Common Preprocessor tính Local Contribution Score
cho mỗi cạnh theo tỷ lệ bán bậc và suy giảm theo khoảng cách hop, loại bỏ
cạnh dư thừa trước khi ghi xuống store, giúp giảm đáng kể disk growth. Thứ
hai, DLG-IDS Top-N edge selection thay thế chiến lược fanout cố định trong
subgraph builder bằng cách chỉ giữ N cạnh có trọng số semantic cao nhất cho
mỗi seed flow, cho phép giảm 30-50% inference latency mà không suy giảm
macro-F1. Các kỹ thuật tinh vi hơn (BGML selective forgetting, HERO continual
learning với DiSCo + knowledge distillation, GaLore low-rank optimizer,
RapidStore multi-version backend, RTE cross-shard temporal encoding) được
giữ trong backlog Tier 3 với điều kiện kích hoạt rõ ràng - chỉ triển khai
khi runbook phát hiện đúng triệu chứng (Catastrophic Forgetting, Hot Buffer
churn, OOM khi train rộng, hay I/O contention). Cách tiếp cận tầng này tránh
được vừa rủi ro phình to dữ liệu vừa rủi ro phình to độ phức tạp hệ thống.
```

## 14. Bảng quan hệ ba tài liệu

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
  -> Lean Optimal: GIỮ nguyên kiến trúc cũ (Tier 1), thêm SIGC +
     DLG-IDS Top-N (Tier 2) làm tối ưu rẻ.
  -> Kỹ thuật 2024-2026 còn lại (BGML, HERO, GaLore, RapidStore,
     RTE) là Tier 3 - chỉ bật khi runbook xác nhận triệu chứng.
```

Đọc theo thứ tự: doc này trước (kiến trúc tổng), rồi hai doc còn lại cho chi
tiết từng nhánh.
