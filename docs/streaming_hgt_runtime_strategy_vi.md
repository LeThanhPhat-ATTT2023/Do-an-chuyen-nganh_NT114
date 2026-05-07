# Hướng Xử Lý Graph Phình To Cho HGT Runtime

Tài liệu này chốt lại hướng triển khai online/near-real-time cho pipeline IDS
đang dùng payload embedding, MITRE context và HGT classifier.

Mục tiêu không phải là bỏ graph hay thay HGT bằng student. Mục tiêu là giữ
đúng định hướng heterogeneous graph, nhưng không để graph tăng vô hạn khi hệ
thống chạy liên tục.

## 1. Điểm chính cần phân biệt

Trong dự án hiện tại có hai khái niệm khác nhau:

```text
Three-tier graph:
  Cấu trúc đồ thị gồm flow, packet, MITRE technique/tactic.

HGT num_layers:
  Số lần message passing của model HGT.
```

Graph 3 tầng không đồng nghĩa với HGT 3 layer. Graph 3 tầng là cấu trúc node
type và edge type. HGT 3 layer là độ sâu lan truyền thông tin qua cạnh.

Nếu config HGT là:

```yaml
model:
  num_layers: 3
```

thì runtime subgraph nên lấy vùng ảnh hưởng:

```text
K-hop với K = 3
```

Không nên chỉ lấy 1-hop nếu model đã train với 3 message-passing layers, vì
runtime sẽ thiếu context so với lúc train.

## 2. Kiến trúc runtime được đề xuất

Pipeline online/near-real-time:

```text
Packet stream
-> trích xuất payload_256
-> Student 1D-CNN tạo packet embedding 768-D
-> cosine với MITRE technique embeddings để lấy top-k technique
-> Hot Graph Buffer cập nhật graph nóng
-> build affected K-hop subgraph quanh seed flow, K = HGT num_layers
-> HGT inference trên subgraph nhỏ
-> nếu benign: trả kết quả nhanh
-> nếu suspicious/malicious: push flow_id vào slow-path queue
-> Slow Path hydrate context rộng hơn
-> SLM tạo giải thích/XAI report
```

Student 1D-CNN chỉ thay SecureBERT trong đường online embedding. HGT vẫn là
classifier chính.

## 3. Phương pháp 1: Hot Graph Buffer

### Mục tiêu

Quản lý graph động trong RAM bằng TTL hoặc `max_events` để graph không phình
vô hạn.

### Nguyên tắc

```text
Flow và packet là runtime nodes:
  Có thể bị evict theo TTL/max_events.

MITRE technique và tactic là static knowledge nodes:
  Load một lần, không evict.
```

### Cấu trúc dữ liệu

Nên dùng:

```text
deque event_queue:
  Lưu thứ tự thời gian của flow/packet/event.

dict flow_features:
  flow_id -> vector thống kê flow.

dict packet_embeddings:
  packet_id -> embedding 768-D từ student 1D-CNN.

dict packet_payload_text:
  packet_id -> payload_256 dạng hex text, dùng cho XAI/SLM.

dict flow_to_packets:
  flow_id -> list packet_id.

dict packet_to_flow:
  packet_id -> flow_id.

dict packet_to_mitre:
  packet_id -> list[(technique_id, score)].

dict flow_to_mitre:
  flow_id -> list[(technique_id, score)].

dict technique_features:
  technique_id -> MITRE embedding 768-D.

dict technique_to_tactic:
  technique_id -> tactic_id.
```

### Logic add và evict

```text
add_event(packet_or_flow):
  1. Thêm event vào deque.
  2. Cập nhật feature dict.
  3. Cập nhật adjacency dict.
  4. Nếu packet mới:
       - Gán packet vào flow.
       - Gán packet vào top-k MITRE technique.
       - Cập nhật flow_to_mitre bằng aggregate từ packet_to_mitre.
  5. Gọi evict.

evict:
  1. Nếu vượt max_events hoặc TTL:
       old_event = deque.popleft()
  2. Nếu old_event là packet:
       - Xóa packet_embeddings[packet_id].
       - Xóa packet_to_flow[packet_id].
       - Xóa packet khỏi flow_to_packets[flow_id].
       - Xóa packet_to_mitre[packet_id].
       - Rebuild flow_to_mitre[flow_id].
  3. Nếu old_event là flow:
       - Xóa flow_features[flow_id].
       - Xóa flow_to_packets[flow_id].
       - Xóa flow_to_mitre[flow_id].
       - Xóa các packet con của flow nếu chưa bị xóa.
```

Khuyến nghị dùng đồng thời:

```text
ttl_seconds = 30 đến 300 giây
max_events = giới hạn RAM cứng
max_packets_per_flow = giới hạn số packet giữ cho mỗi flow
max_techniques_per_node = top-k MITRE, ví dụ 5
```

## 4. Phương pháp 2: Affected-Set K-Hop Subgraph Builder

### Thay đổi so với ý tưởng ban đầu

Tên cũ:

```text
Affected-set 1-hop builder
```

Tên chính thức:

```text
Affected-set K-hop builder
```

Với HGT hiện tại:

```text
K = HGT num_layers = 3
```

### Mục tiêu

Không chạy inference trên graph toàn cục. Mỗi lần flow được cập nhật, chỉ tạo
subgraph cục bộ quanh flow đó.

### Logic build_khop_subgraph

```text
input:
  seed_flow_id
  K = 3

hop 0:
  seed flow

hop 1:
  packet trực tiếp của flow
  MITRE technique trực tiếp của flow

hop 2:
  MITRE technique của packet
  packet lân cận trong cùng flow
  tactic của technique

hop 3:
  bổ sung context gần seed flow theo các cạnh hợp lệ
```

### Giới hạn để graph con không nổ kích thước

Cần chặn các hướng mở rộng có tính hub:

```text
Không mặc định mở rộng:
  technique -> tất cả flow khác có cùng technique

Chỉ mở rộng nếu có budget rõ ràng:
  max_neighbor_flows_per_technique
```

Nên giữ:

```text
flow -> packet
packet -> packet gần nhau trong cùng flow
packet -> top-k technique
flow -> top-k technique
technique -> tactic
reverse edges nếu HGT train có reverse edges
```

### Output cho HGT

Nếu dùng model HGT hiện tại trong repo, không bắt buộc dùng PyG `HeteroData`.
Có thể trả về dict:

```text
node_features:
  "flow" -> Tensor[num_flows, flow_dim]
  "packet" -> Tensor[num_packets, 768]
  "technique" -> Tensor[num_techniques, 768]
  "tactic" -> Tensor[num_tactics, 1]

edge_index:
  ("flow", "contains", "packet") -> Tensor[2, E]
  ("packet", "next_packet", "packet") -> Tensor[2, E]
  ("packet", "matches_technique", "technique") -> Tensor[2, E]
  ("flow", "matches_technique", "technique") -> Tensor[2, E]
  ("technique", "belongs_to_tactic", "tactic") -> Tensor[2, E]
  reverse edge types nếu add_reverse_edges = true

edge_weight:
  semantic cosine score cho MITRE edges
```

### Lưu ý quan trọng về tactic node

Model HGT hiện tại dùng embedding index cho tactic. Vì vậy tactic local index
phải ổn định với index đã train. Cách an toàn:

```text
Luôn giữ tactic nodes theo global tactic_to_index order.
```

Nếu chỉ đưa một vài tactic node vào subgraph và đánh lại index tùy tiện, HGT có
thể gán sai tactic embedding.

## 5. Phương pháp 3: On-Demand K-Hop Hydrate Cho Slow Path

### Mục tiêu

Tách đường detection nhanh khỏi đường giải thích chậm.

Fast Path chỉ làm:

```text
embedding -> graph update -> K-hop subgraph -> HGT inference
```

Slow Path chỉ chạy khi có trigger:

```text
HGT output suspicious/malicious
-> queue.put(flow_id)
```

### Worker slow path

```text
worker_slow_path:
  while True:
    flow_id = queue.get()
    context = hydrate_context(flow_id, K_slow)
    prompt = format_xai_prompt(context)
    report = slm_inference(prompt)
    save_report(report)
    queue.task_done()
```

### Dữ liệu hydrate

Nên gồm:

```text
flow metadata:
  src_ip, dst_ip, src_port, dst_port, protocol, duration, packet_count

packet sequence:
  packet_id, timestamp, payload_len_raw, payload hex text

MITRE context:
  top-k technique id
  technique name
  tactic name
  cosine score

graph context:
  packets trong flow
  technique/tactic liên quan
  neighbor flows nếu được phép và còn trong buffer
```

### Cần có cold store

Vì Hot Graph Buffer có TTL, slow path có thể xử lý trễ và mất dữ liệu. Do đó
nên thêm cold store nhẹ:

```text
Hot buffer:
  Phục vụ HGT fast path.

Cold store:
  Lưu event/payload/meta đã evict để SLM hydrate khi cần.
```

Cold store có thể là JSONL, SQLite, Parquet hoặc log file tùy mức độ cần thiết.

## 6. Cấu hình khuyến nghị ban đầu

```yaml
runtime:
  hot_graph:
    ttl_seconds: 60
    max_events: 100000
    max_packets_per_flow: 64
    max_techniques_per_node: 5
    expand_technique_to_flows: false

  subgraph:
    hops: 3
    use_reverse_edges: true
    include_all_tactics_in_global_order: true

  slow_path:
    enabled: true
    hydrate_hops: 2
    queue_max_size: 1000
    cold_store: data/runtime/events.jsonl
```

## 7. Pseudocode tổng hợp

```text
on_packet(packet):
  flow_id = flow_tracker.update(packet)
  payload_256 = extract_payload_256(packet)
  packet_emb = student_cnn(payload_256)
  packet_mitre_topk = cosine_topk(packet_emb, mitre_embeddings, k=5)

  hot_graph.add_flow_or_update(flow_id, flow_features)
  hot_graph.add_packet(
    packet_id,
    flow_id,
    packet_emb,
    packet_mitre_topk,
    payload_hex_text
  )

  subgraph = hot_graph.build_khop_subgraph(
    seed_flow_id=flow_id,
    hops=hgt_num_layers
  )

  logits = hgt(subgraph)
  label, score = policy(logits)

  if label in suspicious_labels or score > threshold:
    slow_path_queue.put(flow_id)

  return label, score
```

## 8. Cách viết trong báo cáo

Có thể trình bày như sau:

```text
Trong môi trường online, việc tích lũy flow và packet vào một graph toàn cục sẽ
làm graph tăng kích thước vô hạn, gây tăng RAM và latency cho HGT inference.
Hệ thống đề xuất Hot Graph Buffer để duy trì graph nóng trong RAM bằng TTL và
max_events. Khi một flow được cập nhật, hệ thống không recompute toàn graph mà
xây dựng affected-set K-hop subgraph quanh flow đó, với K bằng số layer của HGT.
Với cấu hình HGT 3 layer, runtime builder lấy subgraph 3-hop có giới hạn, giữ
MITRE technique/tactic như knowledge nodes tĩnh và chặn các hướng mở rộng hub để
tránh graph con phình to. Kết quả HGT fast path chỉ dùng cho detection nhanh;
nếu flow bị đánh dấu suspicious/malicious, slow path mới hydrate context rộng
hơn và đưa vào SLM để sinh giải thích XAI.
```

## 9. Hướng đánh giá

Cần đo các chỉ số:

```text
Detection:
  accuracy, macro-F1, recall từng lớp tấn công

Runtime:
  latency trung bình mỗi packet/flow
  p95 latency
  RAM của Hot Graph Buffer
  số node/edge trung bình mỗi K-hop subgraph

Graph quality:
  top-k MITRE coverage
  số semantic edges mỗi flow
  tỷ lệ flow có context MITRE hợp lệ

XAI:
  thời gian hydrate slow path
  tỷ lệ trigger có đủ context
  manual spot-check MITRE explanation
```
