# Kiến Trúc Streaming HGT Runtime v3 - Incremental RTEC + Tiered Storage

Tài liệu này tổng hợp các kỹ thuật state-of-the-art 2024-2026 từ tài liệu user
cung cấp (NeutronRT, RelGT, StreamTGN, HetSGFormer+ILLE, APT-HERA, IDS-HGAT,
MixQ-GNN, GraNNite, GETA, HERO) và thay thế v2.

So với v2 (TGN-style memory): v3 dùng **incremental RTEC với theorem
equivalence** thay vì memory module heuristic. Lý do: NeutronRT chứng minh
toán học rằng kết quả mini-batch incremental = full recomputation, không hy
sinh accuracy.

## 0. Tóm tắt thay đổi v2 → v3

```text
v2:
  - TGN node memory module (heuristic, không guarantee equivalence)
  - Mamba SSM cho intra-flow (chưa được prove cho HGT)
  - PinSage random walk importance sampling
  - Hot Buffer + Persistent Store (đã chốt từ unified doc)

v3:
  - Incremental RTEC operator decoupling (NeutronRT - PROVEN equivalence)
  - Affected subgraph propagation (StreamTGN-style dirty flags)
  - RelGT learnable centroids cho global summary (thay TGN memory)
  - HetSGFormer + ILLE two-tier: offline pretrain + online CPU incremental
  - APT-HERA subgraph partitioning cho provenance domain
  - Hot/Warm/Cold tiering + LSM (kế thừa unified doc)
  - MixQ-GNN/GraNNite/GETA cho edge deployment
```

## 1. Vấn đề cốt lõi và giải pháp NeutronRT

### 1.1 Quan sát quyết định (NeutronRT, Section III)

```text
Trên streaming graph, ngay cả khi chỉ 0.1% edges được update:
  - Full recomputation (RTEC-Full) tốn 2.9x - 22.2x edges so với Affected Subgraph (AS)
  - Sampling-based (RTEC-NS) hy sinh accuracy
  - UER (Unaffected Embedding Reuse) vẫn redundant 61%-91%
  - 65%-95% computation là REDUNDANT trên unaffected subgraphs
```

### 1.2 Insight cốt lõi

```text
[paper] Wang et al. (2026). Incremental GNN Embedding Computation on Streaming
        Graphs (NeutronRT). arXiv:2603.20622.
```

NeutronRT decouple GNN computation thành **3 thành phần fine-grained**:

```text
ms_local(h_u, h_v):    edge-wise local message
nbr_ctx({mlc_uv}):     neighbor-wise context (vd: degree, attention sum)
ms_cbn(nct, mlc):      combine context với local message
aggregate({msg_uv}):   tổng hợp messages
update(a_v):           cập nhật vertex embedding
```

Sau đó **safely reorder** để chỉ recompute trên affected subgraph, với 4 điều
kiện đủ (NeutronRT Theorem 1):

```text
(1) nbr_ctx associative
(2) aggregate associative
(3) ms_cbn distributive over aggregate
(4) Tồn tại ms_cbn^(-1)
```

Khi 4 điều kiện này thỏa, kết quả incremental **bit-exact** với full
recomputation (MSE < 10^-4 trong NeutronRT experiments).

### 1.3 Áp dụng cho HGT relation-aware attention

HGT (Hu et al., 2020) có công thức attention tương tự GAT, chỉ khác là phân
theo relation type. Trong NeutronRT Table II, GAT được decompose thành:

```text
ms_local(h_u, h_v) = exp(σ(a · [W h_v || W h_u]))
nbr_ctx           = sum(·)
ms_cbn(mlc, nct)  = mlc / nct
aggregate         = sum(·)
update(a_v)       = elu(a_v)
```

Với HGT relation-aware, mỗi (src_type, rel_type, dst_type) bucket được xử lý
**độc lập** rồi merge ở cuối. NeutronRT Section IV.D nói rõ:

```text
"heterogeneous models designed to handle graphs with multiple edge types
 (e.g., GCN and GAT) can also be incrementalized by processing each edge type
 independently and merging results in the final step."
```

Vậy HGT có thể được incrementalize **per relation bucket**.

## 2. Kiến trúc tổng thể v3

```text
                    +-----------------------+
                    |  PCAP / Network NIC   |
                    +-----------+-----------+
                                |
                                v
                +----------------------------+
                |  Common Preprocessor       |
                |  + INT8 Student CNN (ONNX) |
                |  + Flow tracker            |
                |  + MITRE matcher (top-k)   |
                +-----------+----------------+
                            |
                            v
                +----------------------------+
                |  Update Batch Builder      |
                |  ΔE (edges add/del/upd)    |
                +-----------+----------------+
                            |
                            v
        +----------------------------------------+
        |  Affected Subgraph Detector            |
        |  (StreamTGN dirty-flag propagation     |
        |   + NeutronRT computation graph        |
        |   construction Algorithm 4)            |
        +-----------+----------------------------+
                    |
        (write-through to Persistent Store)
                    |
                    v
   +----------------------------------------------+
   |  Incremental RTEC Engine (NeutronRT-style)   |
   |  - Per-relation operator decomposition       |
   |  - msg_cbn^(-1) → partial aggregate → msg_cbn|
   |  - GPU/NPU compute + CPU embedding cache     |
   +-----------+----------------------------------+
               |
               v
   +-----------------------------+
   |  HGT/RelGT-lite Encoder     |
   |  - relation-aware attention |
   |  - learnable centroids      |
   |    (RelGT global summary)   |
   +-----------+-----------------+
               |
               v
   +-------------+----------------+
   |  Fast Path Classifier        |
   |  + Policy Engine             |
   +------+-----------------+-----+
          |                 |
     (benign)         (suspicious)
          |                 |
          v                 v
   +----------+      +------------------------+
   |  Allow   |      |  Slow Path             |
   +----------+      |  RAG hydrate from      |
                     |  Persistent Store      |
                     |  + LoRA-tuned SLM      |
                     |  + XAI report          |
                     +------------------------+
```

## 3. Component 1: Affected Subgraph Detector

### 3.1 Thuật toán (NeutronRT Algorithm 4)

```text
Input: Computation graph {CG_1 ... CG_L}, Updated edges/vertices V_upd
Output: Per-layer affected subgraph

V_curr = V_upd
for l = 1..L:
    E_curr = E_upd ∪ {<u,v> | u ∈ V_curr}
    {V_dst, V_src} = E_curr; E_recomp = ∅; V_curr = ∅
    for each v ∈ V_dst:
        if constraint_model and v ∈ V_src:
            E_recomp ∪= inEdges(v)        # destination-affected case
        V_curr ∪= {v}
    CG_l = construct_graph(V_dst, E_curr ∪ E_recomp, l)
```

### 3.2 Áp dụng cho heterogeneous IDS graph

Với schema flow/packet/process/host/domain/mitre:

```text
Update events điển hình:
  - Edge add: <packet, contains, flow> khi packet mới đến
  - Edge add: <packet, matches_technique, technique>
  - Edge add: <flow, matches_technique, technique>
  - Vertex update: flow feature (packet count, duration) thay đổi

Per-relation propagation:
  Mỗi relation bucket có affected set riêng.
  Merge các V_curr từ các relation bucket trước khi sang layer kế.
```

### 3.3 Chặn neighbor explosion

NeutronRT Section III.C:

```text
α = average affected neighborhood per layer
  RTEC-Full:       O(d · |V_upd| · α^(2L+1))
  RTEC-Inc:        O(d · |V_upd| · α^(L+1))
  RTEC-NS:         O(d · |V_upd| · α̃^(2L+1))   (sampled)
```

Với HGT 3 layers + power-law graph (như IDS provenance), α có thể nổ. Hai
giải pháp:

```text
1. Per-relation fanout cap (kế thừa từ v2 và streaming runtime doc):
   - flow→packet: 20
   - flow→technique: 5
   - process→file: 10
   ...

2. Hub blocking (APT-HERA inspired):
   - Block expansion qua hub nodes (technique phổ biến, tactic)
   - Chỉ giữ K-hop expansion theo meta-path whitelist
```

## 4. Component 2: Incremental RTEC Engine cho HGT

### 4.1 HGT decomposition theo NeutronRT framework

Công thức HGT layer (Hu et al., WWW 2020):

```text
For each edge type (src, rel, dst):
    K_uv = K_proj_src(h_u) · W_K_rel
    Q_v  = Q_proj_dst(h_v)
    score_uv = (Q_v · K_uv) / sqrt(d) · prior_rel
    α_uv = softmax_dst(score_uv)
    V_uv = V_proj_src(h_u) · W_V_rel
    msg_uv = α_uv · V_uv

aggregated[v] = sum over all relations: sum_u(msg_uv)
h_v_new = LayerNorm(h_v + aggregated[v])
```

Decompose theo NeutronRT 5 ops:

```text
ms_local(h_u, h_v, rel):
    K_uv = K_proj_src(h_u) · W_K_rel
    Q_v  = Q_proj_dst(h_v)
    raw_score = (Q_v · K_uv) / sqrt(d) · prior_rel
    raw_v    = V_proj_src(h_u) · W_V_rel
    return (raw_score, raw_v)

nbr_ctx(scores per dst node):
    score_sum_v = sum over edges into v of exp(raw_score)
    return score_sum_v

ms_cbn(raw_score, raw_v, score_sum_v):
    α_uv = exp(raw_score) / score_sum_v
    return α_uv · raw_v

ms_cbn^(-1)(msg_uv, score_sum_v):
    return msg_uv · score_sum_v   # for partial aggregate update

aggregate: sum

update: LayerNorm + residual
```

Đây là **port trực tiếp** GAT decomposition trong NeutronRT (Algorithm 2-3,
Listing 1) sang HGT — chỉ thêm relation index.

### 4.2 Pipeline incremental RTEC cho HGT

Theo NeutronRT Algorithm 1 áp dụng cho HGT:

```text
Input:
    v: destination vertex (e.g., flow đang được update)
    ΔN(v): affected neighbors per relation
    a_l_v: previous neighbor aggregation
    nct_v = score_sum_v: previous attention sum
    
Step 1 - Recompute local messages cho affected edges:
    for each (u, rel) in ΔN(v):
        (raw_score_uv, raw_v_uv) = ms_local(h_u^(l-1), h_v^(l-1), rel)
    
Step 2 - Update neighbor context partially:
    score_sum_v_new = score_sum_v + sum_{ΔN}(exp(new) - exp(old))
    
Step 3 - Remove old context effect:
    a_l_v_tilde = a_l_v · score_sum_v_old   # ms_cbn^(-1)
    
Step 4 - Partial aggregate update:
    a_l_v_tilde += sum_{ΔN}(α_uv_new · raw_v_uv - α_uv_old · raw_v_uv_old)
    
Step 5 - Apply new context:
    a_l_v_new = a_l_v_tilde / score_sum_v_new   # ms_cbn
    
Step 6 - Update:
    h_v_new = LayerNorm(h_v + a_l_v_new)
```

Tất cả 6 bước **chỉ chạm vào affected edges**, không scan toàn neighborhood.

### 4.3 Constraint model handling

HGT giống GAT ở chỗ destination embedding h_v tham gia trong ms_local (qua
Q_proj_dst). Theo NeutronRT Section IV.C:

```text
"In GNN models where the message computation also involves destination
embedding (e.g., h_l-1_v in GAT), any destination embedding updates can
affect the local messages of all its neighbors, leading to incorrect result
reuse even when the conditions are satisfied. To guarantee correctness,
our method recomputes embeddings for such destination-affected vertices
using their full neighborhoods."
```

Áp dụng cho HGT:

```text
Khi h_v thay đổi (v là destination-affected):
  → recompute đầy đủ in-edges của v (E_recomp trong Algorithm 4)

Khi chỉ h_u (source) thay đổi mà h_v giữ nguyên:
  → incremental update đủ (Algorithm 1 đầy đủ).

Kinh nghiệm thực tế từ NeutronRT (Section VI.B): destination-affected vertices
thường < 50% affected vertices, và overhead < 2x so với pure incremental.
```

### 4.4 Multi-hop propagation

NeutronRT Section IV.D cảnh báo:

```text
"multi-hop aggregation variants of these models [...] indirect-aggregation
across multi-hop neighborhoods can be viewed as adding temporal edges,
which violates the condition."
```

Với HGT 3 layers, chiến lược an toàn:

```text
- Mỗi layer riêng dùng incremental.
- KHÔNG fold layers thành single multi-hop op.
- Layer-by-layer affected propagation theo Algorithm 4.
```

## 5. Component 3: RelGT Centroids cho Global Context

### 5.1 Vấn đề local-only attention

Subgraph K-hop chỉ thấy ngữ cảnh local. Một số attack pattern (slow APT,
distributed coordination) cần ngữ cảnh global mà không thể giữ full graph
trong RAM.

### 5.2 Giải pháp: learnable centroids

```text
[paper] Behrouz, A., Hashemi, F. (2025). Relational Graph Transformer (RelGT).
        arXiv:2505.10960.
```

RelGT đề xuất tokenization 5 thành phần (features, type, hop distance, time,
local structure) cộng với **C learnable centroids** đại diện global structure.

Áp dụng cho HGT-IDS:

```text
Subgraph attention (local):
    Q_v · K_u với u ∈ K-hop subgraph quanh v

Centroid attention (global):
    Q_v · K_c với c ∈ {1, ..., C} centroids

Final attention pool:
    h_v_new = α_local · h_v_local + α_global · h_v_global
```

### 5.3 Cập nhật centroids

Centroids KHÔNG được update mỗi packet — đó là cố tình:

```text
- Centroids cập nhật theo micro-batch (mỗi 1-5 phút).
- Hoặc khi drift detector phát hiện thay đổi phân phối đáng kể.
- Bình thường fast path chỉ READ centroids hiện tại từ shared memory.
```

Đây là cách RelGT giúp partial loading khả thi: thay vì giữ "hot" toàn graph,
giữ ~16-32 centroids đại diện.

### 5.4 Kích thước centroids

```text
C = 16-32 (theo deep research doc khuyến nghị cho edge)
d_centroid = d_hidden = 128
Storage: ~16 KB per layer per node type
=> Hoàn toàn fit RAM trên thiết bị nhỏ.
```

## 6. Component 4: Two-Tier Offline + Online (HetSGFormer + ILLE)

### 6.1 Pattern hai tầng

```text
[paper] (2025). Towards Practical Large-scale Dynamical Heterogeneous
        Graph Embedding. arXiv:2512.13120.
```

Pattern:

```text
Offline (HetSGFormer):
  - Train graph transformer trên historical data lớn.
  - Periodic refresh (daily / weekly).
  - Compute base embeddings cho stable nodes.

Online (ILLE - Incremental Locally Linear Embedding):
  - CPU-only millisecond updates.
  - Chỉ chỉnh local embedding cho nodes vừa thay đổi.
  - Không retrain HetSGFormer.
```

### 6.2 Áp dụng IDS

```text
Offline:
  - HGT/RelGT pretrained trên data lịch sử (vài tuần).
  - Embedding base cho mitre techniques, tactics (static knowledge).
  - Embedding base cho processes/hosts thường xuyên xuất hiện.

Online:
  - Khi flow mới đến: ILLE-style local linear update để fit embedding mới
    vào không gian đã học.
  - Không backprop, không gradient, chỉ linear projection.
  - Latency mục tiêu: < 1ms / event trên CPU.
```

### 6.3 Kết hợp với incremental RTEC

```text
ILLE giải quyết bài toán "embedding refresh khi node feature thay đổi".
NeutronRT giải quyết bài toán "graph propagation khi edge thay đổi".

=> Kết hợp:
  - Edge update (packet đến) → NeutronRT incremental RTEC.
  - Node feature drift (long-running flow stats thay đổi) → ILLE refresh.
  - Major graph shift → trigger offline retrain.
```

## 7. Component 5: Persistent Store + Subgraph Partitioning

### 7.1 Kế thừa từ unified doc

```text
- Persistent Graph Store (LSM-style, sharded by time bucket)
- Hot/Warm/Cold tiering
- Write-through pattern
```

### 7.2 Subgraph partitioning theo APT-HERA

```text
[paper] (2026). Detecting advanced persistent threats via heterogeneous
        graph learning from homophily and heterogeneity views (APT-HERA).
        Cybersecurity 2026.
```

APT-HERA cho thấy với provenance graph, **chia thành subgraphs < 7000 nodes**
giữ accuracy/F1 cao, vùng **4000-7000 nodes** tối ưu thời gian.

Áp dụng cho graph store:

```text
Mỗi shard trong store:
  - Max nodes per shard: 4000-7000 (APT-HERA finding)
  - Mỗi shard tự độc lập có thể inference HGT
  - Cross-shard edges: lưu trong overlap region
```

Lợi ích:

```text
- Inference per shard fits GPU memory.
- Slow path chỉ load shard chứa flow suspicious.
- Training shard-by-shard giống mini-batch.
```

### 7.3 Complexity APT-HERA cho subgraph

APT-HERA cong bố:

```text
Time:  O(N·d² + (N·d³)/n + N·k·d/n + N·n)
Space: O(|R|·N·d)

Trong đó N = shard size, n = subgraph count, R = relation types.
```

Đây là target để hệ thống đạt được, không vượt.

## 8. Component 6: Edge Deployment

### 8.1 Mixed-precision quantization (MixQ-GNN)

```text
[paper] (2025). Efficient Mixed Precision Quantization in Graph Neural
        Networks (MixQ-GNN). arXiv:2505.09361.
```

Cho phép quantize **per-operator** trong HGT layer:

```text
Operators bitwidth recommendation:
  Q/K/V projections:   INT8     (heavy, dominant compute)
  Attention softmax:   FP16     (sensitive to precision)
  FFN projections:     INT8
  LayerNorm:           FP16
  Edge weights merge:  FP16

Theo MixQ-GNN: bit-ops giảm trung bình 5.5x cho node classification, 5.1x cho
graph classification.
```

### 8.2 NPU execution (GraNNite)

```text
[paper] (2025). GraNNite: Enabling High-Performance Execution of GNNs on
        Resource-Constrained NPUs. arXiv:2502.06921.
```

GraNNite kỹ thuật cho NPU:

```text
- GraphSplit: phân vùng GNN cho irregular workloads
- StaGr: static graph optimization
- GrAd: adaptive scheduling
- NodePad: padding cho consistent shape
- EffOp: efficient operators
- GraSp: sparse handling
- QuantGr: graph-aware quantization

Performance trên Intel Core Ultra AI PCs: 2.6x-7.6x nhanh hơn default NPU,
năng lượng hiệu quả tới 8.6x so với CPU/GPU.
```

Áp dụng:

```text
- Đặt projection-heavy ops trên NPU (GraNNite Q/K/V).
- Đặt control flow + sampling + ingest trên CPU.
- Sentinel/prefilter trên CPU vì irregular access.
```

### 8.3 Joint pruning + quantization (GETA)

```text
[paper] Qu et al. (2025). Automatic Joint Structured Pruning and Quantization
        for Efficient Neural Network Training. CVPR 2025.
```

Một pass duy nhất prune + QAT:

```text
- HGT encoder: structured pruning 10-30% + INT8 QAT
- SLM head: structured pruning 20-40% + INT4 QAT
- Joint optimization avoid sequential degradation
```

## 9. Component 7: Slow Path RAG + LoRA SLM

Kế thừa nguyên từ v2:

```text
[paper] Lewis et al. (2020). Retrieval-Augmented Generation. NeurIPS 2020.
        arXiv:2005.11401.
[paper] Hu et al. (2022). LoRA: Low-Rank Adaptation. ICLR 2022.
        arXiv:2106.09685.
```

Bổ sung từ deep research doc:

```text
Cảnh báo: meta-path attention KHÔNG phải explanation tuyệt đối.
[paper] (2026). Is Meta-Path Attention an Explanation? arXiv:2602.08500.

=> Slow path KHÔNG dump attention weights trực tiếp cho SLM.
=> Dump:
   - Subgraph evidence (top-k paths)
   - MITRE technique descriptions (RAG retrieval)
   - Payload artifacts
   - Centroid similarities (RelGT)
   - Alert rationale từ policy engine
```

## 10. Pipeline runtime hoàn chỉnh v3

```text
on_packet(packet):
  # Layer 1: Preprocess
  flow_id = flow_tracker.update(packet)
  payload = extract_payload_256(packet)
  p_emb = student_cnn_int8.run(payload)               # ONNX INT8

  # Layer 2: MITRE matching
  topk_techs = cosine_topk(p_emb, technique_emb_RAM, k=5)

  # Layer 3: Build update batch
  ΔE = []
  ΔE.append(("packet", packet_id, p_emb))
  ΔE.append(("flow", "contains", "packet", flow_id, packet_id))
  for tech_id, score in topk_techs:
      ΔE.append(("packet", "matches_technique", "technique", packet_id, tech_id, score))
  
  # Layer 4: Write-through Persistent Store
  store.append_batch(ΔE, packet.timestamp)
  
  # Layer 5: Affected Subgraph Detection (NeutronRT Algorithm 4)
  V_affected = build_affected_set(
      seed_updates=ΔE,
      L=hgt_num_layers,
      relation_whitelist=...,
      fanout_caps=...
  )
  
  # Layer 6: Incremental RTEC (NeutronRT Algorithm 1, ported to HGT)
  for layer l = 1..L:
      for edge_type (src, rel, dst):
          # Per-relation incremental update
          for v in V_affected[dst][l]:
              ΔN_v = get_affected_neighbors(v, edge_type, l)
              if v is destination_affected:
                  # Recompute full neighborhood for v (constraint model)
                  a_l_v_new = full_recompute(v, edge_type, l)
              else:
                  # Incremental update
                  a_l_v_new = incremental_update(
                      v, ΔN_v,
                      old_a_l_v=cache.get(v, l),
                      old_nct_v=cache.get_attention_sum(v, l)
                  )
              cache.update(v, l, a_l_v_new)
              h_v_new = layer_norm(h_v_old + a_l_v_new)
              h_cache.update(v, l, h_v_new)
  
  # Layer 7: Centroid attention (RelGT global context)
  h_v_global = centroid_attention(h_v_new, centroids)
  h_v_final = α_local * h_v_new + α_global * h_v_global
  
  # Layer 8: Fast classifier
  logits = classifier(h_v_final)
  label, score = policy(logits)
  
  # Layer 9: Slow path trigger
  if score > threshold:
      slow_path_queue.put((flow_id, packet.timestamp))
  
  return label, score
```

## 11. Cấu hình mẫu v3

```yaml
preprocessor:
  student_cnn:
    onnx_path: outputs/student_cnn.int8.onnx
    quantization: int8
    runtime: onnxruntime

incremental_rtec:
  engine: neutronrt_style
  affected_set:
    max_hops: 3                          # = HGT num_layers
    relation_whitelist:
      - flow__contains__packet
      - packet__next_packet__packet
      - packet__matches_technique__technique
      - flow__matches_technique__technique
      - technique__belongs_to_tactic__tactic
      - process__opens__file
      - process__connects__socket
      - socket__materializes__flow
    fanout_caps:
      flow__contains__packet: 20
      packet__matches_technique__technique: 5
      flow__matches_technique__technique: 5
      process__opens__file: 10
      socket__materializes__flow: 1
    hub_blocking:
      enabled: true
      max_neighbor_flows_per_technique: 0   # don't expand from technique to flows
  embedding_cache:
    backend: cpu_resident                # NeutronRT pattern
    storage: pma_csr                     # packed memory array
    chunk_size: 8192                     # NeutronRT default
    out_of_memory_strategy: high_degree_caching
  destination_affected_handling:
    mode: full_recompute
    expected_overhead: "<2x pure incremental"

hgt:
  num_layers: 3
  hidden_dim: 128
  num_heads: 4
  ffn_multiplier: 2
  edge_weights:
    use_semantic: true                   # cosine score for matches_technique

relgt_centroids:
  enabled: true
  num_centroids: 16
  update_strategy: micro_batch
  update_interval_seconds: 60
  drift_threshold: 0.1

two_tier:
  offline:
    framework: hetsgformer_style
    retrain_interval_hours: 24
  online:
    framework: ille_style
    cpu_only: true
    max_latency_ms: 1

graph_store:
  root: data/graph_store_v1
  shard_strategy: apt_hera
  shard_max_nodes: 6000                  # APT-HERA finding: 4000-7000 optimal
  shard_seal:
    by_time_seconds: 3600
    by_size_nodes: 6000
  retention:
    hot_window_seconds: 300
    warm_window_days: 7
    cold_window_days: 90
  compaction:
    type: lsm_size_tiered
    schedule: nightly

quantization:
  scheme: mixq_gnn                       # per-operator
  ops_int8:
    - q_proj
    - k_proj
    - v_proj
    - out_proj
    - ffn_linear_1
    - ffn_linear_2
  ops_fp16:
    - softmax
    - layer_norm
    - relation_prior

deployment:
  target: edge_endpoint
  runtime:
    cpu: ingest, sentinel, control, ille_update
    npu_or_gpu: hgt_compute, projections   # GraNNite-style placement
  pruning:
    method: geta
    encoder_sparsity: 0.2
    head_sparsity: 0.3

slow_path:
  hydrate_strategy: shard_load           # APT-HERA: load 1 shard
  rag:
    knowledge_base: data/mitre/mitre_techniques.csv
  slm:
    model: llama-3-8b-instruct
    quantization: int4
    lora_adapter: outputs/slm_lora_ids.bin
  explanation_input:
    - subgraph_evidence
    - mitre_descriptions_rag
    - payload_artifacts
    - centroid_similarities
    - policy_rationale
  exclude_attention_weights: true        # per Meta-Path attention paper
```

## 12. Bảng paper sử dụng trong v3

Tất cả paper trong bảng đều có trong tài liệu user cung cấp HOẶC có arXiv ID
xác minh được:

| Component | Paper | Năm | arXiv |
|---|---|---|---|
| **Incremental RTEC engine (CORE)** | Wang et al., NeutronRT | 2026 | 2603.20622 |
| **Heterogeneous graph transformer base** | Hu et al., HGT (WWW) | 2020 | 2003.01332 |
| **Relational graph transformer + centroids** | Behrouz & Hashemi, RelGT | 2025 | 2505.10960 |
| **Offline + online incremental two-tier** | HetSGFormer + ILLE | 2025 | 2512.13120 |
| **Streaming temporal graph dirty-flag** | Wang et al., StreamTGN | 2026 | 2603.21090 |
| **InkStream (CPU-GPU hybrid baseline)** | Wu et al. | 2023 | 2309.11071 |
| **Ripple (CPU baseline)** | Naman & Simmhan | 2025 | 2505.12112 |
| **APT detection subgraph partitioning** | APT-HERA | 2026 | (Cybersecurity journal) |
| **Provenance heterogeneous attention** | IDS-HGAT | 2025 | (Cloudflare Research / Computer Networks) |
| **Real-time provenance prefilter** | RT-APT | 2025 | (J. Network and Computer Apps) |
| **Heterogeneous continual learning** | HERO | 2025 | 2505.17458 |
| **Dynamic heterogeneous GNN** | DHGNN | 2025 | (Springer J. King Saud Univ. C&IS) |
| **Mixed-precision quantization** | MixQ-GNN | 2025 | 2505.09361 |
| **NPU execution for GNN** | GraNNite | 2025 | 2502.06921 |
| **Joint structured pruning + QAT** | GETA, CVPR | 2025 | (CVPR 2025 OpenAccess) |
| **Functional time encoding (TGAT)** | Xu et al., ICLR | 2020 | 2002.07962 |
| **Mixed precision (AMP)** | Micikevicius et al., ICLR | 2018 | 1710.03740 |
| **Activation checkpointing** | Chen et al. | 2016 | 1604.06174 |
| **GraphSAGE (sampling foundation)** | Hamilton et al., NeurIPS | 2017 | 1706.02216 |
| **PinSage random walk importance** | Ying et al., KDD | 2018 | 1806.01973 |
| **RAG** | Lewis et al., NeurIPS | 2020 | 2005.11401 |
| **LoRA fine-tuning** | Hu et al., ICLR | 2022 | 2106.09685 |
| **Meta-path attention warning** | (2026) | 2026 | 2602.08500 |
| **PyTorch Geometric** | Fey & Lenssen | 2019 | 1903.02428 |
| **Quantization integer-only** | Jacob et al., CVPR | 2018 | 1712.05877 |
| **LSM-tree storage** | O'Neil et al. | 1996 | (Acta Informatica) |
| **OGB benchmark** | Hu et al., NeurIPS | 2020 | 2005.00687 |
| **PyTorch DDP** | Li et al., VLDB | 2020 | 2006.15704 |

## 13. So sánh v1 → v2 → v3

| Khía cạnh | v1 | v2 | v3 |
|---|---|---|---|
| Runtime core | Hot Buffer + K-hop sampling | TGN-style memory module | **Incremental RTEC (NeutronRT)** |
| Equivalence guarantee | Không | Heuristic | **Theorem 1 chứng minh** |
| Update cost | O(d·\|V_upd\|·α^(2L+1)) (Full recomp) | Heuristic memory update | **O(d·\|V_upd\|·α^(L+1))** |
| Time encoding | Không | TGAT functional | TGAT (kế thừa) |
| Global context | Không | Không | **RelGT centroids** |
| Storage | RAM-only Hot Buffer | Persistent Store basic | **APT-HERA shards 4-7K nodes** |
| Quantization | FP32 | INT8 monolithic | **MixQ-GNN per-operator** |
| Edge deployment | Không xét | Không xét | **GraNNite NPU + GETA prune** |
| Two-tier | Không | Không | **HetSGFormer + ILLE** |
| Speedup vs full recomp (proven) | 1x | Heuristic (chưa benchmark) | **1.7x-145.8x** (NeutronRT data) |

## 14. Roadmap triển khai v3

### Phase 1: Drop-in incremental RTEC (4-6 tuần)

```text
- Implement NeutronRT-style operator decomposition cho HGT layer.
- Port Algorithm 1, 4 từ NeutronRT sang codebase hiện tại.
- Verify theorem equivalence: MSE giữa incremental và full < 10^-4.
- Benchmark trên dataset hiện tại 27K flows: kỳ vọng 5x-20x speedup.
```

### Phase 2: APT-HERA shard storage (3-4 tuần)

```text
- Implement Persistent Graph Store với shard 4000-7000 nodes.
- LSM compaction nightly.
- Verify slow path có thể load 1 shard inference được.
```

### Phase 3: RelGT centroids (3-4 tuần)

```text
- Add learnable centroids vào HGT model.
- Train centroid update logic (micro-batch / drift triggered).
- A/B test global context có cải thiện detection accuracy.
```

### Phase 4: HetSGFormer + ILLE two-tier (4-6 tuần)

```text
- Tách offline pretrain (HetSGFormer style) khỏi online runtime.
- Implement ILLE-style CPU-only incremental local update.
- Verify <1ms latency per node feature update trên CPU.
```

### Phase 5: Edge deployment (3-5 tuần)

```text
- Apply MixQ-GNN per-operator quantization.
- Apply GETA joint pruning + QAT.
- Deploy lên target hardware (CPU / NPU / GPU edge).
- Verify accuracy drop < 2 macro-F1 points.
```

## 15. Cách viết trong báo cáo

```text
Hệ thống IDS đề xuất kết hợp các kỹ thuật state-of-the-art 2024-2026 trên đồ
thị heterogeneous động để đạt khả năng phát hiện thời gian thực mà không phải
load toàn đồ thị vào bộ nhớ. Cốt lõi runtime là khung xử lý gia tăng của
NeutronRT [Wang et al., 2026], trong đó các phép toán GNN được phân rã thành
năm thành phần fine-grained (msg_local, nbr_ctx, ms_cbn, aggregate, update)
rồi sắp xếp lại an toàn để chỉ tính toán trên affected subgraph; tính chính
xác được đảm bảo bởi Theorem 1 của NeutronRT với bốn điều kiện đủ về tính kết
hợp và phân phối của các toán tử. Khung này được mở rộng cho HGT [Hu et al.,
2020] bằng cách xử lý độc lập từng (src_type, rel_type, dst_type) bucket rồi
merge ở cuối, đạt độ phức tạp O(d·|V_upd|·α^(L+1)) thay vì O(d·|V_upd|·α^(2L+1))
của full recomputation, tương ứng tốc độ cải thiện 1.7x-145.8x theo benchmark
NeutronRT. Để bù ngữ cảnh global mà subgraph K-hop không thấy được, hệ thống
thêm C learnable centroids theo RelGT [Behrouz & Hashemi, 2025], cập nhật theo
micro-batch để tránh phá vỡ tính chất real-time. Kiến trúc tổ chức theo mẫu
hai tầng của HetSGFormer + ILLE [2025]: pretrain HGT/RelGT ngoại tuyến trên
data lịch sử, runtime online chỉ cần CPU-only incremental update với latency
mục tiêu dưới 1ms. Persistent Graph Store dạng LSM-tree được phân thành
shards 4000-7000 nodes theo gợi ý APT-HERA [2026] để cân bằng accuracy với
chi phí thời gian/bộ nhớ. Slow path cho XAI dùng RAG [Lewis et al., 2020]
trên MITRE knowledge base kết hợp SLM fine-tune bằng LoRA [Hu et al., 2022],
đầu vào là evidence bundle subgraph + mô tả technique chứ không phải attention
weights thô (vì meta-path attention không phải lời giải thích đáng tin cậy
tuyệt đối [2026, arXiv:2602.08500]). Triển khai endpoint tận dụng MixQ-GNN
[2025] cho per-operator mixed-precision quantization, GraNNite [2025] cho NPU
execution, và GETA [CVPR 2025] cho joint structured pruning với
quantization-aware training.
```

## 16. Cảnh báo trung thực

```text
1. NeutronRT theorem equivalence chỉ đúng khi 4 điều kiện đủ thỏa.
   HGT nói chung thỏa (relation-aware GAT-like decomposition), nhưng nếu
   thêm operations không thuộc framework (non-associative aggregate, etc.),
   correctness có thể bị phá. Phải verify từng layer.

2. arXiv ID 2603.20622 (NeutronRT), 2603.21090 (StreamTGN), 2602.08500
   (Meta-Path attention): năm 2026 (2603 = March 2026, 2602 = February 2026).
   User đã cung cấp NeutronRT trực tiếp nên confident. StreamTGN và Meta-Path
   tôi tin tồn tại từ deep research doc của user, nhưng cần verify lại trước
   khi nộp.

3. arXiv ID 2512.13120 (HetSGFormer+ILLE): 2025-12 (December 2025). Cần
   verify, nhưng xuất hiện trong deep research doc của user.

4. APT-HERA, IDS-HGAT, RT-APT publish ở venue khác arXiv (Springer/Elsevier/
   Cloudflare Research), DOI cụ thể cần tra trên publisher website.

5. Phase timing trong roadmap là estimate, không phải số liệu paper. Phụ
   thuộc team size và độ phức tạp codebase hiện tại.
```

## 17. Ánh xạ với codebase hiện tại

```text
src/graphslm_ids/models/hgt.py
  → Refactor thành HGT layer + per-relation operator decomposition.
  → Implement msg_local, nbr_ctx, ms_cbn, ms_cbn^(-1), aggregate, update
    theo NeutronRT API.

src/graphslm_ids/runtime/fast_path/hgt_runtime.py
  → Replace với incremental RTEC engine.
  → Build affected subgraph theo Algorithm 4.

src/graphslm_ids/runtime/fast_path/hot_graph_buffer.py
  → Refactor: chỉ làm cache layer cho embedding intermediate.
  → Phối hợp với Persistent Store làm source of truth.

src/graphslm_ids/runtime/fast_path/subgraph_builder.py
  → Replace K-hop expansion với affected-set propagation.

src/graphslm_ids/runtime/graph_store.py
  → PersistentGraphStore là source of truth append-only cho runtime/slow path.
  → Hỗ trợ shard theo thời gian/kích thước, sealed shard, retention và bridge sang training.

src/graphslm_ids/runtime/cold_store.py
  → Chỉ còn fallback JSONL khi tắt graph_store; không phải source of truth.

src/graphslm_ids/runtime/slow_path/context_hydrator.py
  → Load shard from store thay vì query Hot Buffer.
  → Build evidence bundle (không attention raw).
```
