# Ghi chú sơ đồ kiến trúc FL based IDS

## Luồng chính

1. Chia train/validation/test trước; chỉ partition phần train thành 5-10 client mô phỏng.
2. Mỗi client giữ raw PCAP/flow, tiền xử lý, MSEE/MITRE/EACS, graph cục bộ và local labels.
3. Local neighbor/subgraph sampling chọn seed flow, 2-hop neighbors và fanout 5-10 theo relation; sampler không vượt biên client.
4. Client train HGT rồi gửi model delta Δw_k = w_k - W_t. Client Byzantine chèn poisoning vào delta trong thí nghiệm.
5. Coordinator chỉ xử lý model delta: kiểm tra schema/round, norm/clipping, cosine, sign, geometric distance, spectral outlier và history; không dùng raw graph/evidence/MITRE.
6. Update được accept hoặc quarantine; robust aggregation tạo global model W_(t+1), sau đó broadcast về client.
7. Evaluation nhận metrics only; runtime dùng approved checkpoint tại gateway/client để inference và tạo XAI cục bộ.

## Điểm cần nhấn mạnh trong luận văn

- Đây là mô phỏng cross-silo FL bằng một benchmark dataset được phân vùng; không tuyên bố dữ liệu thật đã được thu thập từ nhiều tổ chức.
- Mũi tên raw PCAP/graph/features/labels/evidence không đi vào coordinator.
- Poisoning hook là experiment-only, dùng để tạo client Byzantine và đo attack impact.
- ASR chỉ dùng cho targeted misclassification/backdoor; untargeted attack dùng macro-F1 drop, recall drop, loss increase và attack impact.
- Secure aggregation và per-client screening là hai giả định khác nhau; secure aggregation chỉ để hướng mở rộng/ablation.
- Robust aggregation là cơ chế được đánh giá trên các attack đã chọn, không phải bảo đảm chống mọi poisoning attack.

