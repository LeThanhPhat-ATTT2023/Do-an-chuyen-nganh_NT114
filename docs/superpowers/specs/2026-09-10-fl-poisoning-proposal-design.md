# Chỉnh đề cương theo hướng Federated Learning và poisoning attack

## Mục tiêu

Hiệu chỉnh đề cương hiện có để phản ánh đúng định hướng của giảng viên: xây dựng FL-based IDS trên nền HGT, mô phỏng nhiều client bằng một bộ dữ liệu thực nghiệm được phân vùng, dùng sampling cục bộ tại client, và lấy model poisoning attack cùng Byzantine-robust aggregation làm trọng tâm nghiên cứu.

## Quyết định nội dung

- Giữ nguyên tên tiếng Việt hiện tại.
- Để trống trường tên tiếng Anh.
- Giữ nền tảng GraphSLM IDS gồm heterogeneous graph, MSEE, HGT, EACS, MITRE và XAI; XAI là lớp hỗ trợ giải thích, không phải câu hỏi nghiên cứu chính.
- Diễn đạt rõ: thực tế mỗi client đại diện cho một gateway/tổ chức có dữ liệu riêng; trong thực nghiệm, một bộ dữ liệu chuẩn hóa được xem như dữ liệu tổng hợp từ nhiều miền và chia thành các client mô phỏng. PCAP/dữ liệu thô không truyền về server.
- Sampling được đặt ở client: neighbor/subgraph sampling theo mini-batch để giới hạn bộ nhớ và chi phí huấn luyện HGT. Sampling không thay thế bước phân vùng dữ liệu client và không tạo ra trao đổi dữ liệu thô.
- Mỗi round FL gồm: server phát global model; client nhận model, lấy subgraph cục bộ và train local; client gửi model delta `Δw_k = w_k - w_t`; server đo tín hiệu đáng ngờ chỉ từ delta/update, mô phỏng hoặc chặn poisoning; server aggregate robust; server phát global model mới.
- Threat model gồm client Byzantine thực hiện sign-flip, ALIE, Min-Max/Min-Sum và ScaleSign; đánh giá dưới IID và non-IID, nhiều tỷ lệ client độc hại.
- Baseline gồm centralized HGT, local-only, FedAvg và một tập robust aggregators phù hợp. Cơ chế đa tín hiệu chỉ dùng đặc trưng update/model, là phương án đề xuất, không khẳng định an toàn tuyệt đối.
- Threat model mặc định mô phỏng 5-10 client, một hoặc một số client Byzantine, attacker biết global model và giao thức update; collusion chỉ là biến mở rộng. Untargeted poisoning là cấu hình chính; targeted misclassification/backdoor chỉ dùng khi có kịch bản và metric riêng.
- Giữ đánh giá random/temporal split, noisy/clean labels, macro-F1, attack success rate, recall, false-positive rate, communication cost, training time và explanation stability.

## Phạm vi sửa đề cương

1. Sửa phần **Nội dung đề tài** để nối trực tiếp bài toán data silo, FL, local sampling và poisoning.
2. Sửa **Mục tiêu** để poisoning/robust aggregation đứng trước XAI.
3. Sửa **Đối tượng và phạm vi** để phân biệt mô phỏng client với triển khai FL xuyên tổ chức thật.
4. Bổ sung sampling vào **Nội dung 2** và luồng server-client vào phương pháp.
5. Làm rõ **Nội dung 3** là đánh giá tấn công; **Nội dung 4** là phòng thủ/tổng hợp.
6. Cập nhật **Nội dung 6** và kế hoạch thực hiện theo các round FL, attack evaluation và defense ablation.
7. Thêm trang **Sơ đồ kiến trúc FL-based IDS đề xuất** sau phần tài liệu tham khảo.

## Kiến trúc sơ đồ

Sơ đồ có sáu vùng: (1) dữ liệu/miền quản trị mô phỏng, (2) client FL với graph construction và local sampling, (3) FL coordinator, (4) poisoning attack injection và robust aggregation, (5) HGT inference/XAI runtime, (6) đánh giá và feedback. Mũi tên model delta đi từ client lên coordinator; global model đi ngược xuống client. Dữ liệu thô, graph và evidence chỉ nằm trong client.

Sampling được mô tả bằng seed flow, neighbor sampling theo từng relation, số hop và fanout; cấu hình khởi đầu dự kiến là 2-hop với fanout 5-10 node cho mỗi relation, sau đó chọn trên validation. Sampler không vượt qua biên client và được đo bằng memory, sampling time, training time và macro-F1.

## Đầu ra

- DOCX chỉnh sửa, giữ tên tiếng Anh trống.
- PDF xuất từ DOCX.
- Sơ đồ PNG và SVG; kèm Mermaid source để dễ chỉnh sửa.
- Canvas preview hiển thị luồng kiến trúc và các điểm kiểm soát poisoning.

## Kiểm tra nghiệm thu

- Đọc lại toàn bộ nội dung sau chỉnh sửa: không còn câu diễn đạt rằng các client gửi dữ liệu thô hoặc server gom PCAP.
- Kiểm tra các thuật ngữ FL, sampling, poisoning, Byzantine-robust aggregation nhất quán.
- Render mọi trang DOCX/PDF thành ảnh và kiểm tra không clipping, overlap, font lỗi, bảng lỗi hoặc sơ đồ quá nhỏ.
- Mở lại file DOCX/PDF sau khi xuất để xác nhận file không hỏng; kiểm tra sơ đồ đủ nhãn và mũi tên.
