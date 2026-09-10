# FL Poisoning Proposal Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chỉnh đề cương tiếng Việt theo hướng FL-based IDS với local sampling và model poisoning là trọng tâm, đồng thời tạo bộ sơ đồ kiến trúc có thể chỉnh sửa và chèn vào hồ sơ.

**Architecture:** Giữ cấu trúc đề cương hiện có, sửa có trọng tâm các đoạn về FL, client simulation, sampling, threat model, defense và evaluation. Sơ đồ mới mô tả vòng lặp server-client: raw data/graph ở client, local graph sampling, local HGT training, malicious update injection, server-side signal checks, robust aggregation, global model distribution, inference và XAI.

**Tech Stack:** OOXML/DOCX, PyMuPDF, Pillow, SVG, Mermaid, Canvas SDK (`cursor/canvas`), PowerShell, Python runtime `D:\v\nt114\Scripts\python.exe`.

**Spec:** `docs/superpowers/specs/2026-09-10-fl-poisoning-proposal-design.md`

## Global Constraints

- Giữ nguyên tên tiếng Việt hiện tại.
- Để trống trường tên tiếng Anh.
- Không diễn đạt client gửi PCAP, graph, feature, nhãn hoặc evidence về server; client chỉ gửi model delta `Δw_k = w_k - w_t` và metadata/metrics tối thiểu của giao thức.
- Một bộ dữ liệu thực nghiệm được dùng như dữ liệu tổng hợp từ nhiều miền và được chia thành client mô phỏng.
- Sampling chỉ diễn ra cục bộ ở client dưới dạng seed-flow neighbor/subgraph mini-batch 2-hop, fanout 5-10 theo relation; không vượt qua biên client.
- Poisoning gồm sign-flip, ALIE, Min-Max/Min-Sum và ScaleSign.
- Đánh giá IID/non-IID, nhiều tỷ lệ client độc hại, random/temporal split và noisy/clean labels.
- Không sửa các thay đổi mã nguồn không liên quan trong repository.
- Mọi DOCX/PDF cuối phải được render thành ảnh và kiểm tra trực quan trước bàn giao.

---

### Task 1: Chuẩn bị nội dung chỉnh sửa

**Files:**
- Read: `DeCuongKLTN_NCP.docx`
- Read: `DeCuongKLTN_NCP.pdf`
- Read: `docs/architecture/system_architecture_academic.md`
- Create: `outputs/fl_proposal_revision_content.txt`

**Interfaces:**
- Consumes: Văn bản 5 trang hiện có và kiến trúc GraphSLM v3_ob.
- Produces: Bản nội dung đã hiệu chỉnh để đưa vào DOCX, gồm các đoạn mới cho nội dung đề tài, mục tiêu, phạm vi, nội dung 2-6 và kế hoạch.

- [ ] **Step 1: Trích xuất đoạn văn và lập ánh xạ thay thế**

  Xác định các đoạn bắt đầu bằng `Nội dung đề tài:`, `Mục tiêu:`, `Đối tượng nghiên cứu:`, `Phạm vi nghiên cứu:`, `Nội dung 2:`, `Nội dung 3:`, `Nội dung 4:`, `Nội dung 6:` và bảng kế hoạch. Giữ các đoạn về HGT/MSEE/EACS/MITRE; thay đoạn FL bằng wording client-server và sampling cục bộ.

- [ ] **Step 2: Viết nội dung thay thế bằng tiếng Việt**

  Dùng các ý chính sau trong bản nội dung:

  ```text
  Trong thực nghiệm, một bộ dữ liệu chuẩn hóa được xem là dữ liệu tổng hợp từ nhiều gateway hoặc miền quản trị. Bộ dữ liệu được phân vùng thành các client mô phỏng để tái hiện data silo; đây là cách mô phỏng có kiểm soát, không hàm ý các tổ chức chia sẻ PCAP với nhau.

  Ở mỗi round, server phát global HGT model cho các client được chọn. Mỗi client giữ graph và nhãn cục bộ, chọn seed flow rồi dùng neighbor/subgraph sampling 2-hop với fanout theo relation để tạo mini-batch, huấn luyện local model, rồi gửi model delta `Δw_k = w_k - w_t`. Server không nhận dữ liệu thô; server kiểm tra các tín hiệu bất thường chỉ từ delta/model update, tổng hợp bằng FedAvg hoặc cơ chế Byzantine-robust, và phát global model mới.

  Trọng tâm an toàn là model poisoning do client Byzantine. Các kịch bản sign-flip, ALIE, Min-Max/Min-Sum và ScaleSign được chèn ở bước gửi update, dưới IID và non-IID, với nhiều tỷ lệ client độc hại. Hiệu quả phòng thủ được đánh giá bằng cách so sánh FedAvg với Median, Trimmed Mean, Multi-Krum, SignGuard hoặc phương pháp phù hợp và cơ chế đa tín hiệu đề xuất.
  ```

- [ ] **Step 3: Rà soát nội dung**

  Chạy tìm kiếm các cụm dễ gây hiểu nhầm: `gom dữ liệu`, `gửi dữ liệu`, `chia sẻ PCAP`, `cắt dữ liệu gửi`, `English`, `TBD`, `TODO`. Sửa để mọi câu đều nhất quán với Global Constraints.

- [ ] **Step 4: Commit nội dung trung gian**

  ```powershell
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' add -- 'outputs/fl_proposal_revision_content.txt'
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' commit -m 'docs: draft FL poisoning proposal content'
  ```

### Task 2: Dựng sơ đồ kiến trúc FL-based IDS

**Files:**
- Create: `outputs/fl_based_ids_architecture.mmd`
- Create: `outputs/fl_based_ids_architecture.svg`
- Create: `outputs/fl_based_ids_architecture.png`
- Create: `outputs/fl_based_ids_architecture_notes.md`

**Interfaces:**
- Consumes: Nội dung Task 1 và `docs/architecture/system_architecture_academic.md`.
- Produces: Sơ đồ vector, ảnh raster và Mermaid source với cùng một luồng.

- [ ] **Step 1: Viết Mermaid source**

  Sơ đồ phải có các vùng và node sau:

  - `Data silos / client simulation`: raw PCAP/flow, MITRE index, client partition.
  - `Client k`: graph builder, local feature store, neighbor/subgraph sampler, HGT local train, optional malicious client.
  - `FL coordinator`: client selection, global model broadcast, update receive.
  - `Poisoning & defense`: attack injector gồm bốn attack, update signal checks gồm schema/round validity, norm/clipping, cosine, sign, geometric distance, spectral outlier và client history; robust aggregation và client trust score. Server không dùng MITRE/evidence/raw graph.
  - `Global evaluation`: centralized HGT, local-only, FedAvg, robust aggregators, random/temporal, clean/noisy grading.
  - `Runtime`: HGT inference, policy engine, evidence-grounded XAI, SOC feedback.

  Mũi tên phải ghi rõ `raw data stays local`, `local model update`, `global model`, `poisoned update`, `filtered updates` và `deploy checkpoint`.

- [ ] **Step 2: Tạo bản SVG**

  Dùng bố cục trái-phải hoặc trên-dưới, chữ đủ lớn để chèn vào trang A4 landscape. Màu trung tính: data xanh nhạt, process trắng, model cam nhạt, defense đỏ nhạt, store xanh lá, decision tím nhạt. Không dùng gradient hoặc hiệu ứng trang trí.

- [ ] **Step 3: Tạo bản PNG**

  Render SVG ở độ phân giải tối thiểu 2400 x 1400 px, nền trắng, kiểm tra chữ tiếng Việt và ký hiệu mũi tên.

- [ ] **Step 4: Viết ghi chú sơ đồ**

  Ghi rõ: client giữ raw data/graph; sampling nằm trong client; poisoning được chèn vào update; server không xem dữ liệu thô; robust aggregation là lớp thí nghiệm, không tuyên bố an toàn tuyệt đối.

- [ ] **Step 5: Commit bộ sơ đồ**

  ```powershell
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' add -- 'outputs/fl_based_ids_architecture.mmd' 'outputs/fl_based_ids_architecture.svg' 'outputs/fl_based_ids_architecture.png' 'outputs/fl_based_ids_architecture_notes.md'
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' commit -m 'docs: add FL IDS architecture diagram'
  ```

### Task 3: Chỉnh DOCX và chèn sơ đồ

**Files:**
- Read: `DeCuongKLTN_NCP.docx`
- Create: `outputs/DeCuongKLTN_FL_Poisoning_Revised.docx`
- Modify: `outputs/fl_proposal_revision_content.txt`
- Use: `outputs/fl_based_ids_architecture.png`

**Interfaces:**
- Consumes: Nội dung Task 1 và ảnh Task 2.
- Produces: DOCX chỉnh sửa, giữ tên tiếng Anh trống và có trang sơ đồ kiến trúc ở cuối.

- [ ] **Step 1: Mở bản DOCX bằng OOXML/python-docx runtime**

  Nếu `python-docx` chưa có trong `D:\v\nt114`, cài đúng gói vào môi trường này trước khi authoring. Không sửa file nguồn trực tiếp; ghi ra `outputs/DeCuongKLTN_FL_Poisoning_Revised.docx`.

- [ ] **Step 2: Giữ metadata đầu trang và xóa tên tiếng Anh**

  Giữ `TÊN ĐỀ TÀI` và tên Việt hiện tại. Thay dòng tên tiếng Anh bằng `Tên tiếng Anh:` không có nội dung sau dấu hai chấm. Giữ các trường thông tin cá nhân dạng điền sau.

- [ ] **Step 3: Thay nội dung theo ánh xạ Task 1**

  Sửa các đoạn FL, mục tiêu, phạm vi, nội dung nghiên cứu và lịch. Không thêm khẳng định kết quả chưa đo; giữ các số liệu dataset hiện có ở mức `CIC-IoT-2023-style, 18 lớp`.

- [ ] **Step 4: Thêm trang sơ đồ**

  Thêm page break, heading `Sơ đồ kiến trúc FL based IDS đề xuất`, ảnh PNG căn giữa, caption và 4 ghi chú ngắn về local data, sampling, poisoning update và robust aggregation.

- [ ] **Step 5: Render DOCX**

  Chạy renderer của Documents skill với `--emit_pdf` vào thư mục QA riêng để tạo ảnh trang và PDF tạm.

- [ ] **Step 6: Commit DOCX**

  ```powershell
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' add -- 'outputs/DeCuongKLTN_FL_Poisoning_Revised.docx'
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' commit -m 'docs: revise proposal for FL poisoning study'
  ```

### Task 4: Xuất PDF cuối

**Files:**
- Create: `outputs/DeCuongKLTN_FL_Poisoning_Revised.pdf`
- Read: `outputs/DeCuongKLTN_FL_Poisoning_Revised.docx`

**Interfaces:**
- Consumes: DOCX đã QA.
- Produces: PDF cuối có nội dung và sơ đồ giống bản DOCX.

- [ ] **Step 1: Xuất PDF từ DOCX renderer**

  Dùng PDF do renderer tạo từ DOCX; nếu renderer chỉ tạo PDF tạm, copy/đổi tên thành output PDF cuối sau khi kiểm tra.

- [ ] **Step 2: Kiểm tra cấu trúc PDF**

  Dùng PyMuPDF mở lại PDF, xác nhận số trang > 0, text layer tồn tại và trang sơ đồ có caption.

- [ ] **Step 3: Commit PDF cuối**

  ```powershell
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' add -- 'outputs/DeCuongKLTN_FL_Poisoning_Revised.pdf'
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' commit -m 'docs: export revised FL proposal PDF'
  ```

### Task 5: Canvas preview và nghiệm thu cuối

**Files:**
- Create: `C:\Users\Le Thanh Phat\.cursor\projects\empty-window\canvases\fl-based-ids-architecture.canvas.tsx`
- Read: `outputs/DeCuongKLTN_FL_Poisoning_Revised.docx`
- Read: `outputs/DeCuongKLTN_FL_Poisoning_Revised.pdf`

**Interfaces:**
- Consumes: Sơ đồ Task 2.
- Produces: Canvas preview và báo cáo QA cuối trong hội thoại.

- [ ] **Step 1: Viết canvas bằng `cursor/canvas`**

  Dùng `useHostTheme()` và raw SVG inline; không import npm package, không dùng gradient, emoji hoặc box-shadow. Canvas phải hiển thị 6 vùng kiến trúc và các mũi tên update/global model.

- [ ] **Step 2: Kiểm tra canvas**

  Mở canvas, xác nhận không blank và không có lỗi build; xem nội dung ở mức 100%.

- [ ] **Step 3: Render mọi trang đầu ra**

  Render DOCX/PDF thành PNG. Kiểm tra từng trang: không clipping/overlap, chữ tiếng Việt đúng, tên tiếng Anh trống, sơ đồ đọc được, bảng kế hoạch không vỡ.

- [ ] **Step 4: Chạy tìm kiếm nghiệm thu**

  ```powershell
  rg -n -i "gửi.*PCAP|gửi.*dữ liệu thô|server.*gom|chia sẻ.*PCAP|Tên tiếng Anh: .+|TBD|TODO" 'outputs/DeCuongKLTN_FL_Poisoning_Revised.docx' 'outputs/DeCuongKLTN_FL_Poisoning_Revised.pdf'
  ```

  Expected: không có dòng nào mô tả client gửi dữ liệu thô; dòng tên tiếng Anh chỉ có nhãn rỗng; không có placeholder mới ngoài các trường cá nhân có sẵn.

- [ ] **Step 5: Commit nghiệm thu**

  ```powershell
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' add -- 'docs/superpowers/plans/2026-09-10-fl-poisoning-proposal.md'
  git -c safe.directory='D:/Tai_lieu_nam_ba/Hoc_ky_2/Do_an_chuyen_nganh/Do-an-chuyen-nganh_NT114' -C 'Do-an-chuyen-nganh_NT114' commit -m 'docs: record FL proposal artifact plan'
  ```

## Self-review

- Spec coverage: client simulation, local sampling, FL round, four poisoning attacks, robust aggregation, XAI support, output formats and render QA are covered by Tasks 1-5.
- Placeholder scan: no `TBD`, `TODO` or unspecified implementation step appears in the plan; personal metadata fields remain intentionally blank in the source proposal and are not new technical placeholders.
- Type/file consistency: all Task 2 outputs are consumed by Task 3; Task 3 DOCX is consumed by Task 4 and Task 5; canvas consumes the Task 2 diagram semantics.
