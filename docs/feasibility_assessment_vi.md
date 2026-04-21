# Danh gia tinh thuc tien va do kha thi

## 1) Ket luan nhanh
Huong de tai "Context-rich, Explainable IDS voi Heterogeneous Graph + MITRE + LLM embedding" la kha thi, co gia tri khoa hoc ro, va phu hop voi tai nguyen may tinh i5/16GB neu tach pipeline thanh 2 che do:

1. Offline nhe dau (distillation va graph training) theo lo
2. Online nhe cuoi (student CNN + HGT nho + fast classifier)

## 2) Tai sao huong nay khac phuc diem yeu cua GNN4ID

### Diem yeu tham chieu tu GNN4ID
1. Packet node dung payload 1500 bytes, dan den input chieu cao va chi phi tinh toan lon.
2. Pipeline nghieng ve notebook/offline, chua toi uu cho van hanh real-time nghiem ngat.
3. Can bang du lieu bang over/under-sampling manh de huan luyen, co nguy co sai lech phan bo.

### Cai tien de xuat
1. Cat payload ve 256 bytes de tang toc va giam nhieu.
2. Distill SecureBERT -> 1D-CNN embedding 768 chieu de giu ngu nghia nhung van nhanh khi suy luan.
3. Gan MITRE theo cosine tren khong gian embedding de them context chien thuat.
4. Tach Fast Path (<10ms) va Slow Path (XAI 1-2s) de dong thoi dam bao tac chien va giai thich.

## 3) Kha thi theo tai nguyen Asus Vivobook i5/16GB

### Co the lam tot tren may hien tai
1. Trich payload 256 bytes tu PCAP.
2. Tao teacher targets theo lo nho (batch 16-32 neu RAM han che).
3. Train student 1D-CNN voi MSE + cosine tren CPU/GPU nho.
4. Huan luyen HGT ban nho tren tap mau da loc.

### Co the bi nghen
1. SecureBERT encoding toan bo CIC-IoT2023 se ton thoi gian lon neu chi dung CPU.
2. HGT tren do thi day dac va cua so thoi gian lon de qua tai RAM.
3. SLM sinh XAI neu dung model qua lon se vuot nguong do tre.

### Cach giam rui ro
1. Dung chunking theo file/time-window ngay tu dau.
2. Distillation va teacher embedding chay offline theo queue, khong chen vao online path.
3. Dung student CNN + ONNX Runtime cho online embedding.
4. Gioi han so node/canh moi window, tao co che truot cua so.
5. SLM quantized 4-bit, co timeout va fallback template.

## 4) KPI de bao ve truoc hoi dong

### Hieu nang
1. Latency fast path p95 < 10ms cho moi graph window.
2. Throughput dat nguong dat ra theo kich ban thu nghiem (vi du >= 500 flow/s tren may local).

### Do chinh xac
1. F1-macro cao hon baseline GNN4ID trong cung bo du lieu/phan chia.
2. AUC va recall lop tan cong quan trong duoc cai thien ro.

### XAI va MITRE
1. Ty le mapping MITRE hop le (manual spot-check) dat muc chuan noi bo.
2. Bao cao Slow Path co thong tin: ky thuat, bang chung packet/flow, confidence.

## 5) Lo trinh trien khai de xuat (thuc dung)
1. Tuan 1-2: Hoan tat extractor payload 256 + metadata, tao teacher_targets ban dau.
2. Tuan 3-4: Train va benchmark student 1D-CNN; xuat ONNX.
3. Tuan 5-7: Dung hetero graph 3 tang + tactical edge MITRE.
4. Tuan 8-10: Train HGT, tao fast classifier, do latency.
5. Tuan 11-12: Tich hop SLM Slow Path cho bao cao XAI.
6. Tuan 13-14: Ablation va thong ke.
7. Tuan 15-16: Viet luan van, chot demo, chay tong duyet.

## 6) Tieu chi dung/sua huong som
1. Neu student CNN khong dat do tuong dong embedding: thuong tokenization payload chua dung, can dieu chinh bieu dien byte->text.
2. Neu latency HGT vuot muc: giam kich thuoc graph, giam so layer/head, hoac chuyen mot phan logic sang rule-based gating.
3. Neu MITRE mapping nhieu false positive: nang nguong cosine, them rerank bang flow context.
