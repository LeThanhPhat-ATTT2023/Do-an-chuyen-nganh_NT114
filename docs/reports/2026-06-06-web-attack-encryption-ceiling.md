# Chẩn đoán quyết định: trần macro-F1 do web-attack mã hóa TLS

**Ngày:** 2026-06-06
**Bối cảnh:** sau khi hgt_v3fam đạt test macro-F1 = 0.766 (< GNN4ID 0.854, mục tiêu 0.9),
điều tra tại sao cụm web-attack (CommandInjection/XSS/Uploading_Attack) không tách được.

---

## 0. TL;DR

> **Traffic web-attack trong CIC-IoT-2023 (subset 14GB) chạy ~95% trên HTTPS/TLS (port 443).
> Nội dung tấn công phân biệt nằm trong ciphertext → không model nào trích được.
> Đây là trần THÔNG TIN của dữ liệu. macro-F1 ≈ 0.85 là trần thực; 0.9 gần như bất khả thi
> trên setup 18-lớp này, và train tập lớn hơn KHÔNG cứu được (cùng phương pháp capture).**

---

## 1. Chuỗi bằng chứng

### 1.1 Separability web bằng packet_x (byte-4gram, đã chuẩn hóa độ dài)
LR trên packet feature pooled theo flow, 4 lớp web (random 70/30):

| Pooling | web macro-F1 | CmdInj | SQLi | Upload | XSS |
|---|---|---|---|---|---|
| MEAN | 0.591 | 0.562 | 0.848 | 0.408 | 0.547 |
| MAX | 0.575 | 0.539 | 0.826 | 0.410 | 0.523 |
| MEAN+MAX | 0.601 | 0.571 | 0.846 | 0.434 | 0.551 |

→ MAX **không** hơn MEAN ⇒ không có tín hiệu plaintext sparse nào để aggregation cứu.
Đổi pooling/attention-MIL **không phải** lời giải.

### 1.2 Separability bằng exact-match literal trên payload thô
Script: `scripts/diagnostics/web_literal_separability.py`. Quét 53 literal tấn công
high-precision trên payload TCP thô, max-pool lên flow, LR:

| | web macro-F1 | CmdInj | SQLi | Upload | XSS |
|---|---|---|---|---|---|
| exact-literal | **0.394** | 0.222 | 0.835 | 0.192 | 0.326 |

→ TỆ HƠN byte-4gram. "Discriminator" hàng đầu là **backtick `` ` `` (0.64–0.75 ở mọi lớp)** —
xác suất byte ngẫu nhiên trong payload ~200B. Chữ ký thật nhiều ký tự gần như vắng:
`%3cscript` (XSS) = 0.03, `union%20select` (SQLi) ≈ 0.00.

### 1.3 Xác minh giao thức (quyết định)
Đếm port đích + nhận dạng HTTP/TLS trên packet có payload:

| Lớp | port 443 | HTTP plaintext | TLS/mã hóa |
|---|---|---|---|
| CommandInjection | 2960 (chủ đạo) | 320 (4%) | ~7680 (96%) |
| XSS | 3143 (chủ đạo) | 263 (3%) | ~7737 (97%) |
| Uploading_Attack | 1033 (chủ đạo) | 164 (5%) | ~2959 (95%) |
| SqlInjection | 2588 + port80:677 | 1348 (17%) | ~6652 (83%) |

→ **CmdInj/XSS/Upload ~95% mã hóa.** SQLi có nhiều plaintext nhất (17%) ⇒ là lớp web
DUY NHẤT tách được (~0.83), khớp hoàn hảo với 1.1 và 1.2.

---

## 2. Giải thích thống nhất

| Quan sát | Nguyên nhân |
|---|---|
| byte-4gram cap 0.59 | payload mã hóa = entropy cao, byte gần đều |
| MAX không hơn MEAN | không tồn tại tín hiệu plaintext sparse |
| exact-literal 0.39 < byte 0.59 | literal thật vắng; chỉ trùng byte ngẫu nhiên |
| SQLi tách được, CmdInj/XSS/Upload không | SQLi nhiều plaintext (port 80) hơn hẳn |
| **GNN4ID cũng cap 0.8537, cũng fail web** | **đụng cùng trần thông tin — không phải nó giỏi hơn** |
| v3_fam family-filter vô hiệu (+0.01) | không có tín hiệu plaintext family để lọc |

---

## 3. Trần thực & phân rã macro-F1 18 lớp

| Nhóm | Số lớp | F1 đạt được (thực) | Ghi chú |
|---|---|---|---|
| Mạnh sẵn (DDoS-flood×2, ACK-frag, Recon×3, Benign) | 7 | ~0.97 | đã tốt |
| Volumetric (Mirai, ICMP-Flood, ICMP-Frag) | 3 | hiện 0.25–0.60 → **~0.95 nếu windowing** | đói mẫu, sửa được |
| Mid (BrowserHijack, PingSweep, VulnScan, Backdoor) | 4 | ~0.85 | ổn |
| SqlInjection | 1 | ~0.85 | có plaintext |
| **CmdInj / XSS / Upload (mã hóa)** | 3 | **~0.4–0.6 — TRẦN CỨNG** | **không sửa bằng content** |

Ước lượng trần tổng (sau khi sửa volumetric): **~0.85–0.87.** Để chạm 0.9 cần cụm 3 lớp
mã hóa đạt ~0.8 — bất khả thi bằng feature nội dung.

---

## 3b. Đã ĐO: encrypted-traffic side-channel (Phương án C)

Script `scripts/diagnostics/web_sidechannel_separability.py` — đặc trưng KHÔNG cần
giải mã: signed packet-length sequence (32 gói đầu) + TLS record-size sequence
(header TLS 5-byte cleartext) + burst/volume/timing aggregates. RandomForest:

| | web macro-F1 | CmdInj | SQLi | Upload | XSS |
|---|---|---|---|---|---|
| side-channel | **0.606** | 0.601 | 0.951 | 0.378 | 0.494 |
| (baseline byte-4gram) | 0.591 | 0.562 | 0.848 | 0.408 | 0.547 |
| **cụm mã hóa CmdInj/XSS/Upload** | **0.514** | — | — | — | — |

→ Side-channel là hướng ĐÚNG, có ích thật (CmdInj ↑, SQLi ↑0.95, tín hiệu trực giao),
NHƯNG **không phá vỡ** confusion CmdInj↔XSS↔Upload (cụm vẫn ~0.51) — vì 3 đòn tấn công
cùng endpoint, cấu trúc request/response tương tự. Deep sequence model (1D-CNN/transformer
trên chuỗi dài) có thể nhích thêm nhưng bằng chứng không cho thấy chạm 0.8 cho cụm này.

## 3c. Con đường thật tới >0.9 (Phương án B — lượng hóa)

Gộp 3 lớp mã hóa bất-khả-phân thành 1 siêu lớp "WebAttack" (đúng nghĩa SOC), giữ SQLi
riêng (vì nó tách được nhờ plaintext). Lỗi CmdInj↔XSS↔Upload là lỗi NỘI BỘ cụm → khi gộp,
chúng biến mất. Ước lượng từ per-class F1 hiện tại + windowing cứu volumetric:

| Cấu hình | macro-F1 (ước lượng) |
|---|---|
| 18 lớp hiện tại | 0.766 |
| 18 lớp + windowing | ~0.82 |
| **16 lớp (gộp CmdInj/XSS/Upload) + windowing** | **~0.92** |

(Số chính xác cần một lần inference dump confusion matrix — chưa chạy; ước lượng dựa trên
việc lỗi web là nội-cụm và windowing đưa Mirai 0.25→~0.95, ICMP_Frag 0.60→~0.95.)

## 3d. ĐỘT PHÁ: gốc rễ thật là LABEL POLLUTION, không phải mã hóa

Test TLS-handshake (#4) lộ ra: pcap web-attack bị **traffic nền IoT→cloud** áp đảo
(SNI: connectivity.smartthings.com, apicom.netatmo.net, api.amazonalexa.com — GIỐNG HỆT
ở cả CmdInj/XSS/Upload). Test isolation (#5) xác nhận:

Đòn tấn công THẬT là **HTTP plaintext tới web app DVWA cục bộ**:
```
CommandInjection → POST /dvwa/vulnerabilities/exec/    (192.168.137.13:80)
XSS              → GET  /dvwa/vulnerabilities/xss_d/    (192.168.137.13:80)
Uploading_Attack → POST /dvwa/vulnerabilities/upload/   (192.168.137.13:80)
SqlInjection     → GET  /dvwa/vulnerabilities/sqli/?id=1%27+or+%271%27%3D%271  (192.168.137.4:80)
```

| Cấu hình | cụm web macro-F1 |
|---|---|
| Data bẩn (lẫn nhiễu nền cloud) | 0.51 |
| **Flow tấn công THẬT (HTTP plaintext tới victim)** | **0.80** (char-ngram LR, ~200 mẫu/lớp) |

**Phân rã nhãn lớp CommandInjection (4109 flow):** ~187 (4.5%) là tấn công DVWA thật;
~95% là nhiễu nền (IoT→cloud TLS + Vera /port_3480 polling benign). `infer_label_from_path`
gán "CommandInjection" cho TẤT CẢ → model bị bắt phân biệt nhiễu-nền giống hệt nhau giữa
các lớp → bất khả. macro-F1 0.766 phản ánh **nhãn ô nhiễm**, KHÔNG phải giới hạn model.

→ Mã hóa KHÔNG phải rào chắn thật: payload tấn công (URL/params/body tới DVWA) là **plaintext**.
Rào chắn thật là **nhiễu nền nuốt tín hiệu + nhãn theo-file thô**.

## 3e. Đường tới 18-lớp ≥0.9 (GIỮ 18 lớp, honest)

1. **Cô lập flow tấn công thật** theo đích (victim web app, HTTP plaintext) — tách khỏi cloud noise.
2. **Feature payload plaintext** (URL path + query + body) cho flow HTTP — nơi attack string thật
   nằm, không mã hóa, generalize (shell-meta/script-tag/SQL-meta).
3. **Nhiễu nền trong pcap tấn công → gán Benign** (đúng bản chất; data-cleaning chính danh).
Kỳ vọng: cụm web 0.4→~0.90, volumetric→0.95 (windowing) ⇒ **18-class ~0.90+ trung thực.**

(Cảnh báo cũ về "trần mã hóa ~0.85" ở trên BỊ THAY THẾ bởi phát hiện này — trần thật do
label pollution, gỡ được. Giữ phần trên làm lịch sử điều tra.)

## 4. Phương án (cần quyết định)

**A. Trung thực 0.85 + đóng khung học thuật mạnh.**
Vượt GNN4ID khiêm tốn (windowing + recipe ổn định ⇒ ~0.86), và **đóng góp thật là
GIẢI THÍCH trần** (phát hiện mã hóa + temporal-split honesty + MSEE provenance) —
điều XG-NID/PacketCLIP/GNN4ID không làm. Paper trung thực thường mạnh hơn con số 0.9 giả.

**B. Báo cáo ở độ hạt dữ liệu hỗ trợ ⇒ >0.9 hợp lệ.**
Cụm mã hóa không tách được về vật lý ⇒ gộp CmdInj/XSS/Upload thành 1 siêu lớp
"Web-App-Attack" (đúng nghĩa SOC). macro-F1 trên 15–16 lớp có thể **vượt 0.9 một cách trung thực**,
vì đã bỏ phân biệt bất khả thi. Đây là con đường thật tới >0.9.

**C. Đặc trưng hành vi TLS (không chắc).**
Đặc trưng rò qua TLS: chuỗi kích thước TLS record, timing, hướng, JA3 handshake.
Có thể nâng cụm mã hóa từ ~0.5 lên ~0.65–0.75. KHÔNG đủ chạm 0.9, rủi ro cao.

**D. Đổi dữ liệu.** Tìm capture có web-attack plaintext, hoặc dataset khác. Scope lớn.

---

## 5. Khuyến nghị

Kết hợp **A + B**: làm các thắng chắc chắn (windowing cứu volumetric, bỏ DRW ổn định
training) để đạt ~0.86 trên 18 lớp một cách lành mạnh; ĐỒNG THỜI báo cáo biến thể gộp
siêu-lớp (B) để có một con số >0.9 trung thực. Đóng góp lõi của luận án chuyển từ
"thắng số" sang "giải thích trần + giao thức đánh giá honest" — vững và khó bác hơn.

**Cảnh báo quan trọng:** KHÔNG scale lên tập lớn để mong 0.9 — trần này là phương pháp
luận (mã hóa), không phải số mẫu. Tập lớn sẽ cap ở cùng chỗ.
