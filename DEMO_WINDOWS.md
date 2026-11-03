# Demo trên laptop Windows (không cần robot, không cần webcam)

Giao diện chạy trong trình duyệt: **tải ảnh bàn cờ lên → thư viện nhận diện thế cờ → gợi ý nước đi**. Dùng đúng thư viện `XiangqiBrain` (xem [API.md](API.md)), chỉ khác là YOLO chạy bằng ONNX trên CPU thay vì TensorRT trên robot.

---

## 1. Lấy gói demo

Chạy `training/package.ipynb` trên Colab tới **mục 7** → tải về **`xiangqi_demo_windows.zip`** (≈ 60 MB). Trong gói có sẵn:

| Thành phần | |
|---|---|
| Mã nguồn thư viện + `tools/demo_app.py` | giao diện demo |
| `models/best.onnx` | mô hình YOLO11n (Bước 2) |
| `engine/pikafish.exe` + `engine/pikafish.nnue` | Pikafish cho Windows, chạy ngay không cần cài |
| `run_demo.bat` | bấm đúp để chạy |

## 2. Cài Python (một lần)

1. Tải **Python 3.11** tại https://www.python.org/downloads/windows/ (bản *Windows installer 64-bit*).
2. Khi cài, **tích ô "Add python.exe to PATH"** rồi bấm *Install Now*.

## 3. Chạy

1. Giải nén `xiangqi_demo_windows.zip` vào thư mục **không có dấu, không có khoảng trắng**, ví dụ `D:\xiangqi_demo\`.
2. Vào `D:\xiangqi_demo\xiangqi\`, **bấm đúp `run_demo.bat`**.
   - Lần đầu: tự cài thư viện Python (cần Internet, 3–5 phút).
   - Các lần sau: mở ngay (≈ 10 giây).
3. Trình duyệt tự mở **http://127.0.0.1:7860**. Cửa sổ đen phải để mở trong suốt buổi demo.

Cửa sổ đen in `[brain] Engine: Pikafish ...` là đúng. Nếu in `CẢNH BÁO: không có Pikafish` thì xem mục 6.

---

## 4. Chuẩn bị ảnh demo

Chụp bằng điện thoại, ảnh nào cũng được tải lên (jpg, png).

**Cách chụp để nhận diện tốt**
- Chụp **thẳng từ trên xuống**, cả bàn cờ nằm trọn trong ảnh, chừa một chút viền.
- **Phía quân Đỏ ở cạnh dưới ảnh** (thư viện mặc định coi cạnh dưới là phía Đỏ).
- Đủ sáng, không bóng tay/điện thoại đè lên quân; không để tay trong khung hình.
- Giữ điện thoại cùng một chỗ cho cả ván (kê lên chồng sách hoặc giá đỡ).

**Bộ ảnh cho một ván (khuyên dùng)** — chụp lần lượt, đặt tên theo thứ tự:

| Ảnh | Chụp lúc |
|---|---|
| `00_khai_cuoc.jpg` | Bày xong thế khai cuộc |
| `01_nguoi.jpg` | Sau khi bạn (Đỏ) đi nước 1 |
| `02_nguoi.jpg` | Sau khi đi giúp máy nước đầu **và** bạn đi nước 2 |
| … | 8–10 lượt là đủ |

Muốn diễn phần **tự sửa lỗi**: chụp thêm vài ảnh *có lỗi cố ý* — che một quân bằng ngón tay, đặt một quân lệch khỏi giao điểm, hoặc đi một nước sai luật (tượng qua sông).

> Mẹo: chơi thử một ván với giao diện trước buổi demo và **giữ lại đúng bộ ảnh đó**. Hôm bảo vệ chỉ cần tải lại theo thứ tự — không phụ thuộc ánh sáng phòng.

---

## 5. Kịch bản demo (7–10 phút)

**Thẻ 1 · Phân tích một ảnh**
1. Tải `00_khai_cuoc.jpg`, chọn *Đỏ đi tiếp*, bấm **Phân tích** → thấy 32 quân được khoanh, bàn cờ vẽ lại, mũi tên nước gợi ý, thời gian xử lý.
2. Tải ảnh **có lỗi cố ý** + dán FEN của lượt trước vào ô *FEN lượt trước* → thư viện suy ra nước vừa đi và **tô đỏ ô đã tự sửa**. Đây là điểm khác biệt chính của đề tài.
3. Tải ảnh **đi sai luật** → thư viện báo *Cần chụp lại* thay vì đi theo thế cờ sai.

**Thẻ 2 · Chơi cả ván**
1. Chọn *Máy cầm Đen*, bấm **Ván mới**.
2. Tải lần lượt `01_nguoi.jpg`, `02_nguoi.jpg`, … — mỗi lần máy trả lời nước đi (ô gắp → ô đặt), biên bản ván cờ ghi bằng ký hiệu tiếng Việt.

**Slide số liệu** (từ các notebook): mAP@0.5 = 0.968 · ảnh → FEN đúng ~78% · biết thế trước: suy ra đúng nước ~95%, **0% sai mà không phát hiện**.

---

## 6. Lỗi thường gặp

| Hiện tượng | Cách sửa |
|---|---|
| `'py' is not recognized` / `'python' is not recognized` | Cài lại Python, nhớ tích **Add python.exe to PATH** |
| `pip install` lỗi mạng | Lần đầu cần Internet. Thử lại ở mạng khác |
| `DLL load failed ... onnxruntime` | Cài **Microsoft Visual C++ Redistributable (x64)**: https://aka.ms/vs/17/release/vc_redist.x64.exe rồi chạy lại |
| `CẢNH BÁO: không có Pikafish` | Kiểm tra có `engine\pikafish.exe` và `engine\pikafish.nnue`; phần mềm diệt virus có thể đã cách ly `pikafish.exe` → cho phép (allow) rồi chạy lại |
| Trình duyệt không tự mở | Tự mở http://127.0.0.1:7860 |
| `Address already in use` / cổng 7860 bận | Đóng cửa sổ demo cũ, hoặc chạy `py tools\demo_app.py --port 7861` |
| Luôn báo *Cần chụp lại* | Ảnh nghiêng / thiếu sáng / thiếu một phần bàn; xem ảnh *Quân nhận diện được* để biết thiếu quân nào |
| Thế cờ bị lật ngược (Đỏ ở trên) | Chụp lại sao cho phía quân Đỏ ở cạnh dưới ảnh |
| Muốn cài lại thư viện Python | Xoá file `.deps_ok` rồi chạy lại `run_demo.bat` |

## 7. Chạy từ mã nguồn (không dùng gói)

```bat
git clone https://github.com/JulianNguyen05/xiangqi-sorting-jetmax
cd xiangqi-sorting-jetmax
py -m pip install -r requirements-demo.txt
REM chép best.onnx vào models\ ; chép pikafish.exe + pikafish.nnue vào engine\
set PYTHONUTF8=1
py tools\demo_app.py
```
