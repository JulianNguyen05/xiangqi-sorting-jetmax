# Notebook: từ dữ liệu đến thư viện "ảnh bàn cờ → nước đi"

Thư mục này chứa toàn bộ quy trình xây dựng thư viện **XiangqiBrain**: nhận ảnh bàn cờ tướng (kèm thế cờ lượt trước), trả về nước đi tiếp theo dưới dạng ô gắp → ô đặt. Thư viện chạy trên robot JetMax (Jetson Nano) hoặc trên laptop qua giao diện demo.

Mọi notebook chạy trên **Google Colab**; dữ liệu và kết quả lưu ở Drive `MyDrive/datn/`. Bấm nút *Open in Colab* ở đầu mỗi notebook.

## Thứ tự các notebook

```
Bước 1 build_dataset ──► Bước 2 train_yolo11 ──► Bước 3 jetmax_deploy
                                   │
                                   ├──► Bước 4 board_to_fen ──► Bước 5 rules_engine ──► Bước 6 package
                                   │
image_preprocessing (minh hoạ xử lý ảnh, chạy độc lập bất kỳ lúc nào)
```

| Notebook | Việc chính | Đầu vào | Đầu ra | GPU |
|---|---|---|---|---|
| `build_dataset.ipynb` (Bước 1) | Gộp 4 bộ dữ liệu Roboflow, chuẩn hoá nhãn về 14 lớp, polygon/pose → bbox, gom ảnh trùng (pHash + union-find), chia tập theo nhóm (StratifiedGroupKFold ~80/10/10) | API key Roboflow | `merged_v1.zip` | Không |
| `train_yolo11.ipynb` (Bước 2) | Huấn luyện YOLO11n hai giai đoạn, đánh giá trên tập test, test xoay 0/90/180/270°, xuất ONNX | `merged_v1.zip` | `best.pt`, `best.onnx` | T4 |
| `jetmax_deploy.ipynb` (Bước 3) | Giới thiệu robot JetMax, phần cứng, tầm với, kế hoạch triển khai; kiểm tra `best.onnx`; hướng dẫn đưa mô hình + engine lên robot | `best.onnx`, `pikafish_jetson.tar.gz` | Hướng dẫn, kiểm tra ONNX | Không |
| `board_to_fen.ipynb` (Bước 4) | Khớp lưới 9×10 từ tâm quân, đánh giá ảnh → FEN trên tập test, quét ngưỡng tin cậy | `best.onnx`, tập test | Bảng số liệu, hình | Không |
| `rules_engine.ipynb` (Bước 5) | Luật cờ, perft, kiểm tra thế cờ, suy ra nước vừa đi và **tự sửa lỗi nhận dạng**; biên dịch Pikafish, gọi qua UCI | — | `pikafish_jetson.tar.gz`, số liệu | Không |
| `package.ipynb` (Bước 6) | Kiểm thử thư viện `XiangqiBrain`, đóng gói một file cho robot và gói demo Windows | `best.onnx`, Pikafish | `xiangqi_to_robot.zip`, `xiangqi_demo_windows.zip` | Không |
| `image_preprocessing.ipynb` | Minh hoạ **từng bước xử lý ảnh** trên 3 ảnh thật trong `samples/`: letterbox, HSV, phân ngưỡng, hình thái học, Hough, homography, màu quân; **tăng cường ảnh tự động** (`vision/enhance.py`: lọc song phương, Gray-World, CLAHE, gamma) | `samples/*.jpg/png` | Hình minh hoạ | Không |

## Ảnh mẫu (`samples/`)

Ảnh chụp thật bàn cờ của đề tài (tấm nhựa, đường kẻ đỏ), dùng trong `image_preprocessing.ipynb`, `tests/test_brain.py` và giao diện demo.

| Ảnh | Nội dung |
|---|---|
| `board_empty.jpg` | Bàn cờ trống |
| `start_position.jpg` | Thế khai cuộc, 32 quân trên bàn |
| `disorganized_location.png` | Quân để lộn xộn quanh bàn (vùng chờ) |

## 14 lớp (đúng thứ tự id)

| id | lớp | FEN | id | lớp | FEN |
|---|---|---|---|---|---|
| 0 | black_advisor | a | 7 | red_advisor | A |
| 1 | black_bishop | b | 8 | red_bishop | B |
| 2 | black_cannon | c | 9 | red_cannon | C |
| 3 | black_king | k | 10 | red_king | K |
| 4 | black_knight | n | 11 | red_knight | N |
| 5 | black_pawn | p | 12 | red_pawn | P |
| 6 | black_rook | r | 13 | red_rook | R |

## Bước 1 — Dữ liệu (`merged_v1`)

Sau làm sạch: **6 808 ảnh, 112 938 quân, 4 265 nhóm ảnh trùng**. Chia theo nhóm nên không nhóm nào bị tách giữa các tập (nếu giữ cách chia gốc, 1 044 ảnh bị rò rỉ).

| Nguồn | train | val | test | tổng |
|---|---|---|---|---|
| super (312s-workspace/xiangqi-super-dataset) | 1837 | 240 | 241 | 2318 |
| scara (312s-workspace/xiangqi-scara-yolo26) | 1335 | 216 | 183 | 1734 |
| new_chinese_chess (quoc-rnmzr, v1) | 1000 | 128 | 125 | 1253 |
| duytranle (duytranle/xiangqi-ubjij, v18) | 1184 | 155 | 164 | 1503 |
| **Tổng** | **5356** | **739** | **713** | **6808** |

## Bước 2 — Huấn luyện

| Run | Xuất phát | Thay đổi chính | Lý do |
|---|---|---|---|
| `yolo11n_640` | `yolo11n.pt` (COCO) | 100 epoch, `degrees=10`, `fliplr=0` | Bài toán ban đầu: quân trên băng chuyền, lệch góc nhẹ |
| `yolo11n_640_rot` | `yolo11n_640/best.pt` | 60 epoch, **`degrees=180`**, `flipud=0`, AdamW `lr0=0.001` | Quân nằm xoay bất kỳ 0–360° |

Không lật ảnh (`fliplr=0`, `flipud=0`) vì lật biến chữ Hán thành ảnh gương. Xoay không đảo chữ nên dùng được tới 180°.

Kết quả `yolo11n_640_rot` trên **713 ảnh test (11 950 quân)**:

| mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Tham số | GFLOPs |
|---|---|---|---|---|---|
| **0.964** | 0.815 | 0.990 | 0.959 | 2.6 M | 6.5 |

| mAP@0.5 khi xoay ảnh test | 0° | 90° | 180° | 270° |
|---|---|---|---|---|
| `yolo11n_640` | 0.955 | 0.762 | 0.904 | 0.761 |
| `yolo11n_640_rot` | **0.964** | **0.965** | **0.963** | **0.965** |

ONNX (opset 12) khớp PyTorch: sai lệch lớn nhất 1.07e-3, lớp dự đoán trùng 100%.

## Bước 4, 5 — Từ ảnh đến nước đi

| Chỉ số (229 ảnh test có bàn cờ, ngưỡng 0.5) | Kết quả |
|---|---|
| Ảnh → FEN đúng hoàn toàn (end-to-end) | 78.2% |
| Biết thế lượt trước → suy ra đúng nước đi | **95.2%** (ngưỡng 0.25: 96.9%) |
| Không chắc → yêu cầu chụp lại | 4.8% |
| Sai mà không phát hiện | **0%** |
| Bộ sinh nước đi (perft 1–4 từ khai cuộc) | 44 / 1 920 / 79 666 / 3 290 240 — khớp giá trị chuẩn |

## Tiền xử lý ảnh (`image_preprocessing.ipynb`)

Gọi đúng các hàm của thư viện trên ảnh thật:

| Bước | Hàm | Kết quả trên `start_position.jpg` |
|---|---|---|
| Letterbox 640, BGR→RGB, CHW, chia 255 | `vision/pieces.py: letterbox()` | r = 0.5, đệm 80 px trên/dưới |
| HSV + phân ngưỡng đỏ hai khoảng H | `vision/board_detect.py: red_mask()` | 4.4% điểm ảnh là đỏ |
| Phép đóng 5×5, đường viền lớn nhất | `_quad()` (dự phòng) | — |
| HoughLinesP → 2 họ đường | `_line_families()` | 2 họ đường ~90° |
| Homography + 90 giao điểm | `locate()` | 17 đường, sai lệch 0.61 mm |
| Màu quân (đỏ / đen) | `vision/pieces.py: ink_color()` | — |
| Đo ảnh, tự bật khử nhiễu / cân bằng trắng / CLAHE | `vision/enhance.py: enhance()` | Ảnh tốt: không bước nào (giữ nguyên) |

Tăng cường ảnh (`brain.enhance` trong `config/robot.yaml`, mặc định `mode: auto`, chỉ cho ảnh định vị bàn cờ). Trên ảnh suy giảm được mô phỏng:

| Ảnh | `auto` áp dụng | Định vị bàn: trước → sau |
|---|---|---|
| 3 ảnh mẫu, ảnh hơi tối | không | giữ nguyên |
| Rất tối (L 85) | CLAHE | 16 đường → 14 đường (đều tìm được) |
| Ám vàng | Gray-World | 17 → 17 đường |
| Nhiễu σ = 22 / 30 / 40 | lọc song phương | **mất bàn → tìm được** |
| Rất tối + nhiễu σ = 16 | lọc + CLAHE | mất bàn → vẫn mất (chuyển sang `lattice` / chụp lại) |

## Mang sang robot

- Toàn bộ trong **một file** `xiangqi_to_robot.zip` (Bước 6): mã nguồn, `best.onnx`, Pikafish cho ARM, `install.sh`.
- Trên robot: `cd ~/xiangqi && sh install.sh` — tự build `best.engine` (TensorRT FP16) trên chính Jetson:
  `/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16`
- Ultralytics không chạy trên Python 3.6 (JetPack 4.x), phía robot dùng TensorRT/onnxruntime trực tiếp; tiền xử lý và NMS tự cài đặt trong `vision/pieces.py`.
- Hướng dẫn gọi thư viện: `API.md`; chạy demo trên Windows: `DEMO_WINDOWS.md`.

## Hình trong báo cáo lấy từ đâu

| Chương báo cáo | Notebook |
|---|---|
| Chương 3 (robot, phần cứng, tầm với) | `jetmax_deploy.ipynb` |
| Chương 4: hình 4.2–4.10 (dữ liệu) | `build_dataset.ipynb` |
| Chương 4: hình 4.12–4.14 (kích thước quân, tăng cường) | `train_yolo11.ipynb` |
| Chương 4: hình 4.1, 4.11, 4.15–4.22 (xử lý ảnh) | `image_preprocessing.ipynb` |
| Chương 5 (huấn luyện, đánh giá, ONNX) | `train_yolo11.ipynb`, `jetmax_deploy.ipynb` |
| Chương 6 (ảnh → FEN, luật cờ, Pikafish) | `board_to_fen.ipynb`, `rules_engine.ipynb` |
| Chương 7 (thư viện, đóng gói) | `package.ipynb` |
