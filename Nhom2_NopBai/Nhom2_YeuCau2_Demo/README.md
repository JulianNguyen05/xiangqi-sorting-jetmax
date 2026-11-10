# Yêu cầu 2 - Nghiên cứu và áp dụng mô hình học sâu: YOLO11n cho robot JetMax chơi cờ tướng

> **Yêu cầu của bài thi:** nghiên cứu và áp dụng mô hình học sâu phù hợp vào bài toán thực tế liên ngành đã chọn.
>
> **Nhóm thực hiện:** mô hình **YOLO11n** (mạng nơ-ron tích chập phát hiện đối tượng) nhận dạng 14 loại quân cờ tướng; áp dụng vào bài toán liên ngành **học sâu + robot + trò chơi**: robot JetMax nhìn bàn cờ, hiểu thế cờ và chọn nước đi.

## 1. Tóm tắt kết quả

| Hạng mục | Giá trị |
|---|---|
| Mô hình | YOLO11n: 2,6 triệu tham số, 6,5 GFLOPs, tệp 5,5 MB |
| Dữ liệu | `merged_v1` của Yêu cầu 1: 6 808 ảnh, 14 lớp |
| Huấn luyện | Học chuyển giao 2 giai đoạn trên GPU Tesla T4 (Google Colab) |
| mAP@0.5 trên tập test (713 ảnh, 11 950 quân) | **0,964** (mục tiêu 0,95) |
| mAP@0.5:0.95 / Precision / Recall | 0,815 / 0,990 / 0,959 |
| Khi xoay ảnh 90 / 180 / 270 độ | 0,965 / 0,963 / 0,965 (không giảm) |
| Ảnh - thế cờ đúng hoàn toàn | 78,2% |
| Biết thế cờ lượt trước, suy ra đúng nước vừa đi | **95,2%**; 4,8% xin chụp lại; 0% sai mà không phát hiện |

## 2. Nội dung thư mục

```
Nhom2_YeuCau2_Demo/
├── README.md              tệp này
├── requirements.txt       thư viện cho các tập lệnh trong src/
├── models/
│   ├── best.pt            mô hình đã huấn luyện (PyTorch)
│   └── best.onnx          cùng mô hình ở định dạng ONNX opset 12 (cho robot và giao diện demo)
├── runs/detect/
│   ├── yolo11n_640/       giai đoạn 1: args.yaml, results.csv, weights/
│   └── yolo11n_640_rot/   giai đoạn 2: args.yaml, results.csv, weights/
├── src/
│   ├── train_yolo11.py    huấn luyện hai giai đoạn
│   ├── evaluate.py        đánh giá trên tập test, xuất ONNX
│   └── predict.py         nhận dạng quân cờ trên ảnh
├── notebooks/             notebook Colab đã chạy, có đầy đủ output và hình
│   ├── train_yolo11.ipynb         chọn mô hình, kiến trúc, huấn luyện, đánh giá, xuất ONNX
│   ├── jetmax_deploy.ipynb        robot JetMax, Jetson Nano, đưa mô hình lên robot
│   ├── board_to_fen.ipynb         từ quân cờ đến thế cờ (FEN)
│   ├── rules_engine.ipynb         luật cờ, tự sửa lỗi nhận dạng, engine Pikafish
│   ├── package.ipynb              đóng gói thư viện
│   ├── image_preprocessing.ipynb  các bước xử lý ảnh hỗ trợ
│   └── README.md
└── xiangqi/               thư viện XiangqiBrain và giao diện demo
    ├── vision/  board/  brain/  config/  models/  samples/  tests/  tools/
    ├── run_demo.bat       mở giao diện demo trên Windows
    ├── API.md             cách gọi thư viện
    └── DEMO_WINDOWS.md    hướng dẫn chạy demo
```

## 3. Vì sao chọn YOLO11n

Bài toán cần **vị trí và loại** của tối đa 32 quân trong một ảnh, nên phải dùng mô hình phát hiện đối tượng. Mô hình phải chạy trên Jetson Nano của robot (GPU 128 nhân, RAM 4 GB, TensorRT 8.0, Python 3.6), nên phải nhẹ.

| Mô hình | mAP@0.5:0.95 (COCO) | Tham số (triệu) | GFLOPs | Nhận xét |
|---|---|---|---|---|
| Faster R-CNN R50-FPN | 37,0 | 41,8 | 134,4 | Nặng gấp khoảng 20 lần |
| EfficientDet-D0 | 34,3 | 3,9 | 2,5 | Nhẹ nhưng kém 5 điểm mAP |
| RT-DETR-R18 | 46,5 | 20 | 60 | Chính xác nhất, quá nặng cho Jetson Nano |
| YOLOv8n | 37,3 | 3,2 | 8,7 | Thế hệ trước |
| **YOLO11n** | **39,5** | **2,6** | **6,5** | **Được chọn** |
| YOLO11s | 47,0 | 9,4 | 21,6 | Dự phòng, chi phí gấp 3,3 lần |

## 4. Kiến trúc mô hình

| Khối | Việc | Thành phần chính |
|---|---|---|
| Backbone | Trích đặc trưng, thu nhỏ ảnh dần (bước 2 đến 32) | Conv 3x3 + BatchNorm + SiLU, C3k2, SPPF, C2PSA |
| Neck | Trộn đặc trưng giữa các tỉ lệ | FPN (trên xuống) + PAN (dưới lên) |
| Head | Dự đoán hộp và lớp ở 3 lưới 80x80, 40x40, 20x20 | Không dùng anchor; hai nhánh riêng cho hộp (DFL) và lớp (sigmoid) |

Mỗi ảnh 640 x 640 cho 80² + 40² + 20² = **8 400 dự đoán**. Đầu ra ONNX là tensor `(1, 18, 8400)`: 4 toạ độ + 14 điểm lớp. Phép lọc hộp trùng NMS không nằm trong mạng; trên robot nó được tự cài đặt trong `xiangqi/vision/pieces.py`.

Sơ đồ kiến trúc: `notebooks/images/yolo11n_architecture.png`.

## 5. Huấn luyện hai giai đoạn

| Giai đoạn | Xuất phát | Thay đổi chính | Lý do |
|---|---|---|---|
| 1: `yolo11n_640` | `yolo11n.pt` (COCO) | 100 epoch, xoay +-10 độ | Bài toán ban đầu: quân lệch góc nhẹ |
| 2: `yolo11n_640_rot` | `best.pt` của giai đoạn 1 | 60 epoch, **xoay +-180 độ**, AdamW, lr0 = 0,001 | Quân nằm xoay bất kỳ |

| Tham số | Giá trị | Lý do |
|---|---|---|
| `imgsz` | 640 | Quân có cạnh trung vị khoảng 48 px, đủ để đọc chữ |
| `batch` | 32 | Vừa bộ nhớ GPU T4 |
| `fliplr`, `flipud` | 0 | Lật làm chữ Hán thành ảnh gương, chữ không có thật |
| `mosaic`, `close_mosaic` | 1,0 và 10 | Ghép 4 ảnh; tắt ở 10 epoch cuối |
| `hsv_h/s/v` | 0,015 / 0,7 / 0,4 | Chịu được ánh sáng khác nhau |
| `patience` | 20 | Dừng sớm nếu không cải thiện |

Hàm mất mát: `L = 7,5 x box (CIoU) + 0,5 x cls (BCE) + 1,5 x dfl`. Tệp `best.pt` là epoch có `fitness = 0,1 x mAP@0.5 + 0,9 x mAP@0.5:0.95` cao nhất trên tập valid.

## 6. Kết quả theo lớp trên tập test

| Lớp | P | R | mAP@0.5 | Lớp | P | R | mAP@0.5 |
|---|---|---|---|---|---|---|---|
| black_advisor | 1,000 | 0,967 | 0,965 | red_advisor | 0,997 | 0,940 | 0,944 |
| black_bishop | 0,985 | 0,965 | 0,962 | red_bishop | 0,996 | 0,950 | 0,964 |
| black_cannon | 0,987 | 0,937 | 0,934 | red_cannon | 0,988 | 0,951 | 0,964 |
| black_king | 0,997 | 0,989 | 0,995 | red_king | 0,991 | 0,953 | 0,955 |
| black_knight | 0,983 | 0,970 | 0,980 | red_knight | 0,985 | 0,978 | 0,980 |
| black_pawn | 0,994 | 0,971 | 0,974 | red_pawn | 0,994 | 0,964 | 0,975 |
| black_rook | 0,985 | 0,942 | 0,952 | red_rook | 0,983 | 0,949 | 0,954 |

Precision cao hơn Recall: mô hình rất ít báo nhầm, lỗi chủ yếu là bỏ sót. Hai lớp dưới 0,95 là `black_cannon` và `red_advisor`; cả hai bị bỏ sót chứ không bị nhận thành quân khác.

## 7. Cách chạy

### 7.1. Cài đặt (một lần)

```bat
cd Nhom2_YeuCau2_Demo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2. Nhận dạng quân cờ trên ảnh (không cần GPU)

```bat
python src\predict.py xiangqi\samples\start_position.jpg
python src\predict.py xiangqi\samples --conf 0.5
```

Chương trình in số quân của từng lớp và lưu ảnh đã vẽ khung vào `runs\detect\predict\`.

### 7.3. Đánh giá lại trên tập test

```bat
python src\evaluate.py
python src\evaluate.py --export
```

Cần cơ sở dữ liệu ở `..\Nhom2_YeuCau1\Nhom2_YeuCau1\`. Lệnh thứ hai xuất thêm `best.onnx` (opset 12). Chạy bằng CPU được nhưng chậm; có GPU thì thêm `--device 0`.

### 7.4. Huấn luyện lại (cần GPU)

```bat
python src\train_yolo11.py --epochs 3
python src\train_yolo11.py
python src\train_yolo11.py --stage 2
```

Lệnh đầu chạy thử 3 epoch để kiểm tra môi trường. Lệnh thứ hai chạy đủ hai giai đoạn (100 + 60 epoch). Kết quả ghi vào `runs\detect\`. Cách thuận tiện hơn là mở `notebooks\train_yolo11.ipynb` trên Google Colab với GPU T4.

### 7.5. Giao diện demo: ảnh bàn cờ - thế cờ - nước đi

```bat
cd xiangqi
run_demo.bat
```

Trình duyệt mở địa chỉ `http://127.0.0.1:7860`. Tải ảnh bàn cờ lên, giao diện hiển thị quân nhận dạng được, thế cờ (FEN), nước đối thủ vừa đi và nước đi gợi ý. Hướng dẫn chi tiết: `xiangqi\DEMO_WINDOWS.md`. Cách gọi thư viện từ chương trình khác: `xiangqi\API.md`.

### 7.6. Kiểm thử thư viện

Bộ kiểm thử nằm trong `xiangqi\tests\` (`test_rules.py`, `test_brain.py`, `selftest.py`). Cách chạy và kết quả mong đợi: xem `xiangqi\RUNBOOK.md`.

## 8. Áp dụng vào bài toán liên ngành: robot JetMax chơi cờ tướng

| Bước của robot | Thành phần | Kỹ thuật |
|---|---|---|
| Nhìn | `vision/pieces.py` | **YOLO11n** nhận dạng quân; trên robot chạy bằng TensorRT FP16 |
| Hiểu | `vision/board_detect.py`, `vision/lattice.py`, `board/fen.py` | Định vị 90 giao điểm của bàn cờ, gán quân vào ô, sinh chuỗi FEN |
| Kiểm tra | `board/rules.py`, `brain/game.py` | Luật cờ phát hiện và tự sửa lỗi nhận dạng |
| Quyết định | `brain/uci.py` | Engine Pikafish (tìm kiếm alpha-beta + mạng NNUE) |
| Hành động | chương trình điều khiển robot | Nhận nước đi dạng ô gắp - ô đặt |

Các ràng buộc của robot quyết định thiết kế:

| Ràng buộc của JetMax | Quyết định |
|---|---|
| GPU nhỏ (khoảng 472 GFLOPS FP16) | Dùng bản nano, 6,5 GFLOPs mỗi ảnh |
| TensorRT 8.0.1.6 | Xuất ONNX opset 12; build `best.engine` trên chính robot bằng `trtexec --fp16` |
| Python 3.6, không cài được Ultralytics | Tự viết letterbox, giải mã đầu ra và NMS |
| Không có Internet | Mang sang robot một tệp `xiangqi_to_robot.zip`, chạy `sh install.sh` |
| Tầm với 104 - 289 mm, bàn cờ 360 x 260 mm | Robot chỉ đi quân ở phần bàn với tới được (tối đa khoảng 70% diện tích) |

Mô hình đi qua ba định dạng: `best.pt` (huấn luyện trên Colab), `best.onnx` (định dạng mở, mang sang robot), `best.engine` (TensorRT biên dịch trên chính Jetson Nano). ONNX khớp PyTorch: sai lệch lớn nhất 1,07e-3, lớp dự đoán trùng 100%.

## 9. Hạn chế

- Mọi số liệu đo trên ảnh công khai; chưa đo trên ảnh từ camera của robot.
- Chưa đo tốc độ TensorRT FP16 trên Jetson Nano.
- Cánh tay không với tới toàn bộ bàn cờ.
