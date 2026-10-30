# Huấn luyện mô hình nhận diện quân cờ tướng (YOLO11n)

Mô hình nhận diện **14 lớp** quân cờ (7 loại × 2 màu), chạy trên Jetson Nano của robot JetMax.

## Notebook

| Notebook | Việc |
|---|---|
| `build_dataset.ipynb` | Tải 4 bộ dataset từ Roboflow Universe, đổi tên lớp về 14 lớp chuẩn, chuyển polygon/pose về bbox, gom ảnh trùng (phash + union-find), chia train/val/test theo nhóm (StratifiedGroupKFold, ~80/10/10), xuất `merged_v1.zip` |
| `train_yolo11.ipynb` | Huấn luyện, đánh giá, so sánh trên tập test xoay, xuất ONNX |

Chạy trên Google Colab (GPU T4), dữ liệu và kết quả lưu ở Drive `MyDrive/datn/`.

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

## Dữ liệu (`merged_v1`)

| Nguồn | train | val | test |
|---|---|---|---|
| super (312s-workspace/xiangqi-super-dataset) | 1841 | 254 | 223 |
| scara (312s-workspace/xiangqi-scara-yolo26) | 1423 | 151 | 160 |
| new_chinese_chess (quoc-rnmzr, v1) | 1005 | 124 | 124 |
| duytranle (duytranle/xiangqi-ubjij, v18) | 1200 | 146 | 157 |
| **Tổng** | **5469** | **675** | **664** |

## Các lần huấn luyện

| Run | Xuất phát | Thay đổi chính | Lý do |
|---|---|---|---|
| `yolo11n_640` | `yolo11n.pt` (COCO) | 100 epoch, `degrees=10`, `fliplr=0` | Bài toán ban đầu: quân trên băng chuyền, lệch góc nhẹ |
| `yolo11n_640_rot` | `yolo11n_640/best.pt` | 60 epoch, **`degrees=180`**, `flipud=0`, AdamW `lr0=0.001` | Bài toán mới: quân trong khay nằm xoay bất kỳ 0–360° |

Không lật ảnh (`fliplr=0`, `flipud=0`) vì lật biến chữ Hán thành ảnh gương. Xoay thì không đảo chữ nên dùng được tới 180°.

## Mang sang robot

- `weights/best.onnx` (opset 12) → build TensorRT engine **trên chính Jetson của robot**:
  `/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16`
- Ultralytics không chạy trên Python 3.6 (JetPack 4.x), phía robot dùng TensorRT/onnxruntime trực tiếp.
