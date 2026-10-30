# xiangqi-sorting-jetmax

Robot JetMax nhìn bàn cờ và **vùng chờ** (quân rải cạnh bàn), nhận diện quân, rồi gắp từng quân đặt lên đúng giao điểm để tạo **thế khai cuộc** hoặc **thế cờ bất kỳ (FEN)**.

**Ngày đầu có robot: làm theo [RUNBOOK.md](RUNBOOK.md).**

## Luồng xử lý

```
tư thế quan sát (1 hay nhiều view, có thể trượt)
  -> ảnh -> [YOLO11 x 4 góc xoay] + [màu chữ HSV] + [tâm bằng Hough circle]  -> quân (px)
  -> H_top (ảnh -> robot, mặt trên quân)                                       -> quân (mm thế giới)
  -> H_table (ảnh -> robot, mặt bàn) -> ghép ảnh nhìn thẳng -> dò lưới         -> 90 giao điểm (mm thế giới)
  -> quân trong / ngoài bàn -> sửa nhãn theo số lượng quân (Hungarian)
  -> ghép ô đích (FEN) <-> quân cùng loại gần nhất -> thứ tự (hàng cuối: a c e g i trước)
  -> gắp / đặt (tự chọn vị trí thanh trượt) -> chụp lại kiểm tra
```

## Cấu trúc

| Đường dẫn | Việc |
|---|---|
| `main.py` | Chương trình chính (`--scan-only`, `--hover`, `--max-moves`, `--fen`, chạy giả lập `--images`) |
| `config/board.yaml` | Kích thước bàn cờ (lưới 220 × 300 mm, sông 30 mm), quân 25 × 10 mm |
| `config/robot.yaml` | Topic ROS, giới hạn an toàn, độ cao, thanh trượt, các view quan sát, model |
| `config/calib.yaml` | Sinh bởi `tools/calibrate.py` (homography mỗi tư thế, z gắp, vector trượt) |
| `board/geometry.py`, `board/fen.py` | Toạ độ 90 giao điểm; đọc FEN |
| `vision/board_detect.py` | Định vị bàn không cần marker (Hough → ghép mẫu lưới → dò từng đường) |
| `vision/pieces.py` | YOLO11 (onnxruntime / cv2.dnn / TensorRT), xoay 4 góc, màu, tâm quân |
| `vision/world.py` | Ảnh → thế giới, ghép nhiều view, phân vùng chờ / trên bàn |
| `vision/calib_math.py` | Tính hiệu chỉnh camera, vector thanh trượt |
| `planner/assign.py` | Hungarian, sửa nhãn theo số lượng quân, kế hoạch và thứ tự đặt |
| `robot/motion.py` | Tay máy + hút + thanh trượt trong hệ thế giới; backend giả lập cho laptop |
| `robot/camera.py`, `robot/jog.py`, `robot/config.py` | Camera ROS / ảnh file; lái bằng phím; đọc config |
| `tools/check_env.py` | Kiểm tra môi trường robot + thử model |
| `tools/build_engine.py` | Build TensorRT engine trên Jetson (không cần mạng, không cần pycuda) |
| `tools/pack.py` | Đóng gói chép qua NoMachine; `--results` gói kết quả mang về |
| `config/robot.local.yaml` | Số đo riêng trên robot, ghi đè `robot.yaml` (mẫu: `robot.local.example.yaml`) |
| `tools/teleop.py` | Lái tay máy, ghi điểm đo |
| `tools/capture.py` | Chụp ở tư thế quan sát, kiểm tra camera thấy bàn |
| `tools/calibrate.py` | Hiệu chỉnh camera (chạm 6-9 quân), đo thanh trượt, kiểm tra |
| `tools/test_board_photo.py` | Thử định vị bàn trên ảnh |
| `training/` | Notebook dựng dataset và train YOLO11n |

## Thử trên laptop (không cần robot)

```bash
pip install numpy opencv-python pyyaml onnxruntime
python3 tools/test_board_photo.py anh.jpg
python3 main.py --images anh.jpg --photo-calib --scan-only        # cần models/best.onnx
python3 main.py --images anh.jpg --photo-calib --scan-only --no-model
```
Kết quả: `runs/<thời gian>/scene.jpg`.

## Quy ước

- Hệ **thế giới** = hệ robot JetMax khi thanh trượt ở 0 (mm). Robot = thế giới − `slider_vec` × bước trượt.
- Hệ **bàn cờ**: gốc a0, u theo cột a → i (27.5 mm), v theo hàng 0 → 9 (33.75 mm, sông 30 mm). Phía Đỏ (hàng 0) là phía gần robot.
- Mã quân theo FEN: `K A B N R C P` (Đỏ), chữ thường cho Đen. Thứ tự 14 lớp model: `training/README.md`.
- Python ≥ 3.6 (Jetson Nano JetPack 4.x), không cần ultralytics/scipy trên robot.
