# xiangqi-sorting-jetmax

Nhận diện thế cờ tướng từ ảnh và đưa ra nước đi tiếp theo, chạy được trên laptop lẫn robot JetMax (Jetson Nano).

- **Thư viện chính `XiangqiBrain`**: ảnh bàn cờ (+ thế cờ trước, FEN) → thế cờ hiện tại → nước đi tiếp theo. Xem [API.md](API.md).
- **Demo robot**: JetMax nhìn bàn và vùng chờ (quân rải cạnh bàn), gắp quân đặt lên đúng giao điểm, hoặc đi nước do thư viện trả về. Xem [RUNBOOK.md](RUNBOOK.md).
- **Demo trên laptop Windows (không cần robot, không cần webcam)**: tải ảnh lên trình duyệt → nước đi. Xem [DEMO_WINDOWS.md](DEMO_WINDOWS.md).

Đồ án môn Xử lý ảnh – Nhóm 2.

## Phần cứng

- Bàn cờ nhựa theo bộ (36 × 26 cm, lưới 220 × 300 mm, sông 30 mm), quân Ø25 × 10 mm.
- Robot JetMax (Jetson Nano, JetPack 4.x), camera gắn trên tay máy, thanh trượt.
- Robot **không có Internet**: mọi thứ được đóng gói trên laptop rồi chép qua NoMachine.
- Tay máy không với tới toàn bộ bàn, nên demo robot giới hạn ở các nước đầu phía một bên sông.

## Quy trình (chuỗi notebook trong `training/`)

| Bước | Notebook | Kết quả |
|---|---|---|
| 1 | Dựng dataset (gộp các dataset Roboflow) | Dataset 14 lớp quân |
| 2 | Train YOLO11n | `best.pt` / `best.onnx` |
| 3 | `jetmax_deploy.ipynb` | **Một thư mục thư viện duy nhất** đóng gói sẵn, chép lên robot một lần rồi chạy `install.sh` |

## Luồng xử lý

**Ảnh → nước đi (`XiangqiBrain`)**

```
ảnh -> [YOLO11 x 4 góc xoay] + [màu chữ HSV] + [tâm bằng Hough circle]  -> quân (px)
    -> khớp lưới 9 x 10 (định vị bàn hoặc lattice từ tâm quân)          -> thế cờ (FEN)
    -> so với thế trước: suy ra nước vừa đi, sửa lỗi nhận diện bằng luật cờ
    -> Pikafish (UCI) / engine Python dự phòng                          -> nước đi tiếp theo
```

**Robot gắp / đặt**

```
tư thế quan sát (1 hay nhiều view, có thể trượt)
  -> quân (px) -> H_top (mặt trên quân)                        -> quân (mm thế giới)
  -> H_table (mặt bàn) -> ghép ảnh nhìn thẳng -> dò lưới       -> 90 giao điểm (mm thế giới)
  -> quân trong / ngoài bàn -> sửa nhãn theo số lượng quân (Hungarian)
  -> ghép ô đích (FEN) <-> quân cùng loại gần nhất -> thứ tự (hàng cuối: a c e g i trước)
  -> gắp / đặt (tự chọn vị trí thanh trượt) -> chụp lại kiểm tra
```

## Cấu trúc

**Thư viện (ảnh → nước đi)**

| Đường dẫn | Việc |
|---|---|
| `brain/xiangqi_brain.py` | **`XiangqiBrain.analyze()` / `step()`**: ảnh (+ thế trước) → nước đi tiếp theo |
| `brain/uci.py`, `brain/simple.py`, `brain/game.py` | Điều khiển Pikafish qua UCI; engine dự phòng Python; `GameTracker` cho ván người – robot |
| `board/rules.py` | Luật cờ: nước hợp lệ, perft, kiểm tra thế cờ, suy ra nước vừa đi, ký hiệu tiếng Việt |
| `board/geometry.py`, `board/fen.py` | Toạ độ 90 giao điểm; đọc FEN |
| `vision/pieces.py` | YOLO11 (onnxruntime / cv2.dnn / TensorRT), xoay 4 góc, màu, tâm quân |
| `vision/board_detect.py` | Định vị bàn không cần marker (Hough → ghép mẫu lưới → dò từng đường) |
| `vision/lattice.py` | Khớp lưới 9 × 10 từ tâm quân, chốt hướng bằng luật cờ (không cần biết kiểu bàn) |

**Robot**

| Đường dẫn | Việc |
|---|---|
| `main.py` | Chương trình gắp / đặt (`--scan-only`, `--hover`, `--max-moves`, `--fen`, giả lập `--images`) |
| `vision/world.py` | Ảnh → thế giới, ghép nhiều view, phân vùng chờ / trên bàn |
| `vision/calib_math.py` | Tính hiệu chỉnh camera, vector thanh trượt |
| `planner/assign.py` | Hungarian, sửa nhãn theo số lượng quân, kế hoạch và thứ tự đặt |
| `robot/motion.py` | Tay máy + hút + thanh trượt trong hệ thế giới; backend giả lập cho laptop |
| `robot/camera.py`, `robot/jog.py`, `robot/config.py` | Camera ROS / ảnh file; lái bằng phím; đọc config |

**Cấu hình**

| Đường dẫn | Việc |
|---|---|
| `config/board.yaml` | Kích thước bàn và quân |
| `config/robot.yaml` | Topic ROS, giới hạn an toàn, độ cao, thanh trượt, các view quan sát, model |
| `config/robot.local.yaml` | Số đo riêng trên robot, ghi đè `robot.yaml` (mẫu: `robot.local.example.yaml`) |
| `config/calib.yaml` | Sinh bởi `tools/calibrate.py` (homography mỗi tư thế, z gắp, vector trượt) |

**Công cụ**

| Đường dẫn | Việc |
|---|---|
| `install.sh` | Cài trên robot sau khi chép gói: build `best.engine`, kiểm tra Pikafish, tự kiểm tra |
| `tools/demo_app.py`, `run_demo.bat` | Giao diện demo trong trình duyệt (Windows / Colab) |
| `tools/brain_demo.py` | Chạy thử thư viện trên ảnh hoặc camera robot |
| `tools/check_env.py` | Kiểm tra môi trường robot + thử model |
| `tools/build_engine.py` | Build TensorRT engine trên Jetson (không cần mạng, không cần pycuda) |
| `tools/build_pikafish.sh` | Build Pikafish cho Jetson (aarch64, link tĩnh) trên máy có mạng |
| `tools/bench_engine.py` | Đo tốc độ engine trên robot |
| `tools/pack.py` | Đóng gói chép qua NoMachine; `--results` gói kết quả mang về |
| `tools/teleop.py`, `tools/capture.py` | Lái tay máy, ghi điểm đo; chụp ở tư thế quan sát |
| `tools/calibrate.py` | Hiệu chỉnh camera (chạm 6–9 quân), đo thanh trượt, kiểm tra |
| `tools/test_board_photo.py` | Thử định vị bàn trên ảnh |
| `tests/test_rules.py`, `tests/test_brain.py` | Tự kiểm tra luật cờ + engine; `XiangqiBrain` trên ảnh mẫu |

## Thử nhanh trên laptop

```bash
pip install numpy opencv-python pyyaml onnxruntime

# Thư viện: ảnh -> nước đi
python3 tools/brain_demo.py anh.jpg
run_demo.bat                                   # Windows: mở giao diện trình duyệt

# Phần robot, chạy giả lập trên ảnh
python3 tools/test_board_photo.py anh.jpg
python3 main.py --images anh.jpg --photo-calib --scan-only        # cần models/best.onnx
python3 main.py --images anh.jpg --photo-calib --scan-only --no-model
```
Kết quả giả lập: `runs/<thời gian>/scene.jpg`.

## Quy ước

- Hệ **thế giới** = hệ robot JetMax khi thanh trượt ở 0 (mm). Robot = thế giới − `slider_vec` × bước trượt.
- Hệ **bàn cờ**: gốc a0, u theo cột a → i (27.5 mm), v theo hàng 0 → 9 (33.75 mm, sông 30 mm). Phía Đỏ (hàng 0) gần robot.
- Mã quân theo FEN: `K A B N R C P` (Đỏ), chữ thường cho Đen. Thứ tự 14 lớp model: `training/README.md`.
- Python ≥ 3.6 trên robot, không cần ultralytics/scipy.

## Thành viên – Nhóm 2

| Thành viên | GitHub |
|---|---|
| Trọng | |
| Nguyễn Gia Khiêm | [@NgKhiem69](https://github.com/NgKhiem69) |
| Huỳnh Thanh Trình | [@ThanhTrinh31](https://github.com/ThanhTrinh31) |
