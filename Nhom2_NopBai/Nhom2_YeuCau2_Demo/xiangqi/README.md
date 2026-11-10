# xiangqi-sorting-jetmax

Thư viện **XiangqiBrain**: nhận ảnh bàn cờ tướng (kèm thế cờ lượt trước), trả về nước đi tiếp theo dưới dạng ô gắp → ô đặt. Thư viện chạy trên robot JetMax (Jetson Nano) hoặc trên laptop qua giao diện demo.

- **Gọi thư viện**: [API.md](API.md)
- **Demo trên laptop Windows** (không cần robot, không cần webcam, tải ảnh lên trình duyệt): [DEMO_WINDOWS.md](DEMO_WINDOWS.md)
- **Ngày đầu có robot**: [RUNBOOK.md](RUNBOOK.md)
- **Notebook, dữ liệu, số liệu chi tiết**: [training/README.md](training/README.md)

Đồ án môn Xử lý ảnh – Nhóm 2.

## Kết quả chính

| | |
|---|---|
| Mô hình | YOLO11n, 14 lớp, 2.6 M tham số. mAP@0.5 = **0.964** trên 713 ảnh test, giữ ~0.96 khi xoay 90/180/270° |
| Ảnh → FEN đúng hoàn toàn | 78.2% (229 ảnh test có bàn cờ) |
| Biết thế lượt trước → suy ra đúng nước đi | **95.2%**; 4.8% còn lại báo chụp lại, **0%** sai mà không phát hiện |
| Luật cờ | perft 1–4 từ khai cuộc khớp giá trị chuẩn |

## Luồng xử lý

```
ảnh -> tăng cường tự động (khử nhiễu / Gray-World / CLAHE, chỉ khi cần)
    -> [YOLO11 x 4 góc xoay] + [màu chữ HSV] + [tâm bằng Hough circle]  -> quân (px)
    -> định vị bàn (mặt nạ đỏ -> Hough -> homography -> 90 giao điểm)
       hoặc dự phòng: khớp lưới 9 x 10 từ tâm quân (lattice)            -> thế cờ (FEN)
    -> so với thế lượt trước: suy ra nước vừa đi, tự sửa lỗi nhận dạng bằng luật cờ
       (không chắc -> yêu cầu chụp lại)
    -> Pikafish (UCI) / engine Python dự phòng                         -> nước đi: ô gắp -> ô đặt
```

Trên robot, phần gắp / đặt (`main.py`) đi tiếp:

```
toạ độ ảnh -> H_top / H_table -> mm thế giới -> quân trong / ngoài bàn (Hungarian)
           -> thứ tự gắp -> gắp / đặt (tự chọn vị trí thanh trượt) -> chụp lại kiểm tra
```

## Notebook (`training/`)

Mọi notebook chạy trên Google Colab, dữ liệu lưu ở Drive `MyDrive/datn/`.

```
Bước 1 build_dataset ──► Bước 2 train_yolo11 ──► Bước 3 jetmax_deploy
                                   │
                                   └──► Bước 4 board_to_fen ──► Bước 5 rules_engine ──► Bước 6 package

image_preprocessing (minh hoạ xử lý ảnh, chạy độc lập)
```

| Notebook | Việc | Đầu ra |
|---|---|---|
| `build_dataset.ipynb` | Gộp 4 bộ Roboflow, chuẩn hoá 14 lớp, gom ảnh trùng, chia tập theo nhóm | `merged_v1.zip` (6 808 ảnh) |
| `train_yolo11.ipynb` | Huấn luyện YOLO11n hai giai đoạn (giai đoạn 2 xoay 180°), xuất ONNX | `best.pt`, `best.onnx` |
| `jetmax_deploy.ipynb` | Giới thiệu robot, tầm với, kiểm tra ONNX, hướng dẫn đưa lên robot | — |
| `board_to_fen.ipynb` | Khớp lưới 9×10, đánh giá ảnh → FEN, quét ngưỡng | Số liệu, hình |
| `rules_engine.ipynb` | Luật cờ, perft, suy ra nước vừa đi; build Pikafish cho Jetson | `pikafish_jetson.tar.gz` |
| `package.ipynb` | Kiểm thử `XiangqiBrain`, đóng gói | `xiangqi_to_robot.zip`, `xiangqi_demo_windows.zip` |
| `image_preprocessing.ipynb` | Minh hoạ từng bước xử lý ảnh trên ảnh trong `samples/` | Hình |

## Chạy nhanh

**Laptop Windows**: giải nén `xiangqi_demo_windows.zip`, chạy `run_demo.bat`.

**Laptop / Colab (Python)**

```bash
pip install numpy opencv-python pyyaml onnxruntime
python3 tools/brain_demo.py samples/start_position.jpg   # ảnh -> FEN -> nước đi
python3 tools/demo_app.py --share                        # giao diện web (Colab)
```

**Robot** (không có Internet, mọi file đi qua NoMachine). Làm đúng thứ tự trong [RUNBOOK.md](RUNBOOK.md):

| Bước | Lệnh / việc | Xong khi |
|---|---|---|
| 0. Trên laptop | `python tests/selftest.py`, rồi `python tools/pack.py` (hoặc `training/package.ipynb`) | Có `dist/xiangqi_to_robot.zip` |
| 1. Chép sang robot | Kéo thả zip qua NoMachine, rồi `python3 -m zipfile -e ~/Desktop/xiangqi_to_robot.zip ~/` và `cd ~/xiangqi && sh install.sh` | `install.sh` chạy xong |
| 2. Kiểm tra môi trường | `python3 tools/check_env.py`; nếu không nạp được model: `python3 tools/build_engine.py` | `[OK] nạp model`, selftest ra `TẤT CẢ OK` |
| 3. Đo robot | `python3 tools/teleop.py --wide`, ghi giới hạn vào `config/robot.local.yaml` | — |
| 4. Tư thế quan sát | `python3 tools/capture.py --arm 0 -160 210 --slider 0 --name thu1` | Thấy đủ 4 góc bàn và vùng chờ |
| 5. Hiệu chỉnh camera | `python3 tools/calibrate.py`, `--slider`, `--verify` | `rms_top_mm` < 1.5, bay trúng tâm mọi quân |
| 6. Kiểm tra nhìn | `python3 main.py --scan-only` | `runs/<mới nhất>/scene.jpg` đúng 32 quân |
| 7. Chạy không hút | `python3 main.py --hover --max-moves 3` | Đầu hút đúng tâm |
| 8. Chạy thật | `python3 main.py --max-moves 1` → `python3 main.py` → `python3 main.py --fen "<FEN>"` | `KIỂM TRA: đúng x/32 ô` |
| 9. Mang kết quả về | `python3 tools/pack.py --results` | Có `dist/results_<giờ>.zip` |

Chơi cả ván với thư viện: `python3 tools/brain_demo.py --camera` (người đi quân giúp robot).

Trên robot **chỉ sửa `config/robot.local.yaml`**; chép code mới sang không làm mất `robot.local.yaml`, `calib.yaml`, `best.engine`. Tay máy không với tới toàn bộ bàn, nên demo gắp / đặt giới hạn ở các nước đầu phía một bên sông.

## Cấu trúc

**Thư viện**

| Đường dẫn | Việc |
|---|---|
| `brain/xiangqi_brain.py` | **`XiangqiBrain.analyze()` / `step()`**: ảnh (+ thế trước) → nước đi tiếp theo |
| `brain/uci.py`, `brain/simple.py`, `brain/game.py` | Pikafish qua UCI; engine dự phòng Python; `GameTracker` cho ván người – robot |
| `board/rules.py` | Luật cờ: nước hợp lệ, perft, kiểm tra thế cờ, suy ra nước vừa đi, ký hiệu tiếng Việt |
| `board/geometry.py`, `board/fen.py`, `board/draw.py` | Toạ độ 90 giao điểm; đọc FEN; vẽ bàn cờ |
| `vision/pieces.py` | YOLO11 (onnxruntime / cv2.dnn / TensorRT), letterbox + NMS tự cài, xoay 4 góc, màu quân, tâm quân |
| `vision/board_detect.py` | Định vị bàn không cần marker (mặt nạ đỏ → Hough → homography → dò từng đường) |
| `vision/lattice.py` | Khớp lưới 9 × 10 từ tâm quân, chốt hướng bằng luật cờ (dự phòng) |
| `vision/enhance.py` | Tăng cường ảnh tự động: lọc song phương, Gray-World, CLAHE, gamma |

**Robot**

| Đường dẫn | Việc |
|---|---|
| `main.py` | Gắp / đặt (`--scan-only`, `--hover`, `--max-moves`, `--fen`, giả lập `--images`) |
| `vision/world.py`, `vision/calib_math.py` | Ảnh → thế giới, ghép nhiều view, vùng chờ / trên bàn; hiệu chỉnh camera, thanh trượt |
| `planner/assign.py` | Hungarian, sửa nhãn theo số lượng quân, thứ tự đặt |
| `robot/motion.py` | Tay máy + hút + thanh trượt; backend giả lập cho laptop |
| `robot/camera.py`, `robot/jog.py`, `robot/config.py` | Camera ROS / ảnh file; lái bằng phím; đọc config |

**Cấu hình**

| Đường dẫn | Việc |
|---|---|
| `config/board.yaml` | Kích thước bàn (lưới 220 × 300 mm, sông 30 mm) và quân (Ø25 × 10 mm) |
| `config/robot.yaml` | Topic ROS, giới hạn an toàn, độ cao, thanh trượt, view quan sát, model, `brain.enhance` |
| `config/robot.local.example.yaml` | Mẫu số đo riêng trên robot; chép thành `robot.local.yaml` để ghi đè |
| `config/calib.yaml` | Sinh ra trên robot bởi `tools/calibrate.py` |

**Công cụ và kiểm thử**

| Đường dẫn | Việc |
|---|---|
| `install.sh` | Cài trên robot: build `best.engine`, kiểm tra Pikafish, tự kiểm tra |
| `run_demo.bat`, `tools/demo_app.py` | Giao diện demo trong trình duyệt (tải ảnh lên → thế cờ → nước đi) |
| `tools/brain_demo.py` | Chạy thư viện trên ảnh hoặc camera robot |
| `tools/check_env.py`, `tools/bench_engine.py` | Kiểm tra môi trường robot; đo tốc độ engine |
| `tools/build_engine.py`, `tools/build_pikafish.sh` | Build TensorRT engine trên Jetson; build Pikafish aarch64 |
| `tools/pack.py` | Đóng gói chép qua NoMachine; `--results` gói kết quả mang về |
| `tools/teleop.py`, `tools/capture.py`, `tools/calibrate.py` | Lái tay máy; chụp ở tư thế quan sát; hiệu chỉnh camera và thanh trượt |
| `tools/test_board_photo.py` | Thử định vị bàn trên ảnh |
| `tests/` | `selftest.py` (tự kiểm tra tổng, không di chuyển robot, phải ra `TẤT CẢ OK`), `test_rules.py` (luật cờ + engine), `test_brain.py` (`XiangqiBrain` trên ảnh mẫu) |

**Khác**

| Đường dẫn | Việc |
|---|---|
| `models/` | `best.onnx` (laptop) / `best.engine` (robot, do `install.sh` build) |
| `samples/` | Ảnh thật: `board_empty.jpg`, `start_position.jpg`, `disorganized_location.png` |
| `docs/` | Đề cương, báo cáo; `robot-reference/` là mã gốc của JetMax (`jetmax_control`) để tham khảo |

## Quy ước

- Bàn cờ nhựa theo bộ (36 × 26 cm). Hệ **bàn cờ**: gốc a0, cột a → i cách 27.5 mm, hàng 0 → 9 cách 33.75 mm (sông 30 mm). Phía Đỏ (hàng 0) gần robot.
- Hệ **thế giới** = hệ robot JetMax khi thanh trượt ở 0 (mm). Robot = thế giới − `slider_vec` × bước trượt.
- Mã quân theo FEN: `K A B N R C P` (Đỏ), chữ thường cho Đen. Thứ tự 14 lớp: [training/README.md](training/README.md).
- Robot: Python 3.6 (JetPack 4.x), không cần ultralytics/scipy; suy luận bằng TensorRT / onnxruntime trực tiếp.

## Thành viên – Nhóm 2

| Thành viên | GitHub |
|---|---|
| Trọng | |
| Nguyễn Gia Khiêm | [@NgKhiem69](https://github.com/NgKhiem69) |
| Huỳnh Thanh Trình | [@ThanhTrinh31](https://github.com/ThanhTrinh31) |
