# RUNBOOK: ngày đầu có robot (robot KHÔNG có Internet)

Robot không vào mạng được: **không `git`, không `pip install`**. Mọi file đi qua **NoMachine** (kéo thả).
Code đã viết để chạy với những gì có sẵn trên Jetson: Python 3, OpenCV, numpy, ROS, TensorRT.

Làm **đúng thứ tự**; mỗi bước có tiêu chí "xong khi". Lỗi thì xem mục Xử lý sự cố ở cuối.

---

## 0. Chuẩn bị TRÊN LAPTOP (hôm nay)

- [ ] `models/best.onnx` là model mới (`yolo11n_640_rot`, opset 12). Xóa `best_op12.onnx` và `best.engine` cũ.
- [ ] Chạy thử: `python tests/selftest.py` → phải ra `TẤT CẢ OK`.
- [ ] Đóng gói:
  ```powershell
  python tools/pack.py
  ```
  → tạo `dist/xiangqi_to_robot.zip` (~10 MB: code + model + ảnh mẫu).
- [ ] Bàn cờ dán phẳng lên bìa cứng. Mang thước kẻ, băng dính, 32 quân.

## 1. Chép code sang robot (5 phút)

1. Mở NoMachine, kết nối robot.
2. **Kéo thả** `dist/xiangqi_to_robot.zip` vào cửa sổ NoMachine. File thường nằm ở `~/Desktop` hoặc `~/Downloads` của robot. Nếu kéo thả không được: menu NoMachine (Ctrl+Alt+0) → *Devices* → *Transfer files* → *Upload*.
3. Mở Terminal trên robot (trong NoMachine):
   ```bash
   cd ~ && python3 -m zipfile -e ~/Desktop/xiangqi_to_robot.zip ~/
   cd ~/xiangqi
   cp config/robot.local.example.yaml config/robot.local.yaml     # CHỈ lần đầu
   ```
   Không cần lệnh `unzip`. Python tự giải nén vào `~/xiangqi`.

**Quy tắc:** trên robot **chỉ sửa `config/robot.local.yaml`**, không sửa `config/robot.yaml`. Mỗi khi chép code mới từ laptop sang (lặp lại bước 1.2–1.3, bỏ lệnh `cp`), các file `robot.local.yaml`, `calib.yaml`, `best.engine` trên robot **được giữ nguyên**.

Sửa file trên robot: `gedit config/robot.local.yaml &` (cửa sổ soạn thảo trong NoMachine), hoặc `nano`.

## 2. Kiểm tra môi trường (10–25 phút)

```bash
cd ~/xiangqi
python3 tools/check_env.py
python3 tests/selftest.py        # phải ra TẤT CẢ OK (không di chuyển robot)
```

### 2a. ROS / camera

| `[LỖI]` | Cách sửa |
|---|---|
| node jetmax_control không chạy | Khởi động lại robot (node tự chạy khi bật), hoặc `roslaunch jetmax_control jetmax_control.launch` |
| `/usb_cam/image_rect_color` không có | `roslaunch usb_cam usb_cam-test.launch` (hoặc file launch camera của Hiwonder) |
| `jetmax_control.msg` lỗi | `source ~/ros/devel/setup.bash` (đường dẫn workspace ROS trên máy có thể khác) |
| `yaml` thiếu | rospy cần yaml nên gần như chắc chắn có sẵn. Nếu vẫn thiếu, báo mình |

### 2b. Model đã chạy được chưa?

Xem dòng `== Thử model` của `check_env.py`:
- `[OK] nạp model ... backend onnxruntime` hoặc `cv2.dnn` → dùng luôn, sang bước 3. Nếu chạy chậm (> 5 s/ảnh) thì vẫn nên làm 2c.
- `[LỖI] model: Không nạp được model` → **làm 2c**. Đây là trường hợp hay gặp nhất: OpenCV 4.1 của JetPack không đọc được YOLO11, và robot không có onnxruntime.

### 2c. Build TensorRT engine (chắc chạy, không cần mạng, ~15 phút)

```bash
python3 tools/build_engine.py            # models/best.onnx -> models/best.engine (FP16)
python3 tools/check_env.py               # phải thấy: backend tensorrt (ctypes cudart)
```
Code tự dùng `models/best.engine` nếu có. **Không cần pycuda.** Trong lúc chờ build, làm song song bước 3.

Lỗi build → `python3 tools/build_engine.py --fp32`. Mỗi lần đổi `best.onnx` thì phải build lại; `check_env.py` sẽ báo nếu engine không khớp model.

**Xong khi** `check_env.py` in `[OK] nạp model` và `tests/selftest.py` ra `TẤT CẢ OK`.

## 3. Đo robot bằng tay (15 phút)

```bash
python3 tools/teleop.py --wide
```
Phím: `w/a/s/d` = x, y   `r/f` = lên/xuống   `x/z` = trượt   `t` = đổi bước   Enter = ghi điểm vào `captures/teleop_points.txt`   `q` = thoát.

Ghi vào **`config/robot.local.yaml`** (xem ví dụ có sẵn trong file):
- [ ] `limits.x`, `limits.y`, `limits.r_max`: đưa đầu hút ra xa nhất / gần nhất ở z = -10 và z = 50. Lấy giá trị **thu vào ~10 mm** so với mức tay còn với tới.
- [ ] Thanh trượt: bấm `x` vài lần, xem robot trượt theo hướng nào. Đặt bàn cờ **dọc theo hướng trượt**. Robot không có thanh trượt → `slider: {enabled: false}`.
- [ ] z chạm mặt quân: hạ đầu hút chạm nhẹ một quân, đọc z (bước 5 sẽ đo lại chính xác).

## 4. Chọn tư thế quan sát (15 phút)

Đặt bàn cờ cạnh robot, vùng chờ là một dải ~10 × 40 cm dọc cạnh dài của bàn.

```bash
python3 tools/capture.py --arm 0 -160 210 --slider 0 --name thu1
```
Mở `captures/thu1.jpg` (nhấp đúp trong trình quản lý file của robot). Tăng z tới mức cao nhất tay còn lên được, chỉnh x, y cho camera nhìn thẳng xuống bàn.

- Nếu **một ảnh thấy hết** bàn cờ và vùng chờ: dùng 1 view.
- Nếu không: dùng **cùng một tư thế tay**, thêm view ở vị trí trượt khác (ví dụ 0 và 2500), các view chồng lấn khoảng 1/4.

Ghi vào `config/robot.local.yaml` → `views:`, rồi `python3 tools/capture.py` để chụp lại tất cả.

**Xong khi** các ảnh gộp lại thấy đủ 4 góc bàn cờ và toàn bộ vùng chờ.

## 5. Hiệu chỉnh camera (15 phút / tư thế tay)

1. Đặt **6–9 quân đúng giữa giao điểm**, rải đều khắp khung hình: 4 góc ảnh, giữa, các cạnh.
2. `python3 tools/calibrate.py`
3. Mở `captures/calib_pose_discs.jpg` để biết thứ tự số. Với từng quân: lái đầu hút **chạm nhẹ đúng tâm** mặt quân → Enter. Từ quân thứ 4, tay tự bay tới gần, chỉ cần chỉnh 1–2 mm (bước 1 mm). **Đừng làm quân xê dịch.**
4. Đọc kết quả:
   - `rms_top_mm` **< 1.5** (tốt < 1.0). Có điểm `per_point_mm` > 2.5 → làm lại.
   - `board_scale` ≈ 1.00 (±0.02). Lệch nhiều → kích thước trong `config/board.yaml` chưa đúng.
   - `top_vs_table_mm` khoảng 0–6.
5. Có thanh trượt: `python3 tools/calibrate.py --slider` (để vài quân trong ảnh).
6. Kiểm tra: `python3 tools/calibrate.py --verify`. Tay bay tới trên từng quân, đầu hút phải nằm **đúng tâm** (nhìn ngang).

Kết quả lưu ở `config/calib.yaml` trên robot.

**Xong khi** `--verify` bay trúng tâm mọi quân.

## 6. Kiểm tra nhìn (5 phút)

Rải 32 quân ở vùng chờ (một lớp, ngửa mặt, cách nhau ≥ 5 mm), bàn cờ để trống.
```bash
python3 main.py --scan-only
```
Mở `runs/<mới nhất>/scene.jpg`. Kiểm tra: lưới bàn khớp, đủ 32 quân, nhãn đúng, **a0 ở bên trái phía Đỏ** (phía gần robot).
- a0 nằm bên phải → thêm `board: {mirror: true}` vào `robot.local.yaml`.
- Phía Đỏ bị đảo đầu → sửa `board.red_near_world` thành một điểm gần phía muốn làm Đỏ.
- Nhìn chậm (> 5 s/view) → `detector: {tta_rotations: [0]}` (model mới đã chịu được quân xoay).

## 7. Chạy thử không hút (5 phút)

```bash
python3 main.py --hover --max-moves 3
```
Tay bay tới trên quân rồi tới trên ô, hạ xuống còn cách 15 mm. Nhìn ngang: phải **đúng tâm**.

## 8. Chạy thật

```bash
python3 main.py --max-moves 1        # 1 quân
python3 main.py --max-moves 5
python3 main.py                      # cả bàn, khai cuộc
python3 main.py --fen "<FEN>"        # thế cờ bất kỳ
```
Trước khi chạy, chương trình in bảng quân + kế hoạch rồi chờ:
- `Enter`: chạy.
- `#5=N`: sửa nhãn quân #5 thành Mã Đỏ (mã FEN: `K A B N R C P` cho Đỏ, chữ thường cho Đen).
- `r`: nhìn lại.
- `q`: thoát.

**Khi demo không dùng `--yes`**: màn hình xác nhận là lớp bảo vệ cuối nếu model nhận nhầm.

Cuối mỗi lần chạy, robot chụp lại và in `KIỂM TRA: đúng x/32 ô, lệch tâm TB ... mm`. Mọi thứ lưu trong `runs/<thời gian>/`.

## 9. Mang kết quả về laptop (cuối buổi)

```bash
python3 tools/pack.py --results      # -> dist/results_<giờ>.zip (runs/, captures/, config/)
```
Kéo file từ robot về laptop qua NoMachine (*Transfer files* → *Download*). Trên laptop, chép `config/robot.local.yaml` và `config/calib.yaml` từ gói này vào repo rồi commit, để giữ số đo cho lần sau.

Gặp lỗi cần mình xem: gửi `runs/<lần chạy>/log.txt` + `scene.jpg`, hoặc cả file `results_*.zip`.

## 10. Số liệu cho báo cáo

- **Nhận diện:** 10 lần rải quân ngẫu nhiên + `main.py --scan-only`. Đếm số quân đúng nhãn trước (cột tin cậy) và sau sửa theo số lượng (log `sửa nhãn theo số lượng quân`).
- **Định vị đặt quân:** dòng `KIỂM TRA ... lệch tâm TB/max` sau mỗi lần xếp.
- **Thời gian:** log có giờ từng bước.

---

## Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách sửa |
|---|---|
| `Không nạp được model` | Bước 2c (build engine) |
| `engine ... KHÔNG khớp model` | Đã đổi `best.onnx`: `python3 tools/build_engine.py` |
| `Tay máy chưa tới đích (lệch ...)` | Điểm ngoài tầm động học: thu hẹp `limits`, đặt bàn gần robot hơn |
| `ngoài tầm với (kể cả khi trượt)` | Dời bàn hoặc vùng chờ lại gần |
| Không định vị được bàn cờ | Xem `runs/.../scene.jpg`: bàn có lọt các view không, có bị lóa không |
| Hút không lên quân | Thêm `heights: {z_pick: <giảm 0.5-1 mm>}` vào `robot.local.yaml` hoặc hiệu chỉnh lại; tăng `suck_wait` |
| Đặt quân bị nảy / lệch | Tăng `place_gap` nếu quân đập xuống, giảm nếu thả từ cao |
| Đặt lệch đều một phía | Hiệu chỉnh lại (bước 5); quân hiệu chỉnh có nằm đúng giao điểm không |
| Lệch tăng dần theo hướng trượt | Đo lại `calibrate.py --slider` |
| Nhận nhầm nhiều quân | Kiểm tra ánh sáng (không lóa), thử `detector: {conf: 0.25}` |
| Thanh trượt không chạy | `slider: {enabled: false}` nếu robot không có thanh trượt |
| Chép code mới làm mất số đo | Số đo phải nằm trong `robot.local.yaml` / `calib.yaml` (không bị ghi đè), không phải `robot.yaml` |
| Ctrl+C khi đang chạy | Robot tắt hút, nâng tay lên, về chỗ cũ |
