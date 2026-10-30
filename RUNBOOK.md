# RUNBOOK: ngày đầu có robot

Làm **đúng thứ tự**. Mỗi bước có tiêu chí "xong khi" rõ ràng. Gặp lỗi thì xem mục Xử lý sự cố ở cuối.
Thời gian ước tính: khoảng 2 giờ cho bước 1-7, sau đó chạy thật.

---

## 0. Chuẩn bị TRƯỚC khi tới (hôm nay)

- [ ] Code: `git push` repo, hoặc chép thư mục `xiangqi/` vào USB.
- [ ] Model: chép `best.onnx` (bản train lại nếu xong, không thì bản cũ `best_op12.onnx`) vào `models/best.onnx`.
- [ ] Bàn cờ: dán phẳng lên bìa cứng hoặc tấm xốp. Nếp gấp gây lệch vài mm, mà khe giữa 2 quân chỉ 2.5 mm.
- [ ] Mang: thước kẻ, băng dính, 32 quân, laptop có SSH/VNC.
- [ ] Trên laptop, chạy thử để chắc code không lỗi:
  `python3 tests/selftest.py` (phải ra `TẤT CẢ OK`), rồi
  `python3 main.py --images samples/start_position.jpg --photo-calib --hint 1280 480 --scan-only`

## 1. Kết nối và kiểm tra môi trường (10 phút)

```bash
scp -r xiangqi jetson@<IP>:~/          # hoặc git clone
ssh jetson@<IP>
cd ~/xiangqi
python3 tools/check_env.py
python3 tests/selftest.py        # phải ra TẤT CẢ OK (không di chuyển robot)
```

**Xong khi** mọi dòng `[LỖI]` được xử lý:

| Lỗi | Cách sửa |
|---|---|
| node jetmax_control không chạy | Bật lại robot, hoặc `roslaunch jetmax_control jetmax_control.launch` |
| `/usb_cam/image_rect_color` không có | `roslaunch usb_cam usb_cam-test.launch` (hoặc launch camera của Hiwonder) |
| `jetmax_control.msg` lỗi | `source ~/ros/devel/setup.bash` (thêm vào `~/.bashrc`) |
| `yaml` thiếu | `pip3 install pyyaml` |
| Không có backend model nào | Xem mục 1b |

### 1b. Chạy model trên Jetson (chọn cách đầu tiên chạy được)

1. **onnxruntime** có sẵn → dùng ngay `models/best.onnx`.
2. **cv2.dnn** (OpenCV ≥ 4.5 đọc được YOLO11 opset 12) → `check_env.py` sẽ báo nạp được hay không.
3. **TensorRT** (nhanh nhất, chắc chạy được trên JetPack):
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --fp16   # ~10 phút
   pip3 install pycuda     # nếu check_env báo thiếu
   ```
   Sau đó sửa `config/robot.yaml`: `detector.model: models/best.engine`

**Xong khi** `check_env.py` in `[OK] nạp model ... backend ...`.

## 2. Đo robot bằng tay (15 phút)

```bash
python3 tools/teleop.py --wide
```
Phím: `w/a/s/d` = x, y   `r/f` = lên/xuống   `x/z` = trượt   `t` = đổi bước   Enter = ghi điểm vào `captures/teleop_points.txt`.

Đo và **ghi vào `config/robot.yaml`**:
- [ ] `limits.x`, `limits.y`, `limits.r_max`: đưa đầu hút ra xa/gần nhất ở độ cao z = -10 và z = 50. Lấy giá trị **thu vào ~10 mm** so với mức tay còn với tới được.
- [ ] Thanh trượt: bấm `x` vài lần, xem robot trượt theo hướng nào so với bàn. Đặt bàn cờ **dọc theo hướng trượt**.
- [ ] z chạm mặt quân: hạ đầu hút xuống chạm nhẹ một quân, đọc z. (Bước hiệu chỉnh sẽ đo lại chính xác.)

## 3. Chọn tư thế quan sát (15 phút)

Đặt bàn cờ và vùng chờ (dải ~10 × 40 cm dọc cạnh dài của bàn) cạnh robot.

```bash
python3 tools/capture.py --arm 0 -160 210 --slider 0 --name thu1
```
Mở `captures/thu1.jpg`. Tăng z tới mức cao nhất tay còn lên được, chỉnh x, y cho camera nhìn thẳng xuống bàn.

- Nếu **một ảnh thấy hết** bàn cờ và vùng chờ: dùng 1 view.
- Nếu không: dùng **cùng một tư thế tay**, thêm view ở vị trí trượt khác (ví dụ 0 và 2500). Các view nên chồng lấn khoảng 1/4.

Ghi vào `config/robot.yaml` → `views:`, rồi chạy `python3 tools/capture.py` để chụp lại tất cả.

**Xong khi** gộp các ảnh lại thấy đủ 4 góc bàn cờ và toàn bộ vùng chờ.

## 4. Hiệu chỉnh camera (15 phút / tư thế tay)

1. Đặt **6-9 quân đúng giữa giao điểm**, rải đều khắp khung hình: 4 góc ảnh, giữa, và các cạnh.
2. Chạy:
   ```bash
   python3 tools/calibrate.py
   ```
3. Mở `captures/calib_pose_discs.jpg` để biết thứ tự số. Với từng quân: lái đầu hút **chạm nhẹ đúng tâm** mặt quân, rồi Enter. Từ quân thứ 4, tay tự bay tới gần, chỉ cần chỉnh 1-2 mm. **Đừng làm quân xê dịch.**
4. Đọc kết quả:
   - `rms_top_mm` **< 1.5** (tốt < 1.0). Nếu `per_point_mm` có điểm > 2.5 → làm lại.
   - `board_scale` ≈ 1.00 (± 0.02). Nếu lệch → kích thước trong `config/board.yaml` chưa đúng.
   - `top_vs_table_mm` khoảng 0-6.
5. Nếu dùng thanh trượt: `python3 tools/calibrate.py --slider` (để vài quân trong ảnh).
6. Kiểm tra: `python3 tools/calibrate.py --verify`. Tay bay tới trên từng quân, đầu hút phải nằm **đúng tâm** (nhìn ngang).

**Xong khi** `--verify` bay trúng tâm mọi quân.

## 5. Kiểm tra nhìn (5 phút)

Rải 32 quân ở vùng chờ (một lớp, ngửa mặt, cách nhau ≥ 5 mm), bàn cờ để trống.
```bash
python3 main.py --scan-only
```
Mở `runs/<mới nhất>/scene.jpg`. Kiểm tra: lưới bàn khớp, đủ 32 quân, nhãn đúng, **a0 ở bên trái phía Đỏ** (phía gần robot).

- Nếu a0 ở bên phải: đặt `board.mirror: true` trong `config/robot.yaml`.
- Nếu phía Đỏ bị đảo đầu: sửa `board.red_near_world` thành một điểm gần phía muốn làm Đỏ.

## 6. Chạy thử không hút (5 phút)

```bash
python3 main.py --hover --max-moves 3
```
Tay bay tới trên quân rồi tới trên ô, hạ xuống còn cách 15 mm. Nhìn ngang: phải **đúng tâm**.

## 7. Chạy thật

```bash
python3 main.py --max-moves 1        # 1 quân
python3 main.py --max-moves 5
python3 main.py                      # cả bàn, khai cuộc
python3 main.py --fen "<FEN>"        # thế cờ bất kỳ
```
Trước khi chạy, chương trình in bảng quân và kế hoạch, rồi chờ:
- `Enter`: chạy.
- `#5=N`: sửa nhãn quân #5 thành Mã Đỏ (mã FEN: K A B N R C P đỏ / chữ thường cho đen).
- `r`: nhìn lại.
- `q`: thoát.

Cuối mỗi lần chạy, robot chụp lại và in `KIỂM TRA: đúng x/32 ô, lệch tâm TB ... mm`. Mọi thứ lưu trong `runs/<thời gian>/`.

## 8. Số liệu cho báo cáo

- **Nhận diện:** 10 lần rải quân ngẫu nhiên + `main.py --scan-only`. Đếm số quân đúng nhãn **trước** (cột tin cậy) và **sau** sửa theo số lượng, theo log `sửa nhãn theo số lượng quân`.
- **Định vị đặt quân:** dòng `KIỂM TRA ... lệch tâm TB/max` sau mỗi lần xếp.
- **Thời gian:** log có giờ từng bước.

---

## Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách sửa |
|---|---|
| `Tay máy chưa tới đích (lệch ...)` | Điểm ngoài tầm động học: thu hẹp `limits`, đặt bàn gần robot hơn |
| `ngoài tầm với (kể cả khi trượt)` | Quân / ô nằm ngoài vùng với: dời bàn hoặc vùng chờ lại gần |
| Không định vị được bàn cờ | Xem `runs/.../scene.jpg` hoặc `mosaic`: bàn có lọt các view không, có bị lóa không |
| Hút không lên quân | Giảm `heights.z_pick` 0.5-1 mm (ép sâu hơn), tăng `suck_wait` |
| Đặt quân bị nảy / lệch | Tăng `place_gap` nếu quân đập xuống, giảm nếu thả từ cao |
| Đặt lệch đều một phía | Hiệu chỉnh lại (mục 4), kiểm tra quân hiệu chỉnh có nằm đúng giao điểm không |
| Lệch tăng dần theo hướng trượt | Đo lại `--slider` |
| Nhận nhầm nhiều quân | Kiểm tra ánh sáng (không lóa), giảm `detector.conf` xuống 0.25 |
| Thanh trượt không chạy | `slider.enabled: false` nếu robot không có thanh trượt |
| Ctrl+C khi đang chạy | Robot tắt hút, nâng tay lên, về chỗ cũ |
