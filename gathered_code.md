# GATHERED SOURCE CODE - XIANGQI SORTING JETMAX

- **Thư mục gốc:** `D:\2026_2027\Chuyen-de\xiangqi-sorting-jetmax`
- **Tổng số file đã gôm:** 14

## Mục lục

### Nhóm 1: Camera & Thu thập hình ảnh
- [tests\camera_check.py](#testscamera_checkpy)
- [tests\capture_images.py](#testscapture_imagespy)
- [tests\capture_group.py](#testscapture_grouppy)
- [tests\scan_capture.py](#testsscan_capturepy)

### Nhóm 2: Mô hình & Dữ liệu
- [training\README.md](#trainingreadmemd)
- [training\build_dataset.ipynb](#trainingbuild_datasetipynb)
- [training\train_yolo11.ipynb](#trainingtrain_yolo11ipynb)
- [runs\yolo11n_640\results.csv](#runsyolo11n_640resultscsv)
- [robot\models\best.onnx](#robotmodelsbestonnx)
- [robot\models\best.pt](#robotmodelsbestpt)

### Nhóm 3: Nhận diện & Điều khiển gắp (Kinematics, Calibration & Pick-Place)
- [tests\pick_place.py](#testspick_placepy)
- [docs\robot-reference\waste_classification_stepper.py](#docsrobot-referencewaste_classification_stepperpy)
- [tests\analyze_images.py](#testsanalyze_imagespy)

### Nhóm 4: Tổng quan dự án
- [README.md](#readmemd)

---

## Nhóm 1: Camera & Thu thập hình ảnh

### `tests\camera_check.py`

*Dung lượng: 2.84 KB | Số dòng: 64*

```python
#!/usr/bin/env python3
# Kiểm tra camera JetMax khi đang chạy (chạy trên JetMax, cần node camera /usb_cam đang bật):
#   python3 camera_check.py            # đo 10 giây
#   python3 camera_check.py --secs 20
#
# Đo: độ phân giải, FPS thực, khoảng cách lớn nhất giữa hai khung (giật hình), độ trôi độ sáng
# (tự động phơi sáng/đèn nhấp nháy), nhiễu theo thời gian (độ lệch chuẩn từng điểm ảnh qua các khung).
# Hãy để CẢNH TĨNH (băng chuyền dừng, không có vật chuyển động) trong lúc đo để số liệu có nghĩa.

import argparse, time
import numpy as np
import rospy
from sensor_msgs.msg import Image

TOPIC = '/usb_cam/image_rect_color'
frames, stamps, means, arrival = [], [], [], []
info = {}


def on_image(msg):
    info['w'], info['h'], info['enc'] = msg.width, msg.height, msg.encoding
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    gray = img.mean(axis=2)
    means.append(float(gray.mean()))
    stamps.append(msg.header.stamp.to_sec())
    arrival.append(time.time())
    if len(frames) < 30 and len(means) % 3 == 0:
        frames.append(gray.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secs', type=float, default=10)
    a = ap.parse_args()
    rospy.init_node('camera_check')
    rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
    print('Đang đo %.0f giây từ %s (giữ cảnh tĩnh)...' % (a.secs, TOPIC))
    t0 = time.time()
    while time.time() - t0 < a.secs and not rospy.is_shutdown():
        time.sleep(0.05)
    if len(arrival) < 5:
        print('Gần như không nhận được ảnh. Kiểm tra: rostopic list | grep usb_cam')
        return

    dt = np.diff(arrival)
    fps = (len(arrival) - 1) / (arrival[-1] - arrival[0])
    print('Độ phân giải: %dx%d, mã hóa: %s' % (info['w'], info['h'], info['enc']))
    print('FPS thực nhận được: %.1f   (khoảng cách khung lớn nhất: %.0f ms)' % (fps, dt.max() * 1000))
    m = np.array(means)
    print('Độ sáng trung bình: %.1f, độ lệch chuẩn theo thời gian: %.2f, dao động max-min: %.1f' % (
        m.mean(), m.std(), m.max() - m.min()))
    if m.max() - m.min() > 6:
        print('  -> Độ sáng dao động đáng kể: có thể do tự động phơi sáng hoặc đèn nhấp nháy.')
    if len(frames) >= 5:
        tn = float(np.mean(np.std(np.stack(frames), axis=0)))
        print('Nhiễu theo thời gian (độ lệch chuẩn mỗi điểm ảnh): %.2f mức xám' % tn)
        if tn > 4:
            print('  -> Nhiễu cao: kiểm tra ánh sáng, độ lợi (gain) của camera.')
    if fps < 15:
        print('  -> FPS thấp: băng chuyền nhanh sẽ khó bám vật; cân nhắc giảm độ phân giải hoặc tắt hiển thị.')


if __name__ == '__main__':
    main()
```

---

### `tests\capture_images.py`

*Dung lượng: 1.85 KB | Số dòng: 56*

```python
#!/usr/bin/env python3

# Bấm phím cách để lưu, q để thoát. Ảnh lưu vào ~/capture/,
# mỗi lần lưu ghi thêm một dòng vào positions.csv (tên file, x, y, z).

import os, time, csv
import cv2, numpy as np, rospy
from sensor_msgs.msg import Image
from jetmax_control.msg import JetMax

TOPIC = '/usb_cam/image_rect_color'
OUT = os.path.expanduser('~/capture')
os.makedirs(OUT, exist_ok=True)
state = {'img': None, 'pos': (0.0, 0.0, 0.0)}

def on_image(msg):
    state['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()  # RGB

def on_status(msg):
    state['pos'] = (msg.x, msg.y, msg.z)

rospy.init_node('capture_images')
rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
rospy.Subscriber('/jetmax/status', JetMax, on_status, queue_size=1)

csv_path = os.path.join(OUT, 'positions.csv')
is_new = not os.path.exists(csv_path)
f = open(csv_path, 'a', newline='')
w = csv.writer(f)
if is_new:
    w.writerow(['file', 'x', 'y', 'z'])

n, t0 = 0, time.time()
print('Space = lưu ảnh, q = thoát')
while not rospy.is_shutdown():
    if state['img'] is None:
        if time.time() - t0 > 5:
            print('Chưa nhận được ảnh từ', TOPIC, '- kiểm tra rostopic list')
            t0 = time.time()
        time.sleep(0.05)
        continue
    bgr = cv2.cvtColor(state['img'], cv2.COLOR_RGB2BGR)
    show = bgr.copy()
    cv2.putText(show, 'saved: %d' % n, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('capture', show)
    k = cv2.waitKey(30) & 0xFF
    if k == ord(' '):
        name = '%d.jpg' % int(time.time() * 1000)
        cv2.imwrite(os.path.join(OUT, name), bgr)
        w.writerow([name] + ['%.1f' % v for v in state['pos']])
        f.flush()
        n += 1
    elif k == ord('q'):
        break
f.close()
cv2.destroyAllWindows()
```

---

### `tests\capture_group.py`

*Dung lượng: 9.04 KB | Số dòng: 188*

```python
#!/usr/bin/env python3
# Chụp ảnh theo NHÓM điều kiện để phân tích chất lượng ảnh (dùng chung với analyze_images.py).
# Chạy trên JetMax, cần node jetmax_control (nếu dùng --pose) và camera đang bật.
#
# Ví dụ:
#   python3 capture_group.py static                              # chụp không giới hạn số ảnh, dừng bằng q/Esc
#   python3 capture_group.py static --n 15                       # hoặc đặt số ảnh tối đa
#   python3 capture_group.py belt_mid --n 30 --interval 0.5      # băng chuyền chạy, tự chụp mỗi 0.5 giây
#   python3 capture_group.py static --pose 0 -200 80             # đổi điểm chụp (mặc định: DEFAULT_POSE)
#   python3 capture_group.py static --no-move                    # không đưa tay máy đi đâu, chụp tại chỗ
#
# Ảnh lưu ở ~/capture_scan/<nhóm>/ . Sau đó: python3 analyze_images.py ~/capture_scan
#
# Mặc định script khóa phơi sáng và cân bằng trắng của camera cho cả buổi chụp (tắt bằng --no-lock).
# Thông số cũ vẫn trả lại khi chạy:  v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=3
#                                    v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature_auto=1

import os, time, argparse, subprocess
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

TOPIC = '/usb_cam/image_rect_color'
ROOT = os.path.expanduser('~/capture_scan')
DEV = '/dev/video0'
DEFAULT_POSE = (0.0, -220.4, 50.3)      # điểm chụp đã chọn bằng joint_teleop.py
WIN = 'capture_group (Space/Enter = chup, q/Esc = dung)'
X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0   # giống scan_capture.py

state = {'img': None, 'n': 0}


def on_image(msg):
    state['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
    state['n'] += 1


def v4l2(ctrl):
    r = subprocess.run(['v4l2-ctl', '-d', DEV, '--set-ctrl=' + ctrl],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        print('  Không đặt được %s: %s' % (ctrl, r.stderr.strip()))


def lock_camera(exposure, wb):
    print('Khóa camera: phơi sáng thủ công = %d, cân bằng trắng = %d K' % (exposure, wb))
    v4l2('exposure_auto=1')                    # phải sang thủ công trước khi đặt giá trị
    v4l2('exposure_absolute=%d' % exposure)
    v4l2('white_balance_temperature_auto=0')
    v4l2('white_balance_temperature=%d' % wb)
    time.sleep(1.0)                            # chờ camera ổn định


def grab():
    t0 = time.time()
    n0 = state['n']
    while state['n'] < n0 + 2:                 # đợi khung hình mới
        if time.time() - t0 > 3:
            raise RuntimeError('Không nhận được ảnh từ ' + TOPIC)
        time.sleep(0.01)
    return cv2.cvtColor(state['img'], cv2.COLOR_RGB2BGR)


def preview(rgb, group, saved, total, extra=''):
    """Hiện ảnh trực tiếp kèm tâm ngắm và số đo độ nét/độ sáng (chữ không dấu vì font OpenCV). Trả về phím bấm."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr[h // 4: 3 * h // 4, w // 4: 3 * w // 4], cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()       # tính ở vùng giữa ảnh; càng cao càng nét
    vis = bgr.copy()
    cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
    cv2.line(vis, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
    cv2.rectangle(vis, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 255), 1)   # vùng đo độ nét
    label = '%s  %d/%d' % (group, saved, total) if total > 0 else '%s  %d' % (group, saved)
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, 'Sharp %.0f  Bright %.0f' % (sharp, bgr.mean()), (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, extra, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.imshow(WIN, vis)
    return cv2.waitKey(30) & 0xFF


def go_pose(x, y, z):
    if not (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN):
        raise SystemExit('Pose (%.1f, %.1f, %.1f) ngoài giới hạn an toàn, không đi.' % (x, y, z))
    from jetmax_control.msg import SetJetMax
    pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    rospy.sleep(1.0)
    pub.publish(SetJetMax(x=x, y=y, z=z, duration=2.0))
    rospy.sleep(2.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('group', help='tên nhóm/thư mục, ví dụ static, belt_slow, belt_mid, belt_fast')
    ap.add_argument('--n', type=int, default=0, help='số ảnh tối đa; 0 (mặc định) = không giới hạn, dừng bằng q/Esc hoặc Ctrl+C')
    ap.add_argument('--interval', type=float, default=0, help='>0: tự chụp mỗi N giây; 0: bấm Enter mỗi ảnh')
    ap.add_argument('--pose', type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=list(DEFAULT_POSE),
                    help='điểm chụp của tay máy (mặc định %s)' % (DEFAULT_POSE,))
    ap.add_argument('--no-move', action='store_true', help='không di chuyển tay máy, chụp tại vị trí hiện tại')
    ap.add_argument('--exposure', type=int, default=60, help='exposure_absolute (đơn vị phụ thuộc camera)')
    ap.add_argument('--wb', type=int, default=4000, help='nhiệt độ màu cân bằng trắng (K)')
    ap.add_argument('--no-preview', action='store_true', help='không mở cửa sổ xem trước, chụp bằng dòng lệnh')
    ap.add_argument('--no-lock', action='store_true', help='không khóa phơi sáng/cân bằng trắng')
    a = ap.parse_args()

    rospy.init_node('capture_group')
    rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
    if not a.no_lock:
        lock_camera(a.exposure, a.wb)
    if not a.no_move:
        print('Đưa tay máy tới (%.1f, %.1f, %.1f), hãy chắc đường đi không vướng vật cản...' % tuple(a.pose))
        go_pose(*a.pose)
    rospy.sleep(0.5)

    out = os.path.join(ROOT, a.group)
    os.makedirs(out, exist_ok=True)
    start = len([f for f in os.listdir(out) if f.lower().endswith('.jpg')])
    print('Lưu vào %s (đã có %d ảnh)' % (out, start))

    gui = not a.no_preview and bool(os.environ.get('DISPLAY'))
    if gui:
        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        except cv2.error:
            gui = False
    if gui:
        print('Cửa sổ xem trước đã mở: Space/Enter = chụp, q/Esc = dừng.')
        print('Chỉnh trực tiếp: [ ] = giảm/tăng phơi sáng, , . = giảm/tăng nhiệt độ màu, a = về tự động.')
    else:
        print('Không mở được cửa sổ xem trước (chạy trong terminal của NoMachine, không phải SSH), dùng dòng lệnh.')

    def save(n):
        img = grab()
        name = '%s_%03d.jpg' % (a.group, start + n)
        cv2.imwrite(os.path.join(out, name), img)
        print('  đã lưu %s (độ sáng %.0f)' % (name, img.mean()))

    saved = 0
    last = time.time()
    exp, wb, auto = a.exposure, a.wb, a.no_lock
    try:
        while (a.n <= 0 or saved < a.n) and not rospy.is_shutdown():
            if gui:
                if state['img'] is None:
                    cv2.waitKey(50)
                    continue
                extra = 'AUTO' if auto else 'Exp %d  WB %dK' % (exp, wb)
                key = preview(state['img'], a.group, saved, a.n, extra)
                if key in (ord('q'), 27):
                    break
                if key in (ord('['), ord(']')):
                    exp = max(1, int(exp * 0.7)) if key == ord('[') else min(5000, max(exp + 1, int(exp * 1.4)))
                    v4l2('exposure_auto=1')
                    v4l2('exposure_absolute=%d' % exp)
                    auto = False
                elif key in (ord(','), ord('.')):
                    wb = max(2800, wb - 200) if key == ord(',') else min(6500, wb + 200)
                    v4l2('white_balance_temperature_auto=0')
                    v4l2('white_balance_temperature=%d' % wb)
                    auto = False
                elif key == ord('a'):
                    v4l2('exposure_auto=3')
                    v4l2('white_balance_temperature_auto=1')
                    auto = True
                due = a.interval > 0 and time.time() - last >= a.interval
                if due or (a.interval <= 0 and key in (13, 32)):
                    save(saved)
                    saved += 1
                    last = time.time()
            else:
                if a.interval > 0:
                    time.sleep(a.interval)
                elif input('Ảnh %s [Enter = chụp, q = dừng] ' % (('%d/%d' % (saved + 1, a.n)) if a.n > 0 else ('%d' % (saved + 1)))).strip().lower() == 'q':
                    break
                save(saved)
                saved += 1
    except KeyboardInterrupt:
        pass
    if gui:
        cv2.destroyAllWindows()
    print('Xong: %d ảnh trong %s' % (saved, out))


if __name__ == '__main__':
    main()
```

---

### `tests\scan_capture.py`

*Dung lượng: 5.98 KB | Số dòng: 168*

```python
#!/usr/bin/env python3
# Test kết hợp: thanh trượt -> tay máy -> chụp ảnh, tại từng điểm dừng.
#
# Cách điều khiển (khớp với các file hiện có của bạn):
#   - Tay máy: qua ROS topic (/jetmax/command), giống test_move.py. KHÔNG tạo hiwonder.JetMax()
#     trong script này vì node jetmax_control đã giữ bus servo.
#   - Thanh trượt: hiwonder.Stepper(1) trực tiếp, giống stepper_teleop.py (goto là bước TƯƠNG ĐỐI).
#   - Ảnh: /usb_cam/image_rect_color (RGB), lưu ~/capture_scan/ + scan.csv
#
# Yêu cầu: node jetmax_control và usb_cam đang chạy (roslaunch như bình thường).
# Mỗi điểm dừng hỏi [Enter = chạy, q = dừng]. Kết thúc/lỗi/q: tay máy về chỗ cũ, thanh trượt về 0.

import os, csv, time
import cv2
import numpy as np
import rospy
import hiwonder
from sensor_msgs.msg import Image
from jetmax_control.msg import SetJetMax, JetMax

TOPIC = '/usb_cam/image_rect_color'
OUT = os.path.expanduser('~/capture_scan')

# ---- Kế hoạch quét (chỉnh ở đây) ----
SLIDER_POS = [0, 1000, 2000]                 # bước, 0 = home (ví dụ hãng đi tối đa ~5200)
ARM_OFFSETS = [(0, 0, 0), (20, 0, 0), (-20, 0, 0),
               (0, 20, 0), (0, -20, 0), (0, 0, -20)]   # mm, tính từ vị trí ban đầu
DUR = 2.0                                    # giây mỗi lần di chuyển tay máy

# ---- Giới hạn cứng (từ jetmax.py: z <= 225, bán kính xy >= 50 nếu không lệnh bị bỏ qua im lặng) ----
SLIDER_MIN, SLIDER_MAX = 0, 5200
X_LIM, Y_LIM, Z_LIM = (-120, 120), (-230, -100), (-12, 225)   # bảo thủ, nới dần sau khi thử
R_MIN = 50.0
TOL = 5.0                                    # mm, sai lệch cho phép giữa đích và vị trí thực


class Stop(Exception):
    pass


state = {'img': None, 'n': 0, 'pos': None}


def on_image(msg):
    state['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
    state['n'] += 1


def on_status(msg):
    state['pos'] = (msg.x, msg.y, msg.z)


def get_pos():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z


def ask(text):
    if input(text + ' [Enter = chạy, q = dừng] ').strip().lower() == 'q':
        raise Stop()


def check_target(x, y, z):
    ok = (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
          and (x * x + y * y) ** 0.5 >= R_MIN)
    if not ok:
        raise ValueError('Đích (%.1f, %.1f, %.1f) ngoài giới hạn an toàn, bỏ qua' % (x, y, z))


def move_arm(abs_pub, target):
    check_target(*target)
    abs_pub.publish(SetJetMax(x=target[0], y=target[1], z=target[2], duration=DUR))
    rospy.sleep(DUR + 0.7)
    cur = get_pos()
    err = max(abs(cur[i] - target[i]) for i in range(3))
    if err > TOL:
        print('  CẢNH BÁO: tay máy lệch %.1f mm so với đích (lệnh có thể bị chặn bởi giới hạn)' % err)
    return cur


def move_slider(stepper, pos, new):
    new = max(SLIDER_MIN, min(SLIDER_MAX, new))
    d = new - pos
    if d == 0:
        return pos
    stepper.set_mode(stepper.SLEEP)
    stepper.goto(d)
    time.sleep(abs(d) / 1000.0 + 0.5)
    return new


def grab():
    t0 = time.time()
    n0 = state['n']
    while state['n'] < n0 + 2:           # đợi khung hình mới sau khi dừng hẳn
        if time.time() - t0 > 3:
            raise RuntimeError('Không nhận được ảnh từ ' + TOPIC)
        time.sleep(0.02)
    return cv2.cvtColor(state['img'], cv2.COLOR_RGB2BGR)


def main():
    os.makedirs(OUT, exist_ok=True)
    rospy.init_node('scan_capture')
    abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
    rospy.Subscriber('/jetmax/status', JetMax, on_status, queue_size=1)
    rospy.sleep(1.0)

    start = get_pos()
    print('Vị trí tay máy ban đầu =', start)
    print('Thanh trượt sẽ về HOME. Hãy chắc tay máy đang nâng cao và không vướng vật cản.')
    ask('Bắt đầu')

    stepper = hiwonder.Stepper(1)
    pos = 0
    csv_path = os.path.join(OUT, 'scan.csv')
    new_file = not os.path.exists(csv_path)
    f = open(csv_path, 'a', newline='')
    w = csv.writer(f)
    if new_file:
        w.writerow(['file', 'slider', 'x', 'y', 'z'])
    run = int(time.time())
    n = 0

    try:
        stepper.go_home(True)            # chặn cho tới khi về home (không có timeout, canh nút nguồn)
        stepper.set_mode(stepper.EN)
        stepper.set_div(stepper.DIV_1_4)

        for sp in SLIDER_POS:
            ask('Thanh trượt -> %d' % sp)
            move_arm(abs_pub, start)     # luôn về tư thế cao ban đầu trước khi trượt
            pos = move_slider(stepper, pos, sp)
            print('  thanh trượt = %d' % pos)

            for off in ARM_OFFSETS:
                target = (start[0] + off[0], start[1] + off[1], start[2] + off[2])
                ask('  Tay máy offset %s' % (off,))
                try:
                    cur = move_arm(abs_pub, target)
                except ValueError as e:
                    print('  ', e)
                    continue
                img = grab()
                name = '%d_s%d_%03d.jpg' % (run, pos, n)
                cv2.imwrite(os.path.join(OUT, name), img)
                w.writerow([name, pos, '%.1f' % cur[0], '%.1f' % cur[1], '%.1f' % cur[2]])
                f.flush()
                n += 1
                print('  đã lưu', name)
    except Stop:
        print('Dừng theo yêu cầu')
    finally:
        f.close()
        try:
            abs_pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
            rospy.sleep(2.5)
            if pos > 0:
                stepper.goto(-pos)       # quay lại home như ví dụ hãng
                time.sleep(pos / 1000.0 + 0.5)
            stepper.set_mode(stepper.EN)
        finally:
            print('Đã về vị trí ban đầu, đã lưu %d ảnh vào %s' % (n, OUT))


if __name__ == '__main__':
    main()
```

---

## Nhóm 2: Mô hình & Dữ liệu

### `training\README.md`

*Dung lượng: 0 B | Số dòng: 0*

```markdown

```

---

### `training\build_dataset.ipynb`

*File Jupyter Notebook | Dung lượng: 3.49 MB*

<!-- Cell 1 (Markdown) -->
<a href="https://colab.research.google.com/github/JulianNguyen05/xiangqi-sorting-jetmax/blob/main/training/build_dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- Cell 2 (Markdown) -->
# Mount Drive và tạo thư mục

```python
# Cell 3 (Code)
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/datn'
RAW_DRIVE = f'{BASE}/raw'          # bản nén .zip của từng bộ (lưu lâu dài)
RAW_LOCAL = '/content/raw'         # bản giải nén trên máy Colab (đọc nhanh)
for d in ['raw', 'merged', 'runs', 'own_images']:
    os.makedirs(f'{BASE}/{d}', exist_ok=True)
os.makedirs(RAW_LOCAL, exist_ok=True)
print('OK:', BASE)
```

<!-- Cell 4 (Markdown) -->
# Cài thư viện

```python
# Cell 5 (Code)
!pip install -q roboflow imagehash pyyaml
```

<!-- Cell 6 (Markdown) -->
# Lấy API key


```python
# Cell 7 (Code)
from google.colab import userdata
from roboflow import Roboflow

API_KEY = userdata.get('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=API_KEY)
print('Đã kết nối Roboflow')
```

<!-- Cell 8 (Markdown) -->
## Khai báo 4 bộ dataset
Danh sách 4 dataset trên Roboflow Universe.

| Tên ngắn | Link |
|---|---|
| super | https://universe.roboflow.com/312s-workspace/xiangqi-super-dataset |
| scara | https://universe.roboflow.com/312s-workspace/xiangqi-scara-yolo26 |
| new_chinese_chess | https://universe.roboflow.com/quoc-rnmzr/new-chinese-chess-ayph4/dataset/1 |
| duytranle | https://universe.roboflow.com/duytranle/xiangqi-ubjij/dataset/18 |

```python
# Cell 9 (Code)
DATASETS = {
    'super':             {'workspace': '312s-workspace', 'project': 'xiangqi-super-dataset',   'version': None},
    'scara':             {'workspace': '312s-workspace', 'project': 'xiangqi-scara-yolo26',    'version': None},
    'new_chinese_chess': {'workspace': 'quoc-rnmzr',     'project': 'new-chinese-chess-ayph4', 'version': 1},
    'duytranle':         {'workspace': 'duytranle',      'project': 'xiangqi-ubjij',           'version': 18,
                          'format': 'yolov8'},   # dự án keypoint không hỗ trợ yolov11
}
```

<!-- Cell 10 (Markdown) -->
### Tải về, giải nén và sao lưu lên Drive
Với mỗi bộ: (1) đã có trên Colab thì bỏ qua; (2) đã có file .zip trên Drive thì giải nén; (3) chưa có thì tải từ Roboflow (định dạng yolov11) rồi nén lưu lên Drive. Phiên bản đã tải được ghi vào versions_used.json để đưa vào báo cáo.

```python
# Cell 11 (Code)
import shutil, json

def latest_version(project):
    vs = project.versions()
    return max(int(str(v.version).split('/')[-1]) for v in vs)

VERSIONS_FILE = f'{BASE}/raw/versions_used.json'
versions_used = json.load(open(VERSIONS_FILE)) if os.path.exists(VERSIONS_FILE) else {}
paths = {}

for name, cfg in DATASETS.items():
    out_dir  = f'{RAW_LOCAL}/{name}'
    zip_base = f'{RAW_DRIVE}/{name}'

    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f'[{name}] đã có trên Colab, bỏ qua')
    elif os.path.exists(zip_base + '.zip'):
        print(f'[{name}] giải nén từ Drive...')
        shutil.unpack_archive(zip_base + '.zip', out_dir)
    else:
        print(f'[{name}] tải từ Roboflow...')
        project = rf.workspace(cfg['workspace']).project(cfg['project'])
        ver = cfg['version'] or latest_version(project)
        fmt = cfg.get('format', 'yolov11')
        project.version(ver).download(fmt, location=out_dir)
        shutil.make_archive(zip_base, 'zip', out_dir)
        versions_used[name] = {**cfg, 'version': ver, 'format': fmt}
        with open(VERSIONS_FILE, 'w') as f:
            json.dump(versions_used, f, indent=2, ensure_ascii=False)
        print(f'[{name}] đã tải version {ver} (định dạng {fmt}) và lưu lên Drive')

    paths[name] = out_dir

print('Xong:', list(paths))
```

<!-- Cell 12 (Markdown) -->
### Kiểm tra tên lớp và số ảnh của từng bộ
In danh sách lớp (names trong data.yaml) và số ảnh mỗi tập. Với mỗi bộ: in tên lớp, số ảnh (và số ảnh gốc ước lượng) mỗi tập, bảng số box theo lớp ở từng tập, và các lỗi nhãn.

```python
# Cell 13 (Code)
import yaml, glob, os
import pandas as pd
from collections import Counter

SPLITS = ['train', 'valid', 'test']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
MIN_BOX_WARN = 300          # lớp có ít box hơn mức này sẽ bị cảnh báo
pd.set_option('display.width', 220)

def load_names(yaml_path):
    with open(yaml_path, encoding='utf-8') as f:
        names = yaml.safe_load(f)['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names

def source_id(filename):
    # Roboflow: <tên gốc>.rf.<hash>.jpg -> bản augment cùng một ảnh gốc
    return filename.split('.rf.')[0]

reports = {}
for name, p in paths.items():
    print('=' * 80)
    print(name.upper())
    yaml_files = glob.glob(f'{p}/**/data.yaml', recursive=True)
    if not yaml_files:
        print('  Không thấy data.yaml')
        continue
    root = os.path.dirname(yaml_files[0])
    names = load_names(yaml_files[0])
    print(f'  Số lớp: {len(names)}')
    print(f'  Tên lớp: {names}')

    box = {s: Counter() for s in SPLITS}      # số box theo lớp
    img_with = Counter()                      # số ảnh có chứa lớp (cả 3 tập)
    problems = Counter()

    for s in SPLITS:
        img_files = [f for f in glob.glob(f'{root}/{s}/images/*')
                     if f.lower().endswith(IMG_EXT)]
        n_src = len({source_id(os.path.basename(f)) for f in img_files})
        print(f'  {s:6s}: {len(img_files):5d} ảnh  (~{n_src} ảnh gốc ước lượng)')

        lbl_files = glob.glob(f'{root}/{s}/labels/*.txt')
        img_stems = {os.path.splitext(os.path.basename(f))[0] for f in img_files}
        lbl_stems = {os.path.splitext(os.path.basename(f))[0] for f in lbl_files}
        problems['ảnh không có file nhãn'] += len(img_stems - lbl_stems)
        problems['file nhãn không có ảnh'] += len(lbl_stems - img_stems)

        for lf in lbl_files:
            with open(lf) as f:
                lines = [l.split() for l in f if l.strip()]
            if not lines:
                problems['file nhãn rỗng'] += 1
            seen = set()
            for parts in lines:
                if len(parts) != 5:
                    problems['dòng nhãn không phải bbox (5 giá trị)'] += 1
                    continue
                try:
                    cid = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    problems['dòng nhãn sai định dạng'] += 1
                    continue
                if not (0 <= cid < len(names)):
                    problems['id lớp ngoài phạm vi'] += 1
                    continue
                if w <= 0 or h <= 0:
                    problems['box có w hoặc h <= 0'] += 1
                if x - w/2 < -0.01 or y - h/2 < -0.01 or x + w/2 > 1.01 or y + h/2 > 1.01:
                    problems['box lệch ra ngoài khung ảnh'] += 1
                box[s][names[cid]] += 1
                seen.add(names[cid])
            for c in seen:
                img_with[c] += 1

    df = pd.DataFrame({f'{s}_box': pd.Series(box[s]) for s in SPLITS}).reindex(names).fillna(0).astype(int)
    df['tong_box'] = df.sum(axis=1)
    df['tong_anh_co_lop'] = pd.Series(img_with).reindex(names).fillna(0).astype(int)
    reports[name] = df

    print('\n  Số box theo lớp:')
    print(df.to_string())
    print(f"\n  Tổng box: {df['tong_box'].sum()}")
    weak = df[df['tong_box'] < MIN_BOX_WARN].index.tolist()
    if weak:
        print(f'  ⚠ Lớp ít hơn {MIN_BOX_WARN} box: {weak}')
    if problems:
        print('  ⚠ Vấn đề phát hiện:')
        for k, v in problems.items():
            print(f'     - {k}: {v}')
    else:
        print('  Không phát hiện lỗi nhãn cơ bản.')
```

<!-- Cell 14 (Markdown) -->
### Chẩn đoán dòng nhãn không phải bbox
Với mỗi bộ: đếm dòng 5 giá trị (bbox chuẩn) và các dòng khác, gom theo (lớp, số giá trị mỗi dòng), in kèm một dòng mẫu. Dùng để biết dòng lạ là polygon (segmentation) hay pose (keypoint) và thuộc lớp nào.

```python
# Cell 15 (Code)
from collections import Counter

for name, p in paths.items():
    yf = glob.glob(f'{p}/**/data.yaml', recursive=True)[0]
    root = os.path.dirname(yf)
    names = load_names(yf)
    odd, sample, ok = Counter(), {}, 0
    for lf in glob.glob(f'{root}/*/labels/*.txt'):
        for line in open(lf):
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 5:
                ok += 1
                continue
            key = (names[int(parts[0])], len(parts))
            odd[key] += 1
            sample.setdefault(key, ' '.join(parts[:12]))
    print('=' * 70)
    print(f'{name} | dòng 5 giá trị: {ok} | dòng khác: {sum(odd.values())}')
    for (cls, n), c in odd.most_common(6):
        print(f'  lớp={cls:15s} số giá trị/dòng={n:3d}  số dòng={c}')
        print('     mẫu:', sample[(cls, n)])
```

<!-- Cell 16 (Markdown) -->
## Làm sạch và ánh xạ về 14 lớp chuẩn
Duyệt từng bộ: đổi tên lớp về 14 lớp chuẩn, bỏ dòng `board`, bỏ ảnh có lớp `None`, chuyển polygon (super) và pose (duytranle) về bbox, kẹp box vào khung ảnh. Ảnh và nhãn sạch được gom vào /content/pool kèm manifest.csv (bộ, tập gốc, mã ảnh gốc). In bảng số box theo lớp và thống kê các thao tác đã làm.

```python
# Cell 17 (Code)
import shutil, glob, os, yaml
import pandas as pd
from collections import Counter

SPLITS = ['train', 'valid', 'test']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
POOL = '/content/pool'

CLASSES = [f'{c}_{p}' for c in ['black', 'red']
           for p in ['advisor', 'bishop', 'cannon', 'king', 'knight', 'pawn', 'rook']]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
PIECE_ALIAS = {'elephant': 'bishop', 'chariot': 'rook', 'general': 'king',
               'horse': 'knight', 'guard': 'advisor', 'soldier': 'pawn'}
DROP_LINES = {'board'}      # bỏ dòng nhãn, giữ ảnh
DROP_IMAGES = {'none'}      # bỏ cả ảnh có lớp này
POSE_DATASETS = {'duytranle'}   # nhãn dạng: lớp x y w h + keypoint

def load_names(yaml_path):
    with open(yaml_path, encoding='utf-8') as f:
        names = yaml.safe_load(f)['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names

def source_id(filename):
    return filename.split('.rf.')[0]

def canon(raw):
    n = raw.strip().lower().replace('-', '_').replace(' ', '_')
    if n in DROP_LINES:
        return 'DROP_LINE'
    if n in DROP_IMAGES:
        return 'DROP_IMAGE'
    color, piece = n.split('_', 1)
    name = f'{color}_{PIECE_ALIAS.get(piece, piece)}'
    if name not in CLASS_ID:
        raise ValueError(f'Lớp lạ chưa ánh xạ: {raw}')
    return name

if os.path.exists(POOL):
    shutil.rmtree(POOL)
os.makedirs(f'{POOL}/images'); os.makedirs(f'{POOL}/labels')

rows, cls_count, stats = [], {}, {}
for name, p in paths.items():
    yf = glob.glob(f'{p}/**/data.yaml', recursive=True)[0]
    root = os.path.dirname(yf)
    names = load_names(yf)
    pose = name in POSE_DATASETS
    st, cnt = Counter(), Counter()

    for split in SPLITS:
        img_dir, lbl_dir = f'{root}/{split}/images', f'{root}/{split}/labels'
        if not os.path.isdir(lbl_dir):
            continue
        imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)}
        for lf in os.listdir(lbl_dir):
            stem = os.path.splitext(lf)[0]
            if stem not in imgs:
                continue
            out, kept, skip = [], [], False
            for line in open(f'{lbl_dir}/{lf}'):
                parts = line.split()
                if not parts:
                    continue
                c = canon(names[int(parts[0])])
                if c == 'DROP_LINE':
                    st['dòng board bỏ'] += 1
                    continue
                if c == 'DROP_IMAGE':
                    skip = True
                    break
                vals = list(map(float, parts[1:]))
                if len(vals) == 4:
                    x, y, w, h = vals
                elif pose and len(vals) >= 4:
                    x, y, w, h = vals[:4]; st['pose→bbox'] += 1
                elif len(vals) >= 6 and len(vals) % 2 == 0:
                    xs, ys = vals[0::2], vals[1::2]
                    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                    x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
                    st['polygon→bbox'] += 1
                else:
                    st['dòng lỗi bỏ'] += 1
                    continue
                x0, y0 = max(0.0, x - w / 2), max(0.0, y - h / 2)
                x1, y1 = min(1.0, x + w / 2), min(1.0, y + h / 2)
                w, h = x1 - x0, y1 - y0
                if w <= 0.005 or h <= 0.005:
                    st['box quá nhỏ/lỗi bỏ'] += 1
                    continue
                out.append(f'{CLASS_ID[c]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {w:.6f} {h:.6f}')
                kept.append(c)
            if skip:
                st['ảnh bỏ (lớp None)'] += 1
                continue
            if not out:
                st['ảnh bỏ (không còn box)'] += 1
                continue
            new = f'{name}__{imgs[stem]}'
            shutil.copy(f'{img_dir}/{imgs[stem]}', f'{POOL}/images/{new}')
            with open(f'{POOL}/labels/{os.path.splitext(new)[0]}.txt', 'w') as f:
                f.write('\n'.join(out) + '\n')
            cnt.update(kept)
            rows.append({'file': new, 'dataset': name, 'orig_split': split,
                         'source_id': f'{name}__{source_id(imgs[stem])}', 'n_box': len(out)})
    cls_count[name], stats[name] = cnt, st

manifest = pd.DataFrame(rows)
manifest.to_csv(f'{POOL}/manifest.csv', index=False)

print('Số ảnh sạch theo bộ và tập gốc:')
print(manifest.groupby(['dataset', 'orig_split']).size().unstack(fill_value=0))
table = pd.DataFrame({n: [c.get(k, 0) for k in CLASSES] for n, c in cls_count.items()}, index=CLASSES)
table['tong'] = table.sum(axis=1)
print('\nSố box theo lớp chuẩn:')
print(table)
print('\nThao tác đã thực hiện:')
for n, s in stats.items():
    print(f'  {n}: {dict(s)}')
```

<!-- Cell 18 (Markdown) -->
## Gom nhóm ảnh trùng (cùng ảnh gốc hoặc gần giống nhau)
Tính perceptual hash cho từng ảnh trong /content/pool rồi gom nhóm bằng union-find: ảnh cùng source_id là một nhóm; ảnh cùng bộ chỉ nối khi hash trùng khít (SAME_DS_T); ảnh khác bộ nối khi khoảng cách hash <= CROSS_DS_T. In kích thước nhóm, thành phần của nhóm lớn nhất, số nhóm nối nhiều bộ, và vẽ vài cặp ảnh trùng khác bộ để kiểm tra bằng mắt. Chưa xóa gì.

```python
# Cell 19 (Code)
import numpy as np, imagehash, matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

SAME_DS_T = 0    # ngưỡng phash nối ảnh CÙNG bộ (0 = hash trùng khít; đặt -1 để chỉ gom theo source_id)
CROSS_DS_T = 4   # ngưỡng phash nối ảnh KHÁC bộ
manifest = pd.read_csv(f'{POOL}/manifest.csv')
n = len(manifest)

# 1) perceptual hash
hashes = []
for f in manifest['file']:
    try:
        with Image.open(f'{POOL}/images/{f}') as im:
            hashes.append(int(str(imagehash.phash(im)), 16))
    except Exception as e:
        print('Lỗi ảnh:', f, e); hashes.append(0)
H = np.array(hashes, dtype=np.uint64)
ds = manifest['dataset'].values

# 2) union-find
parent = list(range(n))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

first = {}
for i, s in enumerate(manifest['source_id']):
    if s in first: union(first[s], i)
    else: first[s] = i

def popcount64(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)

for i in range(n - 1):
    d = popcount64(H[i + 1:] ^ H[i])
    thr = np.where(ds[i + 1:] == ds[i], SAME_DS_T, CROSS_DS_T)
    for j in (np.nonzero(d <= thr)[0] + i + 1):
        union(i, int(j))

manifest['group'] = [find(i) for i in range(n)]

# 3) thống kê
g = manifest.groupby('group')
sizes = g.size().sort_values(ascending=False)
multi_ds = g['dataset'].nunique()
multi_sp = g['orig_split'].nunique()
print(f'Ảnh: {n} | nhóm: {len(sizes)} | nhóm lớn nhất: {sizes.head(5).tolist()}')
top = sizes.index[0]
print('Thành phần nhóm lớn nhất:', manifest[manifest.group == top].groupby(['dataset', 'orig_split']).size().to_dict())
print(f'Nhóm trải qua >1 bộ: {(multi_ds > 1).sum()} | nhóm nằm ở >1 tập gốc: {(multi_sp > 1).sum()}')
if sizes.iloc[0] > 0.03 * n:
    print('⚠ Nhóm lớn nhất quá lớn: đặt SAME_DS_T = -1 và chạy lại.')

print('\nCặp bộ có ảnh trùng nhau:')
pair = defaultdict(int)
for gid in multi_ds[multi_ds > 1].index:
    sets = tuple(sorted(manifest.loc[manifest.group == gid, 'dataset'].unique()))
    pair[sets] += 1
for k, v in sorted(pair.items(), key=lambda kv: -kv[1]):
    print(' ', k, v)

# 4) vẽ vài cặp trùng khác bộ để kiểm tra bằng mắt
cross = multi_ds[multi_ds > 1].index[:3]
if len(cross):
    fig, axes = plt.subplots(len(cross), 2, figsize=(8, 3.5 * len(cross)), squeeze=False)
    for r, gid in enumerate(cross):
        sub = manifest[manifest.group == gid].drop_duplicates('dataset').head(2)
        for c, (_, row) in enumerate(sub.iterrows()):
            axes[r][c].imshow(Image.open(f'{POOL}/images/{row.file}'))
            axes[r][c].set_title(row.dataset); axes[r][c].axis('off')
    plt.tight_layout(); plt.show()
```

<!-- Cell 20 (Markdown) -->
## Chia train/val/test theo nhóm và xuất dataset hợp nhất
Chia ~80/10/10 theo ảnh, giữ nguyên mỗi nhóm trong cùng một tập và giữ tỷ lệ các bộ trong từng tập (StratifiedGroupKFold). Kiểm tra tỷ lệ val/test của từng bộ trước; nếu lệch thì dừng, không ghi file. Nếu ổn thì copy ảnh/nhãn vào /content/merged, ghi data.yaml với 14 lớp chuẩn, in số box theo lớp ở mỗi tập, rồi nén lưu bản sao lên Drive (ghi đè merged_v1.zip cũ).

```python
# Cell 21 (Code)
from sklearn.model_selection import StratifiedGroupKFold
import yaml

MERGED = '/content/merged'
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
fold = np.zeros(len(manifest), dtype=int)
for k, (_, te) in enumerate(sgkf.split(manifest, manifest['dataset'], manifest['group'])):
    fold[te] = k
manifest['split'] = np.where(fold == 0, 'test', np.where(fold == 1, 'val', 'train'))

# kiểm tra tỷ lệ trước khi ghi file
share = manifest.groupby(['dataset', 'split']).size().unstack(fill_value=0)
for c in ['train', 'val', 'test']:
    if c not in share: share[c] = 0
pct = share[['val', 'test']].div(share[['train', 'val', 'test']].sum(axis=1), axis=0) * 100
print('Số ảnh theo bộ và tập:'); print(share[['train', 'val', 'test']])
print('\nTỷ lệ val/test mỗi bộ (%):'); print(pct.round(1))
bad = pct[(share[['train', 'val', 'test']].sum(axis=1) >= 100) & ((pct < 5).any(axis=1) | (pct > 20).any(axis=1))]
if len(bad):
    raise ValueError(f'Chia tập lệch ở bộ: {list(bad.index)}. Xem lại bước gom nhóm (nhóm quá lớn?), chưa ghi file.')

if os.path.exists(MERGED):
    shutil.rmtree(MERGED)
for s in ['train', 'val', 'test']:
    os.makedirs(f'{MERGED}/{s}/images'); os.makedirs(f'{MERGED}/{s}/labels')

for _, r in manifest.iterrows():
    stem = os.path.splitext(r.file)[0]
    shutil.copy(f'{POOL}/images/{r.file}', f'{MERGED}/{r.split}/images/{r.file}')
    shutil.copy(f'{POOL}/labels/{stem}.txt', f'{MERGED}/{r.split}/labels/{stem}.txt')

with open(f'{MERGED}/data.yaml', 'w') as f:
    yaml.safe_dump({'path': MERGED, 'train': 'train/images', 'val': 'val/images',
                    'test': 'test/images', 'nc': len(CLASSES), 'names': CLASSES},
                   f, allow_unicode=True, sort_keys=False)
manifest.to_csv(f'{MERGED}/manifest.csv', index=False)

box = {s: Counter() for s in ['train', 'val', 'test']}
for s in box:
    for lf in glob.glob(f'{MERGED}/{s}/labels/*.txt'):
        for line in open(lf):
            if line.strip():
                box[s][CLASSES[int(line.split()[0])]] += 1
print('\nSố box theo lớp và tập:')
print(pd.DataFrame(box, index=CLASSES))

shutil.make_archive(f'{BASE}/merged/merged_v1', 'zip', MERGED)
print('\nĐã lưu', f'{BASE}/merged/merged_v1.zip')
```

<!-- Cell 22 (Markdown) -->
## Kiểm tra nhãn bằng mắt
Vẽ box và tên lớp lên 2 ảnh ngẫu nhiên của mỗi bộ trong tập train của /content/merged, để chắc việc chuyển đổi nhãn (polygon/pose → bbox) và ánh xạ lớp đúng trước khi train.

```python
# Cell 23 (Code)
import cv2, matplotlib.pyplot as plt

picks = []
for dsn in ['super', 'duytranle', 'new_chinese_chess', 'scara']:
    sub = manifest[(manifest.dataset == dsn) & (manifest.split == 'train')]
    picks += sub.sample(2, random_state=7)[['file', 'dataset']].values.tolist()

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

for ax, (f, dsn) in zip(axes.flatten(), picks):
    img = cv2.cvtColor(cv2.imread(f'{MERGED}/train/images/{f}'), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    thick, fs = max(2, W // 300), max(0.5, W / 1100)
    stem = os.path.splitext(f)[0]
    for line in open(f'{MERGED}/train/labels/{stem}.txt'):
        c, x, y, w, h = line.split()
        c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)
        p1 = (int((x - w / 2) * W), int((y - h / 2) * H))
        p2 = (int((x + w / 2) * W), int((y + h / 2) * H))
        color = (255, 60, 60) if CLASSES[c].startswith('red') else (60, 120, 255)
        cv2.rectangle(img, p1, p2, color, thick)
        cv2.putText(img, CLASSES[c].split('_')[1][:3], (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 0), max(1, thick - 1))
    ax.imshow(img); ax.set_title(dsn); ax.axis('off')
plt.tight_layout(); plt.show()
```

---

### `training\train_yolo11.ipynb`

*File Jupyter Notebook | Dung lượng: 13.46 KB*

<!-- Cell 1 (Markdown) -->
<a href="https://colab.research.google.com/github/JulianNguyen05/xiangqi-sorting-jetmax/blob/main/training/train_yolo11.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- Cell 2 (Markdown) -->
## Chuẩn bị môi trường
Kiểm tra GPU, mount Drive và cài ultralytics.

```python
# Cell 3 (Code)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ GPU - đổi runtime sang T4')

from google.colab import drive
drive.mount('/content/drive')

!pip install -q ultralytics
```

```python
# Cell 4 (Code)
import os
BASE = '/content/drive/MyDrive/datn'
for p in [
    f'{BASE}/merged/merged_v1.zip',
    f'{BASE}/runs/yolo11n_640/weights/last.pt',
    f'{BASE}/runs/yolo11n_640/args.yaml',
    f'{BASE}/runs/yolo11n_640/results.csv',
]:
    print(os.path.exists(p), p)
```

<!-- Cell 5 (Markdown) -->
## Giải nén dataset hợp nhất
Copy merged_v1.zip từ Drive về máy Colab (đọc nhanh hơn đọc trực tiếp từ Drive) và kiểm tra số ảnh/nhãn mỗi tập.

```python
# Cell 6 (Code)
import os, shutil

BASE = '/content/drive/MyDrive/datn'
MERGED = '/content/merged'

if not os.path.isdir(f'{MERGED}/train'):
    shutil.unpack_archive(f'{BASE}/merged/merged_v1.zip', MERGED)

for s in ['train', 'val', 'test']:
    print(s, len(os.listdir(f'{MERGED}/{s}/images')), 'ảnh |', len(os.listdir(f'{MERGED}/{s}/labels')), 'nhãn')

print(open(f'{MERGED}/data.yaml').read())
```

<!-- Cell 7 (Markdown) -->
## Kiểm tra nhãn trước khi huấn luyện
Đảm bảo mỗi ảnh đều có file nhãn tương ứng và mọi class id nằm trong khoảng 0-13. Nhãn sai class id sẽ làm quá trình train báo lỗi hoặc cho kết quả sai.

```python
# Cell 8 (Code)
import glob

bad = 0
for s in ['train', 'val', 'test']:
    imgs = {os.path.splitext(f)[0] for f in os.listdir(f'{MERGED}/{s}/images')}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(f'{MERGED}/{s}/labels')}
    print(s, '| ảnh thiếu nhãn:', len(imgs - lbls), '| nhãn thiếu ảnh:', len(lbls - imgs))
    for p in glob.glob(f'{MERGED}/{s}/labels/*.txt'):
        for line in open(p):
            parts = line.split()
            if parts and not (0 <= int(parts[0]) <= 13):
                bad += 1
print('Dòng nhãn có class id sai:', bad)
```

<!-- Cell 9 (Markdown) -->
## Khai báo tham số huấn luyện
Gom các siêu tham số vào một chỗ để dễ chỉnh. Kết quả được lưu thẳng vào Drive (`/datn/runs`) nên không mất khi Colab ngắt kết nối. Tên run `yolo11n_640` khớp với thư mục `runs/` trong dự án.

```python
# Cell 10 (Code)
from ultralytics import YOLO

DATA    = f'{MERGED}/data.yaml'
PROJECT = f'{BASE}/runs'
NAME    = 'yolo11n_640'

EPOCHS  = 100
IMGSZ   = 640
BATCH   = 32
```

<!-- Cell 11 (Markdown) -->
## Nạp mô hình YOLO11n pretrained
Dùng trọng số `yolo11n.pt` đã huấn luyện sẵn trên COCO (transfer learning). Bản `n` (nano) nhỏ và nhanh, phù hợp triển khai lên Jetson Nano. Trọng số sẽ tự tải về lần đầu.

```python
# Cell 12 (Code)
model = YOLO('yolo11n.pt')
```

<!-- Cell 13 (Markdown) -->
## Huấn luyện mô hình
Các tham số đáng chú ý:
- `patience=30`: dừng sớm nếu 30 epoch liền không cải thiện.
- `fliplr=0.0`: **tắt lật ngang** vì lật sẽ làm chữ Hán trên quân cờ bị đảo ngược, gây nhầm lớp.
- `degrees=10`: xoay nhẹ, vì quân cờ trên băng chuyền có thể nằm lệch góc.
- `save_period=10`: lưu checkpoint mỗi 10 epoch để dự phòng khi Colab ngắt.
- `close_mosaic=10`: tắt mosaic ở 10 epoch cuối để mô hình ổn định hơn.

Thời gian ước tính khoảng 1-2 giờ trên T4.

```python
# Cell 14 (Code)
results = model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=30,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    save_period=10,
    workers=2,
    fliplr=0.0,
    degrees=10,
    close_mosaic=10,
    plots=True,
)
```

<!-- Cell 15 (Markdown) -->
## Tiếp tục huấn luyện nếu bị ngắt giữa chừng
Chỉ chạy ô này khi Colab ngắt kết nối lúc đang train. Chạy lại phần chuẩn bị môi trường và giải nén dataset, bỏ qua bước 4, rồi chạy ô dưới để train tiếp từ checkpoint `last.pt`.

```python
# Cell 16 (Code)
# model = YOLO(f'{PROJECT}/{NAME}/weights/last.pt')
# model.train(resume=True)
```

<!-- Cell 17 (Markdown) -->
## Xem kết quả huấn luyện
Đọc `results.csv` và vẽ đường loss cùng mAP theo từng epoch. Loss giảm đều và mAP tăng rồi đi ngang là dấu hiệu huấn luyện tốt. Nếu loss val tăng trong khi loss train giảm thì mô hình đang overfit.

```python
# Cell 18 (Code)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(f'{PROJECT}/{NAME}/results.csv')
df.columns = df.columns.str.strip()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for c in ['train/box_loss', 'val/box_loss', 'train/cls_loss', 'val/cls_loss']:
    ax[0].plot(df['epoch'], df[c], label=c)
ax[0].set_title('Loss'); ax[0].set_xlabel('epoch'); ax[0].legend()

for c in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
    ax[1].plot(df['epoch'], df[c], label=c)
ax[1].set_title('mAP (val)'); ax[1].set_xlabel('epoch'); ax[1].legend()
plt.show()

print(df[['epoch', 'metrics/precision(B)', 'metrics/recall(B)',
          'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].tail(3))
```

<!-- Cell 19 (Markdown) -->
## Đánh giá trên tập test
Tập test chưa từng được dùng khi huấn luyện hay chọn mô hình, nên đây là con số khách quan để báo cáo. Mục tiêu trong đề cương là **mAP@0.5 ≥ 95%**. Bảng bên dưới cho biết từng lớp để phát hiện lớp nào còn yếu.

```python
# Cell 20 (Code)
best = YOLO(f'{PROJECT}/{NAME}/weights/best.pt')
metrics = best.val(data=DATA, split='test', imgsz=IMGSZ, project=PROJECT, name=f'{NAME}_test', exist_ok=True)

print(f'mAP@0.5     : {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Precision   : {metrics.box.mp:.4f} | Recall: {metrics.box.mr:.4f}')

print('\nmAP@0.5 theo lớp:')
for i, name in best.names.items():
    print(f'  {name:15s} {metrics.box.ap50[i]:.4f}')
```

<!-- Cell 21 (Markdown) -->
## Dự đoán thử trên ảnh test
Chọn ngẫu nhiên vài ảnh từ tập test để xem mô hình khoanh vùng và gán nhãn có hợp lý không. Bước này hữu ích để đưa vào báo cáo.

```python
# Cell 22 (Code)
import random
from IPython.display import Image, display

samples = random.sample(glob.glob(f'{MERGED}/test/images/*'), 4)
preds = best.predict(samples, imgsz=IMGSZ, conf=0.5, save=True,
                     project=PROJECT, name=f'{NAME}_pred', exist_ok=True)

for r in preds:
    display(Image(filename=os.path.join(r.save_dir, os.path.basename(r.path))))
```

<!-- Cell 23 (Markdown) -->
## Xuất mô hình sang ONNX
`best.pt` dùng được ngay với Ultralytics, còn ONNX là định dạng trung gian để chuyển sang TensorRT. **Lưu ý:** file TensorRT (`.engine`) phải được tạo trực tiếp trên Jetson, không tạo trên Colab được, vì nó phụ thuộc vào GPU và phiên bản TensorRT của thiết bị.

```python
# Cell 24 (Code)
onnx_path = best.export(format='onnx', imgsz=IMGSZ, simplify=True)
print('Đã xuất:', onnx_path)
print('Các file trong thư mục weights:', os.listdir(f'{PROJECT}/{NAME}/weights'))
```

<!-- Cell 25 (Markdown) -->
## Tải kết quả về máy
Nén thư mục kết quả và tải về, giải nén vào `runs/yolo11n_640/` trong dự án. Bản gốc vẫn còn trên Drive.

```python
# Cell 26 (Code)
from google.colab import files

shutil.make_archive('/content/yolo11n_640', 'zip', f'{PROJECT}/{NAME}')
files.download('/content/yolo11n_640.zip')
```

---

### `runs\yolo11n_640\results.csv`

*Dung lượng: 12.40 KB | Số dòng: 101*

```csv
epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,112.154,1.08811,2.03522,1.03137,0.75366,0.6644,0.78154,0.58506,0.90759,1.09886,0.94196,0.000184791,0.000184791,0.000184791
2,217.761,0.88325,1.00072,0.95006,0.86552,0.81591,0.90605,0.72214,0.73186,0.74927,0.89099,0.000366461,0.000366461,0.000366461
3,319.293,0.84482,0.79796,0.93786,0.90926,0.87549,0.94124,0.75852,0.71325,0.57522,0.8868,0.00054446,0.00054446,0.00054446
4,420.224,0.80391,0.69491,0.92563,0.9284,0.8969,0.94967,0.7707,0.71383,0.50203,0.88696,0.000539487,0.000539487,0.000539487
5,520.107,0.79007,0.64044,0.9219,0.95568,0.91751,0.96039,0.78219,0.68137,0.43488,0.88164,0.000533982,0.000533982,0.000533982
6,623.951,0.77072,0.60669,0.91428,0.95923,0.91844,0.96005,0.78536,0.67765,0.42219,0.88378,0.000528478,0.000528478,0.000528478
7,108.317,0.77062,0.58781,0.91277,0.97142,0.9292,0.96314,0.8003,0.6511,0.39453,0.87214,0.000522974,0.000522974,0.000522974
8,207.321,0.75296,0.55577,0.90912,0.96549,0.93512,0.96398,0.78898,0.68572,0.41504,0.87629,0.000517469,0.000517469,0.000517469
9,305.613,0.74053,0.54243,0.90517,0.97373,0.93561,0.96484,0.80765,0.62696,0.36754,0.86464,0.000511965,0.000511965,0.000511965
10,405.445,0.71716,0.5147,0.89831,0.97622,0.93741,0.966,0.79249,0.69427,0.36658,0.88102,0.00050646,0.00050646,0.00050646
11,504.257,0.70728,0.50359,0.89512,0.98016,0.94241,0.9675,0.81232,0.6384,0.35686,0.86477,0.000500956,0.000500956,0.000500956
12,605.153,0.70089,0.49193,0.89143,0.9815,0.94508,0.96726,0.79632,0.672,0.34805,0.88067,0.000495452,0.000495452,0.000495452
13,703.043,0.72093,0.49961,0.89824,0.9764,0.94336,0.96606,0.79473,0.68552,0.36742,0.87567,0.000489947,0.000489947,0.000489947
14,802.35,0.70127,0.48923,0.8936,0.98356,0.94843,0.96722,0.82057,0.61274,0.3368,0.86008,0.000484443,0.000484443,0.000484443
15,901.997,0.69931,0.47756,0.89009,0.9832,0.94674,0.96711,0.79108,0.72394,0.3522,0.88177,0.000478938,0.000478938,0.000478938
16,1001.18,0.68945,0.47128,0.88816,0.9844,0.95054,0.96765,0.80147,0.66911,0.33039,0.87276,0.000473434,0.000473434,0.000473434
17,1099.23,0.69398,0.46171,0.89061,0.98714,0.95415,0.9688,0.80779,0.6526,0.31964,0.86485,0.00046793,0.00046793,0.00046793
18,1197.67,0.69254,0.45822,0.89051,0.98366,0.9516,0.96765,0.80341,0.6749,0.33121,0.87349,0.000462425,0.000462425,0.000462425
19,1296.1,0.68578,0.4572,0.88819,0.98623,0.95534,0.96878,0.82036,0.62056,0.30639,0.85899,0.000456921,0.000456921,0.000456921
20,1392.86,0.67715,0.44843,0.88458,0.98909,0.95126,0.96844,0.82027,0.61214,0.31357,0.85644,0.000451416,0.000451416,0.000451416
21,1489.89,0.68783,0.44757,0.88739,0.98672,0.95405,0.96816,0.80459,0.67443,0.31917,0.87098,0.000445912,0.000445912,0.000445912
22,1591.38,0.67113,0.44057,0.88589,0.98681,0.95551,0.96964,0.8314,0.58183,0.30213,0.85005,0.000440408,0.000440408,0.000440408
23,1688.98,0.66027,0.43669,0.88104,0.98982,0.95329,0.96824,0.83109,0.58601,0.30209,0.8527,0.000434903,0.000434903,0.000434903
24,1785.24,0.65639,0.4319,0.88268,0.98853,0.95617,0.96949,0.83603,0.57397,0.29591,0.85187,0.000429399,0.000429399,0.000429399
25,1882.14,0.65475,0.42699,0.87936,0.9848,0.95801,0.96931,0.81581,0.6404,0.30533,0.86257,0.000423894,0.000423894,0.000423894
26,1980.04,0.65152,0.41942,0.87767,0.99036,0.95893,0.96972,0.82681,0.5977,0.29671,0.85559,0.00041839,0.00041839,0.00041839
27,2077.2,0.6402,0.41621,0.87658,0.98773,0.96107,0.9696,0.82338,0.61055,0.2906,0.85614,0.000412886,0.000412886,0.000412886
28,2175.18,0.6434,0.41419,0.87743,0.9864,0.9595,0.96867,0.83159,0.58089,0.28745,0.85489,0.000407381,0.000407381,0.000407381
29,2272.76,0.64864,0.40675,0.87905,0.98992,0.96156,0.96964,0.83148,0.57815,0.27925,0.85297,0.000401877,0.000401877,0.000401877
30,2370.17,0.6355,0.40783,0.87496,0.99108,0.95947,0.96915,0.83057,0.58185,0.28909,0.85153,0.000396372,0.000396372,0.000396372
31,2468.5,0.63223,0.40446,0.8764,0.98981,0.95913,0.96855,0.83839,0.56338,0.2721,0.8471,0.000390868,0.000390868,0.000390868
32,2572.54,0.63626,0.39742,0.87309,0.99049,0.95756,0.96967,0.8372,0.57127,0.27929,0.84788,0.000385364,0.000385364,0.000385364
33,2671.43,0.62818,0.3895,0.87404,0.98932,0.96241,0.96928,0.84189,0.55543,0.27699,0.84812,0.000379859,0.000379859,0.000379859
34,2772.04,0.63188,0.39247,0.87296,0.99035,0.96287,0.9698,0.84634,0.54454,0.26386,0.84339,0.000374355,0.000374355,0.000374355
35,2873.65,0.63254,0.39457,0.87516,0.99255,0.9629,0.9701,0.83574,0.57797,0.26805,0.85045,0.00036885,0.00036885,0.00036885
36,2973.58,0.62437,0.38965,0.87033,0.99075,0.96449,0.96855,0.83963,0.56868,0.26982,0.85275,0.000363346,0.000363346,0.000363346
37,3073.56,0.62317,0.38711,0.87229,0.99123,0.9631,0.96968,0.84346,0.55261,0.26495,0.84844,0.000357842,0.000357842,0.000357842
38,3172.87,0.62057,0.38759,0.8713,0.99312,0.96374,0.96895,0.8442,0.55318,0.25866,0.8456,0.000352337,0.000352337,0.000352337
39,3272.48,0.61099,0.38081,0.86745,0.99183,0.9626,0.96949,0.84508,0.5562,0.26508,0.84837,0.000346833,0.000346833,0.000346833
40,3371.66,0.61018,0.37738,0.86923,0.99235,0.96391,0.9702,0.84145,0.55943,0.25905,0.84845,0.000341328,0.000341328,0.000341328
41,3470.35,0.61068,0.3773,0.86909,0.99254,0.96387,0.9704,0.8432,0.56224,0.25945,0.8518,0.000335824,0.000335824,0.000335824
42,3571.78,0.60738,0.37485,0.86809,0.99335,0.96399,0.97001,0.84279,0.55616,0.25876,0.84963,0.00033032,0.00033032,0.00033032
43,3670.88,0.61554,0.37533,0.86996,0.992,0.96315,0.97071,0.84049,0.5676,0.25752,0.85011,0.000324815,0.000324815,0.000324815
44,3770.13,0.60551,0.36767,0.8662,0.99349,0.96356,0.97029,0.8481,0.54508,0.2524,0.84413,0.000319311,0.000319311,0.000319311
45,3869.26,0.59762,0.36372,0.86514,0.99475,0.96274,0.96935,0.83826,0.5743,0.25405,0.84872,0.000313806,0.000313806,0.000313806
46,3968.32,0.59632,0.36441,0.86582,0.99286,0.96457,0.96986,0.8499,0.53864,0.24979,0.84462,0.000308302,0.000308302,0.000308302
47,4067.68,0.60035,0.36348,0.86567,0.99392,0.9645,0.96988,0.84627,0.54865,0.24889,0.84368,0.000302798,0.000302798,0.000302798
48,4165.94,0.59855,0.36232,0.8654,0.99285,0.96498,0.97001,0.84962,0.54156,0.25064,0.84346,0.000297293,0.000297293,0.000297293
49,4264.59,0.59897,0.36425,0.86523,0.99384,0.9644,0.96989,0.84869,0.5392,0.2464,0.84356,0.000291789,0.000291789,0.000291789
50,4362.15,0.59264,0.3583,0.86416,0.99272,0.96621,0.97034,0.84419,0.54671,0.24643,0.8441,0.000286284,0.000286284,0.000286284
51,4460.56,0.58614,0.35508,0.86296,0.99417,0.96545,0.97043,0.85045,0.53679,0.24493,0.84239,0.00028078,0.00028078,0.00028078
52,4561.64,0.5863,0.35301,0.86267,0.99429,0.96601,0.96988,0.85526,0.52482,0.23752,0.84086,0.000275276,0.000275276,0.000275276
53,4659.5,0.5838,0.35061,0.86268,0.99337,0.96629,0.97036,0.85366,0.53263,0.23916,0.84274,0.000269771,0.000269771,0.000269771
54,4756.86,0.58464,0.35053,0.86352,0.99479,0.96669,0.97013,0.84917,0.54374,0.24124,0.84257,0.000264267,0.000264267,0.000264267
55,4854.78,0.57824,0.35026,0.86205,0.99472,0.96689,0.97023,0.85408,0.53033,0.2373,0.84129,0.000258762,0.000258762,0.000258762
56,4951.94,0.58493,0.3462,0.86173,0.9945,0.9663,0.97006,0.85065,0.54766,0.23962,0.84278,0.000253258,0.000253258,0.000253258
57,5049.25,0.58275,0.34723,0.86302,0.99513,0.9667,0.97052,0.85332,0.53573,0.23873,0.8412,0.000247754,0.000247754,0.000247754
58,5146.96,0.57398,0.34248,0.85984,0.99528,0.96545,0.96947,0.8545,0.52906,0.23625,0.84046,0.000242249,0.000242249,0.000242249
59,5244.5,0.57557,0.34378,0.86051,0.99483,0.96646,0.97017,0.85276,0.53354,0.2369,0.84125,0.000236745,0.000236745,0.000236745
60,5342.77,0.57161,0.3403,0.85986,0.99531,0.966,0.97002,0.85346,0.52969,0.2364,0.84042,0.00023124,0.00023124,0.00023124
61,5440.21,0.57251,0.34122,0.85973,0.99598,0.96577,0.97005,0.85306,0.53138,0.23416,0.84027,0.000225736,0.000225736,0.000225736
62,5537.32,0.5723,0.33777,0.85989,0.99469,0.96601,0.96995,0.85346,0.52942,0.23404,0.84238,0.000220232,0.000220232,0.000220232
63,5634.56,0.56579,0.33522,0.85783,0.99594,0.96678,0.97075,0.85279,0.54081,0.23499,0.84128,0.000214727,0.000214727,0.000214727
64,5734.01,0.57215,0.33499,0.85924,0.99521,0.96682,0.9703,0.85369,0.52722,0.23324,0.84122,0.000209223,0.000209223,0.000209223
65,5833.95,0.56763,0.33347,0.8584,0.9954,0.96706,0.97023,0.85458,0.52325,0.23106,0.8404,0.000203718,0.000203718,0.000203718
66,5932.82,0.56491,0.33233,0.85758,0.99547,0.96607,0.97064,0.85679,0.51577,0.23146,0.83834,0.000198214,0.000198214,0.000198214
67,6032.92,0.5675,0.33316,0.85731,0.9951,0.9652,0.97059,0.85538,0.5216,0.23118,0.84025,0.00019271,0.00019271,0.00019271
68,6132.43,0.56241,0.32942,0.85919,0.99567,0.96679,0.97003,0.85732,0.51492,0.2291,0.83885,0.000187205,0.000187205,0.000187205
69,6232.11,0.55866,0.32799,0.85584,0.99508,0.96691,0.97069,0.85722,0.51559,0.22871,0.83816,0.000181701,0.000181701,0.000181701
70,6331.64,0.55405,0.32409,0.85353,0.99569,0.96627,0.97071,0.85724,0.51577,0.22729,0.83776,0.000176196,0.000176196,0.000176196
71,6428.9,0.55777,0.32514,0.85554,0.99545,0.96647,0.97077,0.85605,0.5233,0.22952,0.83889,0.000170692,0.000170692,0.000170692
72,6527.09,0.54982,0.32243,0.85536,0.995,0.96746,0.97063,0.85582,0.5223,0.22774,0.83836,0.000165188,0.000165188,0.000165188
73,6627.37,0.54996,0.32369,0.85558,0.99561,0.96635,0.96999,0.85645,0.51843,0.22808,0.83984,0.000159683,0.000159683,0.000159683
74,6727.63,0.54619,0.32205,0.85213,0.99431,0.96708,0.97073,0.85603,0.52055,0.22747,0.83899,0.000154179,0.000154179,0.000154179
75,6825.72,0.54956,0.31941,0.85335,0.99516,0.9666,0.97101,0.85721,0.51766,0.22575,0.83983,0.000148674,0.000148674,0.000148674
76,6924.25,0.54398,0.31552,0.85187,0.99564,0.96566,0.97014,0.85825,0.50906,0.22462,0.83682,0.00014317,0.00014317,0.00014317
77,7025,0.54457,0.31452,0.85182,0.9946,0.96748,0.97021,0.85878,0.50938,0.22372,0.83745,0.000137666,0.000137666,0.000137666
78,7128.09,0.54231,0.31443,0.85225,0.99559,0.96714,0.97085,0.85855,0.51049,0.22357,0.83731,0.000132161,0.000132161,0.000132161
79,7228.27,0.54034,0.31316,0.85133,0.99555,0.96687,0.97087,0.85895,0.51102,0.22283,0.83778,0.000126657,0.000126657,0.000126657
80,7329.74,0.5386,0.31275,0.85107,0.99607,0.96652,0.97084,0.85785,0.51501,0.22295,0.83984,0.000121152,0.000121152,0.000121152
81,7428.82,0.54715,0.31546,0.85303,0.99474,0.96732,0.97026,0.85863,0.51376,0.22352,0.83935,0.000115648,0.000115648,0.000115648
82,7529.59,0.53745,0.3128,0.85047,0.99532,0.96706,0.97087,0.85863,0.51329,0.22264,0.8389,0.000110144,0.000110144,0.000110144
83,7628.07,0.53722,0.30991,0.84933,0.99485,0.96702,0.97092,0.85728,0.51636,0.22351,0.83911,0.000104639,0.000104639,0.000104639
84,7726.64,0.5321,0.30859,0.8498,0.99576,0.966,0.97101,0.85926,0.51226,0.22302,0.83841,9.91348e-05,9.91348e-05,9.91348e-05
85,7824.66,0.52886,0.30437,0.85031,0.99502,0.96731,0.9702,0.8599,0.5099,0.22224,0.8374,9.36304e-05,9.36304e-05,9.36304e-05
86,7923.47,0.53108,0.30661,0.84977,0.99572,0.96656,0.97026,0.85882,0.51212,0.22119,0.83803,8.8126e-05,8.8126e-05,8.8126e-05
87,8021.51,0.5324,0.30596,0.84956,0.99535,0.96708,0.97019,0.85583,0.52108,0.22355,0.84063,8.26216e-05,8.26216e-05,8.26216e-05
88,8119.55,0.52926,0.30542,0.84794,0.99449,0.96752,0.97091,0.85767,0.51644,0.22242,0.8397,7.71172e-05,7.71172e-05,7.71172e-05
89,108.926,0.53069,0.30638,0.84794,0.99509,0.967,0.97098,0.85779,0.51677,0.2218,0.83912,7.16128e-05,7.16128e-05,7.16128e-05
90,207.267,0.52616,0.30282,0.84765,0.99542,0.96694,0.97096,0.85771,0.51349,0.22033,0.83914,6.61084e-05,6.61084e-05,6.61084e-05
91,301.935,0.53638,0.26287,0.87145,0.99481,0.96521,0.97008,0.85492,0.52275,0.22826,0.84144,6.0604e-05,6.0604e-05,6.0604e-05
92,393.649,0.52892,0.25854,0.86785,0.99405,0.96572,0.96993,0.85567,0.52007,0.22887,0.84,5.50996e-05,5.50996e-05,5.50996e-05
93,485.311,0.52339,0.25709,0.86622,0.99371,0.96561,0.96933,0.85628,0.51908,0.22821,0.84031,4.95952e-05,4.95952e-05,4.95952e-05
94,577.289,0.51708,0.25169,0.8611,0.99422,0.96611,0.96939,0.85645,0.51702,0.22667,0.83921,4.40908e-05,4.40908e-05,4.40908e-05
95,670.129,0.51956,0.25356,0.86545,0.99535,0.96495,0.97009,0.85692,0.51578,0.22527,0.8387,3.85864e-05,3.85864e-05,3.85864e-05
96,761.942,0.52019,0.2535,0.86711,0.99533,0.96482,0.97007,0.85699,0.51448,0.22478,0.83895,3.3082e-05,3.3082e-05,3.3082e-05
97,852.992,0.50961,0.24967,0.85933,0.99487,0.9659,0.97002,0.85679,0.51651,0.22416,0.83999,2.75776e-05,2.75776e-05,2.75776e-05
98,943.813,0.50997,0.24993,0.86261,0.9953,0.96603,0.97,0.8564,0.51945,0.22469,0.84091,2.20732e-05,2.20732e-05,2.20732e-05
99,1035.45,0.51412,0.25068,0.8623,0.99517,0.96628,0.97002,0.85517,0.52075,0.22497,0.84135,1.65688e-05,1.65688e-05,1.65688e-05
100,1127.43,0.50401,0.24688,0.85893,0.99546,0.96587,0.97001,0.85565,0.52055,0.22416,0.84144,1.10644e-05,1.10644e-05,1.10644e-05
```

---

### `robot\models\best.onnx`

- **Loại file:** Model Weights (.onnx)
- **Kích thước file:** 10.18 MB
> ℹ️ *Nội dung file nhị phân được bỏ qua.* 

---

### `robot\models\best.pt`

- **Loại file:** Model Weights (.pt)
- **Kích thước file:** 5.24 MB
> ℹ️ *Nội dung file nhị phân được bỏ qua.* 

---

## Nhóm 3: Nhận diện & Điều khiển gắp (Kinematics, Calibration & Pick-Place)

### `tests\pick_place.py`

*Dung lượng: 8.18 KB | Số dòng: 172*

```python
#!/usr/bin/env python3
# Gắp ở điểm A, đặt ở điểm B bằng đầu hút, CÓ dùng thanh trượt (chưa dùng camera).
# Tay máy điều khiển qua ROS topic (cần node jetmax_control đang chạy), thanh trượt qua hiwonder.Stepper.
#
# Bước 1 (chỉ bay trên A và B, KHÔNG hạ xuống, KHÔNG hút; thanh trượt vẫn chạy thật):
#   python3 pick_place.py --a 0 -210.4 --b 0 -200.4 --zpick -8.8 --zplace -5.4 --sa 1500 --sb 5000 --hover
# Bước 2 (gắp - đặt thật, hỏi Enter trước mỗi pha):
#   python3 pick_place.py --a 0 -210.4 --b 0 -200.4 --zpick -8.8 --zplace -5.4 --sa 1500 --sb 5000
# Khi chắc chắn, thêm --yes để chạy liền không hỏi.
#
# Tọa độ A, B và số bước trượt lấy từ joint_teleop.py (bấm p). Số bước trượt là tính từ HOME của thanh trượt,
# nên script sẽ đưa thanh trượt về HOME trước khi bắt đầu (bỏ qua bằng --no-home nếu chắc chắn nó đang ở home).

import argparse, math, time
import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0   # giống các script trước
TOL = 5.0                      # mm, sai lệch cho phép giữa đích và vị trí thực
FAST, SLOW = 100.0, 40.0       # mm/giây: di chuyển bình thường / khi hạ xuống chạm quân cờ
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0           # bước/giây, dùng để ước lượng thời gian chờ (hãng ghi 1000; để thấp cho chắc)
SLIDER_MIN_Z = 40.0            # chỉ trượt khi tay máy nâng đủ cao

S = {'stepper': None, 'pos': 0, 'enabled': False}


def status():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z


def check(p):
    x, y, z = p
    if not (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN):
        raise ValueError('Điểm (%.1f, %.1f, %.1f) ngoài giới hạn an toàn' % p)


def goto(pub, p, speed=FAST):
    check(p)
    cur = status()
    dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
    t = max(0.5, dist / speed)
    pub.publish(SetJetMax(x=p[0], y=p[1], z=p[2], duration=t))
    rospy.sleep(t + 0.3)
    now = status()
    err = max(abs(now[i] - p[i]) for i in range(3))
    if err > TOL:
        raise RuntimeError('Tay máy chưa tới đích (lệch %.1f mm): lệnh có thể bị chặn bởi giới hạn' % err)


def slide_to(target):
    st = S['stepper']
    if st is None:
        raise RuntimeError('Không có thanh trượt')
    target = int(max(SLIDER_MIN, min(SLIDER_MAX, target)))
    d = target - S['pos']
    if d == 0:
        return
    if status()[2] < SLIDER_MIN_Z:
        raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
    if not S['enabled']:
        st.set_mode(st.SLEEP)                    # chỉ gửi một lần
        S['enabled'] = True
    print('  thanh trượt %d -> %d' % (S['pos'], target))
    st.goto(d)
    time.sleep(abs(d) / SLIDER_SPEED + 0.3)
    S['pos'] = target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, nargs=2, required=True, metavar=('X', 'Y'), help='điểm gắp')
    ap.add_argument('--b', type=float, nargs=2, required=True, metavar=('X', 'Y'), help='điểm đặt')
    ap.add_argument('--zpick', type=float, required=True, help='z khi đầu hút chạm quân cờ ở A')
    ap.add_argument('--zplace', type=float, default=None, help='z khi thả ở B (mặc định = zpick + 3)')
    ap.add_argument('--zsafe', type=float, default=50.0, help='z khi di chuyển ngang')
    ap.add_argument('--sa', type=int, default=0, help='vị trí thanh trượt khi gắp ở A (bước từ home)')
    ap.add_argument('--sb', type=int, default=0, help='vị trí thanh trượt khi đặt ở B (bước từ home)')
    ap.add_argument('--no-home', action='store_true', help='không đưa thanh trượt về home lúc bắt đầu')
    ap.add_argument('--hover', action='store_true', help='chỉ bay trên A và B, không hạ, không hút')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước mỗi pha')
    a = ap.parse_args()
    zplace = a.zplace if a.zplace is not None else a.zpick + 3
    A, B = a.a, a.b
    for s in (a.sa, a.sb):
        if not SLIDER_MIN <= s <= SLIDER_MAX:
            raise SystemExit('Vị trí thanh trượt %d ngoài khoảng %d..%d' % (s, SLIDER_MIN, SLIDER_MAX))

    def ask(text):
        if not a.yes and input(text + ' [Enter = chạy, q = dừng] ').strip().lower() == 'q':
            raise KeyboardInterrupt

    rospy.init_node('pick_place')
    abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    sucker = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
    need_slider = a.sa != 0 or a.sb != 0
    if need_slider:
        try:
            import hiwonder
            S['stepper'] = hiwonder.Stepper(1)
        except Exception as e:
            raise SystemExit('Không khởi tạo được thanh trượt: %s' % e)
    rospy.sleep(1.0)

    start = status()
    print('Vị trí ban đầu:', start)
    print('A=%s (trượt %d)  B=%s (trượt %d)  zpick=%.1f zplace=%.1f zsafe=%.1f %s' % (
        A, a.sa, B, a.sb, a.zpick, zplace, a.zsafe, '[CHỈ BAY, KHÔNG HẠ]' if a.hover else ''))
    for p in [(A[0], A[1], a.zsafe), (B[0], B[1], a.zsafe), (A[0], A[1], a.zpick), (B[0], B[1], zplace)]:
        check(p)                                 # kiểm tra mọi điểm trước khi di chuyển
    try:
        sucker.publish(Bool(data=False))
        if need_slider and not a.no_home:
            ask('Đưa thanh trượt về HOME (vị trí 0). Tay máy phải đang nâng cao và không vướng')
            if start[2] < SLIDER_MIN_Z:
                raise RuntimeError('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
            st = S['stepper']
            st.go_home(True)                     # chặn cho tới khi về home, không có timeout
            st.set_mode(st.EN)
            st.set_div(st.DIV_1_4)
            S['pos'], S['enabled'] = 0, False
        ask('Trượt tới %d và bay tới trên A' % a.sa)
        slide_to(a.sa)
        goto(abs_pub, (A[0], A[1], a.zsafe))
        if a.hover:
            ask('Trượt tới %d và bay tới trên B' % a.sb)
            slide_to(a.sb)
            goto(abs_pub, (B[0], B[1], a.zsafe))
        else:
            ask('Hạ xuống A và hút')
            sucker.publish(Bool(data=True))
            rospy.sleep(0.3)
            goto(abs_pub, (A[0], A[1], a.zpick), SLOW)
            rospy.sleep(0.8)                     # chờ hút bám chắc
            goto(abs_pub, (A[0], A[1], a.zsafe), SLOW)
            ask('Mang tới B (trượt tới %d)' % a.sb)
            slide_to(a.sb)
            goto(abs_pub, (B[0], B[1], a.zsafe))
            ask('Hạ xuống B và thả')
            goto(abs_pub, (B[0], B[1], zplace), SLOW)
            sucker.publish(Bool(data=False))     # thả, mở van xả khí
            rospy.sleep(1.0)
            goto(abs_pub, (B[0], B[1], a.zsafe), SLOW)
        print('Xong')
    except KeyboardInterrupt:
        print('Dừng theo yêu cầu')
    except (ValueError, RuntimeError) as e:
        print('LỖI:', e)
    finally:
        sucker.publish(Bool(data=False))
        rospy.sleep(0.6)
        try:
            cur = status()
            if S['stepper'] is not None and S['pos'] != 0 and cur[2] < SLIDER_MIN_Z:
                abs_pub.publish(SetJetMax(x=cur[0], y=cur[1], z=a.zsafe, duration=1.0))   # nâng lên trước khi trượt về
                rospy.sleep(1.4)
            if S['stepper'] is not None and S['pos'] != 0:
                slide_to(0)
            abs_pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
            rospy.sleep(2.5)
        except Exception as e:
            print('Không tự về được vị trí ban đầu (%s), hãy kiểm tra thủ công' % e)
        if S['stepper'] is not None:
            S['stepper'].set_mode(S['stepper'].EN)
        print('Đã về vị trí ban đầu')


if __name__ == '__main__':
    main()
```

---

### `docs\robot-reference\waste_classification_stepper.py`

*Dung lượng: 10.41 KB | Số dòng: 283*

```python
#!/usr/bin/env python3
import os
import sys
import cv2
import math
import time
import rospy
import numpy as np
import threading
import queue
import hiwonder
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from jetmax_control.msg import SetServo
from yolov5_tensorrt import Yolov5TensorRT

ROS_NODE_NAME = "waste_classification_stepper"

TRT_ENGINE_PATH = os.path.join(sys.path[0], "models/waste_v5_160.trt")
TRT_INPUT_SIZE = 160
TRT_CLASS_NAMES = ('Banana Peel', 'Broken Bones', 'Cigarette End', 'Disposable Chopsticks',
                   'Ketchup', 'Marker', 'Oral Liquid Bottle', 'Plate',
                   'Plastic Bottle', 'Storage Battery', 'Toothbrush', 'Umbrella')
TRT_NUM_CLASSES = 12
WASTE_CLASSES = {
    'food_waste': ('Banana Peel', 'Broken Bones', 'Ketchup'),
    'hazardous_waste': ('Marker', 'Oral Liquid Bottle', 'Storage Battery'),
    'recyclable_waste': ('Plastic Bottle', 'Toothbrush', 'Umbrella'),
    'residual_waste': ('Plate', 'Cigarette End', 'Disposable Chopsticks'),
}
COLORS = {
    'recyclable_waste': (0, 0, 255),
    'hazardous_waste': (255, 0, 0),
    'food_waste': (0, 255, 0),
    'residual_waste': (80, 80, 80)
}

TARGET_POSITION = {
    'recyclable_waste': (0, -180, 5, 5200),
    'hazardous_waste': (0, -180, 5, 4400),
    'food_waste': (0, -180, 5, 3600),
    'residual_waste': (0, -180, 5, 2800)
}

# 现实坐标转换函数
def camera_to_world(cam_mtx, r, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
    # invR * T
    inv_r = np.asmatrix(r_mat).I  # 3*3
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))  # 3*3 dot 3*1 = 3*1
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)  # 3*3 dot 3*1 = 3*1
        # [x,y,1] * invR
        worldPtPlane = np.dot(inv_r, worldPtCam)  # 3*3 dot 3*1 = 3*1
        # zc
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        # zc * [x,y,1] * invR
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        # [X,Y,Z]=zc*[x,y,1]*invR - invR*T
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt


class WasteClassification:
    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = False
        self.moving_box = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.camera_params = None
        self.fps_t0 = time.time()
        self.fps = 0
        self.K = None
        self.R = None
        self.T = None

    def reset(self):
        self.is_running = False
        self.moving_box = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0

    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/card_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)

# 移动控制函数
def moving():
    try:
        c_x, c_y, waste_class_name = state.moving_box # 目标的画面坐标
        x, y, _ = camera_to_world(state.K, state.R, state.T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0] #转换成现实距离
        cur_x, cur_y, cur_z = jetmax.position  # 当前的机械臂位置
        t = math.sqrt(x * x + y * y + 140 * 140) / 140 #计算移动的时间
        new_x, new_y = cur_x + x, cur_y + y #目标的现实坐标
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi # 计算偏转角
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass
        hiwonder.pwm_servo1.set_position(90 - arm_angle, 0.1) #角度补偿

        x, y, z, stepper_position = TARGET_POSITION[waste_class_name] # 读取放置坐标
        jetmax.set_position((new_x, new_y, 50), t) # 移动到目标上方
        rospy.sleep(t + 0.2)

        sucker.set_state(True) # 打开气泵
        jetmax.set_position((new_x, new_y, -12), 0.8) # 吸取目标
        rospy.sleep(0.8)
        jetmax.set_position((new_x, new_y, 30), 0.3)
        rospy.sleep(0.3)
        t = abs(new_x)/120.0 + abs(new_y-cur_y)/120.0
        hiwonder.pwm_servo1.set_position(90, 0.5) #角度回正
        jetmax.set_position((x, y, 50), t) # 移动到放置坐标上方
        rospy.sleep(t)

        stepper.set_mode(stepper.SLEEP) # 设置滑轨使能
        stepper.goto(stepper_position) # 驱动滑轨移动到放置位置
        stepper_time = stepper_position/1000 # 计算需要的时间
        rospy.sleep(stepper_time)

        jetmax.set_position((x, y, z), 0.5) # 到放置位置
        rospy.sleep(0.5)

        sucker.release(3)
        jetmax.set_position((x, y, z + 50), 0.5) 
        rospy.sleep(1)

    finally:
        jetmax.set_position((cur_x, cur_y, cur_z), 1) # 机械臂回到初始位置
        stepper.goto(-stepper_position) # 滑轨回到初始位置
        stepper_time = stepper_position/1000
        rospy.sleep(stepper_time)
        stepper.set_mode(stepper.EN) # 解除滑轨锁定
        state.moving_box = None
        state.runner = None
        print("FINISHED")

# 图像处理函数
def image_proc(image):
    if state.runner is not None:
        return image

    outputs = yolov5.detect(image) # 模型推理
    boxes, confs, classes = yolov5.post_process(image, outputs, 0.5)
    width = image.shape[1]
    height = image.shape[0]
    cards = []
    waste_name = 'None'
    for box, cls_conf, cls_id in zip(boxes, confs, classes):
        x1 = int(box[0] / TRT_INPUT_SIZE * width)
        y1 = int(box[1] / TRT_INPUT_SIZE * height)
        x2 = int(box[2] / TRT_INPUT_SIZE * width)
        y2 = int(box[3] / TRT_INPUT_SIZE * height)
        waste_name = TRT_CLASS_NAMES[cls_id]  # 得到检测到的分类名称
        
        waste_class_name = ''
        for k, v in WASTE_CLASSES.items():
            if waste_name in v:
                waste_class_name = k
                break
        cards.append((cls_conf, x1, y1, x2, y2, waste_class_name))
        cv2.putText(image, waste_class_name, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[waste_class_name], 2)
        cv2.putText(image, waste_name + "{:0.2f}".format(float(cls_conf)), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[waste_class_name], 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[waste_class_name], 3)

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
    else:
        if state.moving_box is None:
            moving_box = max(cards, key=lambda card: card[0])
            conf, x1, y1, x2, y2, waste_class_name = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, waste_class_name
        else:
            l_c_x, l_c_y, l_waste_class_name = state.moving_box
            moving_box = min(cards, key=lambda card: math.sqrt((l_c_x - card[1]) ** 2 + (l_c_y - card[2]) ** 2))

            conf, x1, y1, x2, y2, waste_class_name = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            # 在画面中标记出来
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 6)
            image = cv2.circle(image, (int(c_x), int(c_y)), 1, (255, 255, 255), 10)

            if math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) > 30:
                state.count = 0
            else:
                c_x = l_c_x * 0.2 + c_x * 0.8
                c_y = l_c_y * 0.2 + c_y * 0.8
                state.count += 1
            state.moving_box = c_x, c_y, waste_class_name
            if state.count > 50: #多次检测
                state.count = 0
                print(waste_name,waste_class_name)
                state.runner = threading.Thread(target=moving, daemon=True) #开启移动子线程
                state.runner.start()

    return image


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def image_proc_b():
    ros_image = image_queue.get(block=True)
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    result_img = image_proc(image)
    # fps cal
    fps_t1 = time.time()
    fps_cur = (1.0 / (fps_t1 - state.fps_t0))
    state.fps = fps_cur if state.fps == 0.0 else (state.fps * 0.8 + fps_cur * 0.2)
    state.fps_t0 = fps_t1
    show_fps(result_img, state.fps)
    image_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imshow(ROS_NODE_NAME, image_bgr)
    cv2.waitKey(1)


def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    image_queue = queue.Queue(maxsize=2)
    state = WasteClassification()
    state.load_camera_params()
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()
    stepper = hiwonder.Stepper(1)
    jetmax.go_home(2,2)
    stepper.go_home(True)
    stepper.set_mode(stepper.EN)
    stepper.set_div(stepper.DIV_1_4)
    hiwonder.pwm_servo1.set_position(90, 0.5)
    rospy.sleep(1)
    yolov5 = Yolov5TensorRT(TRT_ENGINE_PATH, TRT_INPUT_SIZE, TRT_NUM_CLASSES)
    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback) # 订阅摄像头画面

    while True:
        try:
            image_proc_b()
            if rospy.is_shutdown():
                sys.exit(0)
        except KeyboardInterrupt:
            break

```

---

### `tests\analyze_images.py`

*Dung lượng: 6.12 KB | Số dòng: 143*

```python
#!/usr/bin/env python3
# Phân tích chất lượng ảnh chụp từ camera JetMax để quyết định có cần tăng cường/xử lý ảnh hay không.
# Chạy trên máy tính (cần: pip install opencv-python numpy):
#   python tests/analyze_images.py data/capture_scan
#   python tests/analyze_images.py data/capture_scan --imgsz 640
#
# Mỗi thư mục con được coi là một NHÓM (ví dụ: static, belt_slow, belt_mid, belt_fast, anh_sang_yeu...),
# ảnh nằm trực tiếp trong thư mục gốc là nhóm ".". Kết quả in ra màn hình và lưu quality_report.csv.
#
# Các ngưỡng cảnh báo là NGƯỠNG THAM KHẢO, phụ thuộc cảnh và băng chuyền của bạn. Hãy so sánh các nhóm
# với nhau (ví dụ độ nét khi băng chuyền chạy so với khi đứng yên) hơn là tin tuyệt đối vào một con số.

import os, sys, csv, argparse
import cv2
import numpy as np

EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
ANALYSIS_SIZE = 640          # cạnh dài khi tính độ nét/nhiễu, để so sánh được giữa các độ phân giải


def noise_sigma(gray):
    """Ước lượng độ lệch chuẩn nhiễu (phương pháp Immerkaer)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(np.mean(np.abs(conv)) * np.sqrt(0.5 * np.pi) / 6.0)


def piece_diameter(gray_small):
    """Ước lượng đường kính quân cờ (px, trên ảnh cạnh dài 640) bằng HoughCircles. Chỉ mang tính tham khảo."""
    blur = cv2.medianBlur(gray_small, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30, minRadius=10, maxRadius=60)
    if circles is None:
        return 0, float('nan')
    d = circles[0, :, 2] * 2
    return len(d), float(np.median(d))


def analyze(path, imgsz):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = ANALYSIS_SIZE / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    b, g, r = [float(c.mean()) for c in cv2.split(small)]
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    n_circ, d640 = piece_diameter(gray)
    return {
        'width': w, 'height': h,
        'sharp_lap': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'sharp_ten': float(np.mean(gx ** 2 + gy ** 2)),
        'brightness': float(gray.mean()),
        'contrast': float(gray.std()),
        'dark_clip_%': float((gray <= 5).mean() * 100),
        'bright_clip_%': float((gray >= 250).mean() * 100),
        'glare_%': float(((hsv[..., 2] >= 245) & (hsv[..., 1] < 40)).mean() * 100),
        'saturation': float(hsv[..., 1].mean()),
        'r_over_g': r / max(g, 1e-6), 'b_over_g': b / max(g, 1e-6),
        'noise_sigma': noise_sigma(gray),
        'n_circles': n_circ,
        'piece_px_640': d640,
        'piece_px_imgsz': d640 * imgsz / ANALYSIS_SIZE if d640 == d640 else float('nan'),
    }


def flags(m, imgsz):
    f = []
    if m['sharp_lap'] < 60:
        f.append('mờ')
    if m['brightness'] < 70:
        f.append('tối')
    if m['brightness'] > 190:
        f.append('quá sáng')
    if m['dark_clip_%'] > 2 or m['bright_clip_%'] > 2:
        f.append('mất chi tiết sáng/tối')
    if m['contrast'] < 35:
        f.append('tương phản thấp')
    if m['noise_sigma'] > 6:
        f.append('nhiễu')
    if m['glare_%'] > 1:
        f.append('chói/phản xạ')
    if m['piece_px_imgsz'] == m['piece_px_imgsz'] and m['piece_px_imgsz'] < 25:
        f.append('quân cờ nhỏ ở imgsz=%d' % imgsz)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--imgsz', type=int, default=640, help='kích thước đầu vào mô hình YOLO sẽ dùng')
    a = ap.parse_args()

    rows, groups = [], {}
    for root, _, files in os.walk(a.folder):
        group = os.path.relpath(root, a.folder)
        for fn in sorted(files):
            if not fn.lower().endswith(EXTS):
                continue
            m = analyze(os.path.join(root, fn), a.imgsz)
            if m is None:
                continue
            m['file'], m['group'] = fn, group
            m['flags'] = ';'.join(flags(m, a.imgsz))
            rows.append(m)
            groups.setdefault(group, []).append(m)
    if not rows:
        print('Không tìm thấy ảnh trong', a.folder)
        return

    keys = ['group', 'file', 'width', 'height', 'sharp_lap', 'sharp_ten', 'brightness', 'contrast',
            'dark_clip_%', 'bright_clip_%', 'glare_%', 'saturation', 'r_over_g', 'b_over_g',
            'noise_sigma', 'n_circles', 'piece_px_640', 'piece_px_imgsz', 'flags']
    out = os.path.join(a.folder, 'quality_report.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in rows:
            w.writerow({k: ('%.2f' % m[k] if isinstance(m[k], float) else m[k]) for k in keys})

    print('%-14s %4s %8s %7s %7s %7s %7s %7s %9s' % (
        'nhóm', 'ảnh', 'độ nét', 'sáng', 'tương', 'nhiễu', 'chói%', 'R/G', 'quân(px)'))
    for g, ms in sorted(groups.items()):
        avg = lambda k: float(np.nanmean([m[k] for m in ms])) if not all(m[k] != m[k] for m in ms) else float('nan')
        print('%-14s %4d %8.1f %7.1f %7.1f %7.2f %7.2f %7.2f %9.1f' % (
            g, len(ms), avg('sharp_lap'), avg('brightness'), avg('contrast'), avg('noise_sigma'),
            avg('glare_%'), avg('r_over_g'), avg('piece_px_imgsz')))
        bad = {}
        for m in ms:
            for fl in m['flags'].split(';'):
                if fl:
                    bad[fl] = bad.get(fl, 0) + 1
        if bad:
            print('   cảnh báo:', ', '.join('%s (%d/%d ảnh)' % (k, v, len(ms)) for k, v in bad.items()))
    print('\nĐã lưu chi tiết từng ảnh:', out)
    print('Độ phân giải ảnh đầu tiên: %dx%d' % (rows[0]['width'], rows[0]['height']))


if __name__ == '__main__':
    main()
```

---

## Nhóm 4: Tổng quan dự án

### `README.md`

*Dung lượng: 153 B | Số dòng: 2*

```markdown
# xiangqi-sorting-jetmax
Real-time Xiangqi piece detection (YOLO11 + OpenCV) and sorting on a moving conveyor belt with a Hiwonder JetMax robotic arm.
```

---

