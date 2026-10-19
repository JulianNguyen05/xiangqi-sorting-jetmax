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
