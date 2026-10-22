#!/usr/bin/env python3
# Quét + nhận diện: thanh trượt -> tay máy -> chụp ảnh -> YOLO11 nhận diện, tại từng điểm dừng.
# Ghép từ scan_capture.py (điều khiển, giới hạn an toàn) và detect_view.py (mô hình).
# CHƯA gắp quân cờ; chỉ di chuyển, chụp, nhận diện, vẽ và ghi kết quả.
#
# Đặt scan_detect.py CÙNG THƯ MỤC với detect_view.py (file này import từ đó).
# Yêu cầu: node jetmax_control và usb_cam đang chạy.
#
# Ví dụ:
#   python3 scan_detect.py --model ~/models/best.onnx
#   python3 scan_detect.py --model ~/models/best.onnx --sliders 0 1500 3000 --conf 0.4
#   python3 scan_detect.py --model ~/models/best.onnx --yes          # không hỏi Enter mỗi điểm
#
# Kết quả trong ~/scan_detect/<thời điểm chạy>/ :
#   raw_*.jpg (ảnh gốc), det_*.jpg (ảnh đã vẽ khung), detections.csv (mỗi dòng 1 quân cờ).
# Kết thúc/lỗi/q/Ctrl+C: tay máy về chỗ cũ, thanh trượt về 0.

import os, csv, time, argparse
import cv2
import numpy as np
import rospy
import hiwonder
from sensor_msgs.msg import Image
from jetmax_control.msg import SetJetMax, JetMax
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]
from detect_view import load_backend, draw

TOPIC = '/usb_cam/image_rect_color'
OUT_ROOT = os.path.expanduser('~/scan_detect')

# ---- Kế hoạch quét (chỉnh ở đây hoặc bằng --sliders) ----
SLIDER_POS = [0, 1000, 2000]                 # bước từ home
ARM_OFFSETS = [(0, 0, 0), (20, 0, 0), (-20, 0, 0),
               (0, 20, 0), (0, -20, 0), (0, 0, -20)]   # mm, tính từ vị trí ban đầu
DUR = 2.0                                    # giây mỗi lần di chuyển tay máy

# ---- Giới hạn an toàn (giống scan_capture.py / pick_place.py) ----
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0                         # bước/giây, dùng ước lượng thời gian chờ (hãng ghi 1000)
SLIDER_MIN_Z = 40.0                          # chỉ trượt khi tay máy nâng đủ cao
X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0
TOL = 5.0                                    # mm


class Stop(Exception):
    pass


state = {'img': None, 'n': 0}


def on_image(msg):
    state['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()   # RGB
    state['n'] += 1


def get_pos():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z


def check_target(x, y, z):
    ok = (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
          and (x * x + y * y) ** 0.5 >= R_MIN)
    if not ok:
        raise ValueError('Đích (%.1f, %.1f, %.1f) ngoài giới hạn an toàn, bỏ qua' % (x, y, z))


def move_arm(pub, target):
    check_target(*target)
    pub.publish(SetJetMax(x=target[0], y=target[1], z=target[2], duration=DUR))
    rospy.sleep(DUR + 0.7)
    cur = get_pos()
    if max(abs(cur[i] - target[i]) for i in range(3)) > TOL:
        print('  CẢNH BÁO: tay máy lệch so với đích (lệnh có thể bị chặn bởi giới hạn)')
    return cur


def move_slider(stepper, pos, new):
    new = int(max(SLIDER_MIN, min(SLIDER_MAX, new)))
    d = new - pos
    if d == 0:
        return pos
    if get_pos()[2] < SLIDER_MIN_Z:
        raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
    stepper.set_mode(stepper.SLEEP)
    stepper.goto(d)                          # bước TƯƠNG ĐỐI
    time.sleep(abs(d) / SLIDER_SPEED + 0.5)
    return new


def grab():
    t0, n0 = time.time(), state['n']
    while state['n'] < n0 + 2:               # đợi khung hình mới sau khi dừng hẳn
        if time.time() - t0 > 3:
            raise RuntimeError('Không nhận được ảnh từ ' + TOPIC)
        time.sleep(0.02)
    return cv2.cvtColor(state['img'], cv2.COLOR_RGB2BGR)   # mô hình train trên ảnh BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='đường dẫn best.pt hoặc best.onnx')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--sliders', type=int, nargs='+', default=SLIDER_POS, help='các vị trí thanh trượt (bước từ home)')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước mỗi điểm dừng')
    ap.add_argument('--no-show', action='store_true', help='không mở cửa sổ hiển thị')
    a = ap.parse_args()
    for s in a.sliders:
        if not SLIDER_MIN <= s <= SLIDER_MAX:
            raise SystemExit('Vị trí thanh trượt %d ngoài khoảng %d..%d' % (s, SLIDER_MIN, SLIDER_MAX))

    def ask(text):
        if not a.yes and input(text + ' [Enter = chạy, q = dừng] ').strip().lower() == 'q':
            raise Stop()

    det = load_backend(os.path.expanduser(a.model), a.conf, a.iou)   # nạp mô hình trước khi động tới robot
    show = not a.no_show and bool(os.environ.get('DISPLAY'))

    out = os.path.join(OUT_ROOT, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)

    rospy.init_node('scan_detect')
    pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
    rospy.sleep(1.0)

    start = get_pos()
    print('Vị trí tay máy ban đầu =', start)
    print('Thanh trượt sẽ về HOME. Hãy chắc tay máy đang nâng cao (z >= %d) và không vướng vật cản.' % SLIDER_MIN_Z)
    if start[2] < SLIDER_MIN_Z:
        raise SystemExit('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
    ask('Bắt đầu')

    stepper = hiwonder.Stepper(1)
    pos = 0
    f = open(os.path.join(out, 'detections.csv'), 'w', newline='')
    w = csv.writer(f)
    w.writerow(['image', 'slider', 'arm_x', 'arm_y', 'arm_z', 'class', 'conf', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy'])
    n = 0
    try:
        stepper.go_home(True)                # chặn cho tới khi về home (không có timeout, canh nút nguồn)
        stepper.set_mode(stepper.EN)
        stepper.set_div(stepper.DIV_1_4)

        for sp in a.sliders:
            ask('Thanh trượt -> %d' % sp)
            move_arm(pub, start)             # luôn về tư thế cao ban đầu trước khi trượt
            pos = move_slider(stepper, pos, sp)
            print('  thanh trượt = %d' % pos)

            for off in ARM_OFFSETS:
                target = (start[0] + off[0], start[1] + off[1], start[2] + off[2])
                ask('  Tay máy offset %s' % (off,))
                try:
                    cur = move_arm(pub, target)
                except ValueError as e:
                    print('  ', e)
                    continue
                img = grab()
                t0 = time.time()
                dets = det(img)
                ms = (time.time() - t0) * 1000
                tag = 's%d_%03d' % (pos, n)
                cv2.imwrite(os.path.join(out, 'raw_%s.jpg' % tag), img)
                vis = draw(img.copy(), dets, det.names, 0, ms)
                cv2.imwrite(os.path.join(out, 'det_%s.jpg' % tag), vis)
                for x1, y1, x2, y2, c, k in dets:
                    name = det.names[int(k)] if int(k) < len(det.names) else str(int(k))
                    w.writerow(['det_%s.jpg' % tag, pos, '%.1f' % cur[0], '%.1f' % cur[1], '%.1f' % cur[2],
                                name, '%.3f' % c, int(x1), int(y1), int(x2), int(y2),
                                int((x1 + x2) / 2), int((y1 + y2) / 2)])
                f.flush()
                print('  %s: %d quân (%.0f ms) %s' % (tag, len(dets), ms,
                                                       ', '.join(sorted(det.names[int(d[5])] for d in dets))))
                if show:
                    cv2.imshow('scan_detect', vis)
                    cv2.waitKey(1)
                n += 1
    except Stop:
        print('Dừng theo yêu cầu')
    except (ValueError, RuntimeError) as e:
        print('LỖI:', e)
    finally:
        f.close()
        try:
            pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
            rospy.sleep(2.5)
            if pos > 0:
                stepper.goto(-pos)           # quay lại home
                time.sleep(pos / SLIDER_SPEED + 0.5)
            stepper.set_mode(stepper.EN)
        finally:
            if show:
                cv2.destroyAllWindows()
            print('Đã về vị trí ban đầu. Đã xử lý %d ảnh, kết quả ở %s' % (n, out))


if __name__ == '__main__':
    main()
