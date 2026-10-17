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
