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