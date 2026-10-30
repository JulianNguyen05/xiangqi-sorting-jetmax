#!/usr/bin/env python3
# Lái tay máy bằng bàn phím để ĐO các thông số ngày đầu: vùng với, z chạm quân, chọn tư thế quan sát.
#   python3 tools/teleop.py              # thoát: về chỗ cũ
#   python3 tools/teleop.py --stay       # thoát: để nguyên
#   python3 tools/teleop.py --no-home    # không đưa thanh trượt về HOME lúc đầu
# Enter: in và ghi điểm hiện tại vào captures/teleop_points.txt (để chép vào config).
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from robot import config                                         # noqa: E402
from robot.jog import Jogger                                     # noqa: E402
from robot.motion import Robot                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--stay', action='store_true')
ap.add_argument('--no-home', action='store_true')
ap.add_argument('--wide', action='store_true', help='nới giới hạn an toàn (x ±200, y -300..200) để dò vùng với thật')
a = ap.parse_args()

import rospy                                                     # noqa: E402
rospy.init_node('xq_teleop', anonymous=True)
cfg = config.robot_cfg()
if a.wide:
    cfg['limits'].update({'x': [-200, 200], 'y': [-300, 200], 'r_max': 400.0})
    print('GIỚI HẠN NỚI RỘNG: đi chậm, quan sát kỹ, tay máy tự từ chối điểm ngoài tầm động học.')
rb = Robot(cfg, config.calib(cfg), backend='ros')
if rb.use_slider and not a.no_home:
    input('Đưa thanh trượt về HOME. Tay máy phải đang cao và không vướng. Enter để chạy...')
    rb.home_slider()
jog = Jogger(rb)
os.makedirs(config.path('captures'), exist_ok=True)
log = open(config.path('captures/teleop_points.txt'), 'a')
try:
    while True:
        r = jog.loop('Enter = ghi điểm hiện tại, q = thoát')
        if r == 'quit':
            break
        x, y, z = rb.status()
        line = '%s  robot=(%.1f, %.1f, %.1f)  slider=%d' % (time.strftime('%H:%M:%S'), x, y, z, rb.s)
        print('  GHI: ' + line)
        log.write(line + '\n')
        log.flush()
finally:
    rb.b.sucker(False)
    if not a.stay:
        rb.park()
    log.close()
