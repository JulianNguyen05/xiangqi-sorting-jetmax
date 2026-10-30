#!/usr/bin/env python3
# Chụp ảnh ở tư thế quan sát, lưu vào captures/, và kiểm tra ngay camera thấy bàn cờ tới đâu.
#   python3 tools/capture.py                        # mọi view trong config/robot.yaml
#   python3 tools/capture.py --view v0
#   python3 tools/capture.py --arm 0 -160 210 --slider 0 --name thu1     # thử một tư thế bất kỳ
#   python3 tools/capture.py --here --name thu2     # chụp ngay tại chỗ, không di chuyển
# Ảnh lưu: captures/<tên>.jpg và captures/<tên>_board.jpg (lưới bàn cờ tìm được, nếu thấy).
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from board.geometry import BoardGeometry                          # noqa: E402
from robot import config                                          # noqa: E402
from vision import board_detect                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--view', default=None)
ap.add_argument('--arm', type=float, nargs=3, default=None)
ap.add_argument('--slider', type=int, default=0)
ap.add_argument('--here', action='store_true')
ap.add_argument('--name', default=None)
ap.add_argument('--no-home', action='store_true')
a = ap.parse_args()

import rospy                                                      # noqa: E402
from robot.camera import RosCamera                                # noqa: E402
from robot.motion import Robot                                    # noqa: E402

rospy.init_node('xq_capture', anonymous=True)
cfg = config.robot_cfg()
cam = RosCamera(cfg)
rb = Robot(cfg, config.calib(cfg), backend='ros')
geom = BoardGeometry.load()
out = config.path('captures')
os.makedirs(out, exist_ok=True)

if a.here:
    views = [{'name': a.name or 'here', 'arm': None}]
elif a.arm:
    views = [{'name': a.name or 'arm', 'arm': a.arm, 'slider': a.slider}]
else:
    views = [v for v in cfg['views'] if a.view in (None, v['name'])]
need_slider = any(v.get('arm') and v.get('slider', 0) for v in views)
if rb.use_slider and need_slider and not a.no_home:
    input('Đưa thanh trượt về HOME (tay máy đang cao, không vướng). Enter...')
    rb.home_slider()
try:
    for v in views:
        if v.get('arm'):
            print('-> view %s: tay %s, trượt %d' % (v['name'], v['arm'], v.get('slider', 0)))
            rb.observe(v)
        img = cam.grab(v['name'])
        name = v['name'] if not a.name or len(views) > 1 else a.name
        p = os.path.join(out, '%s.jpg' % name)
        cv2.imwrite(p, img)
        h, w = img.shape[:2]
        fix = board_detect.locate(img, geom, red_hint=(w / 2.0, float(h)))
        if fix is None:
            print('   %s (%dx%d): KHÔNG thấy bàn cờ trong ảnh này' % (p, w, h))
        else:
            inside = sum(1 for (x, y) in fix.pts_px.values() if 0 <= x < w and 0 <= y < h)
            cv2.imwrite(os.path.join(out, '%s_board.jpg' % name), board_detect.draw(img, fix, geom))
            print('   %s: thấy bàn cờ, %d/90 giao điểm nằm trong ảnh, %d/19 đường dò được, %.1f px/mm' % (
                p, inside, fix.n_lines, board_detect._px_per_mm(fix)))
        print('   vị trí tay lúc chụp: (%.1f, %.1f, %.1f), trượt %d' % (rb.status() + (rb.s,)))
finally:
    if not a.here:
        rb.park()
