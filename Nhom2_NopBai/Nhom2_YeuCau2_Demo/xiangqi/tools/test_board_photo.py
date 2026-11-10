#!/usr/bin/env python3
# Test định vị bàn cờ trên ảnh chụp (không cần robot).
#   python3 tools/test_board_photo.py anh.jpg              -> ghi anh_board.jpg (lưới + 90 giao điểm)
#   python3 tools/test_board_photo.py anh.jpg --hint 480 0  -> phía Đỏ là cạnh TRÊN ảnh
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from board.geometry import BoardGeometry          # noqa: E402
from vision import board_detect                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('images', nargs='+')
ap.add_argument('--hint', type=float, nargs=2, default=None, metavar=('X', 'Y'), help='điểm ảnh gần phía Đỏ')
ap.add_argument('--debug', action='store_true', help='ghi thêm ảnh mặt nạ đỏ và ảnh nắn')
a = ap.parse_args()

geom = BoardGeometry.load()
for path in a.images:
    img = cv2.imread(path)
    if img is None:
        print('Không đọc được', path)
        continue
    dbg = {}
    fix = board_detect.locate(img, geom, red_hint=a.hint, debug=dbg)
    base = os.path.splitext(path)[0]
    if fix is None:
        print('%s: KHÔNG tìm thấy bàn cờ' % path)
        continue
    print('%s: %d/90 giao điểm, %d/19 đường, độ lệch %.2f mm' % (path, len(fix.pts_px), fix.n_lines, fix.rms_mm))
    cv2.imwrite(base + '_board.jpg', board_detect.draw(img, fix, geom))
    if a.debug:
        cv2.imwrite(base + '_mask.png', dbg['mask'])
        cv2.imwrite(base + '_warped.png', dbg['warped'])
