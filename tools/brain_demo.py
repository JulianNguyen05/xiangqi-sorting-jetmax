#!/usr/bin/env python3
# Chạy thử THƯ VIỆN ảnh -> nước đi (brain/xiangqi_brain.py). Không điều khiển tay máy.
#
#   python3 tools/brain_demo.py anh.jpg                       # ảnh -> FEN -> nước tiếp theo cho Đỏ
#   python3 tools/brain_demo.py anh.jpg --side b
#   python3 tools/brain_demo.py anh2.jpg --prev "<FEN lượt trước>"   # suy ra nước vừa đi + sửa lỗi nhận diện
#   python3 tools/brain_demo.py anh.jpg --repeat 3            # đo thời gian (lần 1 chậm do GPU khởi động)
#   python3 tools/brain_demo.py --camera                      # TRÊN ROBOT: chơi cả ván, người đi quân thay tay máy
#
# Ảnh có chú thích lưu ở runs/brain_<giờ>.jpg.
import argparse
import os
import sys
import time

import cv2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from board import rules as R                         # noqa: E402
from brain.xiangqi_brain import XiangqiBrain         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('image', nargs='?')
ap.add_argument('--prev', default=None, help='FEN đã xác nhận ở lượt trước')
ap.add_argument('--side', default='w', help="bên đi tiếp khi không có --prev: 'w' (Đỏ) / 'b' (Đen)")
ap.add_argument('--camera', action='store_true', help='chụp từ camera robot (ROS), chơi cả ván')
ap.add_argument('--depth', type=int, default=None)
ap.add_argument('--locate', default=None, choices=['auto', 'board', 'lattice'])
ap.add_argument('--repeat', type=int, default=1, help='chạy lại N lần trên cùng ảnh để đo thời gian khi đã "nóng máy"')
a = ap.parse_args()


def save(img, read, text):
    out = img.copy()
    for p in read.get('pieces', []):
        cv2.circle(out, (int(p.x), int(p.y)), int(p.r), (0, 200, 0), 2)
        cv2.putText(out, p.code, (int(p.x) - 6, int(p.y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    p = os.path.join(ROOT, 'runs', 'brain_%s.jpg' % time.strftime('%Y%m%d_%H%M%S'))
    cv2.imwrite(p, out)
    return p


brain = XiangqiBrain(depth=a.depth, locate=a.locate)
if not a.camera:
    if not a.image:
        sys.exit('Cần đường dẫn ảnh, hoặc --camera')
    img = cv2.imread(a.image)
    if img is None:
        sys.exit('Không đọc được ảnh ' + a.image)
    for k in range(max(1, a.repeat)):
        t0 = time.time()
        res = brain.analyze(img, prev_fen=a.prev, side=a.side)
        tt = res['read'].get('times', {})
        print('Lần %d: tổng %.0f ms | tăng cường ảnh %.0f | YOLO %.0f | định vị bàn %.0f | engine %.0f ms' % (
            k + 1, (time.time() - t0) * 1000, tt.get('enhance_ms', 0), tt.get('detect_ms', 0), tt.get('locate_ms', 0),
            res.get('think_ms') or 0))
    rd = res['read']
    print('Quân nhận diện : %d (trên bàn: %d, cách định vị: %s)' % (rd['n_pieces'], rd['n_on_board'], rd['method']))
    print('Trạng thái     : %s %s' % (res['status'], '; '.join(res['issues'])))
    if res['last_move']:
        print('Nước vừa đi    : %s (%s), đã sửa %d ô nhận diện' % (res['last_move'], res['last_move_vn'], res['corrected']))
    print('FEN            : %s' % res['fen'])
    if res['best_move']:
        print('NƯỚC TIẾP THEO : %s = %s (%s -> %s), điểm %s, %.0f ms' % (
            res['best_move'], res['best_move_vn'], res['best_move_from'], res['best_move_to'],
            res['score_cp'] if res['mate'] is None else 'chiếu bí sau %d' % res['mate'], res['think_ms']))
    print('Ảnh            :', save(img, rd, '%s %s' % (res['status'], res['best_move_vn'] or '')))
    brain.close()
    sys.exit(0)

# ------------------------------------------------------------------ chơi cả ván bằng camera robot
import rospy                                         # noqa: E402
from robot import config                             # noqa: E402
from robot.camera import RosCamera                   # noqa: E402

rospy.init_node('xq_brain_demo', anonymous=True)
cam = RosCamera(config.robot_cfg())
print('Robot cầm %s. Người đi quân của mình rồi nhấn Enter; robot chỉ NÓI nước đi, bạn đi giúp robot.'
      % ('Đen' if brain.game.robot_side == R.BLACK else 'Đỏ'))
while not rospy.is_shutdown():
    if input('\nEnter = chụp, q = thoát: ').strip().lower() == 'q':
        break
    img = cam.grab()
    res = brain.step(img)
    st = res['status']
    if res.get('human_move'):
        print('Người vừa đi : %s (sửa %d ô)' % (res['human_move']['vn'], res['human_move']['corrected']))
    if st == 'robot_move':
        m = res['robot_move']
        print('ROBOT ĐI     : %s  (%s -> %s)%s  [%.0f ms]' % (m['vn'], m['from'], m['to'],
              ', ăn %s' % m['captures'] if m['captures'] else '', m['think_ms']))
        input('Đi giúp robot nước trên rồi nhấn Enter...')
        brain.confirm(m['uci'])
        print('FEN          :', brain.game.fen)
    elif st == 'retake':
        print('Ảnh chưa chắc chắn -> nhấn Enter để chụp lại')
    elif st == 'waiting':
        print('Chưa thấy nước đi mới.')
    else:
        print('Hết ván:', res.get('result') or res)
        break
    save(img, res.get('read', {}), st)
brain.close()
