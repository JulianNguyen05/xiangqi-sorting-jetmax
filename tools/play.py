#!/usr/bin/env python3
# ROBOT ĐÁNH CỜ VỚI NGƯỜI: nhìn bàn -> thư viện XiangqiBrain tính nước -> tay máy gắp / đặt quân.
# Nước nào có ô nằm ngoài tầm với của tay máy thì robot NÓI nước đó, người đi giúp.
#
# Cần trước: tư thế quan sát trong config (views[0]) + hiệu chỉnh camera (tools/calibrate.py, RUNBOOK mục 5).
#
#   python3 tools/play.py --hover        # CHẠY THỬ: tay chỉ bay tới trên ô gắp / ô đặt rồi lên, không hút
#   python3 tools/play.py                # đánh thật
#   python3 tools/play.py --drop 120 -150    # chỗ bỏ quân bị ăn (toạ độ robot, mm); không có thì người tự nhặt
#   python3 tools/play.py --robot-side red   # robot cầm Đỏ (đi trước)
#
# Mỗi lượt: người đi quân -> Enter. Biên bản ván: runs/play_<thời gian>.csv
import argparse
import os
import sys
import time

import cv2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from board import rules as R                                      # noqa: E402
from board.geometry import square_name                            # noqa: E402
from robot import config                                          # noqa: E402
from vision import world                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--hover', action='store_true', help='chỉ bay tới trên ô, không gắp (chạy thử)')
ap.add_argument('--drop', type=float, nargs=2, default=None, metavar=('X', 'Y'),
                help='chỗ bỏ quân bị ăn, toạ độ thế giới/robot (mm). Mặc định: play.drop trong config, không có thì người nhặt')
ap.add_argument('--view', default=None, help='tên view quan sát (mặc định view đầu tiên)')
ap.add_argument('--robot-side', default='black', choices=['black', 'red'])
ap.add_argument('--depth', type=int, default=None)
a = ap.parse_args()

import rospy                                                      # noqa: E402
from robot.camera import RosCamera                                # noqa: E402
from robot.motion import Robot, MoveError                         # noqa: E402
from brain.xiangqi_brain import XiangqiBrain                      # noqa: E402

rospy.init_node('xq_play', anonymous=True)
cfg = config.robot_cfg()
cal = config.calib(cfg) or {}
views = cfg['views']
view = next((v for v in views if v['name'] == a.view), None) if a.view else views[0]
if view is None:
    sys.exit('Không có view %s trong config' % a.view)
cw = world.Calib(cal)
pose = cw.pose_for(view['arm'])
if pose is None:
    sys.exit('Tư thế quan sát %s %s chưa hiệu chỉnh camera.\n'
             'Ghi tư thế vào config/robot.local.yaml (views) rồi chạy: python3 tools/calibrate.py  (RUNBOOK mục 5)'
             % (view['name'], view['arm']))
drop = a.drop or (cfg.get('play') or {}).get('drop')

cam = RosCamera(cfg)
cam.grab(timeout=10.0)
rb = Robot(cfg, cal, backend='ros')
brain = XiangqiBrain(depth=a.depth)
robot_side = R.BLACK if a.robot_side == 'black' else R.RED
brain.new_game(robot_side=robot_side)

os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
stamp = time.strftime('%Y%m%d_%H%M%S')
log_path = os.path.join(ROOT, 'runs', 'play_%s.csv' % stamp)
with open(log_path, 'w') as fh:
    fh.write('luot,gio,trang_thai,nguoi_di,so_o_sua,robot_di,thuc_hien,tong_ms,yolo_ms,dinh_vi_ms,engine_ms,tay_may_s,kiem_tra\n')
print('Biên bản:', log_path)


def look():
    """Về tư thế quan sát, chụp. Trả về (ảnh, ViewMap)."""
    rb.observe(view)
    return cam.grab(view['name'], timeout=10.0), world.ViewMap(pose, rb.s, cw.slider_vec or rb.vec)


def world_of(px, vm, sq, kind):
    """Ô -> toạ độ thế giới. kind='piece': tâm quân (mặt trên quân, H_top); 'square': giao điểm (mặt bàn, H_table)."""
    if kind == 'piece' and sq in px['pieces']:
        return tuple(vm.top(*px['pieces'][sq]))
    return tuple(vm.table(*px['grid'][sq]))


def execute(m, read, vm):
    """Gắp / đặt cho nước m. Trả về 'robot' (tay máy đi), 'human' (ngoài tầm với / lỗi, người đi giúp)."""
    px = read.get('px')
    if read.get('method') != 'board' or not px:
        print('  Không có lưới bàn cờ chính xác (định vị: %s) -> người đi giúp.' % read.get('method'))
        return 'human'
    fr, to = R.uci_to_move(m['uci'])
    p_from = world_of(px, vm, fr, 'piece')
    p_to = world_of(px, vm, to, 'square')
    p_cap = world_of(px, vm, to, 'piece') if m['captures'] else None
    need = [('ô gắp %s' % m['from'], p_from), ('ô đặt %s' % m['to'], p_to)]
    if p_cap and drop:
        need.append(('chỗ bỏ quân bị ăn', tuple(drop)))
    far = [(n, p) for n, p in need if not rb.reachable(*p)]
    if far:
        print('  NGOÀI TẦM VỚI: %s' % ', '.join('%s (%.0f, %.0f)' % (n, p[0], p[1]) for n, p in far))
        return 'human'
    try:
        if a.hover:
            if p_cap:
                rb.hover_check(*p_cap)
            rb.hover_check(*p_from)
            rb.hover_check(*p_to)
            return 'human'
        if p_cap:
            if drop:
                print('  Ăn quân: nhặt %s ở %s bỏ ra ngoài' % (m['captures'], m['to']))
                rb.pick(*p_cap)
                rb.place(*drop)
            else:
                input('  Robot ăn %s ở %s: NHẶT quân đó ra khỏi bàn rồi nhấn Enter...' % (m['captures'], m['to']))
        rb.pick(*p_from)
        rb.place(*p_to)
        return 'robot'
    except MoveError as e:
        print('  LỖI TAY MÁY: %s' % e)
        rb.safe_stop()
        return 'human'


turn = 0
try:
    print('Robot cầm %s. %s' % ('Đen' if robot_side == R.BLACK else 'Đỏ',
                                'Chế độ CHẠY THỬ (--hover): tay chỉ bay tới trên ô, bạn đi quân giúp robot.' if a.hover else ''))
    first = robot_side == R.RED
    while not rospy.is_shutdown():
        if not first and input('\nĐi quân của bạn rồi nhấn Enter (q = thoát): ').strip().lower() == 'q':
            break
        first = False
        t0 = time.time()
        img, vm = look()
        res = brain.step(img)
        total = (time.time() - t0) * 1000
        st = res['status']
        rd = res.get('read') or {}
        tt = rd.get('times', {})
        hm = res.get('human_move') or {}
        if hm:
            print('Bạn vừa đi  : %s (sửa %d ô)' % (hm['vn'], hm['corrected']))
        done, arm_s, check = '', 0.0, ''
        if st == 'robot_move':
            m = res['robot_move']
            print('ROBOT ĐI    : %s (%s -> %s)%s  [nhìn + tính %.1f s, engine %.1f s]' % (
                m['vn'], m['from'], m['to'], ', ăn %s' % m['captures'] if m['captures'] else '', total / 1000.0,
                (m['think_ms'] or 0) / 1000.0))
            if not rd.get('px'):                             # lượt robot đi trước: step() không đọc ảnh
                rd = brain.read_board(img)
            t1 = time.time()
            done = execute(m, rd, vm)
            arm_s = time.time() - t1
            if done == 'human':
                input('  -> Bạn đi giúp robot nước trên rồi nhấn Enter...')
            brain.confirm(m['uci'])
            if done == 'robot':                              # chụp lại kiểm tra quân đã tới đúng ô
                img2, _ = look()
                rd2 = brain.read_board(img2)
                if rd2.get('board') is not None:
                    want = R.parse_fen(brain.game.fen)[0]
                    diff = [square_name(*k) for k in set(want) | set(rd2['board']) if want.get(k) != rd2['board'].get(k)]
                    check = 'khớp' if not diff else 'lệch %d ô: %s' % (len(diff), ' '.join(sorted(diff)))
                    print('  Kiểm tra sau khi đi: %s  (tay máy %.1f s)' % (check, arm_s))
                    if diff:
                        print('  -> Nếu quân đặt lệch, chỉnh lại bằng tay rồi đi tiếp.')
        elif st == 'retake':
            print('Ảnh chưa chắc chắn -> nhấn Enter để chụp lại')
        elif st == 'waiting':
            print('Chưa thấy nước đi mới.')
        else:
            print('Hết ván:', res.get('result') or st)
        turn += 1
        with open(log_path, 'a') as fh:
            fh.write('%d,%s,%s,%s,%s,%s,%s,%.0f,%.0f,%.0f,%.0f,%.1f,%s\n' % (
                turn, time.strftime('%H:%M:%S'), st, hm.get('uci', ''), hm.get('corrected', ''),
                (res.get('robot_move') or {}).get('uci', ''), done, total, tt.get('detect_ms', 0), tt.get('locate_ms', 0),
                (res.get('robot_move') or {}).get('think_ms') or 0, arm_s, check))
        p = os.path.join(ROOT, 'runs', 'play_%s_%02d.jpg' % (stamp, turn))
        cv2.imwrite(p, img)
        if st == 'game_over':
            break
finally:
    rb.safe_stop()
    try:
        rb.park()
    except Exception:
        pass
    brain.close()
    print('Biên bản:', log_path)
