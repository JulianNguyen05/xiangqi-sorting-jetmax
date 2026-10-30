#!/usr/bin/env python3
# Tự kiểm tra phần mềm (không cần robot, không di chuyển gì). Chạy trên laptop và trên robot:
#   python3 tests/selftest.py
# Dùng 2 ảnh mẫu trong samples/. Mọi dòng phải là [OK].
import itertools
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from board import fen                                         # noqa: E402
from board.geometry import BoardGeometry                      # noqa: E402
from planner import assign                                    # noqa: E402
from robot import config, motion                              # noqa: E402
from vision import board_detect as bd, pieces as pc, world as wd   # noqa: E402

fails = []


def check(name, cond, info=''):
    print(('  [OK]   ' if cond else '  [LỖI]  ') + name + ('  ' + info if info else ''))
    if not cond:
        fails.append(name)


g = BoardGeometry.load()
empty = cv2.imread(os.path.join(ROOT, 'samples/board_empty.jpg'))
start = cv2.imread(os.path.join(ROOT, 'samples/start_position.jpg'))

print('== Bàn cờ')
f1 = bd.locate(empty, g)
check('bàn trống: 90 giao điểm', f1 is not None and len(f1.pts_px) == 90)
f2 = bd.locate(start, g, red_hint=(1280, 480))
check('bàn đủ 32 quân: 90 giao điểm', f2 is not None and len(f2.pts_px) == 90,
      'lệch %.2f mm' % f2.rms_mm if f2 else '')

print('== Quân (không cần model)')
pos = fen.parse(fen.START)
r = 12.5 * 1.04 * bd._px_per_mm(f2)
discs = pc.find_discs(start, r)
check('tìm đủ 32 quân', len(discs) == 32, 'thấy %d' % len(discs))
pts = np.array([f2.pts_px[k] for k in pos])
codes = list(pos.values())
ok = 0
for x, y, rr, c, cf in discs:
    i = int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
    ok += c == ('red' if codes[i].isupper() else 'black')
check('màu Đỏ/Đen đúng', ok == len(discs), '%d/%d' % (ok, len(discs)))

print('== Lập kế hoạch')
C = np.random.RandomState(0).rand(5, 6)
sol = assign.hungarian(C)
best = min(itertools.permutations(range(6), 5), key=lambda pm: sum(C[i, pm[i]] for i in range(5)))
check('Hungarian', abs(sum(C[i, sol[i]] for i in range(5)) - sum(C[i, best[i]] for i in range(5))) < 1e-9)
Hw = np.array([[0.42, 0, -200], [0, -0.42, 520], [0, 0, 1]])
pose = {'H_top': Hw.tolist(), 'H_table': Hw.tolist()}
bw = wd.locate_board([(empty, wd.ViewMap(pose, 0, (0, 0)))], g, wd.apply_h(Hw, 480, 1280))
check('định vị bàn trong hệ thế giới', bw is not None and bw.squares[(0, 0)][0] < bw.squares[(8, 0)][0],
      'a0 bên trái i0' if bw else '')
rng = np.random.RandomState(3)
cs = list(pos.values())
rng.shuffle(cs)
ps = []
for i, c in enumerate(cs):
    pr = np.full(14, 0.005)
    pr[pc.CODES.index(c)] = 0.9
    p = pc.Piece(0, 0, 10, pr)
    p.world = (170 + (i % 3) * 32, -40 + (i // 3) * 32)
    ps.append(p)
wd.classify_where(ps, bw, 12.0)
assign.correct_labels(ps)
P = assign.plan(pos, bw, ps)
check('kế hoạch 32 nước, đúng loại quân', len(P.moves) == 32 and P.ok and all(pos[m.square] == m.code for m in P.moves))
check('hàng cuối: cột b d f h đặt sau cùng', [m.phase for m in P.moves] == sorted(m.phase for m in P.moves))

print('== Điều khiển (giả lập)')
cfg = config.robot_cfg()
cfg['limits'].update({'x': [-300, 300], 'y': [-100, 700], 'r_max': 1e9, 'r_min': 0})
cfg['slider']['enabled'] = False
rb = motion.Robot(cfg, {'z_pick': -8}, backend='fake', log=lambda *a: None)
for m in P.moves[:3]:
    rb.pick(*m.pick_xy)
    rb.place(*m.place_xy)
sucks = [t[1] for t in rb.b.trace if t[0] == 'suck']
check('chuỗi hút/thả', sucks[1:] == [True, False] * 3, str(sucks))

print('== Model (nếu có)')
mp = config.path(cfg['detector']['model'])
if os.path.exists(mp):
    try:
        det = pc.PieceDetector(mp, rotations=(0,))
        found = det.detect(start)
        check('model chạy được (%s)' % det.net.name, True, '%d quân trên ảnh mẫu' % len(found))
    except Exception as e:
        check('model chạy được', False, str(e))
else:
    print('  [--]   chưa có %s' % mp)

print('\nKẾT QUẢ: %s' % ('TẤT CẢ OK' if not fails else 'LỖI: ' + ', '.join(fails)))
sys.exit(1 if fails else 0)
