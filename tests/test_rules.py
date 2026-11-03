#!/usr/bin/env python3
# Tự kiểm tra luật cờ + engine (không cần robot, không cần model). Chạy trên laptop và trên robot:
#   python3 tests/test_rules.py                                  # luật + engine dự phòng
#   python3 tests/test_rules.py --engine ~/pikafish/pikafish     # thêm kiểm tra Pikafish
# Mọi dòng phải là [OK].
import argparse
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from board import rules as R          # noqa: E402
from brain.game import GameTracker     # noqa: E402
from brain.simple import SimpleEngine  # noqa: E402

fails = []


def check(name, cond, info=''):
    print(('  [OK]   ' if cond else '  [LỖI]  ') + name + ('  ' + info if info else ''))
    if not cond:
        fails.append(name)


ap = argparse.ArgumentParser()
ap.add_argument('--engine', default=None, help='đường dẫn Pikafish (pikafish.nnue để cùng thư mục)')
a = ap.parse_args()

print('== Luật cờ')
b, s = R.parse_fen(R.START)
check('đọc / ghi FEN', R.to_fen(b, s) == R.START)
t0 = time.time()
for d, exp in ((1, 44), (2, 1920), (3, 79666)):
    n = R.perft(b, s, d)
    check('perft(%d) = %d' % (d, exp), n == exp, 'nhận %d' % n)
print('  (perft 1-3 mất %.1f s)' % (time.time() - t0))
check('thế khai cuộc hợp lệ', R.is_valid(b, s))
bad, _ = R.parse_fen('rnbakabnr/9/1c5c1/p1c1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w')
check('phát hiện 3 pháo', any(i[0] == 'too_many' for i in R.validate(bad)))
check('ký hiệu tiếng Việt', R.move_to_vn(b, R.uci_to_move('h2e2')) == 'Pháo 2 bình 5')

print('== Suy ra nước vừa đi')
truth = R.make_move(b, R.uci_to_move('h2e2'))
obs = dict(truth)
obs[(1, 7)] = 'r'
del obs[(8, 6)]
r = R.infer_move(b, R.RED, obs)
check('sửa 2 ô nhận diện sai', r['uci'] == 'h2e2' and r['board'] == truth, str(r['candidates'][:3]))
g = GameTracker(robot_side=R.BLACK)
check('GameTracker: người đi', g.observe(truth)['status'] == 'moved')
check('GameTracker: chưa đi', g.observe(truth)['status'] == 'no_change')

print('== Khớp lưới từ tâm quân (vision/lattice.py)')
try:
    import numpy as np
    from vision import lattice as L
    sq = sorted(b)
    th = 0.4
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = (np.array([[f - 4, -(r - 4.5)] for f, r in sq], float) * 40) @ rot.T + 300
    got, fit, pl = L.image_to_board(pts, [b[k] for k in sq], [1.0] * len(sq))
    check('thế khai cuộc xoay 23°: khôi phục đúng FEN', got == b)
except ImportError as e:
    print('  (bỏ qua: thiếu %s)' % e)

print('== Engine dự phòng (Python)')
res = SimpleEngine().best_move(g.fen, depth=2)
check('ra nước hợp lệ', res.move is not None and R.uci_to_move(res.move) in R.legal_moves(g.board, g.side),
      '%s (%.0f ms)' % (res.move, res.time_ms))

if a.engine:
    from brain.uci import UciEngine
    print('== Pikafish')
    nnue = os.path.join(os.path.dirname(os.path.abspath(a.engine)), 'pikafish.nnue')
    with UciEngine(a.engine, eval_file=nnue, timeout=60) as eng:
        check('khởi động', eng.name is not None, eng.name or '')
        check('perft(3) khớp board/rules.py', eng.perft(R.START, 3) == 79666)
        res = eng.best_move(g.fen, depth=8)
        check('ra nước hợp lệ (depth 8)', R.uci_to_move(res.move) in R.legal_moves(g.board, g.side),
              '%s %s cp, %.0f ms' % (res.move, res.score_cp, res.time_ms))

print('\nKẾT QUẢ: %s' % ('TẤT CẢ OK' if not fails else '%d LỖI: %s' % (len(fails), ', '.join(fails))))
sys.exit(1 if fails else 0)
