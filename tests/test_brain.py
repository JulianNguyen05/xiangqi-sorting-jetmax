#!/usr/bin/env python3
# Tự kiểm tra thư viện brain/xiangqi_brain.py (không cần model, không cần robot):
# thay YOLO bằng "detector giả" đặt quân đúng vào giao điểm của ẢNH MẪU THẬT (samples/start_position.jpg).
#   python3 tests/test_brain.py                     # engine Python
#   python3 tests/test_brain.py --engine ~/xiangqi/engine/pikafish
# Mọi dòng phải là [OK].
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from board import rules as R                         # noqa: E402
from board.geometry import BoardGeometry             # noqa: E402
from brain.simple import SimpleEngine                # noqa: E402
from brain.xiangqi_brain import XiangqiBrain         # noqa: E402
from vision import board_detect as bd                # noqa: E402
from vision.pieces import CODES, Piece               # noqa: E402

fails = []


def check(name, cond, info=''):
    print(('  [OK]   ' if cond else '  [LỖI]  ') + name + ('  ' + info if info else ''))
    if not cond:
        fails.append(name)


class FakeDetector(object):
    """Trả về quân đúng như thế cờ `board`, đặt tại giao điểm pixel `pts_px` (+ nhiễu, + lỗi tuỳ chọn)."""
    def __init__(self, pts_px, seed=0):
        self.pts_px, self.board, self.rnd = pts_px, {}, np.random.RandomState(seed)

    def detect(self, bgr):
        out = []
        for sq, code in self.board.items():
            x, y = self.pts_px[sq]
            probs = np.zeros(14)
            probs[CODES.index(code)] = 0.9
            out.append(Piece(x + self.rnd.normal(0, 1.5), y + self.rnd.normal(0, 1.5), 15, probs))
        return out


ap = argparse.ArgumentParser()
ap.add_argument('--engine', default=None)
a = ap.parse_args()

img = cv2.imread(os.path.join(ROOT, 'samples/start_position.jpg'))
fix = bd.locate(img, BoardGeometry.load(), red_hint=(1280, 480))
check('ảnh mẫu: định vị được 90 giao điểm', fix is not None and len(fix.pts_px) == 90)
det = FakeDetector(fix.pts_px)
eng = None
if a.engine:
    from brain.uci import UciEngine
    p = os.path.abspath(os.path.expanduser(a.engine))
    eng = UciEngine(p, eval_file=os.path.join(os.path.dirname(p), 'pikafish.nnue'), timeout=60)
else:
    eng = SimpleEngine(max_depth=2)
depth = 8 if a.engine else 2

for locate in ('board', 'lattice'):
    print('== Ảnh -> thế cờ (locate=%s)' % locate)
    brain = XiangqiBrain(detector=det, engine=eng, verbose=False, locate=locate, red_hint=(1280, 480), depth=depth)
    det.board = R.start_board()
    rd = brain.read_board(img)
    check('thế khai cuộc đọc đúng', rd['board'] == det.board, '%s, %d quân' % (rd['method'], rd['n_on_board']))

print('== analyze(): ảnh + thế trước -> nước tiếp theo')
brain = XiangqiBrain(detector=det, engine=eng, verbose=False, locate='board', red_hint=(1280, 480), depth=depth)
truth = R.make_move(R.start_board(), R.uci_to_move('h2e2'))
det.board = dict(truth)
del det.board[(8, 6)]                                      # YOLO bỏ sót tốt Đen i6
det.board[(1, 7)] = 'r'                                    # và nhận nhầm pháo b7 thành xe
res = brain.analyze(img, prev_fen=R.START)
check('suy ra người vừa đi Pháo 2 bình 5, sửa 2 ô', res['last_move'] == 'h2e2' and res.get('corrected') == 2,
      str(res['last_move_vn']))
check('FEN sau khi sửa đúng', res['board'] == truth)
check('có nước tiếp theo hợp lệ cho Đen', res['status'] == 'ok' and
      R.uci_to_move(res['best_move']) in R.legal_moves(truth, R.BLACK), '%s %s' % (res['best_move'], res['best_move_vn']))
det.board = R.start_board()
res = brain.analyze(img, side='w')
check('không có thế trước: khai cuộc, Đỏ đi', res['status'] == 'ok', '%s %s' % (res['best_move'], res['best_move_vn']))
bad = dict(R.start_board())
bad[(2, 6)] = 'c'
det.board = bad
res = brain.analyze(img, side='w')
check('không có thế trước: phát hiện 3 pháo -> invalid', res['status'] == 'invalid', '; '.join(res['issues']))

print('== step(): robot cầm Đen chơi cả ván')
brain.new_game(robot_side=R.BLACK)
det.board = R.start_board()
check('chưa ai đi -> waiting', brain.step(img)['status'] == 'waiting')
det.board = R.make_move(R.start_board(), R.uci_to_move('b2e2'))
r1 = brain.step(img)
check('người đi -> robot_move', r1['status'] == 'robot_move' and r1['human_move']['uci'] == 'b2e2',
      '%s -> robot: %s' % (r1['human_move'] and r1['human_move']['vn'], r1['robot_move'] and r1['robot_move']['vn']))
brain.confirm(r1['robot_move']['uci'])
det.board = R.make_move(brain.game.board, R.uci_to_move('h0g2'))
r2 = brain.step(img)
check('lượt 2', r2['status'] == 'robot_move' and r2['human_move']['uci'] == 'h0g2')
brain.confirm(r2['robot_move']['uci'])
det.board = {}
check('ảnh không có quân -> retake', brain.step(img)['status'] == 'retake')

print('== tăng cường ảnh (vision/enhance.py)')
from vision import enhance as E                      # noqa: E402
_, inf = E.enhance(img, {'mode': 'auto'})
check('ảnh tốt -> không áp dụng bước nào', inf['steps'] == [], 'L %.0f, nhiễu %.1f, ám %.2f' % (
    inf['stats']['brightness'], inf['stats']['noise'], inf['stats']['cast']))
noisy = np.clip(img.astype(np.int16) + np.random.RandomState(0).normal(0, 22, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
out, inf = E.enhance(noisy, {'mode': 'auto'})
check('ảnh nhiễu -> lọc song phương', inf['steps'] == ['denoise'], 'nhiễu %.1f' % inf['stats']['noise'])
check('ảnh nhiễu: không lọc thì mất bàn, lọc thì tìm lại được',
      bd.locate(noisy, BoardGeometry.load(), red_hint=(1280, 480)) is None and
      bd.locate(out, BoardGeometry.load(), red_hint=(1280, 480)) is not None)
dark = E.gamma(img, 3.0)
check('ảnh tối -> CLAHE', E.enhance(dark, {'mode': 'auto'})[1]['steps'] == ['clahe'])
warm = np.clip(img.astype(np.float32) * np.array([0.75, 0.95, 1.15]), 0, 255).astype(np.uint8)
check('ảnh ám vàng -> cân bằng trắng', E.enhance(warm, {'mode': 'auto'})[1]['steps'] == ['white_balance'])
check('mode off -> giữ nguyên ảnh', E.enhance(noisy, {'mode': 'off'})[0] is noisy)
det.board = R.start_board()
rd = brain.read_board(noisy)
check('thư viện trên ảnh nhiễu: vẫn đọc đúng khai cuộc', rd['board'] == det.board and rd['method'] == 'board',
      '%s, tăng cường: %s' % (rd['method'], rd['enhance']['steps']))
brain.close()

print('\nKẾT QUẢ: %s' % ('TẤT CẢ OK' if not fails else '%d LỖI: %s' % (len(fails), ', '.join(fails))))
sys.exit(1 if fails else 0)
