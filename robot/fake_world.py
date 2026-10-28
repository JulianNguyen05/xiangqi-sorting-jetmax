#!/usr/bin/env python3
# Thế giới giả lập để chạy cả ván khi chưa có robot: bàn cờ vật lý, tay máy giả (FakeMover), camera + YOLO giả đi qua
# ĐÚNG đường nhận diện thật (View -> MultiViewMapper -> ScanObserver -> BoardStabilizer -> Game).
# Chỉ thay những thứ cần phần cứng; mọi logic còn lại là code thật.
#
# Các view giả có phép chiếu phối cảnh riêng (mỗi view thấy khoảng 5-6 cột), điểm dạy có nhiễu bấm chuột ±1.5 px.

import random
import numpy as np

from xiangqi_rules import Position
from board import Board, square_name, parse_square
from executor import Tray, make_resolver
from motion_fake import FakeMover
from vision_multi import View

NAMES = ['%s_%s' % (c, p) for c in ('black', 'red')
         for p in ('advisor', 'bishop', 'cannon', 'king', 'knight', 'pawn', 'rook')]     # giống detect_view.NAMES
CODE = {'a': 'advisor', 'b': 'bishop', 'c': 'cannon', 'k': 'king', 'n': 'knight', 'p': 'pawn', 'r': 'rook'}
CORNERS = {'a0': (0, -96, -130), 'i0': (0, 96, -130), 'a9': (3600, -96, -130), 'i9': (3600, 96, -130)}
SIZE = (640, 480)
VIEW_DEFS = [('trai', 0, (0, 2, 4)), ('giua', 2, (2, 4, 6)), ('phai', 4, (4, 6, 8))]     # (tên, cột đầu, cột dạy)


def _true_h(f0):
    return np.array([[100.0, 0, 70 - 100.0 * f0], [0, 42.0, 45.0], [0, 0.0004, 1.0]])


def _project(h, f, r):
    v = h.dot([f, r, 1.0])
    return v[0] / v[2], v[1] / v[2]


class FakeWorld(object):
    def __init__(self, noise=0.0, seed=None, fail_at=None):
        self.rng = random.Random(seed)
        self.noise = float(noise)
        self.board = Board(CORNERS, z_pick=10.0, z_safe=50.0)
        self.tray = Tray(4400, 100.0, -105.0, col_step=(0.0, -14.0), row_step=(-14.0, 0.0), cols=8, rows=4)
        self.resolve = make_resolver(self.board, self.tray)
        self.physical = dict((square_name(*sq), p) for sq, p in Position.start().to_dict().items())
        self.mover = FakeMover(self.resolve, self.physical, fail_at)
        self.names = NAMES
        self.cur, self.scans = None, 0
        self._h, self.views, self.view_cfg = {}, [], []
        for name, f0, cols in VIEW_DEFS:
            h = _true_h(f0)
            pts = {}
            for f in cols:
                for r in (0, 3, 6, 9):
                    x, y = _project(h, f, r)
                    pts[square_name(f, r)] = [round(float(x) + self.rng.uniform(-1.5, 1.5), 1),
                                              round(float(y) + self.rng.uniform(-1.5, 1.5), 1)]
            cfg = {'name': name, 'slider': int(f0 * 700), 'arm': [0, -150, 120], 'points': pts}
            self.view_cfg.append(cfg)
            self._h[name] = h
            self.views.append(View(name, pts, SIZE, cfg['slider'], cfg['arm']))

    def views_yaml(self):
        """Nội dung views.yaml tương ứng (để thử preflight)."""
        return {'image_size': list(SIZE), 'piece_half': 22, 'views': self.view_cfg}

    # ---- camera + YOLO giả
    def move_to(self, view):
        self.cur = view.name

    def grab(self):
        self.scans += 1
        return None

    def __call__(self, frame):
        h, dets = self._h[self.cur], []
        for label, p in self.physical.items():
            if label.startswith('khay') or self.rng.random() < self.noise:
                continue
            f, r = parse_square(label)
            x, y = _project(h, f, r)
            x1, y1, x2, y2 = float(x) - 20, float(y) - 20, float(x) + 20, float(y) + 20
            cx1, cy1 = max(0.0, x1), max(0.0, y1)
            cx2, cy2 = min(float(SIZE[0]), x2), min(float(SIZE[1]), y2)
            if (cx2 - cx1) < 20 or (cy2 - cy1) < 20:       # gần như nằm ngoài ảnh: YOLO sẽ không phát hiện
                continue
            name = ('red_' if p.isupper() else 'black_') + CODE[p.lower()]
            dets.append((cx1, cy1, cx2, cy2, 0.9, NAMES.index(name)))      # YOLO thật cắt khung theo mép ảnh
        return np.array(dets, np.float32).reshape(-1, 6)

    # ---- người chơi giả
    def matches(self, game):
        """Bàn cờ vật lý (không tính khay) có đang đúng bằng thế cờ game.pos không."""
        want = dict((square_name(*sq), p) for sq, p in game.pos.to_dict().items())
        have = dict((k, v) for k, v in self.physical.items() if not k.startswith('khay'))
        return want == have

    def human_move(self, game, auto=False, input_fn=input, log=print):
        if not self.matches(game):
            return                                        # người đã đi rồi, chỉ đang chờ robot nhận ra nước đó
        legal = game.pos.legal_moves()
        if auto:
            m = self.rng.choice(legal)
        else:
            from xiangqi_rules import parse_uci, move_uci
            while True:
                s = input_fn('Nước đi của bạn (vd h2e2, ? = gợi ý, q = thoát): ').strip().lower()
                if s in ('q', 'quit', 'exit'):
                    raise KeyboardInterrupt
                if s == '?':
                    log(' '.join(sorted(move_uci(x) for x in legal)))
                    continue
                try:
                    m = parse_uci(s)
                except ValueError as e:
                    log('LỖI: %s' % e)
                    continue
                if m in legal:
                    break
                log('Nước %s không hợp lệ' % s)
        a, b = square_name(m[0], m[1]), square_name(m[2], m[3])
        self.physical.pop(b, None)                        # người tự bỏ quân bị ăn ra ngoài
        self.physical[b] = self.physical.pop(a)

    def repair(self, game):
        """Mô phỏng người xếp lại bàn cờ cho khớp game.pos sau khi robot lỗi giữa chừng."""
        for k in [k for k in self.physical if not k.startswith('khay')]:
            del self.physical[k]
        for sq, p in game.pos.to_dict().items():
            self.physical[square_name(*sq)] = p
        self.mover.held = None
