#!/usr/bin/env python3
# THƯ VIỆN CHÍNH CỦA ĐỀ TÀI: ảnh bàn cờ (+ thế cờ lượt trước) -> nước đi tiếp theo.
#
# Ghép 3 bước đã làm thành 1 lớp:
#   Bước 2/4  ảnh -> quân (YOLO11n: TensorRT trên robot, ONNX trên laptop) -> lưới 9 x 10 -> thế cờ
#   Bước 5    luật cờ: kiểm tra, suy ra nước đối thủ vừa đi, sửa lỗi nhận diện
#   Bước 5    engine Pikafish (UCI) -> nước đi tiếp theo (dự phòng: engine Python)
#
# Cách 1 — không giữ trạng thái (đúng yêu cầu "ảnh + thế cờ + lượt hiện tại -> nước tiếp theo"):
#     brain = XiangqiBrain()
#     res = brain.analyze(img, prev_fen=fen_luot_truoc)       # hoặc analyze(img, side='w') nếu chưa có thế trước
#     res['best_move'] -> 'h2e2', res['best_move_vn'] -> 'Pháo 2 bình 5', res['fen'] -> FEN đã sửa lỗi
#
# Cách 2 — robot chơi cả ván (giữ trạng thái ván cờ):
#     brain = XiangqiBrain(robot_side='b')
#     while True:
#         res = brain.step(camera.read())
#         if res['status'] == 'robot_move':  tay máy đi res['robot_move'] rồi gọi brain.confirm(res['robot_move']['uci'])
#         elif res['status'] == 'retake':    chụp lại
#         elif res['status'] == 'game_over': dừng
#
# Python 3.6, chỉ cần numpy + OpenCV (+ TensorRT hoặc onnxruntime cho YOLO).
import os
import time

import numpy as np

from board import rules as R
from board.geometry import NF, NR, square_name
from brain.game import GameTracker
from brain.simple import SimpleEngine
from brain.uci import UciEngine

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DEFAULTS = {
    'model': 'models/best.onnx',        # tự dùng models/best.engine nếu có (vision/pieces.load_backend)
    'conf': 0.25,                       # ngưỡng tốt nhất đo ở Bước 4 (board_to_fen.ipynb, mục 9)
    'iou': 0.5,
    'rotations': [0],                   # 1 lần chạy YOLO: Bước 2 cho thấy mô hình không giảm độ chính xác khi quân xoay
    'locate': 'auto',                   # 'board' (bàn của đề tài) | 'lattice' (bàn bất kỳ) | 'auto' (thử board rồi lattice)
    'red_hint': None,                   # điểm ảnh gần phía quân Đỏ (mặc định: cạnh dưới ảnh)
    'engine': ['engine/pikafish', 'engine/pikafish.exe', '~/pikafish/pikafish'],   # thử lần lượt; không có thì engine Python
    'depth': 8,
    'movetime_ms': None,                # đặt số này thì engine tính theo thời gian thay vì độ sâu
    'threads': 2,
    'hash_mb': 64,
    'robot_side': 'b',                  # robot cầm quân Đen, người đi trước
    'max_cost': 2,                      # số ô nhận diện sai tối đa còn tự sửa (Bước 5, mục 5)
    'enhance': {'mode': 'auto', 'apply_to': ['board']},   # tăng cường ảnh (vision/enhance.py, image_preprocessing.ipynb)
}


def _abs(p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_settings(overrides=None):
    """DEFAULTS <- mục 'brain' của config/robot.yaml (+ robot.local.yaml) <- tham số truyền vào."""
    s = dict(DEFAULTS)
    try:
        from robot import config
        s.update(config.robot_cfg().get('brain') or {})
    except Exception:                                   # chạy ngoài repo / thiếu yaml: dùng mặc định
        pass
    s.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return s


class XiangqiBrain(object):
    def __init__(self, detector=None, engine=None, verbose=True, **kw):
        """detector: đối tượng có .detect(bgr) -> [Piece] (mặc định: YOLO11n theo cấu hình).
        engine: đối tượng có .best_move(fen, moves=, depth=, movetime_ms=), hoặc đường dẫn file Pikafish
                (mặc định: theo cấu hình, Pikafish -> engine Python).
        kw: ghi đè cấu hình (conf, depth, robot_side, locate, ...)."""
        if isinstance(engine, (str, list, tuple)):          # truyền đường dẫn engine thay vì đối tượng
            kw['engine'], engine = engine, None
        self.cfg = load_settings(kw)
        self.verbose = verbose
        self.detector = detector or self._make_detector()
        self.engine = engine or self._make_engine()
        self._geom = None
        self.new_game(robot_side=self.cfg['robot_side'])

    # ------------------------------------------------------------------ khởi tạo
    def _log(self, *a):
        if self.verbose:
            print('[brain]', *a)

    def _make_detector(self):
        from vision.pieces import PieceDetector
        c = self.cfg
        d = PieceDetector(_abs(c['model']), conf=c['conf'], iou=c['iou'], rotations=tuple(c['rotations']))
        self._log('YOLO:', type(d.net).__name__, _abs(c['model']))
        return d

    def _make_engine(self):
        c = self.cfg
        cands = c['engine'] if isinstance(c['engine'], (list, tuple)) else [c['engine']]
        for p in cands:
            p = _abs(p)
            if not os.path.exists(p):
                continue
            try:
                if not os.access(p, os.X_OK):
                    os.chmod(p, 0o755)                  # giải nén bằng python -m zipfile làm mất quyền chạy
                eng = UciEngine(p, eval_file=os.path.join(os.path.dirname(p), 'pikafish.nnue'),
                                threads=c['threads'], hash_mb=c['hash_mb'], timeout=60)
                self._log('Engine:', eng.name, '(%s)' % p)
                return eng
            except Exception as e:
                self._log('Không chạy được %s: %s' % (p, str(e).splitlines()[0][:120]))
        self._log('CẢNH BÁO: không có Pikafish -> dùng engine Python (yếu hơn nhiều)')
        return SimpleEngine(max_depth=3)

    @property
    def geom(self):
        if self._geom is None:
            from board.geometry import BoardGeometry
            self._geom = BoardGeometry.load()
        return self._geom

    # ------------------------------------------------------------------ ảnh -> thế cờ
    def read_board(self, bgr):
        """Ảnh -> thế cờ (CHƯA dùng luật cờ). Trả về dict:
        board {(f, r): mã} hoặc None, method 'board'/'lattice'/None, pieces (danh sách Piece), n_pieces, n_on_board,
        enhance {'stats': số đo ảnh, 'steps': các bước tăng cường đã áp dụng},
        px (chỉ khi method='board') {'grid': {(f, r): (x, y) giao điểm}, 'pieces': {(f, r): (x, y) tâm quân}, 'cell_px'}."""
        from vision import enhance
        t0 = time.time()
        ecfg = self.cfg.get('enhance') or {'mode': 'off'}
        good, einfo = enhance.enhance(bgr, ecfg)
        t1 = time.time()
        targets = enhance.settings(ecfg)['apply_to'] or []
        img_pieces = good if 'pieces' in targets else bgr
        img_board = good if 'board' in targets else bgr
        pieces = self.detector.detect(img_pieces)
        t2 = time.time()
        pts = np.array([[p.x, p.y] for p in pieces], np.float64).reshape(-1, 2)
        codes = [p.code for p in pieces]
        confs = [p.conf for p in pieces]
        out = {'board': None, 'method': None, 'pieces': pieces, 'n_pieces': len(pieces), 'n_on_board': 0,
               'enhance': einfo}
        if einfo['steps']:
            self._log('tăng cường ảnh:', ', '.join(einfo['steps']), '->', ', '.join(targets))
        mode = self.cfg['locate']
        if mode in ('auto', 'board'):
            self._last_px = None
            b = self._by_board_detect(img_board, pts, codes, confs)
            if b is not None:
                out.update(board=b, method='board', px=self._last_px)
        if out['board'] is None and mode in ('auto', 'lattice') and len(pieces) >= 6:
            from vision import lattice
            b, fit, pl = lattice.image_to_board(pts, codes, confs)
            if b is not None:
                out.update(board=b, method='lattice')
        if out['board'] is not None:
            out['n_on_board'] = len(out['board'])
        out['times'] = {'enhance_ms': (t1 - t0) * 1000, 'detect_ms': (t2 - t1) * 1000, 'locate_ms': (time.time() - t2) * 1000}
        return out

    def _by_board_detect(self, bgr, pts, codes, confs):
        """Bàn cờ của đề tài (tấm nhựa kẻ đỏ): định vị 90 giao điểm rồi gán quân vào giao điểm gần nhất."""
        try:
            from vision import board_detect
            fix = board_detect.locate(bgr, self.geom, red_hint=self.cfg['red_hint'])
        except Exception as e:
            self._log('board_detect lỗi:', e)
            return None
        if fix is None or len(fix.pts_px) < NF * NR:
            return None
        pts_px = self._snap_grid(fix.pts_px, pts)
        keys = sorted(pts_px)
        grid = np.array([pts_px[k] for k in keys], np.float64)
        # cỡ ô (px) = trung vị khoảng cách tới giao điểm gần nhất
        dd = np.linalg.norm(grid[:, None] - grid[None], axis=2)
        np.fill_diagonal(dd, np.inf)
        cell = float(np.median(dd.min(axis=1)))
        board, best, ppx = {}, {}, {}
        for p, c, cf in zip(pts, codes, confs):
            d = np.linalg.norm(grid - p, axis=1)
            i = int(np.argmin(d))
            if d[i] > 0.45 * cell:                       # ngoài bàn (vùng chờ, quân đã bị ăn)
                continue
            sq = keys[i]
            if sq not in board or cf > best[sq]:
                board[sq], best[sq], ppx[sq] = c, cf, (float(p[0]), float(p[1]))
        fixed = self._fix_orientation(board)
        rot = fixed is not board
        name = (lambda k: (NF - 1 - k[0], NR - 1 - k[1])) if rot else (lambda k: k)
        # toạ độ ảnh của 90 giao điểm và của tâm từng quân, đặt tên ô theo đúng hướng thế cờ (cho robot gắp/đặt)
        self._last_px = {'grid': dict((name(k), (float(v[0]), float(v[1]))) for k, v in pts_px.items()),
                         'pieces': dict((name(k), v) for k, v in ppx.items()), 'cell_px': cell}
        return fixed

    def _snap_grid(self, pts_px, pts):
        """Lưới tìm từ đường kẻ có thể lệch đúng 1 hàng / 1 cột (đường biên bị quân che, khung viền ngoài bị nhận nhầm
        là đường lưới). Thử dịch lưới -1/0/+1 theo mỗi trục (ngoại suy tuyến tính ở mép), chọn cách có NHIỀU QUÂN nằm
        đúng giao điểm nhất; chỉ đổi khi hơn hẳn (>= 2 quân)."""
        if len(pts) < 6:
            return pts_px
        P = dict((k, np.asarray(v, np.float64)) for k, v in pts_px.items())

        def ext1(get, i, n):                       # ngoại suy tuyến tính theo một trục, i có thể ra ngoài [0, n-1]
            if i < 0:
                return get(0) + i * (get(1) - get(0))
            if i > n - 1:
                return get(n - 1) + (i - n + 1) * (get(n - 1) - get(n - 2))
            return get(i)

        def at(f, r):
            return ext1(lambda ff: ext1(lambda rr: P[(ff, rr)], r, NR), f, NF)

        g0 = np.array([P[k] for k in sorted(P)])
        dd = np.linalg.norm(g0[:, None] - g0[None], axis=2)
        np.fill_diagonal(dd, np.inf)
        cell = float(np.median(dd.min(axis=1)))

        def score(grid):
            G = np.array(list(grid.values()))
            d = np.linalg.norm(pts[:, None] - G[None], axis=2).min(axis=1)
            return int((d < 0.45 * cell).sum())
        best, best_sc = pts_px, score(P)
        base_sc = best_sc
        for sf_ in (-1, 0, 1):
            for sr_ in (-1, 0, 1):
                if sf_ == 0 and sr_ == 0:
                    continue
                g = dict(((f, r), at(f + sf_, r + sr_)) for f in range(NF) for r in range(NR))
                sc = score(g)
                if sc >= base_sc + 2 and sc > best_sc:
                    best, best_sc = g, sc
        if best is not pts_px:
            self._log('chỉnh lưới lệch 1 hàng/cột theo vị trí quân: %d -> %d quân đúng giao điểm' % (base_sc, best_sc))
            best = dict((k, (float(v[0]), float(v[1]))) for k, v in best.items())
        return best

    def _fix_orientation(self, board):
        """red_hint chọn nhầm phía quân Đỏ -> thế cờ bị xoay 180°. Quân Đỏ / Đen do YOLO nhận theo màu, nên so số quân
        đứng sai chỗ theo luật (tướng, sĩ ngoài cung, tượng qua sông, tốt lùi...) của thế cờ và thế xoay 180°;
        thế xoay ít lỗi hơn hẳn thì dùng thế xoay. Hoà thì giữ nguyên."""
        def bad(b):
            return sum(1 for i in R.validate(b) if i[0] == 'bad_square')
        rot = {(NF - 1 - f, NR - 1 - r): c for (f, r), c in board.items()}
        nb, nr = bad(board), bad(rot)
        if nr < nb:
            self._log('đảo hướng bàn cờ 180° (phía Đỏ ngược với red_hint): %d -> %d quân sai chỗ' % (nb, nr))
            return rot
        return board

    # ------------------------------------------------------------------ dùng không giữ trạng thái
    def analyze(self, bgr, prev_fen=None, side=None, depth=None, movetime_ms=None):
        """ẢNH (+ thế cờ lượt trước) -> nước đi tiếp theo.
        prev_fen: FEN đã xác nhận ở lượt trước (bên đi trong FEN = bên VỪA đi trên ảnh này) -> suy ra nước vừa đi,
                  sửa lỗi nhận diện. Không có prev_fen: dùng thẳng thế cờ nhận diện được, side = bên đi tiếp ('w'/'b').
        Trả về dict: status ('ok' | 'retake' | 'invalid' | 'game_over'), fen, board, last_move(+_vn),
                     best_move(+_vn, _from, _to), score_cp, mate, read (kết quả read_board), issues."""
        read = self.read_board(bgr)
        res = {'status': 'retake', 'read': read, 'fen': None, 'board': None, 'issues': [],
               'last_move': None, 'last_move_vn': None, 'best_move': None, 'best_move_vn': None}
        if read['board'] is None:
            res['issues'] = ['không định vị được bàn cờ / quá ít quân']
            return res
        if prev_fen:
            prev, mover = R.parse_fen(prev_fen)
            inf = R.infer_move(prev, mover, read['board'])
            res['corrected'] = inf['cost']
            if inf['cost'] > self.cfg['max_cost'] or (inf['ambiguous'] and inf['move'] is not None):
                res['issues'] = ['ảnh lệch %d ô so với mọi nước hợp lệ -> chụp lại' % inf['cost']]
                return res
            board = inf['board']
            if inf['move'] is not None:
                res['last_move'], res['last_move_vn'] = inf['uci'], R.move_to_vn(prev, inf['move'])
                side = R.other(mover)
            else:
                side = mover                              # chưa ai đi: vẫn lượt của bên trong prev_fen
        else:
            board, side = read['board'], side or R.RED
            iss = R.validate(board, side)
            if iss:
                res['status'], res['issues'] = 'invalid', [i[2] for i in iss]
                res['board'], res['fen'] = board, R.to_fen(board, side)
                return res
        res['board'], res['fen'] = board, R.to_fen(board, side)
        st = R.game_status(board, side)
        if st != 'ongoing':
            res['status'], res['issues'] = 'game_over', ['%s thua (%s)' % ('Đỏ' if side == R.RED else 'Đen', st)]
            return res
        self._suggest(res, board, res['fen'], None, depth, movetime_ms)
        return res

    def _suggest(self, res, board, fen, moves, depth, movetime_ms):
        kw = {'movetime_ms': movetime_ms or self.cfg['movetime_ms']} if (movetime_ms or self.cfg['movetime_ms']) \
            else {'depth': depth or self.cfg['depth']}
        er = self.engine.best_move(fen, moves=moves, **kw)
        if er.move is None:
            res['status'] = 'game_over'
            return res
        cur = R.parse_fen(fen)[0]
        for u in moves or []:
            cur = R.make_move(cur, R.uci_to_move(u))
        mv = R.uci_to_move(er.move)
        res.update(status='ok', best_move=er.move, best_move_vn=R.move_to_vn(cur, mv),
                   best_move_from=square_name(*mv[0]), best_move_to=square_name(*mv[1]),
                   captures=cur.get(mv[1]), score_cp=er.score_cp, mate=er.mate, think_ms=er.time_ms)
        return res

    # ------------------------------------------------------------------ robot chơi cả ván
    def new_game(self, fen=R.START, robot_side=None):
        self.game = GameTracker(fen=fen, robot_side=robot_side or self.cfg['robot_side'],
                                max_cost=self.cfg['max_cost'])
        if hasattr(self.engine, 'new_game'):
            self.engine.new_game()

    def step(self, bgr, depth=None, movetime_ms=None):
        """Một lượt nhìn bàn cờ. status:
        'robot_move' (robot cần đi res['robot_move']; đi xong gọi confirm()), 'waiting' (chưa ai đi),
        'retake' (ảnh không chắc chắn -> chụp lại), 'game_over'."""
        g = self.game
        res = {'status': 'waiting', 'human_move': None, 'robot_move': None, 'fen': g.fen}
        if not g.robot_to_move():
            read = self.read_board(bgr)
            res['read'] = read
            if read['board'] is None:
                res['status'] = 'retake'
                return res
            ob = g.observe(read['board'])
            if ob['status'] == 'game_over':
                res.update(status='game_over', result=ob)
                return res
            if ob['status'] == 'uncertain':
                res.update(status='retake', candidates=ob.get('candidates'))
                return res
            if ob['status'] == 'no_change':
                return res
            res['human_move'] = {'uci': ob['uci'], 'vn': ob['vn'], 'corrected': ob['cost']}
            res['fen'] = g.fen
        if R.game_status(g.board, g.side) != 'ongoing':
            res['status'] = 'game_over'
            return res
        tmp = {}
        self._suggest(tmp, g.board, g.start_fen, g.history, depth, movetime_ms)
        if tmp.get('status') != 'ok':
            res['status'] = 'game_over'
            return res
        res['status'] = 'robot_move'
        res['robot_move'] = {'uci': tmp['best_move'], 'vn': tmp['best_move_vn'], 'from': tmp['best_move_from'],
                             'to': tmp['best_move_to'], 'captures': tmp['captures'], 'score_cp': tmp['score_cp'],
                             'mate': tmp['mate'], 'think_ms': tmp['think_ms']}
        return res

    def confirm(self, move_uci):
        """Gọi sau khi tay máy đã đi xong nước của robot."""
        return self.game.apply(move_uci)

    def close(self):
        if hasattr(self.engine, 'close'):
            self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
