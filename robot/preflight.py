#!/usr/bin/env python3
# Kiểm tra TRƯỚC khi chạy ván thật: đọc configs/board.yaml (kèm khay) và configs/views.yaml, báo ngay mọi chỗ còn thiếu
# hoặc sai (chưa điền số, ô ngoài tầm với, view thiếu điểm, ô cờ không view nào nhìn thấy, khay không đủ chỗ...).
# Thuần Python, không cần ROS; chạy được trên laptop, và chạy lại trên robot sau khi điền số đo.
#
#   python robot/preflight.py                                  # kiểm cấu hình (mặc định configs/board.yaml, configs/views.yaml)
#   python robot/preflight.py --model models/best.onnx         # kiểm thêm file mô hình
#   python robot/preflight.py --engine Pikafish/Pikafish-Windows-x86-64-universal.exe --nnue Pikafish/pikafish.nnue
#                                                              # chạy thử Pikafish: hỏi một nước từ thế cờ ban đầu
# Mã thoát 0 nếu không có LỖI (cảnh báo vẫn cho qua), 1 nếu có LỖI.

import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from board import Board, parse_square, square_name, NF, NR
from executor import Tray

SLIDER_RANGE = (0, 5200)
SLIDER_MIN_Z = 40.0
MIN_TRAY = 30                 # tối đa 30 quân có thể bị ăn (32 quân trừ hai tướng)
FIT_WARN, FIT_ERR = 0.15, 0.30
MODEL_EXT = ('.pt', '.onnx', '.engine', '.trt')
# Giống motion.py. Khi chạy trên robot sẽ lấy thẳng từ motion.py; trên laptop (không có rospy) dùng bản sao này.
_X_LIM, _Y_LIM, _Z_LIM, _R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0


def get_limits():
    try:
        import motion
        return motion.in_limits
    except Exception:
        def in_limits(p):
            x, y, z = p
            return (_X_LIM[0] <= x <= _X_LIM[1] and _Y_LIM[0] <= y <= _Y_LIM[1] and _Z_LIM[0] <= z <= _Z_LIM[1]
                    and (x * x + y * y) ** 0.5 >= _R_MIN)
        return in_limits


def read_yaml(path):
    """Đọc YAML bằng UTF-8 (Windows mặc định cp1252 sẽ lỗi với chú thích tiếng Việt)."""
    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class Report(object):
    def __init__(self):
        self.items = []

    def ok(self, msg):
        self.items.append(('OK', msg))

    def warn(self, msg):
        self.items.append(('CẢNH BÁO', msg))

    def error(self, msg):
        self.items.append(('LỖI', msg))

    @property
    def errors(self):
        return [m for k, m in self.items if k == 'LỖI']

    @property
    def warnings(self):
        return [m for k, m in self.items if k == 'CẢNH BÁO']

    def text(self):
        return '\n'.join('%-9s %s' % (k, m) for k, m in self.items)

    def summary(self):
        return '%d lỗi, %d cảnh báo' % (len(self.errors), len(self.warnings))


def _short(names, n=12):
    names = list(names)
    return ' '.join(names[:n]) + (' ... (+%d ô nữa)' % (len(names) - n) if len(names) > n else '')


# ------------------------------------------------------------------ bàn cờ + khay ------
def check_board(path, limits, rep, slider_range=SLIDER_RANGE):
    if not os.path.exists(path):
        rep.error('Không thấy %s' % path)
        return None
    try:
        d = read_yaml(path)
        corners = dict((k, (v.get('slider'), v.get('x'), v.get('y'))) for k, v in (d.get('corners') or {}).items())
        board = Board(corners, d.get('z_pick'), d.get('z_safe', 50.0), d.get('z_place'))
    except (ValueError, TypeError, AttributeError) as e:
        rep.error('board.yaml: %s' % e)
        return None
    except Exception as e:                                 # YAML sai cú pháp, lỗi bảng mã...
        rep.error('board.yaml không đọc được: %s' % e)
        return None
    if not board.z_pick < board.z_safe:
        rep.error('z_pick (%.1f) phải THẤP hơn z_safe (%.1f)' % (board.z_pick, board.z_safe))
    if board.z_place < board.z_pick:
        rep.error('z_place (%.1f) thấp hơn z_pick (%.1f): sẽ ép quân xuống bàn' % (board.z_place, board.z_pick))
    if board.z_safe < SLIDER_MIN_Z:
        rep.error('z_safe (%.1f) < %d: thanh trượt sẽ không được phép chạy' % (board.z_safe, SLIDER_MIN_Z))
    bad = []
    for f in range(NF):
        for r in range(NR):
            sq = square_name(f, r)
            s, x, y = board.position(sq)
            if not slider_range[0] <= s <= slider_range[1] or \
                    not all(limits((x, y, z)) for z in {board.z_safe, board.z_pick, board.z_place}):
                bad.append(sq)
    if bad:
        rep.error('%d/90 ô ngoài tầm với hoặc giới hạn an toàn: %s' % (len(bad), _short(bad)))
    else:
        rep.ok('board.yaml: cả 90 ô trong tầm với (z_pick=%.1f, z_place=%.1f, z_safe=%.1f)' % (
            board.z_pick, board.z_place, board.z_safe))
    return board


def check_tray(path, board, limits, rep, slider_range=SLIDER_RANGE):
    try:
        tray = Tray.load(path)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        rep.error('khay: %s' % e)
        return None
    except Exception as e:
        rep.error('khay: không đọc được board.yaml: %s' % e)
        return None
    if tray.capacity < MIN_TRAY:
        rep.error('Khay chỉ chứa %d quân, cần >= %d (tăng cols/rows)' % (tray.capacity, MIN_TRAY))
    if board is None:
        return tray
    z = tray.z if tray.z is not None else board.z_place
    bad = []
    for i in range(tray.capacity):
        s, x, y = tray.position(i)
        if not slider_range[0] <= s <= slider_range[1] or not all(limits((x, y, zz)) for zz in {z, board.z_safe}):
            bad.append('khay%d' % i)
    if bad:
        rep.error('%d/%d ô khay ngoài tầm với: %s' % (len(bad), tray.capacity, _short(bad)))
    squares = set(board.position(square_name(f, r)) for f in range(NF) for r in range(NR))
    clash = [i for i in range(tray.capacity) if tray.position(i) in squares]
    if clash:
        rep.error('Khay trùng vị trí với ô bàn cờ: khay%s' % ', khay'.join(str(i) for i in clash[:5]))
    if not bad and not clash and tray.capacity >= MIN_TRAY:
        rep.ok('khay: %d chỗ, đều trong tầm với' % tray.capacity)
    return tray


# ------------------------------------------------------------------ các góc nhìn -------
def check_views(path, limits, rep, slider_range=SLIDER_RANGE):
    """Trả về MultiViewMapper chỉ gồm các view hợp lệ (hoặc None nếu không view nào dùng được)."""
    import numpy as np
    from vision_multi import View, MultiViewMapper
    if not os.path.exists(path):
        rep.error('Không thấy %s' % path)
        return None
    try:
        d = read_yaml(path)
    except Exception as e:
        rep.error('views.yaml không đọc được: %s' % e)
        return None
    size = d.get('image_size') or [640, 480]
    raw = d.get('views') or []
    if not raw:
        rep.error('views.yaml chưa có view nào')
        return None
    good, names = [], set()
    for i, v in enumerate(raw):
        name = str(v.get('name', 'view%d' % i))
        if name in names:
            rep.error('view trùng tên "%s"' % name)
        names.add(name)
        ok = True
        s, arm = v.get('slider'), v.get('arm')
        if s is None or arm is None or len(arm) != 3 or any(c is None for c in arm):
            rep.error('view "%s": chưa điền slider/arm' % name)
            ok = False
        else:
            if not slider_range[0] <= s <= slider_range[1]:
                rep.error('view "%s": slider %s ngoài khoảng %d..%d' % (name, s, slider_range[0], slider_range[1]))
                ok = False
            if not limits(tuple(arm)):
                rep.error('view "%s": arm %s ngoài giới hạn an toàn' % (name, list(arm)))
                ok = False
        pts, bad_key = {}, []
        for k, p in (v.get('points') or {}).items():
            try:
                parse_square(str(k))
            except ValueError:
                bad_key.append(str(k))
                continue
            if p is not None and len(p) == 2 and all(c is not None for c in p):
                pts[str(k)] = p
        if bad_key:
            rep.error('view "%s": tên ô sai: %s' % (name, ' '.join(bad_key)))
            ok = False
        if len(pts) < 4:
            rep.error('view "%s": mới có %d/4 điểm pixel (cần >= 4 giao điểm)' % (name, len(pts)))
            ok = False
        if not ok:
            continue
        try:
            view = View(name, pts, size, s, arm, d.get('edge', 0.03), d.get('max_offset', 0.35),
                        d.get('anchor_y', 0.5), d.get('piece_half', 22.0))
        except (np.linalg.LinAlgError, ValueError) as e:
            rep.error('view "%s": không tính được phép biến đổi (%s), các điểm có thẳng hàng không?' % (name, e))
            continue
        if not np.isfinite(view.fit_error) or view.fit_error > FIT_ERR:
            rep.error('view "%s": sai số dạy %.2f ô > %.2f, hãy dạy lại điểm' % (name, view.fit_error, FIT_ERR))
            continue
        if view.fit_error > FIT_WARN:
            rep.warn('view "%s": sai số dạy %.2f ô (nên < %.2f)' % (name, view.fit_error, FIT_WARN))
        else:
            rep.ok('view "%s": %d điểm, sai số dạy %.2f ô' % (name, len(pts), view.fit_error))
        good.append(view)
    if not good:
        return None
    mapper = MultiViewMapper(good)
    miss = mapper.uncovered()
    if miss:
        extra = ' (còn %d view chưa dùng được)' % (len(raw) - len(good)) if len(good) < len(raw) else ''
        rep.error('%d/90 ô không view hợp lệ nào nhìn thấy: %s%s' % (len(miss), _short(miss), extra))
    else:
        rep.ok('views: %d view phủ kín 90 ô' % len(good))
    return mapper


# ------------------------------------------------------------------ mô hình, engine ----
def check_model(path, rep):
    if not os.path.exists(path):
        rep.error('Không thấy mô hình: %s' % path)
    elif not path.lower().endswith(MODEL_EXT):
        rep.error('Đuôi mô hình lạ (cần %s): %s' % (' '.join(MODEL_EXT), path))
    else:
        rep.ok('mô hình: %s (%.1f MB)' % (os.path.basename(path), os.path.getsize(path) / 1e6))


def check_engine(path, nnue, rep, level='easy', args=None):
    from engine import UciEngine
    from xiangqi_rules import Position
    if not os.path.exists(path):
        rep.error('Không thấy engine: %s' % path)
        return
    if nnue and not os.path.exists(nnue):
        rep.error('Không thấy file nnue: %s' % nnue)
        return
    eng = None
    try:
        t0 = time.time()
        eng = UciEngine(path, level=level, nnue=nnue, args=args, startup_timeout=30.0)
        pos = Position.start()
        mv = eng.choose(pos)
        if not pos.is_legal(mv):
            rep.error('engine trả nước không hợp lệ')
        else:
            rep.ok('engine trả lời được (%.1f giây)' % (time.time() - t0))
    except Exception as e:
        rep.error('engine không chạy được: %s' % e)
    finally:
        if eng is not None:
            eng.close()


# ------------------------------------------------------------------ tổng hợp -----------
def check_all(board_path, views_path, model=None, engine=None, nnue=None, level='easy', limits=None):
    """Trả về (Report, {'board', 'tray', 'mapper'})."""
    rep, limits = Report(), limits or get_limits()
    board = check_board(board_path, limits, rep)
    tray = check_tray(board_path, board, limits, rep)
    mapper = check_views(views_path, limits, rep)
    if model:
        check_model(model, rep)
    if engine:
        check_engine(engine, nnue, rep, level)
    return rep, {'board': board, 'tray': tray, 'mapper': mapper}


def main(argv=None):
    import argparse
    root = os.path.join(HERE, '..', 'configs')
    ap = argparse.ArgumentParser()
    ap.add_argument('--board', default=os.path.join(root, 'board.yaml'))
    ap.add_argument('--views', default=os.path.join(root, 'views.yaml'))
    ap.add_argument('--model')
    ap.add_argument('--engine')
    ap.add_argument('--nnue')
    ap.add_argument('--level', default='easy')
    a = ap.parse_args(argv)
    rep, _ = check_all(a.board, a.views, a.model, a.engine, a.nnue, a.level)
    print(rep.text())
    print('--> %s' % rep.summary())
    return 1 if rep.errors else 0


if __name__ == '__main__':
    sys.exit(main())
