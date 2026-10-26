#!/usr/bin/env python3
# Logic ván cờ giữa người và robot (thuần Python, không cần ROS, không cần robot).
#
#   Người cầm ĐỎ  -> người đi trước, robot cầm ĐEN đi sau.
#   Người cầm ĐEN -> robot cầm ĐỎ và đi trước.
#
# Luồng chạy:
#   robot đến lượt : engine.choose(thế cờ) -> executor.execute(nước đi) -> cập nhật thế cờ
#   người đến lượt : observe() trả về bàn cờ nhìn thấy từ camera {(cột, hàng): 'R', ...}
#                    -> detect_move() so với thế cờ hiện tại để TÌM RA nước người vừa đi (và kiểm tra hợp lệ)
#
# Nhận diện nước đi bằng cách thử mọi nước hợp lệ rồi chọn nước cho bàn cờ giống ảnh nhất, nên chịu được việc YOLO
# bỏ sót/nhận nhầm 1 quân (tolerance) và tự loại các tư thế dở dang (tay che, quân đang cầm lên).
#
# Cách nối với robot thật: viết một lớp kế thừa RobotExecutor, trong execute(info) dùng
#   - info.captured khác None: gắp quân ở info.to đem bỏ ra khay (quân bị ăn) TRƯỚC,
#   - rồi gắp quân ở info.frm đặt xuống info.to.
# Ở bước đó có thể dùng Board.position() + Mover.pick()/place() đã có trong robot/board.py và robot/motion.py.
#
# Chưa cài: luật cấm chiếu liên tục/đuổi liên tục và hòa 60 nước không ăn quân; chỉ có giới hạn tổng số nước (max_plies).

import time
from collections import namedtuple

from xiangqi_rules import Position, RED, BLACK, other, move_uci, sq_name, piece_name

MoveInfo = namedtuple('MoveInfo', 'move uci frm to piece captured by')
DetectResult = namedtuple('DetectResult', 'status move distance candidates')
# status: 'nochange' (bàn cờ chưa đổi) | 'pending' (đã thấy một nước, chờ xác nhận) | 'move' (đã xác nhận)
#         'ambiguous' (nhiều nước cùng khớp) | 'invalid' (không nước hợp lệ nào khớp)


def make_info(pos, move, by):
    f1, r1, f2, r2 = move
    cap = pos.b[r2][f2]
    return MoveInfo(move, move_uci(move), sq_name(f1, r1), sq_name(f2, r2), pos.b[r1][f1],
                    None if cap == '.' else cap, by)


def describe(info):
    s = '%s: %s %s -> %s' % ('Robot' if info.by == 'robot' else 'Người', piece_name(info.piece), info.frm, info.to)
    if info.captured:
        s += ' (ăn %s)' % piece_name(info.captured)
    return s


# ------------------------------------------------------------------ robot --------
class RobotError(Exception):
    pass


class RobotExecutor(object):
    def execute(self, info):
        """Thực hiện nước đi bằng tay máy. Ném RobotError nếu thất bại (khi đó thế cờ KHÔNG được cập nhật)."""
        raise NotImplementedError


class DryRunExecutor(RobotExecutor):
    """Chưa có robot: chỉ in các bước sẽ làm."""

    def __init__(self, log=print):
        self.log = log

    def execute(self, info):
        if info.captured:
            self.log('    [robot] gắp %s ở %s -> bỏ ra khay quân bị ăn' % (piece_name(info.captured), info.to))
        self.log('    [robot] gắp %s ở %s -> đặt xuống %s' % (piece_name(info.piece), info.frm, info.to))


# ------------------------------------------------------------------ nhận diện nước đi
def board_distance(a, b):
    """Số ô khác nhau (ô sai quân tính 2, thiếu/thừa quân tính 1)."""
    return len(set(a.items()) ^ set(b.items()))


def detect_move(pos, observed, tolerance=1):
    """So bàn cờ nhìn thấy với thế cờ `pos` (người đang đến lượt) và tìm nước người vừa đi."""
    prev = pos.to_dict()
    d_prev = board_distance(prev, observed)
    if d_prev == 0:
        return DetectResult('nochange', None, 0, [])
    scored = []
    for m in pos.legal_moves():
        exp = dict(prev)
        exp[(m[2], m[3])] = exp.pop((m[0], m[1]))
        scored.append((board_distance(exp, observed), m))
    if not scored:
        return DetectResult('invalid', None, d_prev, [])
    scored.sort()
    best = scored[0][0]
    ties = [m for d, m in scored if d == best]
    if d_prev <= tolerance and d_prev < best:
        return DetectResult('nochange', None, d_prev, [])      # chỉ là nhiễu nhận diện, người chưa đi
    if best > tolerance:
        return DetectResult('invalid', None, best, [])
    if len(ties) > 1 or d_prev == best:
        return DetectResult('ambiguous', None, best, [move_uci(m) for m in ties])
    return DetectResult('move', ties[0], best, [move_uci(ties[0])])


class BoardStabilizer(object):
    """Gộp nhiều khung hình liên tiếp: ô nào có cùng một quân ở đa số khung hình thì mới tin.
    Trả về None cho tới khi đủ `frames` khung hình, nên tay đang che bàn cờ hay một khung hình hỏng không làm sai nước."""

    def __init__(self, frames=3):
        self.n = max(1, int(frames))
        self.buf = []

    def reset(self):
        self.buf = []

    def update(self, observed):
        self.buf.append(dict(observed))
        self.buf = self.buf[-self.n:]
        if len(self.buf) < self.n:
            return None
        need = self.n // 2 + 1
        count = {}
        for b in self.buf:
            for item in b.items():
                count[item] = count.get(item, 0) + 1
        return dict(k for k, c in count.items() if c >= need)


# ------------------------------------------------------------------ ván cờ ------
class Game(object):
    def __init__(self, human='red', engine=None, executor=None, start=None, tolerance=1, confirm=2, max_plies=400):
        h = str(human).lower()
        if h not in ('red', 'black', 'w', 'b'):
            raise ValueError('human phải là "red" hoặc "black"')
        self.human = RED if h in ('red', 'w') else BLACK
        self.robot = other(self.human)
        self.engine = engine
        self.executor = executor if executor is not None else DryRunExecutor()
        self.pos = start if start is not None else Position.start()
        self.tolerance = tolerance
        self.confirm = max(1, int(confirm))      # số lần liên tiếp phải thấy cùng một nước mới chấp nhận
        self.max_plies = max_plies
        self.history = []
        self._cand = (None, 0)

    def robot_to_move(self):
        return self.pos.turn == self.robot

    def result(self):
        """None nếu còn đang chơi, ngược lại (bên thắng hoặc None nếu hòa, lý do)."""
        if len(self.history) >= self.max_plies:
            return (None, 'max_plies')
        r = self.pos.result()
        if r == 'ongoing':
            return None
        return (other(self.pos.turn), r)

    def _apply(self, move, by):
        info = make_info(self.pos, move, by)
        self.pos = self.pos.make(move)
        self.history.append(info)
        self._cand = (None, 0)
        return info

    def robot_move(self):
        if not self.robot_to_move():
            raise RuntimeError('Chưa đến lượt robot')
        move = self.engine.choose(self.pos)
        if not self.pos.is_legal(move):
            raise RuntimeError('Engine trả nước không hợp lệ: %s' % move_uci(move))
        info = make_info(self.pos, move, 'robot')
        self.executor.execute(info)              # lỗi robot -> ngoại lệ, thế cờ giữ nguyên
        return self._apply(move, 'robot')

    def human_observed(self, observed):
        """Đưa bàn cờ vừa nhìn thấy vào. Trả về (DetectResult, MoveInfo hoặc None)."""
        if self.robot_to_move():
            raise RuntimeError('Đang đến lượt robot')
        r = detect_move(self.pos, observed, self.tolerance)
        if r.status != 'move':
            self._cand = (None, 0)
            return r, None
        cand, n = self._cand
        n = n + 1 if cand == r.move else 1
        self._cand = (r.move, n)
        if n < self.confirm:
            return DetectResult('pending', r.move, r.distance, r.candidates), None
        return r, self._apply(r.move, 'human')


def play(game, observe, log=print, poll=0.3):
    """Chạy cả ván. observe() trả về dict bàn cờ nhìn thấy, hoặc None nếu chưa có dữ liệu ổn định."""
    last = None
    while True:
        res = game.result()
        if res is not None:
            winner, why = res
            if winner is None:
                log('HÒA (quá %d nước)' % game.max_plies)
            else:
                log('%s THẮNG (%s)' % ('Robot' if winner == game.robot else 'Người', why))
            return res
        if game.robot_to_move():
            log('Robot đang nghĩ...')
            log(describe(game.robot_move()))
            log(str(game.pos))
            last = None
            continue
        obs = observe()
        info = None
        if obs is not None:
            r, info = game.human_observed(obs)
            if info is not None:
                log(describe(info))
                log(str(game.pos))
                last = None
            elif r.status in ('ambiguous', 'invalid') and r.status != last:
                log('CẢNH BÁO: %s' % ('nhiều nước cùng khớp %s' % r.candidates if r.status == 'ambiguous'
                                      else 'bàn cờ không khớp nước hợp lệ nào (độ lệch %d)' % r.distance))
            if r.status != 'nochange':
                last = r.status
        if info is None and poll:
            time.sleep(poll)
