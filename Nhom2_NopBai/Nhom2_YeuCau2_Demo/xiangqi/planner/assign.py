#!/usr/bin/env python3
# Lập kế hoạch xếp cờ.
#   1. correct_labels: sửa nhãn theo số lượng quân của bộ cờ (mỗi bên 1 tướng, 2 sĩ, 2 tượng, 2 mã, 2 xe, 2 pháo,
#      5 tốt) bằng bài toán ghép cặp chi phí nhỏ nhất (Hungarian) trên -log(xác suất).
#   2. plan: so thế cờ đích (FEN) với bàn hiện tại, ghép mỗi ô còn thiếu với quân cùng loại gần nhất ở vùng chờ,
#      sắp thứ tự đặt (hàng cuối: cột a c e g i trước, b d f h sau vì khe giữa 2 quân chỉ 2.5 mm).
# Không cần scipy (Jetson có thể không có): tự cài thuật toán Hungarian.

import math

import numpy as np

from board.fen import LIMIT, piece_label
from board.geometry import square_name
from vision.pieces import CODES

FULL_SET = {}
for _c, _n in LIMIT.items():
    FULL_SET[_c] = _n
    FULL_SET[_c.upper()] = _n


def hungarian(cost):
    """Ghép hàng -> cột chi phí nhỏ nhất, ma trận n x m với n <= m. Trả về list cột cho từng hàng."""
    C = np.asarray(cost, np.float64)
    n, m = C.shape
    if n > m:
        raise ValueError('Số hàng phải <= số cột')
    INF = float('inf')
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    res = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            res[p[j] - 1] = j - 1
    return res


def correct_labels(pieces, inventory=None, reject_cost=8.0):
    """Gán lại mã quân cho từng Piece theo số lượng cho phép. Quân không vừa chỗ nào (thừa) bị đánh dấu code=None.
    Trả về danh sách (piece, mã cũ, mã mới) cho các quân bị đổi nhãn."""
    inv = inventory or FULL_SET
    slots = []
    for code, n in sorted(inv.items()):
        slots += [code] * n
    if not pieces:
        return []
    eps = 1e-4
    nP = len(pieces)
    cost = np.full((nP, len(slots) + nP), reject_cost)          # thêm nP ô "loại bỏ"
    for i, p in enumerate(pieces):
        for j, code in enumerate(slots):
            cost[i, j] = -math.log(p.probs[CODES.index(code)] + eps)
    sol = hungarian(cost)
    changed = []
    for i, p in enumerate(pieces):
        old = p.code
        new = slots[sol[i]] if sol[i] < len(slots) else None
        p.final = new
        if new != old:
            changed.append((p, old, new))
    return changed


class Move(object):
    def __init__(self, code, piece, square, place_xy):
        self.code, self.piece, self.square = code, piece, square
        self.pick_xy = tuple(piece.world)
        self.place_xy = tuple(place_xy)
        self.phase = 0

    def __repr__(self):
        return '%s: %s (%.0f,%.0f) -> %s' % (self.code, piece_label(self.code), self.pick_xy[0], self.pick_xy[1],
                                             square_name(*self.square))


class Plan(object):
    def __init__(self):
        self.moves = []          # theo thứ tự thực hiện
        self.done = []           # ô đích đã có đúng quân
        self.blockers = []       # (ô, mã) quân đang nằm sai chỗ trên bàn -> cần dọn tay
        self.missing = {}        # mã -> số quân còn thiếu ở vùng chờ
        self.unreachable = []    # quân / ô ngoài tầm với

    @property
    def ok(self):
        return not self.blockers and not self.missing and not self.unreachable


def plan(target, board, pieces, robot_xy=(0.0, 0.0), reachable=None, back_ranks=(0, 9)):
    """target: {(f, r): mã} từ FEN. board: vision.world.BoardWorld. pieces: Piece đã có .world, .where, .final.
    reachable(x, y) -> bool (None = bỏ qua kiểm tra)."""
    P = Plan()
    on_board = {}
    for p in pieces:
        if p.where and p.where[0] == 'board' and getattr(p, 'final', None):
            on_board[p.where[1]] = p
    for sq, p in on_board.items():
        if target.get(sq) == p.final:
            P.done.append(sq)
        else:
            P.blockers.append((sq, p.final))
    need = [(sq, c) for sq, c in sorted(target.items()) if sq not in P.done]
    staging = [p for p in pieces if p.where and p.where[0] == 'staging' and getattr(p, 'final', None)]
    by_code = {}
    for sq, c in need:
        by_code.setdefault(c, []).append(sq)
    moves = []
    for c, sqs in by_code.items():
        cands = [p for p in staging if p.final == c]
        if len(cands) < len(sqs):
            P.missing[c] = len(sqs) - len(cands)
        if not cands:
            continue
        cost = np.array([[math.hypot(p.world[0] - board.squares[s][0], p.world[1] - board.squares[s][1])
                          for p in cands] for s in sqs])
        if len(sqs) <= len(cands):
            sol = hungarian(cost)
            pairs = [(sqs[i], cands[j]) for i, j in enumerate(sol)]
        else:
            sol = hungarian(cost.T)
            pairs = [(sqs[j], cands[i]) for i, j in enumerate(sol)]
        for s, p in pairs:
            moves.append(Move(c, p, s, board.squares[s]))
    if reachable:
        for m in moves:
            if not reachable(*m.pick_xy):
                P.unreachable.append(('quân', m.code, m.pick_xy))
            if not reachable(*m.place_xy):
                P.unreachable.append(('ô', m.code, square_name(*m.square)))
    # thứ tự: pha 0 trước, pha 1 (cột b d f h ở hàng cuối có 2 hàng xóm sát) sau; trong mỗi pha đi đường ngắn
    for m in moves:
        f, r = m.square
        m.phase = 1 if (r in back_ranks and f % 2 == 1) else 0
    order, cur = [], np.asarray(robot_xy, np.float64)
    for ph in (0, 1):
        rest = [m for m in moves if m.phase == ph]
        while rest:
            m = min(rest, key=lambda q: np.hypot(*(np.array(q.pick_xy) - cur)) + np.hypot(*(np.array(q.place_xy) - np.array(q.pick_xy))))
            order.append(m)
            rest.remove(m)
            cur = np.array(m.place_xy)
    P.moves = order
    return P


def describe(P):
    lines = ['Kế hoạch: %d nước gắp-đặt, %d ô đã đúng sẵn' % (len(P.moves), len(P.done))]
    if P.blockers:
        lines.append('  Quân nằm SAI chỗ trên bàn (dọn ra vùng chờ rồi chạy lại): ' +
                     ', '.join('%s=%s' % (square_name(*s), piece_label(c)) for s, c in P.blockers))
    if P.missing:
        lines.append('  THIẾU quân ở vùng chờ: ' + ', '.join('%s x%d' % (piece_label(c), n) for c, n in P.missing.items()))
    if P.unreachable:
        lines.append('  NGOÀI tầm với: ' + ', '.join('%s %s %s' % u for u in P.unreachable))
    for i, m in enumerate(P.moves):
        lines.append('  %2d. %s%s' % (i + 1, m, '  (pha 2)' if m.phase else ''))
    return '\n'.join(lines)
