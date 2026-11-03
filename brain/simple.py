#!/usr/bin/env python3
# Engine DỰ PHÒNG viết bằng Python thuần (alpha-beta + tìm kiếm tĩnh các nước ăn quân).
# Yếu hơn Pikafish rất nhiều; chỉ dùng khi không chạy được Pikafish trên robot, hoặc để thử nghiệm.
# Cùng giao diện với UciEngine.best_move(); chạy được như 1 engine UCI:  python3 -m brain.simple
import sys
import time

from board import rules as R
from brain.uci import EngineResult

VALUE = {'k': 0, 'r': 600, 'n': 270, 'c': 285, 'b': 120, 'a': 120, 'p': 30}
MATE = 100000


def evaluate(board, side):
    """Điểm theo góc nhìn bên side: chênh lệch vật chất + thưởng tốt qua sông, tốt áp sát cung."""
    s = 0
    for (f, r), c in board.items():
        t = c.lower()
        v = VALUE[t]
        if t == 'p':
            red = R.is_red(c)
            rr = r if red else 9 - r                     # số hàng đã tiến
            if rr >= 5:
                v += 40 + (20 if rr <= 8 and 3 <= f <= 5 else 0)
        s += v if R.is_red(c) else -v
    return s if side == R.RED else -s


class SimpleEngine(object):
    name = 'SimplePy (alpha-beta)'

    def __init__(self, max_depth=4):
        self.max_depth = max_depth

    def _order(self, board, moves):
        # Ăn quân giá trị cao bằng quân giá trị thấp lên trước (MVV-LVA)
        def key(m):
            cap = board.get(m[1])
            return -(VALUE[cap.lower()] * 10 - VALUE[board[m[0]].lower()] // 10) if cap else 0
        return sorted(moves, key=key)

    def _quiesce(self, board, side, alpha, beta, qdepth):
        self.nodes += 1
        stand = evaluate(board, side)
        if stand >= beta or qdepth == 0:
            return stand
        alpha = max(alpha, stand)
        caps = [m for m in R.legal_moves(board, side) if m[1] in board]
        for m in self._order(board, caps):
            sc = -self._quiesce(R.make_move(board, m), R.other(side), -beta, -alpha, qdepth - 1)
            if sc >= beta:
                return sc
            alpha = max(alpha, sc)
        return alpha

    def _search(self, board, side, depth, alpha, beta, ply):
        if time.time() > self.deadline:
            raise TimeoutError
        moves = R.legal_moves(board, side)
        if not moves:
            return -MATE + ply, []                       # hết nước = thua (cờ tướng)
        if depth == 0:
            return self._quiesce(board, side, alpha, beta, 4), []
        self.nodes += 1
        best, best_pv = -MATE * 2, []
        ordered = self._order(board, moves)
        if ply == 0 and self.pv_move in ordered:         # thử nước tốt nhất của vòng trước trước
            ordered.remove(self.pv_move)
            ordered.insert(0, self.pv_move)
        for m in ordered:
            sc, pv = self._search(R.make_move(board, m), R.other(side), depth - 1, -beta, -alpha, ply + 1)
            sc = -sc
            if sc > best:
                best, best_pv = sc, [m] + pv
            alpha = max(alpha, sc)
            if alpha >= beta:
                break
        return best, best_pv

    def best_move(self, fen, moves=None, depth=None, movetime_ms=None, nodes=None):
        board, side = R.parse_fen(fen)
        for u in moves or []:
            board = R.make_move(board, R.uci_to_move(u))
            side = R.other(side)
        t0 = time.time()
        self.deadline = t0 + (movetime_ms / 1000.0 if movetime_ms and not depth else 1e9)
        self.nodes, self.pv_move = 0, None
        legal = R.legal_moves(board, side)
        if not legal:
            return EngineResult(None, None, None, 0, 0, 0, [], 0.0)
        res = (None, [legal[0]], 0)
        for d in range(1, (depth or self.max_depth) + 1):   # đào sâu dần: hết giờ thì dùng kết quả vòng trước
            try:
                sc, pv = self._search(board, side, d, -MATE * 2, MATE * 2, 0)
            except TimeoutError:
                break
            res = (sc, pv, d)
            self.pv_move = pv[0] if pv else None
        sc, pv, d = res
        mate = None
        if sc is not None and abs(sc) > MATE - 1000:
            mate = (MATE - abs(sc) + 1) // 2 * (1 if sc > 0 else -1)
        uci_pv = [R.move_to_uci(m) for m in pv]
        return EngineResult(uci_pv[0], uci_pv[1] if len(uci_pv) > 1 else None,
                            None if mate is not None else sc, mate, d, self.nodes, uci_pv,
                            (time.time() - t0) * 1000.0)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def uci_loop():
    """Giao thức UCI tối thiểu để thử brain/uci.py mà không cần Pikafish."""
    eng = SimpleEngine()
    fen, moves = R.START, []
    for line in sys.stdin:
        tok = line.split()
        if not tok:
            continue
        cmd = tok[0]
        if cmd == 'uci':
            print('id name %s' % eng.name)
            print('option name Hash type spin default 16 min 1 max 1024')
            print('uciok')
        elif cmd == 'isready':
            print('readyok')
        elif cmd == 'position':
            if tok[1] == 'startpos':
                fen, rest = R.START, tok[2:]
            else:
                end = tok.index('moves') if 'moves' in tok else len(tok)
                fen, rest = ' '.join(tok[2:end]), tok[end:]
            moves = rest[1:] if rest and rest[0] == 'moves' else []
        elif cmd == 'go' and 'perft' in tok:
            b, s = R.parse_fen(fen)
            for u in moves:
                b, s = R.make_move(b, R.uci_to_move(u)), R.other(s)
            print('Nodes searched: %d' % R.perft(b, s, int(tok[tok.index('perft') + 1])))
        elif cmd == 'go':
            depth = int(tok[tok.index('depth') + 1]) if 'depth' in tok else None
            mt = int(tok[tok.index('movetime') + 1]) if 'movetime' in tok else None
            r = eng.best_move(fen, moves, depth=depth, movetime_ms=mt or (None if depth else 1000))
            score = 'mate %d' % r.mate if r.mate is not None else 'cp %d' % (r.score_cp or 0)
            print('info depth %d score %s nodes %d pv %s' % (r.depth or 0, score, r.nodes or 0, ' '.join(r.pv)))
            print('bestmove %s' % (r.move or '(none)'))
        elif cmd == 'quit':
            break
        sys.stdout.flush()


if __name__ == '__main__':
    uci_loop()
