#!/usr/bin/env python3
# Luật cờ tướng thuần Python (không cần ROS, không cần thư viện ngoài, chạy được trên Python 3.6 của Jetson).
#
# Quy ước TRÙNG với robot/board.py:
#   cột a..i = chỉ số 0..8 (trái -> phải nhìn từ phía quân ĐỎ), hàng 0..9 (hàng 0 = phía đỏ, hàng 9 = phía đen).
#   Nước đi dạng UCI/UCCI "h2e2" = quân ở h2 đi tới e2. Trong code một nước đi là tuple (f1, r1, f2, r2).
#
# Ký hiệu quân (giống FEN của Pikafish):  K tướng, A sĩ, B tượng, N mã, R xe, C pháo, P tốt
#   chữ HOA = quân ĐỎ (đi trước, 'w'), chữ thường = quân ĐEN ('b').
#
# Luật đã cài: mã bị cản chân, tượng bị chặn mắt và không qua sông, sĩ/tướng trong cung, tốt qua sông mới đi ngang,
# pháo ăn qua đúng một quân, hai tướng không được đối mặt (lộ mặt tướng), không được để tướng mình bị chiếu.
# Hết nước đi hợp lệ = THUA (bị chiếu bí hoặc hết nước đều thua trong cờ tướng).
# CHƯA cài: luật cấm chiếu liên tục / đuổi liên tục, luật hòa 60 nước không ăn quân (xem ghi chú trong game.py).

FILES = 'abcdefghi'
RED, BLACK = 'w', 'b'
EMPTY = '.'
START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'

ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT = ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))
PIECES = 'KABNRCPkabnrcp'


# ------------------------------------------------------------------ tiện ích ----
def sq_name(f, r):
    return '%s%d' % (FILES[f], r)


def parse_sq(s):
    s = s.strip().lower()
    if len(s) != 2 or s[0] not in FILES or not s[1].isdigit():
        raise ValueError('Ô "%s" không hợp lệ (cột a-i, hàng 0-9, ví dụ h2)' % s)
    return FILES.index(s[0]), int(s[1])


def move_uci(m):
    return sq_name(m[0], m[1]) + sq_name(m[2], m[3])


def parse_uci(s):
    s = s.strip().lower().replace(' ', '').replace('-', '')
    if len(s) != 4:
        raise ValueError('Nước đi "%s" không hợp lệ (dạng h2e2)' % s)
    f1, r1 = parse_sq(s[:2])
    f2, r2 = parse_sq(s[2:])
    return (f1, r1, f2, r2)


def side_of(p):
    return RED if p.isupper() else BLACK


def other(side):
    return BLACK if side == RED else RED


def in_palace(f, r, side):
    return 3 <= f <= 5 and (0 <= r <= 2 if side == RED else 7 <= r <= 9)


def has_crossed(r, side):
    """Quân đã qua sông chưa (r là hàng hiện tại)."""
    return r >= 5 if side == RED else r <= 4


def _on(f, r):
    return 0 <= f < 9 and 0 <= r < 10


# ------------------------------------------------------------------ thế cờ -----
class Position(object):
    """Thế cờ + bên đi. b[r][f] là ký tự quân hoặc '.'."""
    __slots__ = ('b', 'turn')

    def __init__(self, board=None, turn=RED):
        self.b = board if board is not None else [[EMPTY] * 9 for _ in range(10)]
        self.turn = turn

    # ---- tạo / chuyển đổi
    @classmethod
    def start(cls):
        return cls.from_fen(START_FEN)

    @classmethod
    def from_fen(cls, fen):
        parts = fen.split()
        if not parts:
            raise ValueError('FEN rỗng')
        rows = parts[0].split('/')
        if len(rows) != 10:
            raise ValueError('FEN phải có 10 hàng, nhận được %d' % len(rows))
        board = [[EMPTY] * 9 for _ in range(10)]
        for i, row in enumerate(rows):
            r, f = 9 - i, 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                elif ch in PIECES:
                    if f >= 9:
                        raise ValueError('Hàng FEN quá dài: %s' % row)
                    board[r][f] = ch
                    f += 1
                else:
                    raise ValueError('Ký tự lạ trong FEN: %s' % ch)
            if f != 9:
                raise ValueError('Hàng FEN không đủ 9 cột: %s' % row)
        turn = BLACK if len(parts) > 1 and parts[1].lower() == 'b' else RED
        return cls(board, turn)

    @classmethod
    def from_dict(cls, d, turn=RED):
        """d = {(f, r): 'R', ...}"""
        board = [[EMPTY] * 9 for _ in range(10)]
        for (f, r), p in d.items():
            board[r][f] = p
        return cls(board, turn)

    def to_dict(self):
        return dict(((f, r), self.b[r][f]) for r in range(10) for f in range(9) if self.b[r][f] != EMPTY)

    def fen(self):
        rows = []
        for r in range(9, -1, -1):
            s, n = '', 0
            for f in range(9):
                p = self.b[r][f]
                if p == EMPTY:
                    n += 1
                else:
                    if n:
                        s += str(n)
                        n = 0
                    s += p
            if n:
                s += str(n)
            rows.append(s)
        return '/'.join(rows) + ' %s - - 0 1' % self.turn

    def copy(self):
        return Position([row[:] for row in self.b], self.turn)

    def piece_at(self, f, r):
        p = self.b[r][f]
        return None if p == EMPTY else p

    def make(self, move):
        """Trả về thế cờ mới sau nước đi (KHÔNG kiểm tra hợp lệ; dùng is_legal trước nếu cần)."""
        f1, r1, f2, r2 = move
        n = self.copy()
        n.b[r2][f2] = n.b[r1][f1]
        n.b[r1][f1] = EMPTY
        n.turn = other(self.turn)
        return n

    def __str__(self):
        out = ['    a b c d e f g h i']
        for r in range(9, -1, -1):
            out.append('%d   %s' % (r, ' '.join(self.b[r])))
            if r == 5:
                out.append('    ~ ~ ~ ~ sông ~ ~ ~ ~')
        out.append('Lượt: %s' % ('ĐỎ (chữ hoa)' if self.turn == RED else 'ĐEN (chữ thường)'))
        return '\n'.join(out)

    # ---- sinh nước đi
    def _targets(self, f, r):
        """Các ô đến 'giả hợp lệ' (chưa xét bị chiếu) của quân tại (f, r)."""
        b = self.b
        p = b[r][f]
        side = side_of(p)
        t = p.upper()
        out = []

        def ok(f2, r2):
            q = b[r2][f2]
            return q == EMPTY or side_of(q) != side

        if t == 'R':
            for df, dr in ORTH:
                f2, r2 = f + df, r + dr
                while _on(f2, r2):
                    q = b[r2][f2]
                    if q == EMPTY:
                        out.append((f2, r2))
                    else:
                        if side_of(q) != side:
                            out.append((f2, r2))
                        break
                    f2 += df
                    r2 += dr
        elif t == 'C':
            for df, dr in ORTH:
                f2, r2 = f + df, r + dr
                while _on(f2, r2) and b[r2][f2] == EMPTY:
                    out.append((f2, r2))
                    f2 += df
                    r2 += dr
                f2 += df          # bỏ qua quân làm "ngòi"
                r2 += dr
                while _on(f2, r2):
                    q = b[r2][f2]
                    if q != EMPTY:
                        if side_of(q) != side:
                            out.append((f2, r2))
                        break
                    f2 += df
                    r2 += dr
        elif t == 'N':
            for df, dr in KNIGHT:
                f2, r2 = f + df, r + dr
                if not _on(f2, r2):
                    continue
                lf, lr = (f, r + dr // 2) if abs(dr) == 2 else (f + df // 2, r)   # chân mã
                if b[lr][lf] == EMPTY and ok(f2, r2):
                    out.append((f2, r2))
        elif t == 'B':
            for df, dr in DIAG:
                f2, r2 = f + 2 * df, r + 2 * dr
                if not _on(f2, r2):
                    continue
                if (r2 > 4) if side == RED else (r2 < 5):      # tượng không qua sông
                    continue
                if b[r + dr][f + df] == EMPTY and ok(f2, r2):  # mắt tượng
                    out.append((f2, r2))
        elif t == 'A':
            for df, dr in DIAG:
                f2, r2 = f + df, r + dr
                if _on(f2, r2) and in_palace(f2, r2, side) and ok(f2, r2):
                    out.append((f2, r2))
        elif t == 'K':
            for df, dr in ORTH:
                f2, r2 = f + df, r + dr
                if _on(f2, r2) and in_palace(f2, r2, side) and ok(f2, r2):
                    out.append((f2, r2))
        elif t == 'P':
            d = 1 if side == RED else -1
            cand = [(f, r + d)]
            if has_crossed(r, side):
                cand += [(f - 1, r), (f + 1, r)]
            for f2, r2 in cand:
                if _on(f2, r2) and ok(f2, r2):
                    out.append((f2, r2))
        return out

    def _king_sq(self, side):
        k = 'K' if side == RED else 'k'
        for r in range(10):
            for f in range(9):
                if self.b[r][f] == k:
                    return f, r
        return None

    def _king_attacked(self, side):
        """Tướng của `side` có đang bị chiếu không (kể cả lộ mặt tướng)."""
        ks = self._king_sq(side)
        if ks is None:
            return False
        kf, kr = ks
        b = self.b
        en = other(side)
        er, ec, eh, ep, ek = ('r', 'c', 'n', 'p', 'k') if side == RED else ('R', 'C', 'N', 'P', 'K')
        # xe, pháo, và tướng đối mặt theo đường thẳng
        for df, dr in ORTH:
            f2, r2 = kf + df, kr + dr
            while _on(f2, r2) and b[r2][f2] == EMPTY:
                f2 += df
                r2 += dr
            if not _on(f2, r2):
                continue
            q = b[r2][f2]
            if q == er or q == ek:
                return True
            f2 += df
            r2 += dr
            while _on(f2, r2) and b[r2][f2] == EMPTY:
                f2 += df
                r2 += dr
            if _on(f2, r2) and b[r2][f2] == ec:
                return True
        # mã (kiểm tra chân mã tính từ phía mã)
        for df, dr in KNIGHT:
            nf, nr = kf + df, kr + dr
            if not _on(nf, nr) or b[nr][nf] != eh:
                continue
            if abs(dr) == 2:
                lf, lr = nf, nr + (-dr) // 2
            else:
                lf, lr = nf + (-df) // 2, nr
            if b[lr][lf] == EMPTY:
                return True
        # tốt: đánh thẳng tới, hoặc đánh ngang nếu đã qua sông
        d_e = 1 if en == RED else -1
        if _on(kf, kr - d_e) and b[kr - d_e][kf] == ep:
            return True
        if has_crossed(kr, en):
            for sf in (kf - 1, kf + 1):
                if _on(sf, kr) and b[kr][sf] == ep:
                    return True
        return False

    def legal_moves(self):
        side = self.turn
        b = self.b
        res = []
        for r in range(10):
            for f in range(9):
                p = b[r][f]
                if p == EMPTY or side_of(p) != side:
                    continue
                for f2, r2 in self._targets(f, r):
                    cap = b[r2][f2]
                    b[r2][f2] = p
                    b[r][f] = EMPTY
                    bad = self._king_attacked(side)
                    b[r][f] = p
                    b[r2][f2] = cap
                    if not bad:
                        res.append((f, r, f2, r2))
        return res

    def is_legal(self, move):
        return tuple(move) in self.legal_moves()

    def in_check(self, side=None):
        return self._king_attacked(side or self.turn)

    def result(self):
        """'ongoing' | 'checkmate' | 'stalemate'. Cả hai kết quả cuối đều là bên đang đến lượt THUA."""
        if self.legal_moves():
            return 'ongoing'
        return 'checkmate' if self.in_check() else 'stalemate'


# ------------------------------------------------------------------ mô tả nước đi
PIECE_VN = {'K': 'tướng', 'A': 'sĩ', 'B': 'tượng', 'N': 'mã', 'R': 'xe', 'C': 'pháo', 'P': 'tốt'}


def piece_name(p):
    return '%s %s' % (PIECE_VN[p.upper()], 'đỏ' if p.isupper() else 'đen')


def perft(pos, depth):
    """Đếm số nút lá, dùng để kiểm tra luật (xem tests/test_rules.py)."""
    moves = pos.legal_moves()
    if depth <= 1:
        return len(moves)
    return sum(perft(pos.make(m), depth - 1) for m in moves)
