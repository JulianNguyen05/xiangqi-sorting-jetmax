#!/usr/bin/env python3
# Luật cờ tướng: sinh nước đi hợp lệ, kiểm tra thế cờ, suy ra nước vừa đi từ 2 lần nhìn bàn cờ.
#
# Quy ước (giống board/geometry.py và board/fen.py):
#   - Ô = (f, r): f = cột 0..8 (a..i), r = hàng 0..9; Đỏ ở hàng 0..4 (phía robot), Đen ở hàng 5..9.
#   - Thế cờ = dict {(f, r): mã quân}; chữ HOA = Đỏ, thường = Đen. K A B N R C P.
#   - Bên đi: 'w' = Đỏ, 'b' = Đen (giống FEN của Pikafish).
#   - Nước đi = ((f0, r0), (f1, r1)); dạng chữ UCI 'h2e2' (cùng ký hiệu Pikafish dùng).
# Python 3.6, không cần thư viện ngoài.

from board.fen import ALIAS, LIMIT, START
from board.geometry import FILES, NF, NR, parse_square, square_name

RED, BLACK = 'w', 'b'
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT = ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))

# Các điểm hợp lệ của Sĩ / Tượng (phía Đỏ; phía Đen lật hàng r -> 9 - r)
ADVISOR_PTS = {(3, 0), (5, 0), (4, 1), (3, 2), (5, 2)}
BISHOP_PTS = {(2, 0), (6, 0), (0, 2), (4, 2), (8, 2), (2, 4), (6, 4)}


def is_red(code):
    return code.isupper()


def side_of(code):
    return RED if code.isupper() else BLACK


def other(side):
    return BLACK if side == RED else RED


def on_board(f, r):
    return 0 <= f < NF and 0 <= r < NR


def in_palace(f, r, red):
    return 3 <= f <= 5 and (0 <= r <= 2 if red else 7 <= r <= 9)


def own_half(r, red):
    """True nếu hàng r thuộc nửa bàn của bên đó (chưa qua sông)."""
    return r <= 4 if red else r >= 5


# ---------------------------------------------------------------------------- FEN

def parse_fen(fen):
    """FEN -> (thế cờ, bên đi). KHÔNG kiểm tra số lượng quân (để còn đọc được thế cờ nhận diện sai)."""
    parts = fen.strip().split()
    rows = parts[0].split('/')
    if len(rows) != NR:
        raise ValueError('FEN phải có %d hàng, nhận %d' % (NR, len(rows)))
    board = {}
    for i, row in enumerate(rows):
        r = NR - 1 - i
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            low = ALIAS.get(ch.lower(), ch.lower())
            if low not in LIMIT:
                raise ValueError('Ký tự quân không hợp lệ: %r' % ch)
            if f >= NF:
                raise ValueError('Hàng %d dài quá 9 cột' % r)
            board[(f, r)] = low.upper() if ch.isupper() else low
            f += 1
        if f != NF:
            raise ValueError('Hàng %d có %d cột, phải đúng 9' % (r, f))
    side = BLACK if len(parts) > 1 and parts[1].lower() == 'b' else RED   # 'w' hoặc 'r' (red) = Đỏ
    return board, side


def to_fen(board, side=RED):
    """Thế cờ -> FEN đầy đủ (Pikafish đọc được)."""
    rows = []
    for r in range(NR - 1, -1, -1):
        s, empty = '', 0
        for f in range(NF):
            c = board.get((f, r))
            if c is None:
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += c
        if empty:
            s += str(empty)
        rows.append(s)
    return '%s %s - - 0 1' % ('/'.join(rows), side)


def board_key(board):
    """Chỉ phần xếp quân của FEN (dùng để so sánh 2 thế cờ)."""
    return to_fen(board).split()[0]


def move_to_uci(move):
    (f0, r0), (f1, r1) = move
    return square_name(f0, r0) + square_name(f1, r1)


def uci_to_move(text):
    text = text.strip().lower()
    i = 2 if text[2].isalpha() else 3          # 'h2e2' hoặc 'a10...' (không xảy ra với 0..9)
    return parse_square(text[:i]), parse_square(text[i:])


# ---------------------------------------------------------------------------- sinh nước đi

def pseudo_moves(board, sq):
    """Nước đi theo đúng cách đi của quân tại sq, CHƯA xét việc tự để tướng bị chiếu."""
    code = board[sq]
    red = is_red(code)
    t = code.lower()
    f, r = sq
    out = []

    def add(nf, nr):
        if not on_board(nf, nr):
            return
        dst = board.get((nf, nr))
        if dst is None or is_red(dst) != red:
            out.append((sq, (nf, nr)))

    if t == 'k':
        for df, dr in ORTHO:
            if in_palace(f + df, r + dr, red):
                add(f + df, r + dr)
    elif t == 'a':
        for df, dr in DIAG:
            if in_palace(f + df, r + dr, red):
                add(f + df, r + dr)
    elif t == 'b':
        for df, dr in DIAG:
            nf, nr = f + 2 * df, r + 2 * dr
            if on_board(nf, nr) and own_half(nr, red) and (f + df, r + dr) not in board:   # không bị cản mắt tượng
                add(nf, nr)
    elif t == 'n':
        for df, dr in KNIGHT:
            leg = (f + df // 2, r) if abs(df) == 2 else (f, r + dr // 2)   # chân mã
            if leg not in board:
                add(f + df, r + dr)
    elif t == 'r':
        for df, dr in ORTHO:
            nf, nr = f + df, r + dr
            while on_board(nf, nr):
                dst = board.get((nf, nr))
                if dst is None:
                    out.append((sq, (nf, nr)))
                else:
                    if is_red(dst) != red:
                        out.append((sq, (nf, nr)))
                    break
                nf, nr = nf + df, nr + dr
    elif t == 'c':
        for df, dr in ORTHO:
            nf, nr = f + df, r + dr
            screen = False
            while on_board(nf, nr):
                dst = board.get((nf, nr))
                if not screen:
                    if dst is None:
                        out.append((sq, (nf, nr)))       # pháo đi như xe khi không ăn
                    else:
                        screen = True                    # gặp ngòi
                elif dst is not None:
                    if is_red(dst) != red:
                        out.append((sq, (nf, nr)))       # ăn quân sau ngòi
                    break
                nf, nr = nf + df, nr + dr
    elif t == 'p':
        fwd = 1 if red else -1
        add(f, r + fwd)
        if not own_half(r, red):                         # đã qua sông: đi ngang được
            add(f - 1, r)
            add(f + 1, r)
    return out


def find_king(board, side):
    k = 'K' if side == RED else 'k'
    for sq, c in board.items():
        if c == k:
            return sq
    return None


def is_attacked(board, sq, by_side):
    """Ô sq có bị bên by_side tấn công không (kể cả luật 2 tướng đối mặt)."""
    f, r = sq
    red = by_side == RED

    def enemy(c, t):
        return c is not None and c.lower() == t and is_red(c) == red

    # Xe, pháo, tướng theo 4 hướng thẳng
    for df, dr in ORTHO:
        nf, nr, dist, screen = f + df, r + dr, 1, False
        while on_board(nf, nr):
            c = board.get((nf, nr))
            if c is not None:
                if not screen:
                    if enemy(c, 'r'):
                        return True
                    if enemy(c, 'k') and (dist == 1 or df == 0):   # tướng kề bên, hoặc 2 tướng đối mặt trên 1 cột
                        return True
                    screen = True
                else:
                    if enemy(c, 'c'):
                        return True
                    break
            nf, nr, dist = nf + df, nr + dr, dist + 1
    # Mã: quân mã ở N = sq - (df, dr), đi bằng (df, dr) tới sq; chân mã nằm sát N
    for df, dr in KNIGHT:
        nf, nr = f - df, r - dr
        if enemy(board.get((nf, nr)), 'n'):
            leg = (nf + df // 2, nr) if abs(df) == 2 else (nf, nr + dr // 2)
            if leg not in board:
                return True
    # Tốt: Đỏ tiến lên (+1), Đen tiến xuống (-1)
    back = -1 if red else 1
    if enemy(board.get((f, r + back)), 'p'):
        return True
    for df in (-1, 1):
        c = board.get((f + df, r))
        if enemy(c, 'p') and not own_half(r, red):
            return True
    # Sĩ, tượng (chỉ đi trong vùng của mình)
    for df, dr in DIAG:
        if enemy(board.get((f + df, r + dr)), 'a') and in_palace(f, r, red):
            return True
        if (enemy(board.get((f + 2 * df, r + 2 * dr)), 'b') and own_half(r, red)
                and (f + df, r + dr) not in board):
            return True
    return False


def in_check(board, side):
    k = find_king(board, side)
    return k is not None and is_attacked(board, k, other(side))


def make_move(board, move):
    """Trả về thế cờ MỚI sau nước đi (không sửa board)."""
    b = dict(board)
    b[move[1]] = b.pop(move[0])
    return b


def legal_moves(board, side):
    """Mọi nước đi hợp lệ của bên side (không tự để tướng bị chiếu, không để 2 tướng đối mặt)."""
    red = side == RED
    out = []
    for sq in [s for s, c in board.items() if is_red(c) == red]:
        for mv in pseudo_moves(board, sq):
            frm, to = mv
            cap = board.get(to)
            board[to] = board.pop(frm)                   # đi thử tại chỗ rồi trả lại (nhanh hơn copy)
            ok = not in_check(board, side)
            board[frm] = board.pop(to)
            if cap is not None:
                board[to] = cap
            if ok:
                out.append(mv)
    return out


def perft(board, side, depth):
    """Đếm số chuỗi nước đi hợp lệ độ sâu depth — phép thử chuẩn để kiểm chứng bộ sinh nước đi."""
    moves = legal_moves(board, side)
    if depth == 1:
        return len(moves)
    total = 0
    for mv in moves:
        total += perft(make_move(board, mv), other(side), depth - 1)
    return total


def game_status(board, side):
    """'ongoing' | 'checkmate' | 'stalemate' (cờ tướng: hết nước đi = thua, kể cả khi không bị chiếu)."""
    if legal_moves(board, side):
        return 'ongoing'
    return 'checkmate' if in_check(board, side) else 'stalemate'


# ---------------------------------------------------------------------------- kiểm tra thế cờ

def validate(board, side=None):
    """Kiểm tra một thế cờ (thường là kết quả nhận diện) có thể xảy ra trong ván thật không.
    Trả về danh sách lỗi (rỗng = hợp lệ). Mỗi lỗi: (mã lỗi, ô hoặc None, mô tả)."""
    issues = []
    count = {}
    for c in board.values():
        count[c] = count.get(c, 0) + 1
    for t, lim in LIMIT.items():
        for code in (t.upper(), t):
            n = count.get(code, 0)
            if n > lim:
                issues.append(('too_many', None, 'Quá nhiều %s: %d > %d' % (code, n, lim)))
    for code in ('K', 'k'):
        if count.get(code, 0) == 0:
            issues.append(('no_king', None, 'Thiếu tướng %s' % ('Đỏ' if code == 'K' else 'Đen')))

    for (f, r), c in sorted(board.items()):
        red = is_red(c)
        t = c.lower()
        rr = r if red else NR - 1 - r                     # hàng nhìn từ phía của chính quân đó
        name = square_name(f, r)
        if t == 'k' and not in_palace(f, r, red):
            issues.append(('bad_square', (f, r), '%s ở %s: tướng phải ở trong cung' % (c, name)))
        elif t == 'a' and (f, rr) not in ADVISOR_PTS:
            issues.append(('bad_square', (f, r), '%s ở %s: sĩ chỉ đứng ở 5 điểm trong cung' % (c, name)))
        elif t == 'b' and (f, rr) not in BISHOP_PTS:
            issues.append(('bad_square', (f, r), '%s ở %s: tượng chỉ đứng ở 7 điểm, không qua sông' % (c, name)))
        elif t == 'p':
            if rr < 3:
                issues.append(('bad_square', (f, r), '%s ở %s: tốt không thể lùi về sau hàng xuất phát' % (c, name)))
            elif rr <= 4 and f % 2 == 1:
                issues.append(('bad_square', (f, r), '%s ở %s: tốt chưa qua sông chỉ ở cột a c e g i' % (c, name)))

    kr, kb = find_king(board, RED), find_king(board, BLACK)
    if kr and kb and kr[0] == kb[0]:
        lo, hi = sorted((kr[1], kb[1]))
        if not any((kr[0], r) in board for r in range(lo + 1, hi)):
            issues.append(('kings_facing', None, 'Hai tướng đối mặt trên cột %s' % FILES[kr[0]]))
    if side is not None and kr and kb and in_check(board, other(side)):
        issues.append(('side_in_check', None, 'Bên vừa đi (%s) đang bị chiếu' %
                       ('Đỏ' if other(side) == RED else 'Đen')))
    return issues


def is_valid(board, side=None):
    return not validate(board, side)


# ---------------------------------------------------------------------------- suy ra nước vừa đi

def diff(a, b):
    """Số ô khác nhau giữa 2 thế cờ (ô trống khác ô có quân, hoặc khác loại quân)."""
    keys = set(a) | set(b)
    return sum(1 for k in keys if a.get(k) != b.get(k))


def infer_move(prev, side, observed, allow_none=True):
    """Thế cờ trước (đã chắc chắn đúng) + bên vừa đi + thế cờ vừa nhận diện (có thể sai vài ô)
    -> nước đi hợp lệ giải thích tốt nhất ảnh mới.

    Ý tưởng: không tin tuyệt đối vào ảnh mới; chỉ có ~44 nước hợp lệ, chọn nước mà thế cờ kết quả
    khác ảnh mới ít ô nhất. Nhờ vậy 1-2 quân nhận diện sai không làm hỏng ván cờ.
    Trả về dict: move (None = chưa đi), uci, board (thế cờ đã sửa), cost (số ô lệch), ambiguous, candidates."""
    cands = []
    if allow_none:
        cands.append((diff(prev, observed), None, prev))
    for mv in legal_moves(prev, side):
        nb = make_move(prev, mv)
        cands.append((diff(nb, observed), mv, nb))
    cands.sort(key=lambda x: x[0])
    best = cands[0]
    ambiguous = len(cands) > 1 and cands[1][0] == best[0]
    return {
        'move': best[1],
        'uci': move_to_uci(best[1]) if best[1] else None,
        'board': best[2],
        'cost': best[0],
        'ambiguous': ambiguous,
        'candidates': [(c, move_to_uci(m) if m else None) for c, m, _ in cands[:5]],
    }


# ---------------------------------------------------------------------------- ký hiệu tiếng Việt

VN_NAME = {'k': 'Tướng', 'a': 'Sĩ', 'b': 'Tượng', 'n': 'Mã', 'r': 'Xe', 'c': 'Pháo', 'p': 'Tốt'}


def move_to_vn(board, move):
    """Ký hiệu truyền thống, ví dụ 'Pháo 2 bình 5', 'Mã 8 tiến 7', 'Xe trước thoái 2'.
    Cột đánh số 1..9 từ PHẢI sang TRÁI theo góc nhìn của bên đi (Đỏ: cột i = 1; Đen: cột a = 1)."""
    (f0, r0), (f1, r1) = move
    code = board[(f0, r0)]
    red = is_red(code)
    t = code.lower()
    col = (lambda f: NF - f) if red else (lambda f: f + 1)
    fwd = (r1 - r0) if red else (r0 - r1)
    name = VN_NAME[t]
    same = sorted([r for (f, r), c in board.items() if c == code and f == f0], reverse=red)
    if len(same) == 2 and t not in ('a', 'b'):           # 2 quân cùng loại trên cùng cột: trước / sau
        name += ' trước' if r0 == same[0] else ' sau'
        head = name
    else:
        head = '%s %d' % (name, col(f0))
    if fwd == 0:
        return '%s bình %d' % (head, col(f1))
    act = 'tiến' if fwd > 0 else 'thoái'
    if t in ('n', 'b', 'a'):                             # quân đi chéo: ghi cột đến
        return '%s %s %d' % (head, act, col(f1))
    return '%s %s %d' % (head, act, abs(fwd))


def start_board():
    return parse_fen(START)[0]
