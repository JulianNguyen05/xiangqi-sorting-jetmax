#!/usr/bin/env python3
# Đọc thế cờ dạng FEN cờ tướng -> {(cột, hàng): mã quân}.
# Mã quân: chữ HOA = Đỏ, thường = Đen. K tướng, A sĩ, B tượng, N mã, R xe, C pháo, P tốt.
# Chấp nhận cả E (tượng) và H (mã) theo một số phần mềm khác.
# FEN liệt kê từ hàng 9 (phía Đen) xuống hàng 0 (phía Đỏ), mỗi hàng từ cột a -> i.

from board.geometry import NF, NR, square_name

START = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
ALIAS = {'e': 'b', 'h': 'n'}
LIMIT = {'k': 1, 'a': 2, 'b': 2, 'n': 2, 'r': 2, 'c': 2, 'p': 5}
NAMES = {'k': ('Tướng', 'Tướng'), 'a': ('Sĩ', 'Sĩ'), 'b': ('Tượng', 'Tượng'), 'n': ('Mã', 'Mã'),
         'r': ('Xe', 'Xe'), 'c': ('Pháo', 'Pháo'), 'p': ('Binh', 'Tốt')}


def piece_label(code):
    red = code.isupper()
    return '%s %s' % (NAMES[code.lower()][0 if red else 1], 'Đỏ' if red else 'Đen')


def parse(fen):
    placement = fen.strip().split()[0]
    rows = placement.split('/')
    if len(rows) != NR:
        raise ValueError('FEN phải có %d hàng, nhận %d' % (NR, len(rows)))
    out, count = {}, {}
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
            code = low.upper() if ch.isupper() else low
            if f >= NF:
                raise ValueError('Hàng %d dài quá 9 cột' % r)
            out[(f, r)] = code
            count[code] = count.get(code, 0) + 1
            f += 1
        if f != NF:
            raise ValueError('Hàng %d có %d cột, phải đúng 9' % (r, f))
    for code, n in count.items():
        if n > LIMIT[code.lower()]:
            raise ValueError('Quá nhiều quân %s (%d)' % (piece_label(code), n))
    if count.get('K', 0) != 1 or count.get('k', 0) != 1:
        raise ValueError('Mỗi bên phải có đúng 1 Tướng')
    return out


if __name__ == '__main__':
    pos = parse(START)
    print('%d quân' % len(pos))
    for (f, r), c in sorted(pos.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(square_name(f, r), c, piece_label(c))
