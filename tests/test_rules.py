#!/usr/bin/env python3
# Kiểm thử luật cờ tướng. Không cần robot/ROS.
#   python3 tests/test_rules.py            # perft độ sâu 1-3 + các tình huống riêng (vài giây)
#   python3 tests/test_rules.py --deep     # thêm perft độ sâu 4 (3.290.240 nút, chậm trên Jetson)
# Số perft từ thế cờ ban đầu của cờ tướng: 44, 1920, 79666, 3290240 (số chuẩn dùng để kiểm tra bộ sinh nước đi).

import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

from xiangqi_rules import Position, perft, move_uci, parse_uci, START_FEN

PERFT = {1: 44, 2: 1920, 3: 79666, 4: 3290240}


def uci_set(fen, only_from=None):
    pos = Position.from_fen(fen)
    ms = [move_uci(m) for m in pos.legal_moves()]
    if only_from:
        ms = [m for m in ms if m.startswith(only_from)]
    return set(ms)


def test_fen_roundtrip():
    assert Position.start().fen() == START_FEN
    p = Position.from_fen('R3k4/R8/9/9/9/9/9/9/9/3K5 b')
    assert Position.from_fen(p.fen()).fen() == p.fen()


def test_flying_general():
    # tướng đỏ không được sang cột d vì sẽ đối mặt tướng đen ở d9
    assert uci_set('3k5/9/9/9/9/9/9/9/9/4K4 w') == {'e0f0', 'e0e1'}


def test_knight_leg():
    free = uci_set('5k3/9/9/9/9/9/9/9/4N4/3K5 w', 'e1')
    assert len(free) == 6, free
    blocked = uci_set('5k3/9/9/9/9/9/9/4p4/4N4/3K5 w', 'e1')    # tốt đen ở e2 cản chân mã
    assert len(blocked) == 4 and 'e1f3' not in blocked and 'e1d3' not in blocked, blocked


def test_cannon_screen():
    ms = uci_set('5k3/9/9/9/4r4/9/4P4/9/4C4/3K5 w', 'e1')
    assert 'e1e2' in ms                # đi không ăn
    assert 'e1e5' in ms                # ăn xe đen qua ngòi là tốt e3
    assert 'e1e3' not in ms and 'e1e4' not in ms


def test_bishop_river():
    ms = uci_set('5k3/9/9/9/9/2B6/9/9/9/3K5 w', 'c4')
    assert ms == {'c4a2', 'c4e2'}, ms


def test_pawn():
    assert uci_set('5k3/9/9/9/9/9/P8/9/9/4K4 w', 'a3') == {'a3a4'}             # chưa qua sông: chỉ đi thẳng
    assert uci_set('5k3/9/9/9/4P4/9/9/9/9/3K5 w', 'e5') == {'e5e6', 'e5d5', 'e5f5'}   # qua sông: thêm đi ngang


def test_checkmate():
    assert Position.from_fen('R3k4/R8/9/9/9/9/9/9/9/3K5 b').result() == 'checkmate'
    assert Position.start().result() == 'ongoing'


def test_perft(deep=False):
    pos = Position.start()
    for d in (1, 2, 3, 4) if deep else (1, 2, 3):
        t = time.time()
        n = perft(pos, d)
        print('  perft(%d) = %d  (chuẩn %d)  %.1fs' % (d, n, PERFT[d], time.time() - t))
        assert n == PERFT[d], 'perft(%d) sai: %d != %d' % (d, n, PERFT[d])


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_') and k != 'test_perft'):
        fn()
        print('OK  ', name)
    test_perft('--deep' in sys.argv)
    print('OK   test_perft')
    print('TẤT CẢ ĐỀU ĐẠT')
