#!/usr/bin/env python3
# Kiểm thử JetMaxExecutor với FakeMover, không cần robot/ROS:  python tests/test_executor.py
import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

from xiangqi_rules import Position
from engine import SimpleEngine
from game import Game, MoveInfo, make_info, RobotError
from board import Board, square_name, parse_square
from executor import JetMaxExecutor, Tray, make_resolver
from motion_fake import FakeMover

CORNERS = {'a0': (0, -96, -130), 'i0': (0, 96, -130), 'a9': (3600, -96, -130), 'i9': (3600, 96, -130)}


def setup(fail_at=None, limits=None):
    board = Board(CORNERS, z_pick=10.0, z_safe=50.0)
    tray = Tray(slider=4400, x=100.0, y=-120.0, cols=8, rows=4)
    resolve = make_resolver(board, tray)
    physical = dict((square_name(*sq), p) for sq, p in Position.start().to_dict().items())
    mv = FakeMover(resolve, physical, fail_at)
    ex = JetMaxExecutor(mv, board, tray, limits=limits, log=lambda *a: None)
    return board, tray, mv, ex, physical


def info_for(uci, pos=None):
    pos = pos or Position.start()
    from xiangqi_rules import parse_uci
    return make_info(pos, parse_uci(uci), 'robot')


def test_simple_move_sequence():
    _, _, mv, ex, phys = setup()
    ex.execute(info_for('h2e2'))
    assert mv.sequence() == [('pick', 'h2'), ('place', 'e2')], mv.sequence()
    assert phys['e2'] == 'C' and 'h2' not in phys


def test_capture_removes_victim_to_tray_first():
    _, _, mv, ex, phys = setup()
    ex.execute(info_for('h2h9'))                       # pháo đỏ ăn xe... h9 là mã đen
    assert mv.sequence() == [('pick', 'h9'), ('place', 'khay0'), ('pick', 'h2'), ('place', 'h9')], mv.sequence()
    assert phys['khay0'] == 'n' and phys['h9'] == 'C'
    assert ex.tray_used == 1


def test_tray_slots_distinct_and_in_order():
    t = Tray(4400, 100.0, -120.0, cols=3, rows=2)
    pos = [t.position(i) for i in range(6)]
    assert len(set(pos)) == 6
    assert pos[0] == (4400, 100.0, -120.0)
    try:
        t.position(6)
        assert False
    except IndexError:
        pass


def test_tray_full_is_refused_before_moving():
    _, tray, mv, ex, _ = setup()
    ex.tray_used = tray.capacity
    try:
        ex.execute(info_for('h2h9'))
        assert False
    except RobotError:
        pass
    assert mv.calls == []


def test_unreachable_square_refused_before_any_motion():
    limits = lambda p: p[0] <= 90.0                    # cột i (x=96) nằm ngoài tầm với
    _, _, mv, ex, _ = setup(limits=limits)
    assert 'i0' in ex.unreachable() and 'a0' not in ex.unreachable()
    try:
        ex.execute(info_for('i0i1'))
        assert False
    except RobotError as e:
        assert 'i0' in str(e) or 'i1' in str(e)
    assert mv.calls == [] and ex.broken is None


def test_failure_midway_stops_marks_broken_and_game_keeps_state():
    board, tray, mv, ex, phys = setup(fail_at=3)       # nước ăn quân: lệnh 3 = gắp quân của mình
    start = Position.from_fen('4k4/9/9/9/9/p8/9/9/9/R3K4 w')      # xe đỏ a0 ăn tốt đen a4
    phys.clear()
    phys.update({'a0': 'R', 'e0': 'K', 'e9': 'k', 'a4': 'p'})
    game = Game('black', SimpleEngine('easy', seed=0), ex, start=start)
    fen = game.pos.fen()
    info = info_for('a0a4', start)
    assert info.captured == 'p'
    try:
        ex.execute(info)
        assert False
    except RobotError:
        pass
    assert mv.safe_stops == 1 and ex.broken
    assert ex.tray_used == 1                           # quân bị ăn đã nằm trong khay
    assert game.pos.fen() == fen
    try:                                               # chưa reset thì không cho đi tiếp
        ex.execute(info_for('e0e1', start))
        assert False
    except RobotError as e:
        assert 'reset' in str(e)
    ex.reset()
    assert ex.broken is None


def test_full_game_physical_board_stays_in_sync():
    for human in ('red', 'black'):
        board, tray, mv, ex, phys = setup()
        game = Game(human, SimpleEngine('easy', seed=3), ex, confirm=1, max_plies=80)
        rng = random.Random(11)
        while game.result() is None:
            if game.robot_to_move():
                game.robot_move()
            else:
                m = rng.choice(game.pos.legal_moves())
                f1, r1, f2, r2 = m
                a, b = square_name(f1, r1), square_name(f2, r2)
                if b in phys:                          # người ăn quân: bỏ quân bị ăn khỏi bàn
                    phys.pop(b)
                phys[b] = phys.pop(a)
                _, info = game.human_observed(game.pos.make(m).to_dict())
                assert info is not None
            board_now = dict((k, v) for k, v in phys.items() if not k.startswith('khay'))
            expect = dict((square_name(*sq), p) for sq, p in game.pos.to_dict().items())
            assert board_now == expect, (human, len(game.history))
        n_tray = sum(1 for k in phys if k.startswith('khay'))
        assert n_tray == ex.tray_used
        assert mv.safe_stops == 0 and mv.held is None
        print('   (%s: %d nước, %d quân robot đã bỏ vào khay)' % (human, len(game.history), n_tray))


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        fn()
        print('OK  ', name)
    print('TẤT CẢ ĐỀU ĐẠT')
