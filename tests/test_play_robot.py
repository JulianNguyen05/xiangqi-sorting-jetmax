#!/usr/bin/env python3
# Chạy cả ván trên thế giới giả lập qua đúng đường code thật:  python tests/test_play_robot.py
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

import play_robot
from xiangqi_rules import Position
from board import square_name

QUIET = lambda *a: None


def run(argv):
    return play_robot.run(play_robot.parse_args(['--fake', '--auto'] + argv), log=QUIET)


def check_consistent(game, ctx):
    pos = Position.start()
    for info in game.history:
        assert pos.is_legal(info.move), info.uci          # mọi nước phát lại từ đầu đều hợp lệ
        pos = pos.make(info.move)
    assert pos.fen() == game.pos.fen()
    w = ctx.world
    assert w.matches(game), 'bàn cờ vật lý lệch thế cờ của game'
    assert w.mover.held is None


def test_full_game_both_colours():
    for human in ('red', 'black'):
        game, ctx = run(['--human', human, '--seed', '5', '--max-plies', '50'])
        check_consistent(game, ctx)
        assert game.history[0].by == ('robot' if human == 'black' else 'human')
        assert len(game.history) == 50 or game.result() is not None
        assert ctx.world.mover.safe_stops == 0


def test_with_detection_noise_and_voting():
    for human in ('red', 'black'):
        game, ctx = run(['--human', human, '--seed', '8', '--noise', '0.02', '--frames', '3', '--max-plies', '40'])
        check_consistent(game, ctx)


def test_robot_failure_then_repair_and_continue():
    game, ctx = run(['--human', 'black', '--seed', '2', '--fake-fail-at', '2', '--max-plies', '20'])
    check_consistent(game, ctx)
    assert ctx.world.mover.safe_stops == 1                  # đã gặp đúng một lần lỗi giữa chừng và xử lý
    assert ctx.executor.broken is None
    assert len(game.history) == 20


def test_start_position_mismatch_is_reported():
    a = play_robot.parse_args(['--fake', '--auto', '--human', 'red', '--seed', '1', '--max-plies', '4'])
    ctx = play_robot.build_fake(a, QUIET)
    for sq in ('a0', 'b0', 'c0'):                           # thiếu 3 quân so với thế cờ ban đầu
        del ctx.world.physical[sq]
    game = play_robot.Game('red', play_robot.create_engine(None, 'easy', None, 1), None)
    logs = []
    try:
        play_robot.verify_start(game, ctx.observer, None, logs.append, tries=2)
        assert False
    except RuntimeError:
        pass
    assert any('thiếu' in l and 'a0=R' in l for l in logs), logs


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        fn()
        print('OK  ', name)
    print('TẤT CẢ ĐỀU ĐẠT')
