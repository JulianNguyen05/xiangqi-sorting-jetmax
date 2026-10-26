#!/usr/bin/env python3
# Kiểm thử nhận diện nước đi, engine, ánh xạ ảnh -> ô cờ và cả ván mô phỏng. Không cần robot/ROS.
#   python3 tests/test_game.py

import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

from xiangqi_rules import Position, move_uci
from engine import SimpleEngine
from game import detect_move, BoardStabilizer, Game, make_info
from vision_board import ImageBoardMapper
from play_sim import build
from game import play


def test_detect_every_move():
    rng = random.Random(0)
    pos = Position.start()
    for _ in range(40):                           # đi ngẫu nhiên 40 nước, kiểm tra mọi nước hợp lệ ở mỗi thế cờ
        for m in pos.legal_moves():
            r = detect_move(pos, pos.make(m).to_dict())
            assert r.status == 'move' and r.move == m, (move_uci(m), r)
        assert detect_move(pos, pos.to_dict()).status == 'nochange'
        moves = pos.legal_moves()
        if not moves:
            break
        pos = pos.make(rng.choice(moves))


def test_detect_noise_never_wrong():
    rng = random.Random(1)
    pos = Position.start()
    wrong = ok = 0
    for _ in range(30):
        moves = pos.legal_moves()
        if not moves:
            break
        for m in moves:
            obs = pos.make(m).to_dict()
            del obs[rng.choice(sorted(obs))]      # YOLO bỏ sót một quân bất kỳ
            r = detect_move(pos, obs, tolerance=1)
            if r.status == 'move':
                if r.move == m:
                    ok += 1
                else:
                    wrong += 1
        pos = pos.make(rng.choice(moves))
    assert wrong == 0, 'nhận nhầm nước đi %d lần' % wrong
    assert ok > 0
    print('   (bỏ sót 1 quân: nhận đúng %d, không bao giờ nhận sai)' % ok)


def test_no_move_with_noise():
    pos = Position.start()
    obs = pos.to_dict()
    del obs[(4, 9)]                               # mất tướng đen trong ảnh, quân đen không thể là nước của người đi đỏ
    assert detect_move(pos, obs).status == 'nochange'
    obs = pos.to_dict()
    del obs[(4, 0)]                               # mất tướng đỏ: có thể là nhiễu, cũng có thể tướng vừa đi mà ảnh không
    assert detect_move(pos, obs).status == 'ambiguous'   # thấy ở ô mới -> không đoán, chờ khung hình sau


def test_invalid_board():
    pos = Position.start()
    obs = pos.to_dict()
    del obs[(0, 3)], obs[(2, 3)], obs[(4, 3)]     # mất 3 tốt: không nước nào khớp
    assert detect_move(pos, obs).status == 'invalid'


def test_stabilizer():
    s = BoardStabilizer(3)
    a = {(0, 0): 'R', (1, 0): 'N'}
    assert s.update(a) is None and s.update(a) is None
    assert s.update(a) == a
    flicker = {(0, 0): 'R'}                       # một khung hình bỏ sót quân vẫn qua được nhờ bỏ phiếu
    assert s.update(flicker) == a


def test_human_confirm():
    g = Game('red', SimpleEngine(seed=0), confirm=2)
    after = g.pos.make(g.pos.legal_moves()[0]).to_dict()
    r, info = g.human_observed(after)
    assert r.status == 'pending' and info is None
    r, info = g.human_observed(after)
    assert r.status == 'move' and info is not None and info.by == 'human'


def test_engine_legal_and_captures():
    # robot (đỏ) có xe ăn được pháo đen không được bảo vệ -> với độ sâu 2 phải ăn khi không "đi lệch"
    pos = Position.from_fen('5k3/9/9/9/4c4/9/9/9/4R4/3K5 w')
    eng = SimpleEngine('hard', seed=0)
    assert move_uci(eng.choose(pos)) == 'e1e5'


def test_uci_engine_with_fake():
    from engine import UciEngine
    fake = os.path.join(HERE, 'fake_uci_engine.py')
    eng = UciEngine(sys.executable, level='hard', args=[fake], seed=0)    # hard: luôn chọn nước tốt nhất
    try:
        pos = Position.start()
        assert eng.options >= {'Threads', 'Hash', 'MultiPV', 'EvalFile'}
        # phải lấy kết quả ở độ sâu LỚN NHẤT (depth 2), không phải depth 1: nước đầu tiên trong danh sách hợp lệ
        assert eng.choose(pos) == pos.legal_moves()[0]
        for _ in range(3):                         # nhiều lần liên tiếp trên nhiều thế cờ khác nhau
            pos = pos.make(eng.choose(pos))
            assert pos.legal_moves()
        easy = UciEngine(sys.executable, level='easy', args=[fake], seed=1)
        try:
            legal = set(pos.legal_moves())
            for _ in range(20):
                assert easy.choose(pos) in legal
        finally:
            easy.close()
    finally:
        eng.close()


def test_image_mapper():
    # ảnh giả: bàn cờ phóng 50 px/ô, góc a0 ở (100, 600), trục y ảnh hướng LÊN khi hàng tăng (xoay lật tuỳ ý)
    corners = {'a0': (100, 600), 'i0': (500, 600), 'a9': (100, 150), 'i9': (500, 150)}
    m = ImageBoardMapper(corners)
    names = ['%s_%s' % (c, p) for c in ('black', 'red')
             for p in ('advisor', 'bishop', 'cannon', 'king', 'knight', 'pawn', 'rook')]
    cx, cy = 100 + 4 * 50, 600 - 0 * 50           # ô e0
    dets = [(cx - 20, cy - 20, cx + 20, cy + 20, 0.9, names.index('red_king')),
            (100 + 7 * 50 - 20, 600 - 2 * 50 - 20, 100 + 7 * 50 + 20, 600 - 2 * 50 + 20, 0.8, names.index('red_cannon')),
            (100 + 3.5 * 50 - 20, 150 - 20, 100 + 3.5 * 50 + 20, 150 + 20, 0.7, names.index('black_rook'))]  # lệch giữa 2 ô
    board, warns = m.board_from_detections(dets, names)
    assert board == {(4, 0): 'K', (7, 2): 'C'}, board
    assert len(warns) == 1


def test_full_game_with_noise():
    for human in ('red', 'black'):
        game, sim = build(human, level='easy', noise=0.03, auto=True, seed=7, max_plies=60, log=lambda *a: None)
        play(game, sim.observe, log=lambda *a: None, poll=0)
        # mọi nước trong lịch sử phải hợp lệ khi phát lại từ đầu, và thế cờ cuối khớp thế cờ vật lý
        pos = Position.start()
        for info in game.history:
            assert pos.is_legal(info.move), info.uci
            pos = pos.make(info.move)
        assert pos.fen() == game.pos.fen()
        robot_first = game.history[0].by
        assert robot_first == ('robot' if human == 'black' else 'human')
        print('   (%s: %d nước, kết quả %s)' % (human, len(game.history), game.result()))


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        fn()
        print('OK  ', name)
    print('TẤT CẢ ĐỀU ĐẠT')
