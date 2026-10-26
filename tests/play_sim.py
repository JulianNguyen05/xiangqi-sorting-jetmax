#!/usr/bin/env python3
# Chơi thử cả ván với robot GIẢ LẬP, không cần robot/camera/ROS.
# Bàn cờ "vật lý" được mô phỏng: robot đi xong thì bàn cờ vật lý đổi theo, người đi (gõ tay hoặc ngẫu nhiên) thì bàn cờ
# vật lý đổi, rồi bàn cờ được đưa qua đúng đường nhận diện thật (có thể thêm nhiễu: bỏ sót quân như YOLO) để
# game.detect_move tìm ra nước người vừa đi.
#
#   python3 tests/play_sim.py --human red                       # bạn cầm ĐỎ đi trước, gõ nước dạng h2e2 (q = thoát)
#   python3 tests/play_sim.py --human black                     # bạn cầm ĐEN, robot cầm ĐỎ đi trước
#   python3 tests/play_sim.py --human red --auto --noise 0.03   # người đi ngẫu nhiên, 3% quân bị bỏ sót mỗi khung hình
#   python3 tests/play_sim.py --level normal --engine ~/pikafish/pikafish --nnue ~/pikafish/pikafish.nnue
# Gõ "?" khi đến lượt bạn để xem các nước hợp lệ.

import os, sys, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

from xiangqi_rules import Position, parse_uci, move_uci
from engine import create_engine, LEVELS
from game import Game, DryRunExecutor, play


class Sim(DryRunExecutor):
    """Vừa là 'robot' (execute), vừa là 'camera' (observe) của bàn cờ vật lý mô phỏng."""

    def __init__(self, noise=0.0, auto=False, rng=None, log=print):
        DryRunExecutor.__init__(self, log)
        self.noise, self.auto, self.rng = noise, auto, rng or random.Random()
        self.physical = Position.start()
        self.game = None
        self.calls = 0

    def execute(self, info):                      # robot đi: bàn cờ vật lý đổi theo
        DryRunExecutor.execute(self, info)
        self.physical = self.physical.make(info.move)

    def _human_move(self):
        legal = self.physical.legal_moves()
        if self.auto:
            return self.rng.choice(legal)
        while True:
            s = input('Nước đi của bạn (vd h2e2, ? = gợi ý, q = thoát): ').strip().lower()
            if s in ('q', 'quit', 'exit'):
                raise KeyboardInterrupt
            if s == '?':
                self.log(' '.join(sorted(move_uci(m) for m in legal)))
                continue
            try:
                m = parse_uci(s)
            except ValueError as e:
                self.log('LỖI: %s' % e)
                continue
            if m in legal:
                return m
            self.log('Nước %s không hợp lệ' % s)

    def observe(self):
        if self.game.pos.fen() == self.physical.fen():    # robot đã tin thế cờ này và người CHƯA đi -> người đi
            self.calls = 0
            self.physical = self.physical.make(self._human_move())
        self.calls += 1
        if self.calls > 500:
            raise RuntimeError('Nhận diện mãi không ra nước người vừa đi (nhiễu quá lớn?)')
        return dict((sq, p) for sq, p in self.physical.to_dict().items() if self.rng.random() >= self.noise)


def build(human, engine_path=None, level='easy', nnue=None, noise=0.0, auto=False, seed=None,
          max_plies=400, log=print):
    rng = random.Random(seed)
    sim = Sim(noise, auto, rng, log)
    eng = create_engine(engine_path, level, nnue, seed)
    game = Game(human, eng, sim, max_plies=max_plies)
    sim.game = game
    return game, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', choices=['red', 'black'], default='red', help='màu quân của người chơi')
    ap.add_argument('--level', choices=sorted(LEVELS), default='easy')
    ap.add_argument('--engine', help='đường dẫn file chạy Pikafish (bỏ trống = engine tự viết)')
    ap.add_argument('--nnue', help='đường dẫn pikafish.nnue (nếu không nằm cạnh file chạy)')
    ap.add_argument('--auto', action='store_true', help='người chơi đi ngẫu nhiên (để chạy thử tự động)')
    ap.add_argument('--noise', type=float, default=0.0, help='xác suất bỏ sót mỗi quân ở mỗi khung hình (0-1)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--max-plies', type=int, default=400)
    a = ap.parse_args()

    game, sim = build(a.human, a.engine, a.level, a.nnue, a.noise, a.auto, a.seed, a.max_plies)
    print('Người cầm %s, robot cầm %s (%s). Engine: %s' % (
        'ĐỎ' if a.human == 'red' else 'ĐEN', 'ĐEN' if a.human == 'red' else 'ĐỎ',
        'người đi trước' if a.human == 'red' else 'robot đi trước', game.engine.name))
    print(game.pos)
    try:
        play(game, sim.observe, poll=0)
    except KeyboardInterrupt:
        print('\nDừng theo yêu cầu')
    finally:
        game.engine.close()


if __name__ == '__main__':
    main()
