#!/usr/bin/env python3
# Theo dõi ván cờ giữa người và robot ở mức THẾ CỜ (chưa có camera):
#   game = GameTracker(robot_side='b')
#   r = game.observe(board_tu_camera)        # người vừa đi -> suy ra nước đi, sửa lỗi nhận diện
#   if r['status'] in ('moved', 'corrected'):
#       res = game.suggest(engine)           # nước tiếp theo cho robot (engine: UciEngine / SimpleEngine)
#       game.apply(res.move)                 # robot đi xong thì ghi vào ván
# Bước nhận diện ảnh -> thế cờ (vision/) sẽ cấp board_tu_camera cho observe().

from board import rules as R


class GameTracker(object):
    def __init__(self, fen=R.START, robot_side=R.BLACK, max_cost=2):
        self.start_fen = fen
        self.board, self.side = R.parse_fen(fen)
        self.robot_side = robot_side
        self.max_cost = max_cost          # số ô lệch tối đa giữa ảnh và thế cờ dự kiến còn chấp nhận được
        self.history = []                 # các nước đã đi, dạng UCI

    @property
    def fen(self):
        return R.to_fen(self.board, self.side)

    def robot_to_move(self):
        return self.side == self.robot_side

    def apply(self, move):
        """Ghi một nước đi (UCI 'h2e2' hoặc ((f0,r0),(f1,r1))) sau khi kiểm tra hợp lệ."""
        mv = R.uci_to_move(move) if isinstance(move, str) else move
        if mv not in R.legal_moves(self.board, self.side):
            raise ValueError('Nước %s không hợp lệ ở thế %s' % (R.move_to_uci(mv), self.fen))
        vn = R.move_to_vn(self.board, mv)
        self.board = R.make_move(self.board, mv)
        self.side = R.other(self.side)
        self.history.append(R.move_to_uci(mv))
        return vn

    def observe(self, observed):
        """Thế cờ vừa nhận diện từ camera -> nước bên đang đến lượt vừa đi.
        status: 'no_change' | 'moved' | 'corrected' (đã sửa vài ô nhận diện sai) | 'uncertain' (chụp lại)
                | 'game_over'."""
        st = R.game_status(self.board, self.side)
        if st != 'ongoing':
            return {'status': 'game_over', 'result': st, 'loser': self.side}
        res = R.infer_move(self.board, self.side, observed)
        if res['cost'] > self.max_cost or (res['ambiguous'] and res['move'] is not None):
            res['status'] = 'uncertain'
            return res
        if res['move'] is None:
            res['status'] = 'no_change' if res['cost'] == 0 else 'uncertain'
            return res
        res['status'] = 'moved' if res['cost'] == 0 else 'corrected'
        res['vn'] = self.apply(res['move'])
        return res

    def suggest(self, engine, **kw):
        """Hỏi engine nước tiếp theo. Gửi FEN gốc + chuỗi nước đã đi để engine biết lặp nước."""
        if not kw:
            kw = {'depth': 10}
        res = engine.best_move(self.start_fen, moves=self.history, **kw)
        return res
