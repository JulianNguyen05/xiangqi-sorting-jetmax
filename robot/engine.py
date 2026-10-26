#!/usr/bin/env python3
# Engine chọn nước đi cho robot. Hai lựa chọn, cùng giao diện:   move = engine.choose(position)
#
#   UciEngine    : chạy Pikafish (engine cờ tướng mã nguồn mở, giao tiếp UCI) như một tiến trình con.
#                  Đã cấu hình NHẸ cho Jetson: 1 luồng, hash 16 MB, giới hạn độ sâu + thời gian suy nghĩ.
#                  Cần: file chạy `pikafish` (bản build cho aarch64/ARM64) và file mạng `pikafish.nnue`
#                  để cạnh file chạy (hoặc truyền --nnue).
#   SimpleEngine : tìm kiếm alpha-beta độ sâu 2-3 tự viết, chỉ tính điểm theo quân. Không cần cài gì, dùng để thử
#                  logic khi chưa có Pikafish, hoặc dự phòng nếu Pikafish lỗi.
#
# "Mức dễ" được làm bằng cách cho engine trả về nhiều nước tốt nhất (MultiPV), rồi với một xác suất nhất định chọn
# NGẪU NHIÊN trong các nước không tệ hơn nước tốt nhất quá `margin` (đơn vị centipawn, 100 ~ một tốt).
# Cách này không phụ thuộc phiên bản engine nên mức khó dễ chỉnh được ngay trong LEVELS bên dưới.

import os, re, sys, random, subprocess, threading, time
try:
    import queue
except ImportError:           # Python 2 (không dùng, để cho chắc)
    import Queue as queue

from xiangqi_rules import Position, RED, BLACK, has_crossed, parse_uci, move_uci

LEVELS = {
    #          độ sâu  thời gian(ms)  số nước ứng viên  chênh lệch tối đa  xác suất đi "lệch"
    'easy':   dict(depth=1, movetime=800, multipv=5, margin=200, p_random=0.40, simple_depth=2),
    'normal': dict(depth=3, movetime=1500, multipv=3, margin=100, p_random=0.20, simple_depth=2),
    'hard':   dict(depth=8, movetime=3000, multipv=1, margin=0, p_random=0.0, simple_depth=3),
}
MATE = 30000


def _pick(cands, cfg, rng):
    """cands: [(move, score)] đã sắp theo điểm giảm dần. Chọn nước theo mức độ."""
    best = cands[0][1]
    if len(cands) > 1 and cfg['p_random'] > 0 and rng.random() < cfg['p_random']:
        pool = [m for m, s in cands if s >= best - cfg['margin']]
        return rng.choice(pool)
    return cands[0][0]


# ------------------------------------------------------------------ Pikafish -----
class UciEngine(object):
    def __init__(self, path, level='easy', nnue=None, threads=1, hash_mb=16, seed=None, startup_timeout=30.0,
                 args=None):
        if level not in LEVELS:
            raise ValueError('level phải là một trong %s' % sorted(LEVELS))
        self.cfg = LEVELS[level]
        self.rng = random.Random(seed)
        self.name = 'pikafish(%s)' % level
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            raise IOError('Không thấy engine: %s' % path)
        self.proc = subprocess.Popen([path] + list(args or []), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1,
                                     cwd=os.path.dirname(path))
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        self._send('uci')
        lines = self._read_until(lambda l: l.strip() == 'uciok', startup_timeout)
        self.options = set()
        for l in lines:
            m = re.match(r'option name (.+?) type', l.strip())
            if m:
                self.options.add(m.group(1))
        self._set('Threads', threads)
        self._set('Hash', hash_mb)
        self._set('MultiPV', self.cfg['multipv'])
        if nnue:
            self._set('EvalFile', os.path.abspath(os.path.expanduser(nnue)))
        self._send('isready')
        self._read_until(lambda l: l.strip() == 'readyok', startup_timeout)
        self._send('ucinewgame')

    # ---- giao tiếp
    def _reader(self):
        for line in iter(self.proc.stdout.readline, ''):
            self.q.put(line)
        self.q.put(None)

    def _send(self, s):
        try:
            self.proc.stdin.write(s + '\n')
            self.proc.stdin.flush()
        except (IOError, OSError, ValueError):
            raise RuntimeError('Engine đã thoát (kiểm tra file nnue và bản build có chạy được trên robot không)')

    def _read_until(self, pred, timeout):
        end = time.time() + timeout
        lines = []
        while True:
            left = end - time.time()
            if left <= 0:
                raise RuntimeError('Engine không trả lời trong %.0f giây' % timeout)
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                raise RuntimeError('Engine không trả lời trong %.0f giây' % timeout)
            if line is None:
                raise RuntimeError('Engine đã thoát. Đầu ra cuối: %s' % ' | '.join(l.strip() for l in lines[-5:]))
            lines.append(line)
            if pred(line):
                return lines

    def _set(self, name, value):
        if name in self.options:
            self._send('setoption name %s value %s' % (name, value))
        else:
            print('CẢNH BÁO: engine không có tùy chọn "%s", bỏ qua' % name, file=sys.stderr)

    # ---- chọn nước
    def choose(self, pos):
        legal = pos.legal_moves()
        if not legal:
            raise ValueError('Không còn nước đi hợp lệ')
        c = self.cfg
        self._send('position fen ' + pos.fen())
        self._send('go depth %d movetime %d' % (c['depth'], c['movetime']))
        limit = c['movetime'] / 1000.0 + 10.0
        try:
            lines = self._read_until(lambda l: l.startswith('bestmove'), limit)
        except RuntimeError:
            self._send('stop')                       # engine chậm hơn dự kiến: ép dừng và lấy kết quả hiện có
            lines = self._read_until(lambda l: l.startswith('bestmove'), 5.0)

        best = {}                                    # multipv -> (độ sâu, điểm, nước)
        for l in lines:
            tok = l.split()
            if not tok or tok[0] != 'info' or 'pv' not in tok or 'score' not in tok:
                continue
            try:
                d = int(tok[tok.index('depth') + 1]) if 'depth' in tok else 0
                k = int(tok[tok.index('multipv') + 1]) if 'multipv' in tok else 1
                i = tok.index('score')
                v = int(tok[i + 2])
                score = v if tok[i + 1] == 'cp' else (MATE - abs(v)) * (1 if v > 0 else -1)
                mv = parse_uci(tok[tok.index('pv') + 1])
            except (ValueError, IndexError):
                continue
            if k not in best or d >= best[k][0]:
                best[k] = (d, score, mv)
        legal_set = set(legal)
        cands = sorted([(mv, s) for d, s, mv in best.values() if mv in legal_set], key=lambda x: -x[1])
        if not cands:                                # không đọc được info: dùng bestmove
            for l in lines:
                if l.startswith('bestmove'):
                    try:
                        mv = parse_uci(l.split()[1])
                        if mv in legal_set:
                            cands = [(mv, 0)]
                    except (ValueError, IndexError):
                        pass
        if not cands:
            return self.rng.choice(legal)
        return _pick(cands, c, self.rng)

    def close(self):
        try:
            self._send('quit')
            self.proc.wait()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


# ------------------------------------------------------------------ tự viết -----
VALUE = {'K': 0, 'A': 200, 'B': 200, 'N': 400, 'R': 900, 'C': 450, 'P': 100}


def evaluate(pos):
    """Điểm theo quân, từ góc nhìn của bên đang đến lượt."""
    s = 0
    for r in range(10):
        row = pos.b[r]
        for f in range(9):
            p = row[f]
            if p == '.':
                continue
            t = p.upper()
            v = VALUE[t]
            if t == 'P' and has_crossed(r, RED if p.isupper() else BLACK):
                v += 100
            s += v if p.isupper() else -v
    return s if pos.turn == RED else -s


class SimpleEngine(object):
    def __init__(self, level='easy', seed=None):
        if level not in LEVELS:
            raise ValueError('level phải là một trong %s' % sorted(LEVELS))
        self.cfg = LEVELS[level]
        self.depth = self.cfg['simple_depth']
        self.rng = random.Random(seed)
        self.name = 'simple(%s, độ sâu %d)' % (level, self.depth)

    def _negamax(self, pos, depth, alpha, beta, ply):
        if depth == 0:
            return evaluate(pos)
        moves = pos.legal_moves()
        if not moves:
            return -MATE + ply                       # bên đến lượt hết nước = thua
        moves.sort(key=lambda m: -VALUE.get(pos.b[m[3]][m[2]].upper(), 0))   # ăn quân trước để cắt tỉa tốt hơn
        best = -MATE - 1
        for m in moves:
            s = -self._negamax(pos.make(m), depth - 1, -beta, -alpha, ply + 1)
            if s > best:
                best = s
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def choose(self, pos):
        moves = pos.legal_moves()
        if not moves:
            raise ValueError('Không còn nước đi hợp lệ')
        self.rng.shuffle(moves)
        scored = [(m, -self._negamax(pos.make(m), self.depth - 1, -MATE - 1, MATE + 1, 1)) for m in moves]
        scored.sort(key=lambda x: -x[1])
        # nếu có nước chiếu bí thì luôn đi, không "đi lệch"
        if scored[0][1] >= MATE - 10:
            return scored[0][0]
        return _pick(scored, self.cfg, self.rng)

    def close(self):
        pass


def create_engine(path=None, level='easy', nnue=None, seed=None):
    """Có đường dẫn Pikafish thì dùng Pikafish; lỗi hoặc không có thì dùng SimpleEngine (có in cảnh báo)."""
    if path:
        try:
            return UciEngine(path, level=level, nnue=nnue, seed=seed)
        except Exception as e:
            print('CẢNH BÁO: không chạy được Pikafish (%s). Dùng engine tự viết thay thế.' % e, file=sys.stderr)
    return SimpleEngine(level=level, seed=seed)
