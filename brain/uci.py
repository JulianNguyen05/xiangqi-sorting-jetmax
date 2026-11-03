#!/usr/bin/env python3
# Điều khiển engine cờ tướng qua giao thức UCI (Pikafish, hoặc brain/simple.py khi chạy dạng UCI).
#
#   with UciEngine('/home/jetmax/pikafish/pikafish', eval_file='/home/jetmax/pikafish/pikafish.nnue') as eng:
#       res = eng.best_move(fen, depth=10)
#       print(res.move, res.score_cp, res.pv)
#
# Python 3.6: không dùng capture_output / text=, đọc stdout bằng 1 luồng riêng để có timeout.

import os
import queue
import subprocess
import threading
import time
from collections import namedtuple

EngineResult = namedtuple('EngineResult', 'move ponder score_cp mate depth nodes pv time_ms')
# move: nước tốt nhất dạng 'h2e2', hoặc None nếu hết nước đi (bị chiếu bí / hết nước)
# score_cp: điểm theo góc nhìn BÊN ĐI, đơn vị 1/100 tốt; mate: số nước tới chiếu bí (âm = bị chiếu bí)


class EngineError(RuntimeError):
    pass


class UciEngine(object):
    def __init__(self, path, eval_file=None, threads=1, hash_mb=16, options=None, timeout=15.0, args=None, cwd=None):
        path = os.path.abspath(path) if os.path.exists(path) else path
        cmd = [path] + list(args or [])
        self.timeout = timeout
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1,
                                     cwd=cwd or os.path.dirname(path) or None)   # Pikafish tìm pikafish.nnue cạnh file chạy
        self._q = queue.Queue()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()
        self.name = None
        self.options = {}
        self.send('uci')
        for line in self._wait_for('uciok'):
            if line.startswith('id name '):
                self.name = line[8:].strip()
            elif line.startswith('option name '):
                self.options[line[12:].split(' type ')[0]] = line
        opts = {'Threads': threads, 'Hash': hash_mb}
        if eval_file:
            opts['EvalFile'] = os.path.abspath(eval_file)
        opts.update(options or {})
        for k, v in opts.items():
            if k in self.options:
                self.send('setoption name %s value %s' % (k, v))
        self.ready()

    # -------------------------------------------------------------- vào / ra
    def _reader(self):
        for line in self.proc.stdout:
            self._q.put(line.rstrip('\n'))
        self._q.put(None)                                  # tiến trình đã thoát

    def send(self, text):
        if self.proc.poll() is not None:
            raise EngineError('Engine đã dừng (mã %s)' % self.proc.returncode)
        self.proc.stdin.write(text + '\n')
        self.proc.stdin.flush()

    def _wait_for(self, prefix, timeout=None):
        """Đọc từng dòng cho tới dòng bắt đầu bằng prefix; trả về mọi dòng đã đọc."""
        deadline = time.time() + (timeout or self.timeout)
        lines = []
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise EngineError('Hết thời gian chờ "%s". Dòng cuối: %s' % (prefix, lines[-3:]))
            try:
                line = self._q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise EngineError('Engine thoát giữa chừng. Thông báo: %s' % '\n'.join(lines[-6:]))
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def ready(self):
        self.send('isready')
        self._wait_for('readyok')

    # -------------------------------------------------------------- lệnh
    def new_game(self):
        self.send('ucinewgame')
        self.ready()

    def set_position(self, fen, moves=None):
        cmd = 'position fen ' + fen
        if moves:
            cmd += ' moves ' + ' '.join(moves)
        self.send(cmd)

    def best_move(self, fen, moves=None, depth=None, movetime_ms=None, nodes=None):
        """Tìm nước đi tốt nhất. Chọn 1 trong: depth (độ sâu), movetime_ms (thời gian), nodes."""
        self.set_position(fen, moves)
        if depth:
            go = 'go depth %d' % depth
        elif nodes:
            go = 'go nodes %d' % nodes
        else:
            go = 'go movetime %d' % (movetime_ms or 1000)
        t0 = time.time()
        self.send(go)
        wait = self.timeout + (movetime_ms or 0) / 1000.0
        lines = self._wait_for('bestmove', timeout=wait if not depth else max(wait, 120))
        dt = (time.time() - t0) * 1000.0
        info = parse_info(lines)
        tok = lines[-1].split()
        move = tok[1] if len(tok) > 1 and tok[1] not in ('(none)', '0000') else None
        ponder = tok[3] if len(tok) > 3 and tok[2] == 'ponder' else None
        return EngineResult(move, ponder, info.get('cp'), info.get('mate'), info.get('depth'),
                            info.get('nodes'), info.get('pv', []), dt)

    def perft(self, fen, depth):
        """Đếm nút perft bằng engine (để đối chiếu với board/rules.py)."""
        self.set_position(fen)
        self.send('go perft %d' % depth)
        lines = self._wait_for('Nodes searched', timeout=300)
        return int(lines[-1].split(':')[1])

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send('quit')
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def parse_info(lines):
    """Lấy thông tin của dòng 'info ... pv ...' cuối cùng (độ sâu lớn nhất)."""
    out = {}
    for line in lines:
        if not line.startswith('info ') or ' pv ' not in line:
            continue
        tok = line.split()
        cur = {}
        i = 1
        while i < len(tok):
            k = tok[i]
            if k == 'score' and i + 2 < len(tok):
                cur['cp' if tok[i + 1] == 'cp' else 'mate'] = int(tok[i + 2])
                i += 3
                continue
            if k in ('depth', 'nodes', 'time', 'nps') and i + 1 < len(tok):
                cur[k] = int(tok[i + 1])
                i += 2
                continue
            if k == 'pv':
                cur['pv'] = tok[i + 1:]
                break
            i += 1
        if 'multipv' in tok and tok[tok.index('multipv') + 1] != '1':
            continue
        out = cur
    return out
