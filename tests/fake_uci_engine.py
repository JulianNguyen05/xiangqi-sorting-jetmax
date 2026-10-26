#!/usr/bin/env python3
# Engine UCI GIẢ (không phải Pikafish) dùng để kiểm thử robot/engine.py::UciEngine mà không cần cài Pikafish.
# Nó in đúng dạng dòng "option", "info ... multipv ... score ... pv ...", "bestmove" như Pikafish: nước tốt nhất
# là nước đầu tiên trong danh sách hợp lệ, các nước sau điểm thấp dần.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'robot')]
from xiangqi_rules import Position, move_uci

pos, multipv = Position.start(), 1


def out(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


for line in sys.stdin:
    tok = line.split()
    if not tok:
        continue
    if tok[0] == 'uci':
        out('id name FakeFish')
        out('option name Threads type spin default 1 min 1 max 1024')
        out('option name Hash type spin default 16 min 1 max 33554432')
        out('option name MultiPV type spin default 1 min 1 max 500')
        out('option name EvalFile type string default pikafish.nnue')
        out('uciok')
    elif tok[0] == 'isready':
        out('readyok')
    elif tok[0] == 'setoption':
        if 'MultiPV' in tok:
            multipv = int(tok[-1])
    elif tok[0] == 'position' and tok[1] == 'fen':
        pos = Position.from_fen(' '.join(tok[2:]))
    elif tok[0] == 'go':
        moves = pos.legal_moves()
        out('info string fake engine')
        for k in range(min(multipv, len(moves))):
            out('info depth 1 seldepth 1 multipv %d score cp %d nodes 10 nps 1000 time 1 pv %s' % (
                k + 1, 10 - 100 * k, move_uci(moves[len(moves) - 1 - k])))        # depth 1: nước "sai" (đảo thứ tự)
            out('info depth 2 seldepth 3 multipv %d score cp %d nodes 99 nps 1000 time 2 pv %s %s' % (
                k + 1, 50 - 100 * k, move_uci(moves[k]), move_uci(moves[(k + 1) % len(moves)])))
        out('bestmove %s ponder %s' % (move_uci(moves[0]), move_uci(moves[1 % len(moves)])))
    elif tok[0] == 'quit':
        break
