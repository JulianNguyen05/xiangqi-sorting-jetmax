#!/usr/bin/env python3
# Đo tốc độ engine trên robot (không cần mạng, không di chuyển tay máy):
#   python3 tools/bench_engine.py --engine ~/pikafish/pikafish --depths 6 8 10 12
#   python3 tools/bench_engine.py --simple                       # engine dự phòng Python
# Ghi kết quả vào runs/bench_engine.csv để mang về đưa vào báo cáo.
import argparse
import csv
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from board import rules as R          # noqa: E402
from brain.simple import SimpleEngine  # noqa: E402
from brain.uci import UciEngine        # noqa: E402

# Khai cuộc + 2 thế giữa trận (lấy từ ván Pikafish tự đánh trong notebook)
FENS = [
    R.START,
    'r1bakab1r/9/1cn3nc1/p1p1p1p1p/9/2P3P2/P3P3P/1C2C1N2/9/RNBAKAB1R b - - 0 1',
    '2bakab2/9/2n1c4/p3p1p1p/2p6/6P2/P1P1P3P/4C1N2/4A4/2BAK1B2 w - - 0 1',
]


def rss_mb(pid):
    try:
        with open('/proc/%d/status' % pid) as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except IOError:
        pass
    return float('nan')


def main():
    ap = argparse.ArgumentParser()
    # gói Bước 6 (robot: engine/pikafish, laptop Windows: engine/pikafish.exe); cài riêng thì ~/pikafish/pikafish
    cands = [os.path.join(ROOT, 'engine', f) for f in ('pikafish', 'pikafish.exe')]
    _def = next((c for c in cands if os.path.exists(c)), os.path.expanduser('~/pikafish/pikafish'))
    ap.add_argument('--engine', default=_def)
    ap.add_argument('--nnue', default=None, help='mặc định: pikafish.nnue cạnh file engine')
    ap.add_argument('--simple', action='store_true')
    ap.add_argument('--depths', type=int, nargs='+', default=[6, 8, 10, 12])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--hash', type=int, default=64)
    a = ap.parse_args()

    if a.simple:
        eng, pid = SimpleEngine(), os.getpid()
        a.depths = [d for d in a.depths if d <= 3] or [1, 2, 3]
    else:
        nnue = a.nnue or os.path.join(os.path.dirname(a.engine), 'pikafish.nnue')
        eng = UciEngine(a.engine, eval_file=nnue, threads=a.threads, hash_mb=a.hash, timeout=60)
        pid = eng.proc.pid
    name = getattr(eng, 'name', 'engine')
    print('Engine:', name)
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    out = os.path.join(ROOT, 'runs', 'bench_engine.csv')
    rows = []
    for i, fen in enumerate(FENS):
        for d in a.depths:
            eng.new_game()                       # xoá bảng băm: mỗi lần đo bắt đầu từ đầu
            t0 = time.time()
            r = eng.best_move(fen, depth=d)
            dt = (time.time() - t0) * 1000
            rows.append([name, i, d, r.move, r.score_cp, r.nodes, round(dt, 1), round(rss_mb(pid), 1)])
            print('thế %d  depth %2d  %-6s  điểm %6s  %8s nút  %8.0f ms  RAM %.0f MB' % tuple(
                [i, d, r.move, r.score_cp, r.nodes, dt, rss_mb(pid)]))
    with open(out, 'w') as fh:
        w = csv.writer(fh)
        w.writerow(['engine', 'position', 'depth', 'move', 'score_cp', 'nodes', 'time_ms', 'rss_mb'])
        w.writerows(rows)
    eng.close()
    print('Đã ghi', out)


if __name__ == '__main__':
    main()
