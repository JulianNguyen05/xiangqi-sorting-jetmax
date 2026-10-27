import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]
import numpy as np
from vision_multi import View, MultiViewMapper, ScanObserver, fit_homography
from board import square_name

NAMES = ['%s_%s' % (c, p) for c in ('black', 'red') for p in ('advisor', 'bishop', 'cannon', 'king', 'knight', 'pawn', 'rook')]
CODE = {'a': 'advisor', 'b': 'bishop', 'c': 'cannon', 'k': 'king', 'n': 'knight', 'p': 'pawn', 'r': 'rook'}
SIZE = (640, 480)


def true_h(f0):
    """Phép chiếu thật lưới -> ảnh của một view: nhìn nghiêng nhẹ, thấy khoảng 5 cột."""
    return np.array([[100.0, 0, 70 - 100.0 * f0], [0, 42.0, 45.0], [0, 0.0004, 1.0]])


def project(h, f, r):
    v = h.dot([f, r, 1.0])
    return v[0] / v[2], v[1] / v[2]


def make_view(name, f0, cols):
    h = true_h(f0)
    pts = {}
    for f in cols:
        for r in (0, 3, 6, 9):
            x, y = project(h, f, r)
            pts[square_name(f, r)] = [x + random.uniform(-1.5, 1.5), y + random.uniform(-1.5, 1.5)]   # sai số bấm chuột
    return View(name, pts, SIZE, slider=int(f0 * 700), arm=(0, -150, 120)), h


def fake_dets(h, board, half=20):
    dets = []
    for (f, r), p in board.items():
        x, y = project(h, f, r)
        name = ('red_' if p.isupper() else 'black_') + CODE[p.lower()]
        dets.append((x - half, y - half, x + half, y + half, 0.9, NAMES.index(name)))
    return np.array(dets, np.float32).reshape(-1, 6)


def random_board(rng, n=32):
    sq = [(f, r) for f in range(9) for r in range(10)]
    rng.shuffle(sq)
    return dict((s, rng.choice('RNBAKCPrnbakcp')) for s in sq[:n])


def test_fit_homography_least_squares():
    h = true_h(2)
    src = [project(h, f, r) for f in (2, 4, 6) for r in (0, 5, 9)]
    dst = [(f, r) for f in (2, 4, 6) for r in (0, 5, 9)]
    H = fit_homography(src, dst)
    for s, d in zip(src, dst):
        v = H.dot([s[0], s[1], 1.0])
        assert abs(v[0] / v[2] - d[0]) < 1e-6 and abs(v[1] / v[2] - d[1]) < 1e-6


def test_three_views_cover_all_and_merge():
    random.seed(3)
    rng = random.Random(5)
    vA, hA = make_view('trai', 0, (0, 2, 4))
    vB, hB = make_view('giua', 2, (2, 4, 6))
    vC, hC = make_view('phai', 4, (4, 6, 8))
    m = MultiViewMapper([vA, vB, vC])
    assert m.uncovered() == [], m.uncovered()
    assert max(v.fit_error for v in m.views) < 0.1, [v.fit_error for v in m.views]
    for _ in range(20):
        truth = random_board(rng)
        results = [(v, fake_dets(h, truth), NAMES) for v, h in ((vA, hA), (vB, hB), (vC, hC))]
        board, seen, warns = m.merge(results)
        assert board == truth, (sorted(set(board.items()) ^ set(truth.items())), warns)
        assert len(seen) == 90


def test_missing_view_reported():
    random.seed(1)
    vA, _ = make_view('trai', 0, (0, 2, 4))
    m = MultiViewMapper([vA])
    miss = m.uncovered()
    assert miss and all(sq[0] in 'efghi' for sq in miss), miss      # chỉ thiếu các cột bên phải


def test_edge_piece_ignored_and_disagreement_warned():
    random.seed(2)
    vA, hA = make_view('trai', 0, (0, 2, 4))
    vB, hB = make_view('giua', 2, (2, 4, 6))
    m = MultiViewMapper([vA, vB])
    # quân ở cột 5 (sát mép phải view A, bị cắt đôi) -> view A bỏ qua, view B vẫn thấy
    x, y = project(hA, 5, 4)
    cut = np.array([[x - 20, y - 20, 640, y + 20, 0.95, NAMES.index('red_rook')]], np.float32)
    cells, _ = vA.cells(cut, NAMES)
    assert cells == {}
    board, _, _ = m.merge([(vA, cut, NAMES), (vB, fake_dets(hB, {(5, 4): 'R'}), NAMES)])
    assert board == {(5, 4): 'R'}
    # hai view bất đồng ở ô (3, 4): cảnh báo, giữ quân có điểm cao hơn
    a = fake_dets(hA, {(3, 4): 'R'})
    b = fake_dets(hB, {(3, 4): 'C'})
    b[0, 4] = 0.5
    board, _, warns = m.merge([(vA, a, NAMES), (vB, b, NAMES)])
    assert board == {(3, 4): 'R'} and any('bất đồng' in w for w in warns), (board, warns)


def test_scan_observer_with_stabilizer():
    from game import BoardStabilizer
    random.seed(4)
    rng = random.Random(9)
    vA, hA = make_view('trai', 0, (0, 2, 4))
    vB, hB = make_view('phai', 4, (4, 6, 8))
    hs = {'trai': hA, 'phai': hB}
    truth = random_board(rng)
    cur = {}

    class Det(object):
        names = NAMES
        def __call__(self, frame):
            return fake_dets(hs[cur['v'].name], truth)

    obs = ScanObserver(MultiViewMapper([vA, vB]), Det(), lambda v: cur.update(v=v), lambda: None, BoardStabilizer(2))
    assert obs() is None            # chưa đủ 2 vòng quét
    assert obs() == truth
