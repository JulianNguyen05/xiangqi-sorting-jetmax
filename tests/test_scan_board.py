#!/usr/bin/env python3
# Kiểm thử scan_board (thế giới giả lập, không cần robot):  python tests/test_scan_board.py
import os, sys, csv, shutil, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

import numpy as np
import yaml
import scan_board as sb
from fake_world import FakeWorld, NAMES
from xiangqi_rules import Position

QUIET = lambda *a: None


def run(argv, log=QUIET):
    return sb.main(['--fake', '--no-save'] + argv, log=log)


def test_perfect_scan_matches_start_position():
    res = run(['--seed', '1'])
    assert len(res) == 1 and res[0].perfect
    d = res[0].diff
    assert len(d['ok']) == 32 and not d['missing'] and not d['wrong'] and not d['extra']
    assert len(res[0].seen) == 90


def test_removed_pieces_reported_as_missing():
    res = run(['--seed', '1', '--fake-remove', 'b0', 'h2'])
    d = res[0].diff
    assert sorted(sb.square_name(*c) for c, _ in d['missing']) == ['b0', 'h2']
    assert len(d['ok']) == 30 and not res[0].perfect
    lines = sb.diff_text(d)
    assert any('b0' in l and 'mã đỏ' in l for l in lines) and any('h2' in l and 'pháo đỏ' in l for l in lines)


def test_compare_and_board_text_markers():
    start = Position.start().to_dict()
    obs = dict(start)
    del obs[(0, 0)]                                  # thiếu xe a0
    obs[(1, 0)] = 'r'                                # nhầm mã b0 thành xe đen
    obs[(4, 4)] = 'p'                                # thừa tốt ở e4
    del obs[(8, 9)]                                  # không thấy xe i9, và ô i9 nằm ngoài mọi view
    seen = set((f, r) for f in range(9) for r in range(10)) - {(8, 9)}
    d = sb.compare(start, obs, seen)
    assert d['missing'] == [((0, 0), 'R')] and d['wrong'] == [((1, 0), 'r', 'N')] and d['extra'] == [((4, 4), 'p')]
    assert d['unseen'] == [(8, 9)]                   # i9 có quân nhưng không view nào thấy
    txt = sb.board_text(start, obs, seen)
    rows = {l.split()[0]: l for l in txt.splitlines() if l and l[0].isdigit()}
    assert 'r!' in rows['0'] and '? ' in rows['0'] and 'p+' in rows['4'] and '~' in rows['9']


def test_parse_expect():
    assert sb.parse_expect('none') is None
    assert sb.parse_expect('start') == Position.start().to_dict()
    fen = Position.start().fen()
    assert sb.parse_expect(fen.split(' ')[0]) == Position.start().to_dict()


def test_repeated_rounds_and_stability_with_noise():
    res = run(['--seed', '4', '--noise', '0.05', '--rounds', '6'])
    assert len(res) == 6
    n, bad = sb.stability(res)
    assert n == 6
    for cell, k in bad:
        assert 1 <= k <= 6
    perfect = sum(1 for r in res if r.perfect)
    assert perfect < 6                              # với nhiễu 5%, khó có chuyện cả 6 vòng đều hoàn hảo


def test_view_subset_leaves_unseen_cells():
    res = run(['--seed', '1', '--view', 'trai'])
    r = res[0]
    assert len(r.scans) == 1 and r.scans[0].view.name == 'trai'
    assert r.diff['unseen']                         # chỉ một view thì có ô không nhìn thấy
    assert all(sb.square_name(*c)[0] in 'efghi' for c in r.diff['unseen'])
    try:
        run(['--view', 'khong_co'])
        assert False
    except SystemExit as e:
        assert 'khong_co' in str(e)


def test_saves_images_csv_and_explains_every_detection():
    d = tempfile.mkdtemp()
    try:
        sb.main(['--fake', '--seed', '2', '--save-dir', d, '--fake-remove', 'e9'], log=QUIET)
        names = sorted(os.listdir(d))
        assert 'detections.csv' in names
        raw = [n for n in names if n.startswith('scan') and not n.endswith('_det.jpg')]
        det = [n for n in names if n.endswith('_det.jpg')]
        assert len(raw) == 3 and len(det) == 3
        rows = list(csv.DictReader(open(os.path.join(d, 'detections.csv'), encoding='utf-8')))
        assert rows and {r['status'] for r in rows} <= {'ok', 'dup', 'edge', 'outside', 'offcentre', 'unknown'}
        assert not any(r['cell'] == 'e9' for r in rows)       # e9 đã bị bỏ
        assert sum(1 for r in rows if r['status'] == 'ok') >= 31
    finally:
        shutil.rmtree(d)


def test_explain_statuses():
    w = FakeWorld(seed=1)
    v = w.views[0]
    cls = NAMES.index('red_rook')
    inside = (50, 25, 90, 65, 0.9, cls)                           # a0, nằm gọn
    clipped = (0, 25, 30, 65, 0.9, cls)                           # chạm mép trái ảnh
    far = (600, 300, 639, 340, 0.9, cls)                          # lệch lưới 9x10 về bên phải
    weird = (50, 100, 90, 140, 0.9, 99)                           # lớp lạ
    twin = (52, 27, 92, 67, 0.5, cls)                             # cùng ô a0 với `inside`, điểm thấp hơn
    recs = v.explain(np.array([inside, clipped, far, weird, twin], np.float32), NAMES)
    st = [r['status'] for r in recs]
    assert st[0] == 'ok' and st[1] == 'edge' and st[2] in ('outside', 'offcentre', 'edge') and st[3] == 'unknown' and st[4] == 'dup'
    assert recs[0]['cell'] == (0, 0) and recs[0]['piece'] == 'R'


def test_offline_replay_matches_live_scan():
    """Chạy --fake lưu ảnh, rồi quét lại offline bằng --images với một detector giả đọc ảnh -> kết quả y hệt."""
    d = tempfile.mkdtemp()
    try:
        live_dir = os.path.join(d, 'live')
        w_live = sb.main(['--fake', '--seed', '7', '--rounds', '2', '--save-dir', live_dir], log=QUIET)
        # đổi các ảnh gốc thành ảnh có mã theo view (độ xám) để detector giả nhận ra view nào
        import cv2
        world = FakeWorld(seed=7)
        code = dict((v.name, 60 + 60 * i) for i, v in enumerate(world.views))
        for fn in os.listdir(live_dir):
            if fn.startswith('scan') and not fn.endswith('_det.jpg'):
                view = fn[len('scan0000_'):-4]
                cv2.imwrite(os.path.join(live_dir, fn), np.full((480, 640, 3), code[view], np.uint8))
        inv = dict((c, n) for n, c in code.items())

        class StubDetector(object):
            names = NAMES

            def __call__(self, frame):
                world.cur = inv[int(round(float(frame.mean())))]
                return world(frame)
        stub = types.ModuleType('detect_view')
        stub.load_backend = lambda path, conf, iou: StubDetector()
        sys.modules['detect_view'] = stub
        vpath = os.path.join(d, 'views.yaml')
        yaml.safe_dump(world.views_yaml(), open(vpath, 'w', encoding='utf-8'))
        out = os.path.join(d, 'replay')
        res = sb.main(['--images', live_dir, '--views', vpath, '--model', 'x.onnx', '--save-dir', out], log=QUIET)
        assert len(res) == 2 and all(r.perfect for r in res)
        assert sorted(os.listdir(out)).count('detections.csv') == 1
        assert len([n for n in os.listdir(out) if n.endswith('_det.jpg')]) == 6
        assert [r.board for r in res] == [r.board for r in w_live]
    finally:
        sys.modules.pop('detect_view', None)
        shutil.rmtree(d)


def test_ruler_and_draw_run():
    import cv2
    img = np.full((480, 640, 3), 128, np.uint8)
    out = sb.draw_ruler(cv2, img)
    assert out.shape == img.shape and (out != img).any()


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        fn()
        print('OK  ', name)
    print('TẤT CẢ ĐỀU ĐẠT')
