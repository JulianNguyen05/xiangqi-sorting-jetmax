#!/usr/bin/env python3
# Kiểm thử check_view_repeat (camera giả lập, không cần robot):  python tests/test_check_view_repeat.py
import os, sys, shutil, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

import cv2
import numpy as np
import yaml
import check_view_repeat as cv
from fake_world import FakeWorld

QUIET = lambda *a: None


def run(argv):
    return cv.main(['--fake', '--no-save'] + argv, log=QUIET)


def test_phase_shift_sign_and_accuracy():
    base = cv.synthetic_texture(cv2, 1)
    for sx, sy in [(3.4, -2.1), (-1.0, 0.5), (12.0, 7.0), (0.0, 0.0)]:
        img = cv2.warpAffine(base, np.float32([[1, 0, sx], [0, 1, sy]]), (640, 480), borderMode=cv2.BORDER_REFLECT)
        dx, dy, resp = cv.phase_shift(cv2, base, img)
        assert abs(dx - sx) < 0.4 and abs(dy - sy) < 0.4 and resp > 0.5, (sx, sy, dx, dy, resp)


def test_small_jitter_is_ok_and_measured_sensibly():
    r = run(['--rounds', '12', '--fake-jitter', '0.3', '--seed', '2'])
    assert r['worst'] == 'OK'
    for s in r['summaries']:
        assert 0.2 < s['rms'] < 1.2                    # độ lệch chuẩn 2D thật = 0,3 x sqrt(2) = 0,42 px
        assert s['cells'] is not None and s['cells'] < 0.05


def test_large_jitter_is_an_error():
    r = run(['--rounds', '10', '--fake-jitter', '6.0', '--seed', '2'])
    assert r['worst'] == 'LỖI'
    assert all(s['cells'] > 0.2 for s in r['summaries'])


def test_drift_detected():
    r = run(['--rounds', '10', '--fake-jitter', '0.2', '--fake-drift', '1.5', '--seed', '3'])
    assert all(s['drift'] > 8 for s in r['summaries']), [s['drift'] for s in r['summaries']]
    assert r['worst'] != 'OK'
    v, why = cv.verdict(r['summaries'][0], 3.0, 0.10)
    assert 'trôi' in why or v == 'LỖI'


def test_pixel_only_thresholds_without_taught_points():
    s = dict(view='x', n=10, mean=(0, 0), max_dev=2.0, rms=1.0, drift=0.5, first_max=2.0, resp_min=0.9, patch=0.1, cells=None)
    assert cv.verdict(s)[0] == 'OK'
    assert cv.verdict(dict(s, max_dev=4.0))[0] == 'CẢNH BÁO'
    assert cv.verdict(dict(s, max_dev=7.0))[0] == 'LỖI'
    assert cv.verdict(dict(s, resp_min=0.02))[0] == 'LỖI'


def test_cell_error_conversion_uses_taught_geometry():
    v = FakeWorld(seed=1).views[0]
    assert abs(cv.cell_error(v, 100.0, 0.0) - 1.0) < 0.1          # 100 px ngang ~ 1 ô
    assert abs(cv.cell_error(v, 0.0, 42.0) - 1.0) < 0.1           # 42 px dọc ~ 1 ô


def test_unrelated_images_flagged_unreliable():
    frames = [cv.synthetic_texture(cv2, 10 + i) for i in range(4)]    # bốn ảnh hoàn toàn khác nhau
    sh = cv.frame_shifts(cv2, frames)
    s = cv.summarize('x', sh)
    assert s['resp_min'] < cv.MIN_RESPONSE and cv.verdict(s)[0] == 'LỖI'


def test_offline_analysis_of_saved_frames_matches_live():
    d = tempfile.mkdtemp()
    try:
        live = cv.main(['--fake', '--rounds', '8', '--fake-jitter', '1.0', '--seed', '5', '--save-dir', d], log=QUIET)
        assert os.path.exists(os.path.join(d, 'repeat.csv'))
        off = cv.main(['--images', d], log=QUIET)
        for a, b in zip(live['summaries'], off['summaries']):
            assert a['view'] == b['view'] and a['n'] == b['n'] == 8
            assert abs(a['max_dev'] - b['max_dev']) < 0.5       # chênh nhẹ do nén JPEG
            assert abs(a['rms'] - b['rms']) < 0.3
    finally:
        shutil.rmtree(d)


def test_view_subset_and_too_few_frames():
    r = run(['--rounds', '5', '--view', 'giua', '--seed', '1'])
    assert [s['view'] for s in r['summaries']] == ['giua']
    import io, contextlib
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            cv.parse_args(['--fake', '--rounds', '2'])
        assert False
    except SystemExit:
        pass


def test_load_pose_views_skips_incomplete():
    import scan_board as sb
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 'views.yaml')
        yaml.safe_dump({'views': [
            {'name': 'a', 'slider': 0, 'arm': [0, -150, 120]},
            {'name': 'b', 'slider': None, 'arm': [0, -150, 120]},
            {'name': 'c', 'slider': 100, 'arm': [0, None, 120]},
            {'name': 'd', 'slider': 100, 'arm': [0, -150, 999]}]}, open(p, 'w', encoding='utf-8'))
        logs = []
        views = sb.load_pose_views(p, None, logs.append)
        assert [v.name for v in views] == ['a'] and len(logs) == 3
    finally:
        shutil.rmtree(d)


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        fn()
        print('OK  ', name)
    print('TẤT CẢ ĐỀU ĐẠT')
