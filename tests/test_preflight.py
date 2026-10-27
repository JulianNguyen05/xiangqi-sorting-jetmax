#!/usr/bin/env python3
# Kiểm thử robot/preflight.py:  python tests/test_preflight.py
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

import yaml
import preflight
from fake_world import FakeWorld, CORNERS

BOARD_OK = {
    'corners': dict((k, {'slider': v[0], 'x': v[1], 'y': v[2]}) for k, v in CORNERS.items()),
    'z_pick': 10.0, 'z_place': 13.0, 'z_safe': 50.0,
    'tray': {'slider': 4400, 'x': 100.0, 'y': -105.0, 'col_step': [0.0, -14.0], 'row_step': [-14.0, 0.0],
             'cols': 8, 'rows': 4, 'z': None},
}


def write(d, name, obj):
    p = os.path.join(d, name)
    with open(p, 'w') as f:
        yaml.safe_dump(obj, f, allow_unicode=True)
    return p


def check(board=None, views=None, **kw):
    d = tempfile.mkdtemp()
    try:
        w = FakeWorld(seed=1)
        b = write(d, 'board.yaml', BOARD_OK if board is None else board)
        v = write(d, 'views.yaml', w.views_yaml() if views is None else views)
        return preflight.check_all(b, v, **kw)
    finally:
        shutil.rmtree(d)


def has(rep, text):
    return any(text in m for m in rep.errors)


def test_valid_config_has_no_errors():
    rep, objs = check()
    assert not rep.errors, rep.text()
    assert objs['board'] and objs['tray'] and objs['mapper']
    assert any('phủ kín 90 ô' in m for _, m in rep.items)


def test_template_with_nulls_reports_not_crashes():
    board = {'corners': dict((k, {'slider': None, 'x': None, 'y': None}) for k in CORNERS),
             'z_pick': None, 'z_place': None, 'z_safe': 50.0}
    views = {'image_size': [640, 480], 'views': [
        {'name': 'trai', 'slider': 0, 'arm': [0, -150, 120], 'points': {'a0': None, 'a9': None, 'd0': None, 'd9': None}},
        {'name': 'giua', 'slider': 1500, 'arm': [0, -150, 120], 'points': {'c0': None}}]}
    rep, objs = check(board=board, views=views)
    assert has(rep, 'board.yaml') and has(rep, 'khay') and has(rep, 'mới có 0/4 điểm')
    assert objs['board'] is None and objs['mapper'] is None


def test_unreachable_squares_listed():
    b = dict(BOARD_OK)
    b['corners'] = dict(b['corners'])
    b['corners']['i0'] = {'slider': 0, 'x': 400.0, 'y': -130.0}       # góc xa tầm với
    b['corners']['i9'] = {'slider': 3600, 'x': 400.0, 'y': -130.0}
    rep, _ = check(board=b)
    assert has(rep, '/90 ô ngoài tầm với') and has(rep, 'e0 e1'), rep.text()


def test_z_order_checked():
    b = dict(BOARD_OK)
    b['z_pick'] = 60.0                                                # cao hơn z_safe
    rep, _ = check(board=b)
    assert has(rep, 'z_pick'), rep.text()


def test_tray_too_small_and_overlapping():
    b = dict(BOARD_OK)
    b['tray'] = dict(BOARD_OK['tray'], cols=3, rows=2)
    rep, _ = check(board=b)
    assert has(rep, 'Khay chỉ chứa 6'), rep.text()
    b['tray'] = dict(BOARD_OK['tray'], slider=0, x=-96.0, y=-130.0)    # khay đè lên ô a0
    rep, _ = check(board=b)
    assert has(rep, 'trùng'), rep.text()


def test_view_with_three_points_and_uncovered_cells():
    w = FakeWorld(seed=1)
    y = w.views_yaml()
    y['views'][2]['points'] = dict(list(y['views'][2]['points'].items())[:3])   # view phải chỉ còn 3 điểm
    rep, objs = check(views=y)
    assert has(rep, 'view "phai": mới có 3/4'), rep.text()
    assert has(rep, 'không view hợp lệ nào nhìn thấy'), rep.text()
    assert objs['mapper'] is not None and len(objs['mapper'].views) == 2


def test_dropping_a_view_leaves_uncovered_cells():
    y = FakeWorld(seed=1).views_yaml()
    y['views'] = y['views'][:1]
    rep, _ = check(views=y)
    assert has(rep, 'không view hợp lệ nào nhìn thấy') and has(rep, 'i'), rep.text()


def test_bad_taught_points_detected():
    y = FakeWorld(seed=1).views_yaml()
    y['views'][0]['points']['a9'] = [600.0, 40.0]                      # bấm nhầm một giao điểm
    rep, _ = check(views=y)
    assert any('sai số dạy' in m for m in rep.errors + rep.warnings), rep.text()


def test_view_pose_out_of_limits_and_bad_square_name():
    y = FakeWorld(seed=1).views_yaml()
    y['views'][0]['arm'] = [0, -150, 999]
    y['views'][1]['slider'] = 9000
    y['views'][2]['points']['z9'] = [10, 10]
    rep, _ = check(views=y)
    assert has(rep, 'arm') and has(rep, 'slider 9000') and has(rep, 'tên ô sai'), rep.text()


def test_model_check(tmp=None):
    d = tempfile.mkdtemp()
    try:
        good = os.path.join(d, 'best.onnx')
        open(good, 'wb').write(b'x')
        rep, _ = check(model=good)
        assert not rep.errors, rep.text()
        rep, _ = check(model=os.path.join(d, 'nope.onnx'))
        assert has(rep, 'Không thấy mô hình')
        weird = os.path.join(d, 'm.txt')
        open(weird, 'wb').write(b'x')
        rep, _ = check(model=weird)
        assert has(rep, 'Đuôi mô hình lạ')
    finally:
        shutil.rmtree(d)


def test_engine_check_with_fake_engine():
    fake = os.path.join(HERE, 'fake_uci_engine.py')
    if not os.path.exists(fake):
        print('   (bỏ qua: không thấy tests/fake_uci_engine.py)')
        return
    rep = preflight.Report()
    from engine import UciEngine  # noqa: F401
    preflight.check_engine(sys.executable, None, rep, args=[fake])
    assert not rep.errors and any('engine trả lời được' in m for _, m in rep.items), rep.text()
    rep = preflight.Report()
    preflight.check_engine(os.path.join(HERE, 'khong_co.exe'), None, rep)
    assert has(rep, 'Không thấy engine')


def test_vietnamese_comments_with_legacy_locale():
    """Windows mặc định cp1252: file YAML có chú thích tiếng Việt (UTF-8) vẫn phải đọc được."""
    import subprocess
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, 'board.yaml'), 'w', encoding='utf-8') as f:
            f.write('# Hiệu chuẩn bàn cờ, đơn vị mm: ờ ở ợ\n' + yaml.safe_dump(BOARD_OK))
        with open(os.path.join(d, 'views.yaml'), 'w', encoding='utf-8') as f:
            f.write('# Các góc nhìn của camera: ờ ở ợ\n' + yaml.safe_dump(FakeWorld(seed=1).views_yaml()))
        env = dict(os.environ, PYTHONUTF8='0', PYTHONCOERCECLOCALE='0', LC_ALL='C', PYTHONIOENCODING='utf-8')
        r = subprocess.run([sys.executable, os.path.join(HERE, '..', 'robot', 'preflight.py'),
                            '--board', os.path.join(d, 'board.yaml'), '--views', os.path.join(d, 'views.yaml')],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert r.returncode == 0, (r.stdout + r.stderr).decode('utf-8', 'replace')
    finally:
        shutil.rmtree(d)


def test_cli_exit_codes():
    d = tempfile.mkdtemp()
    try:
        b = write(d, 'board.yaml', BOARD_OK)
        v = write(d, 'views.yaml', FakeWorld(seed=1).views_yaml())
        assert preflight.main(['--board', b, '--views', v]) == 0
        bad = write(d, 'bad.yaml', {'corners': {}})
        assert preflight.main(['--board', bad, '--views', v]) == 1
    finally:
        shutil.rmtree(d)


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        fn()
        print('OK  ', name)
    print('TẤT CẢ ĐỀU ĐẠT')
