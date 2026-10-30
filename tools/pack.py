#!/usr/bin/env python3
# Đóng gói để chép qua lại giữa laptop và robot bằng NoMachine (robot không có mạng Internet).
#
# Trên LAPTOP:  python tools/pack.py             -> dist/xiangqi_to_robot.zip  (code + model, KHÔNG gồm số đo của robot)
#               python tools/pack.py --no-model  -> chỉ code (nhẹ, khi model trên robot không đổi)
# Trên ROBOT:   python3 tools/pack.py --results  -> dist/results_<giờ>.zip    (runs/, captures/, config/ để mang về)
#
# Giải nén trên robot (không cần lệnh unzip):
#   cd ~ && python3 -m zipfile -e ~/Desktop/xiangqi_to_robot.zip ~/
# Ghi đè code cũ nhưng GIỮ config/robot.local.yaml, config/calib.yaml, models/best.engine trên robot.
import argparse
import os
import sys
import time
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TOP = 'xiangqi'                                        # tên thư mục khi giải nén trên robot

SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'runs', 'captures', 'dist', 'training', '.vscode', '.idea'}
SKIP_FILES = {'config/robot.local.yaml', 'config/calib.yaml'}     # số đo riêng của robot: không ghi đè
SKIP_EXT = ('.pt', '.engine', '.src', '.zip', '.rar', '.pyc')

ap = argparse.ArgumentParser()
ap.add_argument('--results', action='store_true', help='TRÊN ROBOT: gói runs/ captures/ config/ để mang về')
ap.add_argument('--no-model', action='store_true')
ap.add_argument('--with-docs', action='store_true', help='kèm thư mục docs/ (mặc định bỏ cho nhẹ)')
a = ap.parse_args()
os.makedirs(os.path.join(ROOT, 'dist'), exist_ok=True)


def add_tree(zf, rel_dirs, filt):
    n = 0
    for rel in rel_dirs:
        base = os.path.join(ROOT, rel)
        if os.path.isfile(base):
            zf.write(base, os.path.join(TOP, rel))
            n += 1
            continue
        for d, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                full = os.path.join(d, f)
                r = os.path.relpath(full, ROOT).replace(os.sep, '/')
                if filt(r):
                    zf.write(full, TOP + '/' + r)
                    n += 1
    return n


if a.results:
    out = os.path.join(ROOT, 'dist', 'results_%s.zip' % time.strftime('%Y%m%d_%H%M%S'))
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        n = add_tree(zf, [x for x in ('runs', 'captures', 'config') if os.path.exists(os.path.join(ROOT, x))],
                     lambda r: True)
else:
    def keep(r):
        if r in SKIP_FILES or r.endswith(SKIP_EXT):
            return False
        if r.startswith('docs/') and not a.with_docs:
            return False
        if r.startswith('models/') and a.no_model and r != 'models/README.txt':
            return False
        return True
    out = os.path.join(ROOT, 'dist', 'xiangqi_to_robot.zip')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        n = add_tree(zf, ['.'], keep)
    if not a.no_model and not os.path.exists(os.path.join(ROOT, 'models', 'best.onnx')):
        print('CẢNH BÁO: không có models/best.onnx trong gói')
print('Đã tạo %s (%d file, %.1f MB)' % (out, n, os.path.getsize(out) / 1e6))
if not a.results:
    print('Kéo file này vào cửa sổ NoMachine, rồi trên robot:')
    print('  cd ~ && python3 -m zipfile -e <đường dẫn>/xiangqi_to_robot.zip ~/')
