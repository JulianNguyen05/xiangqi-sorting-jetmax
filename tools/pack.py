#!/usr/bin/env python3
# Đóng gói để chép qua lại giữa laptop và robot bằng NoMachine (robot không có mạng Internet).
#
# Trên LAPTOP:  python tools/pack.py             -> dist/xiangqi_to_robot.zip  (code + model + Pikafish, KHÔNG gồm số đo của robot)
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

# CHỈ đóng gói những gì robot cần (danh sách trắng), không duyệt cả repo (.venv, dataset, run Colab... rất nặng)
CODE = ['main.py', 'README.md', 'RUNBOOK.md', 'install.sh', 'board', 'brain', 'config', 'planner', 'robot', 'tests',
        'tools', 'vision', 'samples/board_empty.jpg', 'samples/start_position.jpg', 'models/README.txt']
# Engine cờ tướng build sẵn cho Jetson (tools/build_pikafish.sh jetson) -> trong gói nằm ở engine/
ENGINE_DIR = os.path.join('dist', 'pikafish_jetson')
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'runs', 'captures', 'dist', 'training', '.vscode', '.idea'}
SKIP_FILES = {'config/robot.local.yaml', 'config/calib.yaml'}     # số đo riêng của robot: không ghi đè
SKIP_EXT = ('.pt', '.engine', '.src', '.zip', '.rar', '.pyc')

ap = argparse.ArgumentParser()
ap.add_argument('--results', action='store_true', help='TRÊN ROBOT: gói runs/ captures/ config/ để mang về')
ap.add_argument('--no-model', action='store_true')
ap.add_argument('--with-docs', action='store_true', help='kèm thư mục docs/ (mặc định bỏ cho nhẹ)')
ap.add_argument('--engine-dir', default=ENGINE_DIR, help='thư mục chứa pikafish + pikafish.nnue cho Jetson')
ap.add_argument('--no-engine', action='store_true', help='không kèm Pikafish (khi robot đã có, gói nhẹ hơn ~45 MB)')
a = ap.parse_args()
os.makedirs(os.path.join(ROOT, 'dist'), exist_ok=True)


def add_tree(zf, rel_dirs, filt):
    n = 0
    for rel in rel_dirs:
        base = os.path.join(ROOT, rel)
        if not os.path.exists(base):
            print('  (bỏ qua, không có) ' + rel)
            continue
        print('  + ' + rel)
        if os.path.isfile(base):
            if filt(rel):
                zf.write(base, TOP + '/' + rel)
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
        return True
    out = os.path.join(ROOT, 'dist', 'xiangqi_to_robot.zip')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        n = add_tree(zf, CODE + ([] if a.no_model else ['models/best.onnx']), keep)
        version = ['đóng gói lúc ' + time.strftime('%Y-%m-%d %H:%M')]
        try:
            import subprocess
            version.append('mã nguồn: ' + subprocess.check_output(['git', '-C', ROOT, 'log', '-1', '--format=%h %ci'],
                                                                 stderr=subprocess.DEVNULL).decode().strip())
        except Exception:
            pass
        onnx = os.path.join(ROOT, 'models', 'best.onnx')
        if not a.no_model and os.path.exists(onnx):
            sys.path.insert(0, ROOT)
            from vision.pieces import file_md5
            version.append('models/best.onnx md5 ' + file_md5(onnx))
        edir = os.path.join(ROOT, a.engine_dir) if not os.path.isabs(a.engine_dir) else a.engine_dir
        if not a.no_engine:
            if os.path.exists(os.path.join(edir, 'pikafish')) and os.path.exists(os.path.join(edir, 'pikafish.nnue')):
                print('  + engine/ (từ %s)' % edir)
                for f in ('pikafish', 'pikafish.nnue', 'VERSION.txt'):
                    if os.path.exists(os.path.join(edir, f)):
                        zi = zipfile.ZipInfo.from_file(os.path.join(edir, f), TOP + '/engine/' + f)
                        zi.compress_type = zipfile.ZIP_DEFLATED
                        with open(os.path.join(edir, f), 'rb') as fh:
                            zf.writestr(zi, fh.read())
                        n += 1
                vf = os.path.join(edir, 'VERSION.txt')
                if os.path.exists(vf):
                    version.append('engine: ' + open(vf).read().strip())
            else:
                print('CẢNH BÁO: không thấy %s/pikafish + pikafish.nnue -> gói KHÔNG có engine '
                      '(chạy: sh tools/build_pikafish.sh jetson)' % edir)
        zf.writestr(TOP + '/VERSION.txt', '\n'.join(version) + '\n')
        n += 1
    if not a.no_model and not os.path.exists(os.path.join(ROOT, 'models', 'best.onnx')):
        print('CẢNH BÁO: không có models/best.onnx trong gói')
print('Đã tạo %s (%d file, %.1f MB)' % (out, n, os.path.getsize(out) / 1e6))
if not a.results:
    print('Kéo file này vào cửa sổ NoMachine, rồi trên robot:')
    print('  cd ~ && python3 -m zipfile -e <đường dẫn>/xiangqi_to_robot.zip ~/')
    print('  cd ~/xiangqi && sh install.sh')
