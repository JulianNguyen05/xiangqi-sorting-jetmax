#!/usr/bin/env python3
# Build TensorRT engine TRÊN JETSON (không cần mạng), ~10-20 phút trên Jetson Nano.
#   python3 tools/build_engine.py                  # models/best.onnx -> models/best.engine (FP16)
#   python3 tools/build_engine.py --fp32           # nếu FP16 cho kết quả lạ
# Ghi md5 của file onnx vào models/best.engine.src để lần sau biết engine có khớp model không.
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from robot import config                                   # noqa: E402
from vision.pieces import file_md5                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--onnx', default=None)
ap.add_argument('--fp32', action='store_true')
a = ap.parse_args()
cfg = config.robot_cfg()
onnx = config.path(a.onnx or cfg['detector']['model'])
if not onnx.endswith('.onnx'):
    sys.exit('Cần file .onnx, đang là ' + onnx)
eng = os.path.splitext(onnx)[0] + '.engine'
trtexec = next((p for p in ('/usr/src/tensorrt/bin/trtexec', '/usr/bin/trtexec') if os.path.exists(p)), None)
if not trtexec:
    sys.exit('Không thấy trtexec (TensorRT chưa cài trong JetPack?)')
cmd = [trtexec, '--onnx=' + onnx, '--saveEngine=' + eng, '--workspace=1024']
if not a.fp32:
    cmd.append('--fp16')
print('Chạy: ' + ' '.join(cmd))
print('Jetson Nano mất khoảng 10-20 phút, đừng tắt cửa sổ...')
t0 = time.time()
r = subprocess.call(cmd)
if r != 0 or not os.path.exists(eng):
    sys.exit('trtexec thất bại (mã %d). Thử lại với --fp32, hoặc xem log phía trên.' % r)
with open(eng + '.src', 'w') as f:
    f.write(file_md5(onnx))
print('Xong sau %.0f giây: %s' % (time.time() - t0, eng))
print('Kiểm tra: python3 tools/check_env.py')
