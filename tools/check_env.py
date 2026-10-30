#!/usr/bin/env python3
# KIỂM TRA MÔI TRƯỜNG ROBOT (chạy đầu tiên khi có robot, ~1 phút). Không di chuyển robot.
#   python3 tools/check_env.py
#   python3 tools/check_env.py --image captures/v0.jpg     # chạy thử model trên một ảnh
import argparse
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

ap = argparse.ArgumentParser()
ap.add_argument('--image', default=None)
a = ap.parse_args()
OK, BAD, WARN = '  [OK]  ', '  [LỖI] ', '  [?]   '


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10).decode().strip()
    except Exception as e:
        return 'ERR: %s' % e


print('== Hệ thống')
print(OK + 'Python %s (%s)' % (platform.python_version(), sys.executable))
print(OK + 'JetPack/L4T: ' + sh('head -1 /etc/nv_tegra_release 2>/dev/null || echo "không phải Jetson"'))
for mod in ('numpy', 'cv2', 'yaml'):
    try:
        m = __import__(mod)
        print(OK + '%s %s' % (mod, getattr(m, '__version__', '')))
    except Exception as e:
        print(BAD + '%s: %s  -> pip3 install %s' % (mod, e, {'cv2': 'opencv-python', 'yaml': 'pyyaml'}.get(mod, mod)))
try:
    import cv2
    print((OK if hasattr(cv2, 'dnn') else BAD) + 'cv2.dnn có sẵn')
except Exception:
    pass

print('== Chạy model (cần ít nhất một)')
backs = []
try:
    import onnxruntime as ort
    print(OK + 'onnxruntime %s, providers %s' % (ort.__version__, ort.get_available_providers()))
    backs.append('onnx')
except Exception as e:
    print(WARN + 'onnxruntime: %s' % e)
try:
    import tensorrt as trt
    print(OK + 'tensorrt %s' % trt.__version__)
    try:
        from vision.pieces import _CudaRT
        _CudaRT()
        print(OK + 'libcudart (chạy TensorRT không cần pycuda)')
        backs.append('trt')
    except Exception as e:
        print(BAD + 'libcudart: %s' % e)
except Exception as e:
    print(WARN + 'tensorrt: %s' % e)
print(OK + 'trtexec: ' + sh('ls /usr/src/tensorrt/bin/trtexec 2>/dev/null || echo "không thấy"'))
print('  Robot không có mạng: KHÔNG pip install được. Cách chắc chạy nhất: python3 tools/build_engine.py (RUNBOOK 2c)')

print('== ROS')
try:
    import rospy  # noqa: F401
    print(OK + 'rospy')
    nodes = sh('rosnode list')
    print((OK if 'jetmax' in nodes else BAD) + 'node jetmax_control: %s' % ('chạy' if 'jetmax' in nodes else 'KHÔNG thấy'))
    topics = sh('rostopic list')
    for t in ('/jetmax/status', '/jetmax/command', '/jetmax/end_effector/sucker/command', '/usb_cam/image_rect_color'):
        print((OK if t in topics else BAD) + t)
    print(OK + 'tần số ảnh: ' + sh('timeout 4 rostopic hz /usb_cam/image_rect_color 2>&1 | grep average | tail -1'))
    try:
        from jetmax_control.msg import SetJetMax, JetMax  # noqa: F401
        print(OK + 'jetmax_control.msg')
    except Exception as e:
        print(BAD + 'jetmax_control.msg: %s (đã source ~/ros/devel/setup.bash chưa?)' % e)
    try:
        import hiwonder
        print(OK + 'hiwonder (thanh trượt: hiwonder.Stepper)' + ('' if hasattr(hiwonder, 'Stepper') else ' - KHÔNG có Stepper'))
    except Exception as e:
        print(WARN + 'hiwonder: %s' % e)
except Exception as e:
    print(BAD + 'rospy: %s' % e)

print('== Repo')
from robot import config  # noqa: E402
cfg = config.robot_cfg()
mp = config.path(cfg['detector']['model'])
print((OK if os.path.exists(mp) else BAD) + 'model: %s' % mp)
print((OK if os.path.exists(config.path(config.LOCAL)) else WARN) + 'config/robot.local.yaml: %s' % (
    'có' if os.path.exists(config.path(config.LOCAL)) else 'chưa có -> cp config/robot.local.example.yaml config/robot.local.yaml'))
eng = os.path.splitext(mp)[0] + '.engine'
if os.path.exists(eng) and os.path.exists(mp):
    from vision.pieces import engine_matches
    print((OK if engine_matches(mp, eng) else BAD) + 'engine %s %s' % (
        eng, 'khớp model' if engine_matches(mp, eng) else 'KHÔNG khớp model -> python3 tools/build_engine.py'))
cal = config.calib(cfg)
print((OK if cal else WARN) + 'calib: %s' % ('có, %d tư thế' % len(cal.get('poses', [])) if cal else 'chưa có (RUNBOOK mục 4)'))

if os.path.exists(mp):
    print('== Thử model')
    import cv2
    import numpy as np
    from vision import pieces as pc
    try:
        t0 = time.time()
        det = pc.PieceDetector(mp, cfg['detector']['imgsz'], cfg['detector']['conf'], cfg['detector']['iou'],
                               cfg['detector']['tta_rotations'])
        print(OK + 'nạp model %.1f s, backend %s' % (time.time() - t0, det.net.name))
        img = cv2.imread(a.image) if a.image else np.full((480, 640, 3), 200, np.uint8)
        det.raw(img)
        t0 = time.time()
        ps = det.detect(img)
        dt = time.time() - t0
        print(OK + '%d góc xoay: %.2f s / ảnh, tìm thấy %d quân' % (len(det.rotations), dt, len(ps)))
        if a.image:
            pc.apply_color(img, ps)
            outp = os.path.splitext(a.image)[0] + '_det.jpg'
            cv2.imwrite(outp, pc.draw(img, ps))
            print(OK + 'ảnh kết quả: ' + outp)
    except Exception as e:
        print(BAD + 'model: %s' % e)
