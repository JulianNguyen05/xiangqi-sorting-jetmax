#!/usr/bin/env python3
# HIỆU CHỈNH CAMERA (làm 1 lần cho mỗi tư thế tay quan sát, ~10 phút). Xem RUNBOOK.md mục 4.
#
# Chuẩn bị: bàn cờ nằm trong ảnh ở tư thế quan sát; đặt 6-9 quân ĐÚNG GIỮA các giao điểm, rải đều khắp ảnh
# (4 góc + giữa). Quân để ngửa, không chạm nhau.
#
#   python3 tools/calibrate.py                 # hiệu chỉnh tư thế của view đầu tiên trong config
#   python3 tools/calibrate.py --view v1
#   python3 tools/calibrate.py --slider        # đo vector thanh trượt (mm/bước), sau khi đã hiệu chỉnh camera
#   python3 tools/calibrate.py --verify        # tay bay tới trên từng quân nhìn thấy để kiểm tra bằng mắt
#
# Quy trình: chụp ảnh -> tìm quân -> lần lượt từng quân: lái đầu hút CHẠM NHẸ đúng tâm mặt quân -> Enter.
# Từ quân thứ 4 trở đi, tay tự bay tới gần quân tiếp theo, bạn chỉ chỉnh vài mm.
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from board.geometry import BoardGeometry                          # noqa: E402
from robot import config                                          # noqa: E402
from vision import board_detect, calib_math, pieces as pc, world  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--view', default=None, help='tên view (mặc định view đầu tiên)')
ap.add_argument('--slider', action='store_true', help='đo vector thanh trượt')
ap.add_argument('--slide-steps', type=int, default=1500)
ap.add_argument('--verify', action='store_true', help='bay tới trên các quân thấy được để kiểm tra')
ap.add_argument('--r-px', type=float, default=None, help='bán kính quân trong ảnh (px) nếu không tự đoán được')
ap.add_argument('--no-home', action='store_true')
a = ap.parse_args()

import rospy                                                      # noqa: E402
from robot.camera import RosCamera                                # noqa: E402
from robot.jog import Jogger                                      # noqa: E402
from robot.motion import Robot                                    # noqa: E402

rospy.init_node('xq_calibrate', anonymous=True)
cfg = config.robot_cfg()
geom = BoardGeometry.load()
calib_path = cfg.get('calib_file', 'config/calib.yaml')
cal = config.calib(cfg) or {}
cam = RosCamera(cfg)
rb = Robot(cfg, cal, backend='ros')
views = cfg['views']
view = next((v for v in views if v['name'] == a.view), None) if a.view else views[0]
if view is None:
    sys.exit('Không có view tên %s trong config/robot.yaml' % a.view)
out = config.path('captures')
os.makedirs(out, exist_ok=True)


def save_calib():
    p = config.save(calib_path, cal, header='Sinh tự động bởi tools/calibrate.py. Không sửa tay trừ khi biết rõ.')
    print('Đã lưu ' + p)


def guess_r(img):
    if a.r_px:
        return a.r_px
    h, w = img.shape[:2]
    fix = board_detect.locate(img, geom, red_hint=(w / 2.0, float(h)))
    if fix is not None:
        return 1.04 * geom.piece_d / 2.0 * board_detect._px_per_mm(fix)     # mặt quân cao hơn bàn 10 mm
    print('Không thấy bàn cờ để đoán cỡ quân, dùng mặc định (truyền --r-px nếu tìm quân sai)')
    return min(h, w) / 22.0


def snapshot(tag):
    rb.observe(view)
    img = cam.grab(view['name'])
    r = guess_r(img)
    discs = pc.find_discs(img, r)
    vis = img.copy()
    for i, (x, y, rr, c, cf) in enumerate(discs):
        cv2.circle(vis, (int(x), int(y)), int(rr), (0, 255, 0), 2)
        cv2.putText(vis, str(i + 1), (int(x) - 8, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out, 'calib_%s.jpg' % tag), img)
    cv2.imwrite(os.path.join(out, 'calib_%s_discs.jpg' % tag), vis)
    print('Ảnh: captures/calib_%s_discs.jpg  (%d quân, bán kính ~%.0f px)' % (tag, len(discs), r))
    return img, discs


def order_snake(discs):
    """Sắp thứ tự đi: theo hàng (y) rồi zig-zag theo x để tay đi ngắn."""
    if not discs:
        return []
    rows = sorted(discs, key=lambda d: d[1])
    band = 2.5 * np.median([d[2] for d in discs])
    out_, cur, y0 = [], [], rows[0][1]
    for d in rows:
        if d[1] - y0 > band:
            out_.append(cur)
            cur, y0 = [], d[1]
        cur.append(d)
    out_.append(cur)
    res = []
    for i, row in enumerate(out_):
        res += sorted(row, key=lambda d: d[0], reverse=bool(i % 2))
    return res


try:
    if rb.use_slider and not a.no_home:
        input('Đưa thanh trượt về HOME (tay máy đang cao, không vướng). Enter...')
        rb.home_slider()

    # ------------------------------------------------------------------------------ đo thanh trượt
    if a.slider:
        pose = world.Calib(cal).pose_for(view['arm'])
        if pose is None:
            sys.exit('Tư thế này chưa hiệu chỉnh camera. Chạy tools/calibrate.py trước (không có --slider).')
        Ht = np.array(pose['H_top'])
        img0, d0 = snapshot('slide0')
        s0 = rb.s
        target = min(rb.s_max, s0 + a.slide_steps)
        print('Trượt %d -> %d bước rồi chụp lại (quân phải vẫn nằm trong ảnh)...' % (s0, target))
        v2 = dict(view)
        v2['slider'] = target
        rb.observe(v2)
        img1 = cam.grab(view['name'])
        d1 = pc.find_discs(img1, guess_r(img1))
        p0 = [world.apply_h(Ht, d[0], d[1]) for d in d0]
        p1 = [world.apply_h(Ht, d[0], d[1]) for d in d1]
        vec, n = calib_math.slider_vector(p0, p1, target - s0)
        print('Vector thanh trượt: (%.4f, %.4f) mm/bước, độ dài %.4f mm/bước (%d quân khớp)' % (
            vec[0], vec[1], math.hypot(*vec), n))
        cal['slider_vec'] = [round(vec[0], 6), round(vec[1], 6)]
        save_calib()
        sys.exit(0)

    # ------------------------------------------------------------------------------ kiểm tra
    if a.verify:
        cw = world.Calib(cal)
        pose = cw.pose_for(view['arm'])
        if pose is None:
            sys.exit('Tư thế này chưa hiệu chỉnh.')
        img, discs = snapshot('verify')
        vm = world.ViewMap(pose, rb.s, cw.slider_vec or rb.vec)
        for i, (x, y, r, c, cf) in enumerate(order_snake(discs)):
            wx, wy = vm.top(x, y)
            input('Quân %d/%d (%s) tại thế giới (%.1f, %.1f): Enter để bay tới trên, Ctrl+C để dừng...' % (
                i + 1, len(discs), c, wx, wy))
            rb.hover_check(wx, wy, dz=8.0)
        sys.exit(0)

    # ------------------------------------------------------------------------------ hiệu chỉnh camera
    img, discs = snapshot('pose')
    if len(discs) < 4:
        sys.exit('Chỉ thấy %d quân. Đặt 6-9 quân rải đều trong ảnh rồi chạy lại.' % len(discs))
    discs = order_snake(discs)
    print('Thứ tự chạm: xem số trên ảnh captures/calib_pose_discs.jpg (đã sắp lại theo đường đi).')
    vis = img.copy()
    for i, (x, y, r, c, cf) in enumerate(discs):
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(vis, str(i + 1), (int(x) - 8, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out, 'calib_pose_discs.jpg'), vis)
    jog = Jogger(rb)
    pairs = []
    for i, (x, y, r, c, cf) in enumerate(discs):
        print('\n=== Quân %d/%d (ảnh %.0f, %.0f) ===' % (i + 1, len(discs), x, y))
        rb.lift()
        if len(pairs) >= 3:                                   # đoán vị trí từ các điểm đã có, bay tới trên
            src = np.array([p[0] for p in pairs])
            dst = np.array([p[1][:2] for p in pairs])
            M, _ = cv2.estimateAffine2D(src, dst) if len(pairs) >= 3 else (None, None)
            if M is not None:
                gx, gy = M.dot([x, y, 1.0])
                zt = float(np.median([p[1][2] for p in pairs])) + 12.0
                try:
                    rb.goto((gx, gy, rb.z_safe))
                    rb.goto((gx, gy, zt), rb.slow)
                    print('  Đã bay tới gần (đoán): hạ dần bằng f, căn tâm bằng w/a/s/d (bước 1 mm).')
                except Exception as e:
                    print('  Không tự bay tới được (%s), lái tay.' % e)
        r_ = jog.loop('Chạm NHẸ đầu hút vào TÂM mặt quân -> Enter. q = bỏ quân này')
        if r_ == 'enter':
            pairs.append(((x, y), rb.status()))
            print('  Ghi: ảnh (%.0f, %.0f) <-> robot (%.1f, %.1f, %.1f)' % ((x, y) + pairs[-1][1]))
        rb.lift()
    pose, rep, z_pick = calib_math.solve_pose(img, pairs, geom)
    print('\nKẾT QUẢ:')
    for k, v in rep.items():
        print('  %s: %s' % (k, v))
    if rep['rms_top_mm'] > 2.0:
        print('  CẢNH BÁO: sai số > 2 mm. Có thể chạm lệch tâm hoặc quân bị xê dịch khi chạm: nên làm lại.')
    pose['arm'] = [float(v) for v in view['arm']]
    poses = [p for p in cal.get('poses', []) if max(abs(u - w) for u, w in zip(p['arm'], pose['arm'])) > 3.0]
    cal['poses'] = poses + [pose]
    cal['z_pick'] = round(z_pick, 2)
    cal.setdefault('slider_vec', [0.0, 0.0])
    save_calib()
    print('z_pick = %.2f mm (đã trừ %.1f mm để ép nhẹ đầu hút)' % (z_pick, 0.5))
    if rb.use_slider and cal['slider_vec'] == [0.0, 0.0]:
        print('Tiếp theo: python3 tools/calibrate.py --slider')
finally:
    rb.b.sucker(False)
    rb.park()
