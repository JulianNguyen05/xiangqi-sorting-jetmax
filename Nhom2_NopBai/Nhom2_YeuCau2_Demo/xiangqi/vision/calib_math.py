#!/usr/bin/env python3
# Phần tính toán của bước hiệu chỉnh camera (tách riêng để test được trên laptop).
import math

import numpy as np

from board.geometry import BoardGeometry  # noqa: F401  (gợi ý kiểu)
from vision import board_detect
from vision.world import apply_h, fit_h, fit_similarity


def solve_pose(img, pairs, geom, red_hint=None, z_press=0.5):
    """pairs: [((x_px, y_px) tâm quân trong ảnh, (x, y, z) robot khi đầu hút chạm mặt quân)].
    Quân phải đặt ĐÚNG giữa các giao điểm của bàn cờ, bàn cờ nằm trong ảnh.
    Trả về (pose_dict, report_dict)."""
    if len(pairs) < 4:
        raise ValueError('Cần ít nhất 4 quân đã chạm, mới có %d' % len(pairs))
    img_pts = np.array([p[0] for p in pairs], np.float64)
    rob = np.array([p[1] for p in pairs], np.float64)
    H_top, rms_top, _ = fit_h(img_pts, rob[:, :2], ransac=0)          # bình phương tối thiểu trên MỌI điểm
    res = [float(np.hypot(*(np.array(apply_h(H_top, *q)) - rob[i, :2]))) for i, q in enumerate(img_pts)]
    rep = {'n': len(pairs), 'rms_top_mm': rms_top, 'per_point_mm': [round(v, 1) for v in res]}
    zs = rob[:, 2]
    rep['z_touch_median'], rep['z_touch_spread'] = float(np.median(zs)), float(np.ptp(zs))
    z_pick = float(np.median(zs)) - z_press
    pose = {'H_top': H_top.tolist()}
    h, w = img.shape[:2]
    fix = board_detect.locate(img, geom, red_hint=red_hint or (w / 2.0, float(h)))
    if fix is None:
        rep['board'] = 'KHÔNG thấy bàn cờ -> dùng H_top cho cả mặt bàn (lệch thêm vài mm ở mép ảnh)'
        pose['H_table'] = H_top.tolist()
        return pose, rep, z_pick
    # mỗi quân -> giao điểm gần nhất (trong ảnh)
    sq_mm, used = [], []
    for (x, y), (rx, ry, rz) in pairs:
        k = min(fix.pts_px, key=lambda kk: math.hypot(fix.pts_px[kk][0] - x, fix.pts_px[kk][1] - y))
        sq_mm.append(geom.square_mm(*k))
        used.append(k)
    if len(set(used)) < len(used):
        raise ValueError('Hai quân hiệu chỉnh rơi vào cùng một giao điểm: đặt quân đúng giữa giao điểm, cách xa nhau')
    M, rms_b, scale = fit_similarity(sq_mm, rob[:, :2])
    rep['board_fit_rms_mm'] = rms_b
    rep['board_scale'] = scale                       # ~1.0 nếu bàn cờ đúng kích thước config
    H_table = M.dot(fix.Hinv)
    pose['H_table'] = H_table.tolist()
    # chênh lệch hai mặt phẳng ở tâm và góc ảnh (mm) - để báo cáo, kỳ vọng 0..5 mm
    diffs = []
    for x, y in ((w / 2, h / 2), (0, 0), (w, 0), (0, h), (w, h)):
        a = np.array(apply_h(H_top, x, y))
        b = np.array(apply_h(H_table, x, y))
        diffs.append(float(np.hypot(*(a - b))))
    rep['top_vs_table_mm'] = [round(d, 1) for d in diffs]
    if max(diffs) > 8.0:
        rep['CANH_BAO'] = 'mặt quân và mặt bàn lệch > 8 mm ở góc ảnh: kiểm tra lại quân có nằm đúng giao điểm không'
    return pose, rep, z_pick


def slider_vector(p0, p1, ds, tol=4.0):
    """p0, p1: toạ độ robot (x, y) của cùng các quân trước/sau khi trượt ds bước. Tìm tịnh tiến chung t (ghép
    không cần biết thứ tự), trả về (vec mm/bước, số quân khớp). Quan hệ: robot1 - robot0 = -vec * ds."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    best = (0, None)
    for a in p0:
        for b in p1:
            t = b - a
            moved = p0 + t
            d = np.hypot(*(moved[:, None, :] - p1[None, :, :]).transpose(2, 0, 1))
            n = int((d.min(axis=1) < tol).sum())
            if n > best[0]:
                best = (n, t)
    n, t = best
    if t is None or n < 3:
        raise ValueError('Không ghép được quân giữa 2 lần chụp (cần >= 3 quân thấy ở cả hai)')
    moved = p0 + t
    d = np.hypot(*(moved[:, None, :] - p1[None, :, :]).transpose(2, 0, 1))
    i_ok = np.nonzero(d.min(axis=1) < tol)[0]
    t = np.mean([p1[int(np.argmin(d[i]))] - p0[i] for i in i_ok], axis=0)
    return (-t / float(ds)).tolist(), int(len(i_ok))
