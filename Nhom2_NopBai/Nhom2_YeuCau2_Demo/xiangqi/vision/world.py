#!/usr/bin/env python3
# Quy đổi ảnh -> hệ THẾ GIỚI (mm), ghép nhiều góc chụp, định vị bàn cờ và quân trong cùng một hệ.
#
# Hiệu chỉnh (config/calib.yaml, do tools/calibrate.py sinh ra), theo từng TƯ THẾ TAY quan sát:
#   H_top   : ảnh -> (x, y) robot tại độ cao MẶT TRÊN quân (dùng cho tâm quân)
#   H_table : ảnh -> (x, y) robot tại MẶT BÀN (dùng cho đường lưới bàn cờ)
# Hai mặt phẳng cách nhau 10 mm; camera cao ~25-30 cm nên nếu dùng chung sẽ lệch tới 3-4 mm ở mép ảnh.
# Camera gắn trên tay nên ảnh -> robot không đổi khi chỉ trượt; world = robot + slider_vec * s.

import math

import cv2
import numpy as np

from board.geometry import NF, NR
from vision import board_detect

MOSAIC_S = 3.0          # px/mm của ảnh ghép


def apply_h(H, x, y):
    p = np.asarray(H, np.float64).dot([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def fit_h(src, dst, ransac=3.0):
    """Homography src -> dst (>= 4 cặp). Trả về (H, sai số RMS theo đơn vị dst, mặt nạ inlier)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if len(src) < 4:
        raise ValueError('Cần ít nhất 4 cặp điểm, mới có %d' % len(src))
    use_ransac = ransac and ransac > 0 and len(src) > 4
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC if use_ransac else 0, ransac if use_ransac else 0)
    if H is None:
        raise ValueError('Không tính được homography (các điểm thẳng hàng?)')
    pr = np.array([apply_h(H, *p) for p in src])
    err = np.hypot(*(pr - dst).T)
    return H, float(np.sqrt(np.mean(err ** 2))), (inl.ravel().astype(bool) if inl is not None else np.ones(len(src), bool))


def fit_similarity(src, dst):
    """Phép quay + tịnh tiến + tỉ lệ 2D (Umeyama) src -> dst. Trả về ma trận 3x3 và RMS."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, Sg, Vt = np.linalg.svd(b.T.dot(a) / len(src))
    D = np.eye(2)
    if np.linalg.det(U.dot(Vt)) < 0:
        D[1, 1] = -1
    R = U.dot(D).dot(Vt)
    scale = np.trace(np.diag(Sg).dot(D)) / (a ** 2).sum(1).mean()
    t = md - scale * R.dot(ms)
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = t
    pr = (M[:2, :2].dot(src.T)).T + M[:2, 2]
    return M, float(np.sqrt(np.mean(np.sum((pr - dst) ** 2, 1)))), float(scale)


class Calib(object):
    def __init__(self, data):
        self.data = data or {}
        self.poses = self.data.get('poses', [])
        self.z_pick = self.data.get('z_pick')
        self.slider_vec = tuple(self.data.get('slider_vec', (0.0, 0.0)))

    def pose_for(self, arm, tol=3.0):
        for p in self.poses:
            if max(abs(a - b) for a, b in zip(p['arm'], arm)) <= tol:
                return p
        return None

    def ready_for(self, views):
        missing = [v['name'] for v in views if self.pose_for(v['arm']) is None]
        return missing


class ViewMap(object):
    """Quy đổi cho MỘT ảnh chụp ở view (tư thế tay + vị trí trượt s)."""

    def __init__(self, pose, s, vec):
        self.Ht = np.array(pose['H_top'], np.float64)
        self.Hb = np.array(pose['H_table'], np.float64)
        self.off = np.array([vec[0] * s, vec[1] * s])

    def top(self, x, y):
        return np.array(apply_h(self.Ht, x, y)) + self.off

    def table(self, x, y):
        return np.array(apply_h(self.Hb, x, y)) + self.off

    def table_matrix(self):
        """3x3 ảnh -> thế giới tại mặt bàn."""
        T = np.eye(3)
        T[:2, 2] = self.off
        return T.dot(self.Hb)


# ------------------------------------------------------------------------------------- ghép ảnh
def build_mosaic(shots, s=MOSAIC_S, pad=20.0, mirror=False):
    """shots: [(ảnh BGR, ViewMap)]. Ghép thành ảnh nhìn thẳng của MẶT BÀN trong hệ thế giới.
    Trả về (ảnh ghép, T: thế giới mm -> px ảnh ghép).
    Hệ robot thuận tay phải, z hướng lên: nhìn từ trên xuống thì x sang phải, y hướng LÊN. Ảnh thì hàng tăng
    hướng XUỐNG, nên lật trục y (hàng = -y) để ảnh ghép giống ảnh chụp từ trên xuống, không bị soi gương
    (bước định vị bàn cờ dựa vào chiều này để biết cột a nằm bên trái phía Đỏ). mirror=True nếu hệ robot ngược chiều."""
    corners = []
    for img, vm in shots:
        h, w = img.shape[:2]
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            corners.append(vm.table(x, y))
    corners = np.array(corners)
    x0, y0 = corners.min(0) - pad
    x1, y1 = corners.max(0) + pad
    W, H = int((x1 - x0) * s), int((y1 - y0) * s)
    if W * H > 5000 * 5000:
        raise RuntimeError('Ảnh ghép quá lớn (%dx%d): hiệu chỉnh sai?' % (W, H))
    if mirror:
        T = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1]], np.float64)
    else:
        T = np.array([[s, 0, -x0 * s], [0, -s, y1 * s], [0, 0, 1]], np.float64)
    out = np.full((H, W, 3), 255, np.uint8)
    best = np.zeros((H, W), np.float32)
    for img, vm in shots:
        h, w = img.shape[:2]
        M = T.dot(vm.table_matrix())
        warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        # trọng số = khoảng cách tới mép ảnh gốc (ưu tiên vùng giữa ảnh, ít méo)
        ones = np.zeros((h, w), np.uint8)
        ones[1:-1, 1:-1] = 1
        wgt = cv2.distanceTransform(ones, cv2.DIST_L2, 3)
        ww = cv2.warpPerspective(wgt, M, (W, H), flags=cv2.INTER_LINEAR)
        sel = ww > best
        out[sel] = warped[sel]
        best[sel] = ww[sel]
    return out, T


# ------------------------------------------------------------------------------------- kết quả
class BoardWorld(object):
    """Bàn cờ trong hệ thế giới: toạ độ (x, y) mm của 90 giao điểm + phép đổi thế giới <-> hệ bàn cờ."""

    def __init__(self, fix, T, geom):
        self.fix, self.T, self.geom = fix, T, geom
        Ti = np.linalg.inv(T)
        self.squares = {}
        for k, (px, py) in fix.pts_px.items():
            self.squares[k] = np.array(apply_h(Ti, px, py))
        # thế giới -> mm bàn cờ (để biết quân nằm trong hay ngoài bàn)
        self.W2B = np.linalg.inv(fix.H).dot(T)

    def to_board(self, wx, wy):
        return apply_h(self.W2B, wx, wy)

    def nearest_square(self, wx, wy):
        best, bd = None, 1e9
        for k, p in self.squares.items():
            d = math.hypot(p[0] - wx, p[1] - wy)
            if d < bd:
                best, bd = k, d
        return best, bd

    def inside(self, wx, wy, margin):
        u, v = self.to_board(wx, wy)
        g = self.geom
        return -margin <= u <= g.width + margin and -margin <= v <= g.length + margin


def merge_views(pieces_world, dist=10.0):
    """Gộp cùng một quân thấy ở nhiều view (tâm cách nhau < dist mm): trung bình toạ độ và xác suất."""
    groups = []
    for p in pieces_world:
        for g in groups:
            c = np.mean([q.world for q in g], axis=0)
            if math.hypot(p.world[0] - c[0], p.world[1] - c[1]) < dist:
                g.append(p)
                break
        else:
            groups.append([p])
    out = []
    for g in groups:
        p = max(g, key=lambda q: q.conf)
        p.world = tuple(np.mean([q.world for q in g], axis=0))
        p.probs = np.mean([q.probs for q in g], axis=0)
        out.append(p)
    return out


def classify_where(pieces, board, margin):
    for p in pieces:
        if board.inside(p.world[0], p.world[1], margin):
            k, d = board.nearest_square(*p.world)
            p.where = ('board', k, d)
        else:
            p.where = ('staging', None, None)
    return pieces


def locate_board(shots, geom, red_near_world, debug=None, mirror=False):
    mosaic, T = build_mosaic(shots, mirror=mirror)
    hint = apply_h(T, *red_near_world)
    fix = board_detect.locate(mosaic, geom, red_hint=hint)
    if debug is not None:
        debug['mosaic'], debug['T'] = mosaic, T
    if fix is None:
        return None
    return BoardWorld(fix, T, geom)


def piece_radius_px(vm, img_shape, diameter=25.0):
    """Bán kính quân (px) dự kiến ở giữa ảnh, suy từ H_top."""
    h, w = img_shape[:2]
    a = vm.top(w / 2.0, h / 2.0)
    b = vm.top(w / 2.0 + 10, h / 2.0)
    mm_per_px = np.linalg.norm(b - a) / 10.0
    return diameter / 2.0 / max(mm_per_px, 1e-6)
