#!/usr/bin/env python3
# Định vị bàn cờ (tấm nhựa in đỏ) trong ảnh, KHÔNG cần marker.
#
# B1. Lọc màu đỏ (HSV) -> vùng đỏ lớn nhất -> 4 góc mép ngoài khung viền   => homography thô.
# B2. Nắn ảnh về dạng nhìn thẳng (mm), dò lại TỪNG đường lưới (9 dọc, 10 ngang) theo từng đoạn ngắn,
#     bỏ qua đoạn gần giao điểm (tránh dấu hoa thị vị trí pháo/tốt), khớp đường cong bậc 2 (chịu được nếp gấp nhẹ).
# B3. Giao các đường đã khớp -> 90 giao điểm đo được -> chiếu về ảnh. Lặp lại 2 lần cho hội tụ.
#
# Kết quả: BoardFix gồm toạ độ pixel ĐO ĐƯỢC của 90 giao điểm (dùng trực tiếp, tốt hơn khi tấm bị nhăn)
# và homography mm -> ảnh khớp trên 90 điểm đó (dùng để chiếu điểm bất kỳ, ví dụ tâm quân cờ).
#
# Bàn đối xứng xoay 180° nên không tự biết đầu nào là Đỏ: truyền red_hint = một điểm ảnh nằm GẦN phía Đỏ
# (trên robot: điểm ảnh gần đế robot). Mặc định: cạnh dưới ảnh.

import numpy as np
import cv2

from board.geometry import NF, NR

S = 3.0                 # px/mm của ảnh nắn
SLICE = 5.0             # mm, bề dày mỗi đoạn dò
SKIP = 7.0              # mm, bỏ đoạn dò gần đường vuông góc (giao điểm, hoa thị)
WIN_IN, WIN_OUT = 6.0, 7.0   # mm, cửa sổ tìm đường: phía trong / phía ngoài (với đường ngoài cùng)


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (165, 70, 60), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def _quad(mask):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    res = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = res[0] if len(res) == 2 else res[1]          # OpenCV 3 trả 3 giá trị
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.03 * mask.shape[0] * mask.shape[1]:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float64)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)


def _order(quad, red_hint):
    """Sắp 4 góc thành (a0-ngoài, i0-ngoài, i9-ngoài, a9-ngoài): cạnh 0-1 là cạnh NGẮN phía Đỏ."""
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]                             # theo chiều kim đồng hồ trên ảnh (y hướng xuống)
    side = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    if side[0] + side[2] > side[1] + side[3]:             # muốn cạnh 0-1 và 2-3 là cạnh ngắn
        q = np.roll(q, -1, axis=0)
    # chọn cạnh ngắn nào là phía Đỏ: cạnh có trung điểm gần red_hint hơn
    m01 = (q[0] + q[1]) / 2
    m23 = (q[2] + q[3]) / 2
    if np.linalg.norm(m23 - red_hint) < np.linalg.norm(m01 - red_hint):
        q = np.roll(q, -2, axis=0)
    # đảm bảo a ở bên TRÁI khi đứng phía Đỏ nhìn sang Đen: u x v < 0 trong hệ ảnh
    u = q[1] - q[0]
    v = q[3] - q[0]
    if u[0] * v[1] - u[1] * v[0] > 0:
        q = q[[1, 0, 3, 2]]
    return q


def _peak(profile, min_frac, pick=0):
    """Vị trí đỉnh (px, có nội suy). pick=0: đỉnh mạnh nhất; -1: đỉnh đầu mảng; +1: đỉnh cuối mảng.
    Đường lưới ngoài cùng nằm sát vạch dày của khung viền (cách ~4 mm) nên lấy đỉnh nằm TRONG cùng."""
    if profile.max() < min_frac:
        return None
    if pick == 0:
        i = int(np.argmax(profile))
    else:
        idx = [j for j in range(len(profile)) if profile[j] >= min_frac
               and profile[j] >= profile[max(0, j - 1)] and profile[j] >= profile[min(len(profile) - 1, j + 1)]]
        if not idx:
            return None
        i = idx[0] if pick < 0 else idx[-1]
    r = max(1, int(round(1.0 * S)))
    lo, hi = max(0, i - r), min(len(profile), i + r + 1)
    w = profile[lo:hi].astype(np.float64)
    return (np.arange(lo, hi) * w).sum() / w.sum()


def _fit(points, deg=2, tol=1.2):
    """Khớp x = poly(t) bậc 2, loại điểm lệch > tol mm. points: [(t, x)]."""
    p = np.array(points, np.float64)
    if len(p) < 4:
        return None
    keep = np.ones(len(p), bool)
    coef = None
    for _ in range(3):
        d = deg if keep.sum() >= 8 else 1
        coef = np.polyfit(p[keep, 0], p[keep, 1], d)
        err = np.abs(np.polyval(coef, p[:, 0]) - p[:, 1])
        new = err < tol
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
    return coef


class BoardFix(object):
    def __init__(self, H, pts_px, rms_mm, quad, n_lines):
        self.H = H                         # mm (hệ bàn cờ) -> pixel
        self.Hinv = np.linalg.inv(H)
        self.pts_px = pts_px               # {(f, r): (x, y)} đo được
        self.rms_mm = rms_mm               # độ lệch giữa điểm đo và homography (đo độ nhăn/méo)
        self.quad = quad
        self.n_lines = n_lines             # số đường lưới dò được / 19

    def to_px(self, u, v):
        p = self.H.dot([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def to_mm(self, x, y):
        p = self.Hinv.dot([x, y, 1.0])
        return p[0] / p[2], p[1] / p[2]


def _refine(mask, geom, H):
    m = geom.border_offset + 12.0
    W = int((geom.width + 2 * m) * S)
    L = int((geom.length + 2 * m) * S)
    T = np.array([[1 / S, 0, -m], [0, 1 / S, -m], [0, 0, 1]])           # canvas px -> mm
    Hc = H.dot(T)                                                         # canvas px -> ảnh
    warped = cv2.warpPerspective(mask, Hc, (W, L), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    wf = (warped > 127).astype(np.float32)

    def mm2c(x):
        return int(round((x + m) * S))

    files, ranks = geom.file_lines(), geom.rank_lines()
    half = SLICE / 2
    min_hit = 0.5 * SLICE * S                                             # đường phải phủ >= nửa đoạn dò

    # đường dọc (cột): u = poly(v)
    fcoef = []
    for f, u0 in enumerate(files):
        lo = u0 - (WIN_OUT if f == 0 else WIN_IN)
        hi = u0 + (WIN_OUT if f == NF - 1 else WIN_IN)
        pick = 1 if f == 0 else (-1 if f == NF - 1 else 0)          # cột a: lấy đỉnh phía phải (trong), cột i: phía trái
        pts = []
        for v in np.arange(-2.0, geom.length + 2.0, SLICE):
            if min(abs(v - r) for r in ranks) < SKIP:
                continue
            band = wf[mm2c(v - half):mm2c(v + half), mm2c(lo):mm2c(hi)]
            if band.size == 0:
                continue
            pk = _peak(band.sum(axis=0), min_hit, pick)
            if pk is not None:
                pts.append((v, lo + pk / S))
        fcoef.append(_fit(pts))
    # đường ngang (hàng): v = poly(u)
    rcoef = []
    for r, v0 in enumerate(ranks):
        lo = v0 - (WIN_OUT if r == 0 else WIN_IN)
        hi = v0 + (WIN_OUT if r == NR - 1 else WIN_IN)
        pick = 1 if r == 0 else (-1 if r == NR - 1 else 0)
        pts = []
        for u in np.arange(-2.0, geom.width + 2.0, SLICE):
            if min(abs(u - f) for f in files) < SKIP:
                continue
            band = wf[mm2c(lo):mm2c(hi), mm2c(u - half):mm2c(u + half)]
            if band.size == 0:
                continue
            pk = _peak(band.sum(axis=1), min_hit, pick)
            if pk is not None:
                pts.append((u, lo + pk / S))
        rcoef.append(_fit(pts))

    # giao điểm: lặp điểm bất động u = F(v), v = R(u)
    meas = {}
    for f in range(NF):
        for r in range(NR):
            if fcoef[f] is None or rcoef[r] is None:
                continue
            u, v = files[f], ranks[r]
            for _ in range(5):
                u = np.polyval(fcoef[f], v)
                v = np.polyval(rcoef[r], u)
            meas[(f, r)] = (u, v)          # toạ độ "mm theo homography hiện tại"
    n_lines = sum(c is not None for c in fcoef) + sum(c is not None for c in rcoef)
    return meas, n_lines, warped


def locate(bgr, geom, red_hint=None, debug=None):
    """Trả về BoardFix hoặc None. debug: dict nhận ảnh trung gian ('mask', 'warped')."""
    h, w = bgr.shape[:2]
    if red_hint is None:
        red_hint = (w / 2.0, float(h))
    mask = red_mask(bgr)
    quad = _quad(mask)
    if quad is None:
        return None
    q = _order(quad, np.asarray(red_hint, np.float64))
    H, _ = cv2.findHomography(np.array(geom.border_outer_mm(), np.float64), q)
    pts_px, rms, n_lines, warped = {}, None, 0, None
    for _ in range(2):
        meas, n_lines, warped = _refine(mask, geom, H)
        if len(meas) < 12:
            return None
        src = np.array([geom.square_mm(*k) for k in meas], np.float64)
        dst = []
        for k in meas:
            p = H.dot([meas[k][0], meas[k][1], 1.0])
            dst.append((p[0] / p[2], p[1] / p[2]))
        dst = np.array(dst, np.float64)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        pts_px = dict(zip(meas.keys(), map(tuple, dst)))
    # độ lệch điểm đo so với homography, quy ra mm
    Hinv = np.linalg.inv(H)
    errs = []
    for k, (x, y) in pts_px.items():
        p = Hinv.dot([x, y, 1.0])
        u0, v0 = geom.square_mm(*k)
        errs.append(np.hypot(p[0] / p[2] - u0, p[1] / p[2] - v0))
    rms = float(np.sqrt(np.mean(np.square(errs))))
    if debug is not None:
        debug['mask'], debug['warped'] = mask, warped
    return BoardFix(H, pts_px, rms, q, n_lines)


def draw(bgr, fix, geom):
    img = bgr.copy()
    q = fix.quad.astype(np.int32)
    cv2.polylines(img, [q.reshape(-1, 1, 2)], True, (255, 140, 0), 2)
    for f in range(NF):                                           # lưới theo homography (xanh dương nhạt)
        a, b = fix.to_px(*geom.square_mm(f, 0)), fix.to_px(*geom.square_mm(f, NR - 1))
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (255, 200, 0), 1)
    for r in range(NR):
        a, b = fix.to_px(*geom.square_mm(0, r)), fix.to_px(*geom.square_mm(NF - 1, r))
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (255, 200, 0), 1)
    rad = max(4, int(geom.piece_d / 2 * _px_per_mm(fix)))
    for (f, r), (x, y) in fix.pts_px.items():                     # giao điểm đo được (xanh lá) + vòng tròn cỡ quân
        cv2.circle(img, (int(x), int(y)), 3, (0, 220, 0), -1)
        if r in (0, 9):
            cv2.circle(img, (int(x), int(y)), rad, (0, 180, 0), 1)
    c = np.mean(list(fix.pts_px.values()), axis=0)
    for name, k in (('a0', (0, 0)), ('i0', (8, 0)), ('a9', (0, 9)), ('i9', (8, 9))):
        if k in fix.pts_px:
            x, y = fix.pts_px[k]
            d = (np.array([x, y]) - c)
            d = d / (np.linalg.norm(d) + 1e-9) * 28                  # ghi nhãn hướng ra ngoài bàn
            cv2.putText(img, name, (int(x + d[0]) - 10, int(y + d[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 0), 2)
    cv2.putText(img, '%d/90 diem  %d/19 duong  lech %.2f mm' % (len(fix.pts_px), fix.n_lines, fix.rms_mm),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def _px_per_mm(fix):
    a = np.array(fix.to_px(100, 150))
    b = np.array(fix.to_px(110, 150))
    return np.linalg.norm(b - a) / 10.0
