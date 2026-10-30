#!/usr/bin/env python3
# Định vị bàn cờ (tấm nhựa in đỏ) trong ảnh, KHÔNG cần marker.
#
# B1. Lọc màu đỏ (HSV) -> đoạn thẳng Hough -> 2 họ đường vuông góc -> ghép với mẫu 9 cột x 10 hàng
#     (chịu phối cảnh, chịu thiếu vài đường do quân che)                          => homography thô.
#     Dự phòng: 4 góc mép ngoài khung viền.
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
LINE_HALF = 0.4         # mm, nửa bề rộng nét đường lưới in trên tấm
OUTER_TOL = 2.5         # mm, điểm đo trên đường ngoài cùng lệch hơn mức này so với giá trị suy từ đường trong -> bỏ
MIN_SUPPORT = 0.4       # tỉ lệ đoạn dò thấy đường, dưới mức này coi đường bị che, suy ra từ các đường khác


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
    """Vị trí đường (px, có nội suy).
    pick=0 (đường bên trong): đỉnh mạnh nhất.
    pick=-1 / +1 (đường ngoài cùng, phía trong nằm ở đầu / cuối mảng): đường lưới ngoài cùng có chỗ cách vạch dày
    của khung viền ~4 mm, có chỗ dính sát vào nó (trên mặt nạ thành một khối). Vì vậy lấy MÉP TRONG của khối đỏ
    nằm trong cùng rồi lùi nửa bề rộng đường, đúng trong cả hai trường hợp."""
    if profile.max() < min_frac:
        return None
    if pick == 0:
        i = int(np.argmax(profile))
        r = max(1, int(round(1.0 * S)))
        lo, hi = max(0, i - r), min(len(profile), i + r + 1)
        w = profile[lo:hi].astype(np.float64)
        return (np.arange(lo, hi) * w).sum() / w.sum()
    prof = profile[::-1] if pick > 0 else profile           # quy về: phía trong ở đầu mảng
    level = 0.5 * min_frac * 2                               # nửa chiều cao đầy của đoạn dò
    above = np.nonzero(prof >= level)[0]
    if len(above) == 0:
        return None
    j = int(above[0])                                        # điểm đầu tiên vượt ngưỡng, nội suy tuyến tính
    if j > 0:
        a, b = float(prof[j - 1]), float(prof[j])
        edge = j - 1 + (level - a) / max(b - a, 1e-9)
    else:
        edge = 0.0
    pos = edge + LINE_HALF * S
    return (len(profile) - 1 - pos) if pick > 0 else pos


class _LineFit(object):
    """x = poly(t). Trong vùng có dữ liệu dùng bậc 2 (theo nếp gấp); ngoài vùng (đầu đường bị quân che)
    chỉ ngoại suy TUYẾN TÍNH từ đoạn 60 mm gần nhất, vì ngoại suy bậc 2 lệch rất nhanh."""
    END = 60.0
    MARGIN = 6.0

    def __init__(self, t, x, coef):
        self.coef = coef
        self.n = len(t)                    # số đoạn dò khớp đường (để đánh giá đường có bị che nhiều không)
        self.tmin, self.tmax = float(t.min()), float(t.max())
        self.lo = self._lin(t, x, t <= self.tmin + self.END)
        self.hi = self._lin(t, x, t >= self.tmax - self.END)

    @staticmethod
    def _lin(t, x, sel):
        if sel.sum() >= 2 and np.ptp(t[sel]) > 5:
            return np.polyfit(t[sel], x[sel], 1)
        return np.polyfit(t, x, 1)

    def __call__(self, t):
        if t < self.tmin - self.MARGIN:
            return np.polyval(self.lo, t)
        if t > self.tmax + self.MARGIN:
            return np.polyval(self.hi, t)
        return np.polyval(self.coef, t)


def _fit(points, deg=2, tol=1.2):
    """Khớp x = poly(t), loại điểm lệch > tol mm. points: [(t, x)]. Trả về _LineFit hoặc None."""
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
    return _LineFit(p[keep, 0], p[keep, 1], coef)


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
    fcoef, fsup = [], []
    for f, u0 in enumerate(files):
        lo = u0 - (WIN_OUT if f == 0 else WIN_IN)
        hi = u0 + (WIN_OUT if f == NF - 1 else WIN_IN)
        pick = 1 if f == 0 else (-1 if f == NF - 1 else 0)          # cột a: lấy đỉnh phía phải (trong), cột i: phía trái
        pts, tries = [], 0
        for v in np.arange(-2.0, geom.length + 2.0, SLICE):
            if min(abs(v - r) for r in ranks) < SKIP:
                continue
            band = wf[mm2c(v - half):mm2c(v + half), mm2c(lo):mm2c(hi)]
            if band.size == 0:
                continue
            tries += 1
            pk = _peak(band.sum(axis=0), min_hit, pick)
            if pk is not None:
                pts.append((v, lo + pk / S))
        fcoef.append(_fit(pts))
        fsup.append(fcoef[-1].n / float(max(tries, 1)) if fcoef[-1] else 0.0)
    # đường ngang (hàng): v = poly(u)
    rcoef, rsup = [], []
    for r, v0 in enumerate(ranks):
        lo = v0 - (WIN_OUT if r == 0 else WIN_IN)
        hi = v0 + (WIN_OUT if r == NR - 1 else WIN_IN)
        pick = 1 if r == 0 else (-1 if r == NR - 1 else 0)
        pts, tries = [], 0
        for u in np.arange(-2.0, geom.width + 2.0, SLICE):
            if min(abs(u - f) for f in files) < SKIP:
                continue
            band = wf[mm2c(lo):mm2c(hi), mm2c(u - half):mm2c(u + half)]
            if band.size == 0:
                continue
            tries += 1
            pk = _peak(band.sum(axis=1), min_hit, pick)
            if pk is not None:
                pts.append((u, lo + pk / S))
        rcoef.append(_fit(pts))
        rsup.append(rcoef[-1].n / float(max(tries, 1)) if rcoef[-1] else 0.0)

    # giao điểm: lặp điểm bất động u = F(v), v = R(u)
    meas = {}
    for f in range(NF):
        for r in range(NR):
            if fcoef[f] is None or rcoef[r] is None:
                continue
            u, v = files[f], ranks[r]
            for _ in range(5):
                u = fcoef[f](v)
                v = rcoef[r](u)
            meas[(f, r)] = (u, v)          # toạ độ "mm theo homography hiện tại"
    # đường bị che quá nửa (vd. hàng cuối xếp kín 9 quân): điểm dò lẫn vạch khung viền -> không tin
    weak = set(f for f in range(NF) if fsup[f] < MIN_SUPPORT) | set(('r', r) for r in range(NR) if rsup[r] < MIN_SUPPORT)
    n_lines = NF + NR - len(weak)
    return meas, n_lines, warped, weak


def _line_families(mask):
    """Đoạn thẳng Hough -> 2 họ đường gần vuông góc. Trả về [(góc_độ, segs Nx4, độ dài N), x2] hoặc None."""
    h, w = mask.shape
    L = float(max(h, w))
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=int(0.04 * L),
                           minLineLength=int(0.05 * L), maxLineGap=int(0.06 * L))
    if segs is None or len(segs) < 6:
        return None
    segs = segs.reshape(-1, 4).astype(np.float64)
    d = segs[:, 2:] - segs[:, :2]
    ln = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    hist = np.zeros(180)
    np.add.at(hist, ang.astype(int) % 180, ln)
    hs = sum(np.roll(hist, k) for k in range(-3, 4))
    a1 = int(np.argmax(hs))
    a2 = max([(a1 + 90 + k) % 180 for k in range(-20, 21)], key=lambda a: hs[a])
    fams = []
    for a in (a1, a2):
        sel = np.abs((ang - a + 90.0) % 180.0 - 90.0) < 6.0
        fams.append((a, segs[sel], ln[sel]))
    return fams


def _cluster_lines(fam, center, L):
    """Gộp các đoạn cùng nằm trên một đường. Trả về [(t, điểm, hướng, độ_dài)] sắp theo t
    (t = vị trí giao với đường vuông góc qua tâm ảnh)."""
    a, segs, ln = fam
    th = np.radians(a)
    u0 = np.array([np.cos(th), np.sin(th)])
    n = np.array([-u0[1], u0[0]])
    items = []
    for sg, l in zip(segs, ln):
        p, q = sg[:2], sg[2:]
        u = (q - p) / max(np.linalg.norm(q - p), 1e-9)
        den = n[0] * u[1] - n[1] * u[0]
        if abs(den) < 1e-6:
            continue
        r = p - center
        t = (r[0] * u[1] - r[1] * u[0]) / den
        items.append((t, sg, l))
    items.sort(key=lambda x: x[0])
    tol = 0.006 * L
    groups, cur = [], []
    for it in items:
        if cur and it[0] - cur[-1][0] > tol:
            groups.append(cur)
            cur = []
        cur.append(it)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        total = sum(x[2] for x in g)
        if total < 0.12 * L:
            continue
        pts = []
        for _, sg, l in g:                       # lấy mẫu điểm dọc đoạn, tỉ lệ theo độ dài
            k = max(2, int(l / 10))
            for f in np.linspace(0, 1, k):
                pts.append(sg[:2] * (1 - f) + sg[2:] * f)
        pts = np.array(pts, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        dirv = np.array([vx, vy], np.float64)
        pnt = np.array([x0, y0], np.float64)
        den = n[0] * dirv[1] - n[1] * dirv[0]
        if abs(den) < 1e-6:
            continue
        r = pnt - center
        t = (r[0] * dirv[1] - r[1] * dirv[0]) / den
        out.append((t, pnt, dirv, total))
    out.sort(key=lambda x: x[0])
    return out


def _proj_fit(x, t):
    """t = (a x + b) / (c x + 1), bình phương tối thiểu."""
    A = np.stack([x, np.ones_like(x), -x * t], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
    return sol


def _proj_eval(sol, x):
    a, b, c = sol
    return (a * x + b) / (c * x + 1.0)


def _match_1d(T, X, rel_tol=0.25):
    """Ghép các vị trí đo T (px, tăng dần) với vị trí mẫu X (mm). Trả về (số khớp, {chỉ_số_mẫu: chỉ_số_đo}, px/mm)."""
    T = np.asarray(T, np.float64)
    X = np.asarray(X, np.float64)
    n, m = len(T), len(X)
    if n < 2:
        return 0, {}, 0.0
    dmin = np.min(np.diff(X))
    hyp = []
    for i in range(n):
        for j in range(i + 1, n):
            for p in range(m):
                for q in range(p + 1, m):
                    hyp.append((i, j, p, q))
                    hyp.append((i, j, q, p))          # cho phép chiều ngược
    hyp = np.array(hyp)
    a = (T[hyp[:, 1]] - T[hyp[:, 0]]) / (X[hyp[:, 3]] - X[hyp[:, 2]])
    ok = np.abs(a) > 0.3
    hyp, a = hyp[ok], a[ok]
    if len(hyp) == 0:
        return 0, {}, 0.0
    P = a[:, None] * (X[None, :] - X[hyp[:, 2]][:, None]) + T[hyp[:, 0]][:, None]       # (H, m)
    D = np.abs(P[:, :, None] - T[None, None, :]).min(axis=2)                              # (H, m)
    tol = rel_tol * np.abs(a)[:, None] * dmin
    score = (D < tol).sum(axis=1) - 0.01 * np.where(D < tol, D / tol, 0).sum(axis=1)
    best = None
    for h in np.argsort(-score)[:20]:                # tinh chỉnh bằng phép chiếu 1D (chịu phối cảnh)
        pred = P[h]
        for _ in range(3):
            pairs = {}
            for k in range(m):
                d = np.abs(T - pred[k])
                jj = int(np.argmin(d))
                if d[jj] < rel_tol * abs(a[h]) * dmin:
                    pairs[k] = jj
            if len(pairs) < 3:
                break
            ks = np.array(sorted(pairs))
            sol = _proj_fit(X[ks], T[[pairs[k] for k in ks]])
            pred = _proj_eval(sol, X)
        if len(set(pairs.values())) != len(pairs):
            continue
        res = np.mean([abs(T[pairs[k]] - pred[k]) for k in pairs]) if pairs else 1e9
        key = (len(pairs), -res)
        if best is None or key > best[0]:
            best = (key, pairs, abs(a[h]))
    if best is None:
        return 0, {}, 0.0
    return best[0][0], best[1], best[2]


def _intersect(l1, l2):
    p1, d1 = l1
    p2, d2 = l2
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    r = p2 - p1
    s = (r[0] * d2[1] - r[1] * d2[0]) / den
    return p1 + s * d1


def _coarse_lines(mask, geom, red_hint):
    """Homography thô mm -> ảnh dựa trên chính các đường lưới (không cần khung viền). None nếu không đủ đường."""
    fams = _line_families(mask)
    if fams is None:
        return None
    h, w = mask.shape
    L = float(max(h, w))
    center = np.array([w / 2.0, h / 2.0])
    lines = [_cluster_lines(f, center, L) for f in fams]
    if min(len(x) for x in lines) < 3:
        return None
    files, ranks = geom.file_lines(), geom.rank_lines()
    best = None
    for fa, ra in ((0, 1), (1, 0)):                  # họ nào là cột, họ nào là hàng
        nf, pf, kf = _match_1d([x[0] for x in lines[fa]], files)
        nr, pr, kr = _match_1d([x[0] for x in lines[ra]], ranks)
        if nf < 3 or nr < 3 or not (0.5 < kf / kr < 2.0):
            continue
        if best is None or nf + nr > best[0]:
            best = (nf + nr, fa, ra, pf, pr)
    if best is None or best[0] < 10:
        return None
    _, fa, ra, pf, pr = best
    src, dst = [], []
    for f, i in pf.items():
        for r, j in pr.items():
            p = _intersect(lines[fa][i][1:3], lines[ra][j][1:3])
            if p is not None:
                src.append(geom.square_mm(f, r))
                dst.append(p)
    if len(src) < 8:
        return None
    src, dst = np.array(src, np.float64), np.array(dst, np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.01 * L)
    if H is None:
        return None
    # hướng: hàng 0 (Đỏ) phải gần red_hint; cột a ở bên trái khi đứng phía Đỏ
    def px(Hm, u, v):
        q = Hm.dot([u, v, 1.0])
        return np.array([q[0] / q[2], q[1] / q[2]])
    W_, L_ = geom.width, geom.length
    hint = np.asarray(red_hint, np.float64)
    R180 = np.array([[-1, 0, W_], [0, -1, L_], [0, 0, 1]], np.float64)      # (u, v) -> (W-u, L-v)
    if np.linalg.norm(px(H, W_ / 2, L_) - hint) < np.linalg.norm(px(H, W_ / 2, 0) - hint):
        H = H.dot(R180)
    uv = px(H, W_, 0) - px(H, 0, 0)
    vv = px(H, 0, L_) - px(H, 0, 0)
    if uv[0] * vv[1] - uv[1] * vv[0] > 0:
        H = H.dot(np.array([[-1, 0, W_], [0, 1, 0], [0, 0, 1]], np.float64))   # lật cột a <-> i
    return H


def locate(bgr, geom, red_hint=None, debug=None):
    """Trả về BoardFix hoặc None. debug: dict nhận ảnh trung gian ('mask', 'warped')."""
    h, w = bgr.shape[:2]
    if red_hint is None:
        red_hint = (w / 2.0, float(h))
    mask = red_mask(bgr)
    H = _coarse_lines(mask, geom, red_hint)            # cách chính: dò đường lưới trực tiếp
    if H is None:                                      # dự phòng: 4 góc khung viền
        quad = _quad(mask)
        if quad is None:
            return None
        q = _order(quad, np.asarray(red_hint, np.float64))
        H, _ = cv2.findHomography(np.array(geom.border_outer_mm(), np.float64), q)
    pts_px, rms, n_lines, warped = {}, None, 0, None
    for _ in range(2):
        meas, n_lines, warped, weak = _refine(mask, geom, H)
        if len(meas) < 12:
            return None
        allpx = {}
        for k in meas:
            p = H.dot([meas[k][0], meas[k][1], 1.0])
            allpx[k] = (p[0] / p[2], p[1] / p[2])
        good = [k for k in meas if k[0] not in weak and ('r', k[1]) not in weak]
        inner = [k for k in good if 0 < k[0] < NF - 1 and 0 < k[1] < NR - 1]
        if len(inner) < 12:
            return None
        Hi, _ = cv2.findHomography(np.array([geom.square_mm(*k) for k in inner], np.float64),
                                   np.array([allpx[k] for k in inner], np.float64), cv2.RANSAC, 3.0)
        if Hi is None:
            return None
        Hi_inv = np.linalg.inv(Hi)

        def dev_mm(k):                             # điểm đo lệch bao nhiêu mm so với homography của các đường trong
            q = Hi_inv.dot([allpx[k][0], allpx[k][1], 1.0])
            u0, v0 = geom.square_mm(*k)
            return np.hypot(q[0] / q[2] - u0, q[1] / q[2] - v0)
        good = [k for k in good if k in inner or dev_mm(k) < OUTER_TOL]
        src = np.array([geom.square_mm(*k) for k in good], np.float64)
        dst = np.array([allpx[k] for k in good], np.float64)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return None
        pts_px = dict((k, allpx[k]) for k in good)
    for f in range(NF):                        # giao điểm trên đường bị che: suy từ homography
        for r in range(NR):
            if (f, r) not in pts_px:
                pts_px[(f, r)] = fix_px(H, *geom.square_mm(f, r))
    errs_keys = good
    # độ lệch điểm đo so với homography, quy ra mm
    Hinv = np.linalg.inv(H)
    errs = []
    for k in errs_keys:
        x, y = pts_px[k]
        p = Hinv.dot([x, y, 1.0])
        u0, v0 = geom.square_mm(*k)
        errs.append(np.hypot(p[0] / p[2] - u0, p[1] / p[2] - v0))
    rms = float(np.sqrt(np.mean(np.square(errs))))
    if debug is not None:
        debug['mask'], debug['warped'] = mask, warped
    quad = np.array([fix_px(H, u, v) for u, v in geom.border_outer_mm()])
    return BoardFix(H, pts_px, rms, quad, n_lines)


def fix_px(H, u, v):
    q = H.dot([u, v, 1.0])
    return q[0] / q[2], q[1] / q[2]


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
