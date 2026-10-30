#!/usr/bin/env python3
# Định vị tờ bàn cờ A4 trong ảnh camera bằng 4 marker ArUco ở góc giấy.
# Từ các góc marker nhìn thấy (toạ độ mm đã biết trong configs/board_a4.yaml) tính phép biến đổi phối cảnh
# ảnh <-> mm của mặt bàn cờ, rồi chiếu 90 giao điểm lên ảnh. Nhờ đó KHÔNG cần bấm chuột đọc pixel để điền views.yaml.
#
# Cần OpenCV có module aruco (opencv-contrib). Kiểm tra:  python3 -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
# Tương thích cả API cũ (OpenCV < 4.7: DetectorParameters_create) lẫn mới (ArucoDetector).

import numpy as np

from board import square_name, NF, NR


def _make_detect(dict_name):
    import cv2
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('OpenCV (%s) không có module aruco. Cài opencv-contrib-python (hoặc bản OpenCV có contrib) '
                           'hoặc dùng cách bấm tay trong views.yaml.' % cv2.__version__)
    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        raise RuntimeError('OpenCV không có từ điển %s' % dict_name)
    d = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    if hasattr(aruco, 'ArucoDetector'):                       # OpenCV >= 4.7
        p = aruco.DetectorParameters()
        p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        det = aruco.ArucoDetector(d, p)
        return lambda gray: det.detectMarkers(gray)[:2]
    p = aruco.DetectorParameters_create()                     # OpenCV cũ (4.1 - 4.6)
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return lambda gray: aruco.detectMarkers(gray, d, parameters=p)[:2]


class Fix(object):
    """Phép biến đổi ảnh <-> mm của mặt bàn cờ tính từ các marker nhìn thấy trong MỘT ảnh."""

    def __init__(self, h_img2mm, ids, rms_mm, corners_px):
        self.H = h_img2mm
        self.Hinv = np.linalg.inv(h_img2mm)
        self.ids = sorted(ids)
        self.rms_mm = rms_mm              # sai số khớp của các góc marker (mm); 0 nếu chỉ có 1 marker (không đo được)
        self.corners_px = corners_px      # {id: mảng 4x2 pixel}

    def to_mm(self, x, y):
        v = self.H.dot(np.array([x, y, 1.0]))
        return v[0] / v[2], v[1] / v[2]

    def to_px(self, mx, my):
        """Điểm (mx, my) mm trong hệ BOARD -> pixel. Cũng nhận một tuple (mx, my) làm đối số duy nhất."""
        if my is None:
            mx, my = mx
        v = self.Hinv.dot(np.array([mx, my, 1.0]))
        return v[0] / v[2], v[1] / v[2]

    def px_per_mm(self, mx, my):
        """Tỉ lệ pixel/mm quanh điểm (mx, my) (trung bình theo hai trục)."""
        p0 = np.array(self.to_px(mx, my))
        px = np.array(self.to_px(mx + 10.0, my)) - p0
        py = np.array(self.to_px(mx, my + 10.0)) - p0
        return float((np.hypot(*px) + np.hypot(*py)) / 20.0)


class Locator(object):
    def __init__(self, geom):
        self.geom = geom
        self._detect = _make_detect(geom.dictionary)

    def markers(self, bgr):
        """{id: mảng 4x2 pixel} của các marker thuộc tờ bàn cờ nhìn thấy trong ảnh."""
        import cv2
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        corners, ids = self._detect(gray)
        out = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.reshape(-1)):
            if int(i) in self.geom.centers:
                out[int(i)] = np.asarray(c, dtype=np.float64).reshape(4, 2)
        return out

    def __call__(self, bgr):
        """Trả về Fix, hoặc None nếu không thấy marker nào."""
        import cv2
        found = self.markers(bgr)
        if not found:
            return None
        src, dst = [], []
        for mid, c in found.items():
            for p, q in zip(c, self.geom.marker_corners_mm(mid)):
                src.append(p)
                dst.append(q)
        src, dst = np.array(src, np.float64), np.array(dst, np.float64)
        h, _ = cv2.findHomography(src, dst, 0)
        if h is None:
            return None
        proj = np.hstack([src, np.ones((len(src), 1))]).dot(h.T)
        proj = proj[:, :2] / proj[:, 2:3]
        rms = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1)))) if len(found) > 1 else 0.0
        return Fix(h, found.keys(), rms, found)


# ------------------------------------------------------------------ điểm dạy cho views.yaml -------
def visible_squares(fix, geom, size, piece_half, edge=0.03):
    """{(f, r): (x_px, y_px)} của các giao điểm nằm gọn trong ảnh (cả quân cờ), cùng điều kiện View.covers()."""
    w, h = float(size[0]), float(size[1])
    mx, my = edge * w + piece_half, edge * h + piece_half
    out = {}
    for f in range(NF):
        for r in range(NR):
            x, y = fix.to_px(*geom.square_mm(f, r))
            if mx <= x <= w - mx and my <= y <= h - my:
                out[(f, r)] = (x, y)
    return out


def points_for_view(fix, geom, size, piece_half, edge=0.03, max_points=16):
    """Chọn <= max_points giao điểm nhìn thấy, rải đều (lấy điểm xa nhau nhất lần lượt), dạng {'a0': [x, y]} cho views.yaml."""
    vis = visible_squares(fix, geom, size, piece_half, edge)
    if len(vis) < 4:
        return {}
    keys = sorted(vis)
    pts = np.array([vis[k] for k in keys])
    chosen = [int(np.argmin(pts[:, 0] + pts[:, 1]))]
    dist = np.hypot(*(pts - pts[chosen[0]]).T)
    while len(chosen) < min(max_points, len(keys)):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.hypot(*(pts - pts[i]).T))
    return dict((square_name(*keys[i]), [round(float(pts[i][0]), 1), round(float(pts[i][1]), 1)]) for i in sorted(chosen))


def draw_overlay(bgr, fix, geom, piece_half, edge=0.03):
    """Vẽ lưới 90 giao điểm đã chiếu (xanh = nằm gọn trong ảnh, xám = không) và các marker tìm thấy lên bản sao của ảnh."""
    import cv2
    img = bgr.copy()
    h, w = img.shape[:2]
    vis = visible_squares(fix, geom, (w, h), piece_half, edge)
    for f in range(NF):
        for r in range(NR):
            x, y = fix.to_px(*geom.square_mm(f, r))
            if not (0 <= x < w and 0 <= y < h):
                continue
            ok = (f, r) in vis
            cv2.circle(img, (int(x), int(y)), 3, (0, 200, 0) if ok else (150, 150, 150), -1)
            if f in (0, 4, 8) and r in (0, 5, 9):
                cv2.putText(img, square_name(f, r), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 0) if ok else (180, 180, 180), 1)
    for mid, c in fix.corners_px.items():
        cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (0, 140, 255), 2)
        cv2.putText(img, 'id%d' % mid, (int(c[0][0]), int(c[0][1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
    cv2.putText(img, 'marker %s  sai so %.2f mm  %d/90 o' % (fix.ids, fix.rms_mm, len(vis)), (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img
