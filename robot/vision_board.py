#!/usr/bin/env python3
# Chuyển kết quả nhận diện YOLO (khung + lớp) thành bàn cờ {(cột, hàng): 'R'} để đưa vào game.py.
#
# Cách làm: dùng 4 góc bàn cờ trong ẢNH (pixel tại giao điểm a0, i0, a9, i9) để tính phép biến đổi phối cảnh sang lưới
# 9x10 ô; tâm mỗi khung quân cờ được đổi sang toạ độ lưới rồi làm tròn về giao điểm gần nhất.
# Cần điền configs/vision.yaml. Camera phải ở MỘT TƯ THẾ CỐ ĐỊNH nhìn thấy cả bàn cờ (nếu camera gắn trên tay máy thì
# phải đưa tay máy về đúng tư thế quan sát này mỗi lần chụp, vì 4 góc chỉ đúng cho tư thế đó).
#
# Định dạng đầu vào khớp với robot/detect_view.py: mỗi quân là (x1, y1, x2, y2, conf, cls_id) và `names[cls_id]` là
# 'red_rook', 'black_cannon', ... (đúng thứ tự 14 lớp trong training/build_dataset.ipynb).

import numpy as np

CORNER_KEYS = ('a0', 'i0', 'a9', 'i9')
PIECE_CODE = {'advisor': 'a', 'bishop': 'b', 'cannon': 'c', 'king': 'k', 'knight': 'n', 'pawn': 'p', 'rook': 'r'}


def class_to_piece(name):
    """'red_rook' -> 'R', 'black_cannon' -> 'c'. Trả về None nếu tên lạ."""
    try:
        color, kind = name.strip().lower().split('_', 1)
    except ValueError:
        return None
    code = PIECE_CODE.get(kind)
    if code is None or color not in ('red', 'black'):
        return None
    return code.upper() if color == 'red' else code


def _homography(src, dst):
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.solve(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)


class ImageBoardMapper(object):
    def __init__(self, image_corners, max_offset=0.35, anchor_y=0.5):
        """image_corners: {'a0': (x, y), 'i0': ..., 'a9': ..., 'i9': ...} (pixel).
        max_offset: quân lệch tâm giao điểm quá mức này (đơn vị ô) thì bỏ (đang bị cầm hoặc nằm giữa hai ô).
        anchor_y: vị trí lấy điểm trong khung theo chiều dọc, 0.5 = tâm khung, 1.0 = cạnh dưới."""
        miss = [k for k in CORNER_KEYS if k not in image_corners or image_corners[k] is None]
        if miss:
            raise ValueError('Chưa điền góc ảnh: %s (x, y pixel) trong configs/vision.yaml' % ', '.join(miss))
        src = [tuple(float(v) for v in image_corners[k]) for k in CORNER_KEYS]
        dst = [(0.0, 0.0), (8.0, 0.0), (0.0, 9.0), (8.0, 9.0)]
        self.H = _homography(src, dst)
        self.max_offset = float(max_offset)
        self.anchor_y = float(anchor_y)

    @classmethod
    def load(cls, path):
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(d.get('image_corners') or {}, d.get('max_offset', 0.35), d.get('anchor_y', 0.5))

    def to_grid(self, x, y):
        v = self.H.dot(np.array([x, y, 1.0]))
        return v[0] / v[2], v[1] / v[2]

    def board_from_detections(self, dets, names):
        """Trả về (bàn cờ {(f, r): ký tự}, danh sách cảnh báo)."""
        board, conf_at, warns = {}, {}, []
        for x1, y1, x2, y2, conf, cls in dets:
            k = int(cls)
            name = names[k] if k < len(names) else str(k)
            piece = class_to_piece(name)
            if piece is None:
                warns.append('lớp lạ "%s"' % name)
                continue
            gf, gr = self.to_grid((x1 + x2) / 2.0, y1 + (y2 - y1) * self.anchor_y)
            f, r = int(round(gf)), int(round(gr))
            if not (0 <= f < 9 and 0 <= r < 10):
                warns.append('%s ngoài bàn cờ' % name)
                continue
            if abs(gf - f) > self.max_offset or abs(gr - r) > self.max_offset:
                warns.append('%s lệch tâm ô (%.2f, %.2f)' % (name, gf - f, gr - r))
                continue
            if (f, r) in board:
                warns.append('hai quân cùng một ô (%d,%d), giữ quân tin cậy hơn' % (f, r))
                if conf <= conf_at[(f, r)]:
                    continue
            board[(f, r)] = piece
            conf_at[(f, r)] = conf
        return board, warns


class BoardObserver(object):
    """Ghép camera + YOLO + ánh xạ ô + gộp khung hình thành hàm observe() cho game.play().
      detector   : đối tượng từ detect_view.load_backend(...) (gọi detector(bgr) -> Nx6, có thuộc tính .names)
      grab_frame : hàm không tham số trả về ảnh BGR mới nhất (ví dụ dùng cách đọc topic /usb_cam trong scan_detect.py)
      stabilizer : game.BoardStabilizer"""

    def __init__(self, detector, grab_frame, mapper, stabilizer):
        self.detector, self.grab_frame, self.mapper, self.stab = detector, grab_frame, mapper, stabilizer
        self.last_warnings = []

    def __call__(self):
        dets = self.detector(self.grab_frame())
        board, self.last_warnings = self.mapper.board_from_detections(dets, self.detector.names)
        return self.stab.update(board)
