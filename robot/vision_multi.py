#!/usr/bin/env python3
# Nhận diện bàn cờ từ NHIỀU điểm dừng của camera (mỗi điểm chỉ thấy một phần bàn cờ), gộp thành MỘT bàn cờ {(cột, hàng): 'R'}.
# Thuần Python + numpy, không cần ROS, thử được trên laptop với ảnh/dets giả.
#
# Ý tưởng: KHÔNG ghép ảnh. Mỗi điểm dừng (view) có một phép biến đổi phối cảnh riêng ảnh -> lưới 9x10, tính một lần
# từ >= 4 giao điểm nhìn thấy trong ảnh ở điểm dừng đó. Quân nhận diện được ở view nào thì đổi ngay sang (cột, hàng),
# rồi gộp theo ô. "Bàn cờ ảo" chính là lưới (cột, hàng); đổi sang toạ độ robot đã có board.Board.position().
#
# configs/views.yaml:
#   image_size: [640, 480]
#   piece_half: 22                 # nửa kích thước quân cờ trong ảnh (pixel), dùng để biết ô nào nhìn thấy trọn vẹn
#   views:
#     - name: trai
#       slider: 0                  # vị trí thanh trượt khi chụp (bước từ home)
#       arm: [0, -150, 120]        # x, y, z tay máy khi chụp (mm)
#       points:                    # >= 4 giao điểm (tên ô: [x, y] pixel), rải đều trong ảnh, càng nhiều càng tốt
#         a0: [112, 438]
#         ...
# Quy ước ô giống board.py. Phải dạy lại points mỗi khi đổi slider/arm của view đó.

import numpy as np

from board import parse_square, square_name, NF, NR
from vision_board import class_to_piece


def fit_homography(src, dst):
    """Phép biến đổi phối cảnh src -> dst từ >= 4 cặp điểm (bình phương tối thiểu khi > 4)."""
    if len(src) < 4:
        raise ValueError('Cần ít nhất 4 điểm, đang có %d' % len(src))
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.lstsq(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), rcond=None)[0]
    return np.append(h, 1.0).reshape(3, 3)


def _apply(h, x, y):
    v = h.dot(np.array([x, y, 1.0]))
    return v[0] / v[2], v[1] / v[2]


class View(object):
    def __init__(self, name, points, size, slider=None, arm=None, edge=0.03, max_offset=0.35, anchor_y=0.5,
                 piece_half=22.0):
        """edge: bỏ khung nằm sát mép ảnh (quân bị cắt dở), tính theo tỉ lệ kích thước ảnh.
        piece_half: nửa kích thước quân cờ trong ảnh (pixel); một ô chỉ tính là 'nhìn thấy' khi cả quân nằm gọn trong ảnh."""
        pts = [(parse_square(k), v) for k, v in points.items()]
        src = [tuple(float(c) for c in v) for _, v in pts]
        dst = [(float(f), float(r)) for (f, r), _ in pts]
        self.H = fit_homography(src, dst)
        self.Hinv = np.linalg.inv(self.H)
        self.name, self.slider, self.arm = name, slider, arm
        self.w, self.h = float(size[0]), float(size[1])
        self.mx, self.my = edge * self.w, edge * self.h
        self.pad = float(piece_half)
        self.max_offset, self.anchor_y = float(max_offset), float(anchor_y)
        err = [np.hypot(*(np.array(_apply(self.H, x, y)) - np.array(d))) for (x, y), d in zip(src, dst)]
        self.fit_error = float(max(err))        # sai số lớn nhất của các điểm dạy, đơn vị ô; nên < 0.15

    def covers(self, f, r):
        """Giao điểm (f, r) có nằm gọn trong ảnh của view này không."""
        x, y = _apply(self.Hinv, f, r)
        return (self.mx + self.pad <= x <= self.w - self.mx - self.pad
                and self.my + self.pad <= y <= self.h - self.my - self.pad)

    def centre_penalty(self, x, y):
        """0 ở tâm ảnh, 1 ở góc ảnh: quân ở giữa ảnh ít méo hơn quân ở rìa."""
        return min(1.0, np.hypot((x - self.w / 2) / (self.w / 2), (y - self.h / 2) / (self.h / 2)) / 1.414)

    def cells(self, dets, names):
        """Trả về ({(f, r): (ký tự quân, điểm)}, cảnh báo) cho một khung ảnh của view này."""
        out, warns = {}, []
        for x1, y1, x2, y2, conf, cls in dets:
            k = int(cls)
            name = names[k] if k < len(names) else str(k)
            piece = class_to_piece(name)
            if piece is None:
                warns.append('[%s] lớp lạ "%s"' % (self.name, name))
                continue
            if x1 < self.mx or y1 < self.my or x2 > self.w - self.mx or y2 > self.h - self.my:
                continue                                        # quân bị cắt ở mép ảnh: view khác sẽ thấy rõ hơn
            cx, cy = (x1 + x2) / 2.0, y1 + (y2 - y1) * self.anchor_y
            gf, gr = _apply(self.H, cx, cy)
            f, r = int(round(gf)), int(round(gr))
            if not (0 <= f < NF and 0 <= r < NR):
                continue                                        # ngoài bàn cờ (quân ở khay, tay người...)
            if abs(gf - f) > self.max_offset or abs(gr - r) > self.max_offset:
                warns.append('[%s] %s lệch tâm ô %s (%.2f, %.2f)' % (self.name, name, square_name(f, r),
                                                                      gf - f, gr - r))
                continue
            score = float(conf) * (1.0 - 0.3 * self.centre_penalty(cx, cy))
            if (f, r) not in out or score > out[(f, r)][1]:
                out[(f, r)] = (piece, score)
        return out, warns


class MultiViewMapper(object):
    def __init__(self, views):
        if not views:
            raise ValueError('Chưa có view nào')
        self.views = list(views)

    @classmethod
    def load(cls, path):
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        size = d.get('image_size') or [640, 480]
        views = [View(v['name'], v['points'], size, v.get('slider'), v.get('arm'),
                      d.get('edge', 0.03), d.get('max_offset', 0.35), d.get('anchor_y', 0.5), d.get('piece_half', 22.0))
                 for v in (d.get('views') or [])]
        return cls(views)

    def uncovered(self, views=None):
        """Các ô không view nào nhìn thấy. Phải rỗng thì mới nhìn được toàn bàn cờ."""
        vs = self.views if views is None else views
        return [square_name(f, r) for f in range(NF) for r in range(NR) if not any(v.covers(f, r) for v in vs)]

    def merge(self, results):
        """results: [(view, dets, names)] -> (bàn cờ {(f, r): ký tự}, các ô đã được nhìn thấy, cảnh báo).
        Hai view cùng thấy một ô: lấy quân có điểm cao hơn; nếu khác quân thì cảnh báo."""
        best, warns = {}, []
        seen = set()
        for view, dets, names in results:
            for f in range(NF):
                for r in range(NR):
                    if view.covers(f, r):
                        seen.add((f, r))
            cells, w = view.cells(dets, names)
            warns += w
            for sq, (piece, score) in cells.items():
                if sq in best and best[sq][0] != piece:
                    warns.append('hai view bất đồng ở %s: %s và %s' % (square_name(*sq), best[sq][0], piece))
                if sq not in best or score > best[sq][1]:
                    best[sq] = (piece, score)
        return dict((sq, p) for sq, (p, _) in best.items()), seen, warns


class ScanObserver(object):
    """Giống vision_board.BoardObserver nhưng mỗi lần gọi là một vòng quét đủ các view.
      mapper     : MultiViewMapper
      detector   : detect_view.load_backend(...) (gọi detector(bgr) -> Nx6, có .names)
      move_to    : hàm move_to(view) đưa thanh trượt + tay máy tới view.slider / view.arm rồi đợi yên
      grab_frame : hàm không tham số trả về ảnh BGR mới nhất
      stabilizer : game.BoardStabilizer"""

    def __init__(self, mapper, detector, move_to, grab_frame, stabilizer):
        self.mapper, self.detector, self.move_to, self.grab_frame, self.stab = mapper, detector, move_to, grab_frame, stabilizer
        self.last_warnings, self.last_seen = [], set()

    def __call__(self):
        results = []
        for v in self.mapper.views:
            self.move_to(v)
            results.append((v, self.detector(self.grab_frame()), self.detector.names))
        board, self.last_seen, self.last_warnings = self.mapper.merge(results)
        return self.stab.update(board)


if __name__ == '__main__':
    import sys
    m = MultiViewMapper.load(sys.argv[1] if len(sys.argv) > 1 else 'configs/views.yaml')
    for v in m.views:
        print('%-10s slider=%s arm=%s  sai số dạy tối đa %.2f ô%s' % (
            v.name, v.slider, v.arm, v.fit_error, '  <-- DẠY LẠI ĐIỂM' if v.fit_error > 0.15 else ''))
    miss = m.uncovered()
    print('Ô không view nào thấy: %s' % (', '.join(miss) if miss else 'không có (phủ kín 90 ô)'))
