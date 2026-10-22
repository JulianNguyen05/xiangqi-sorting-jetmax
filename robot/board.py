#!/usr/bin/env python3
# Bàn cờ tướng: tên ô <-> chỉ số, và ánh xạ ô cờ -> (thanh trượt, x, y) của robot từ 4 góc đã dạy.
# Thuần Python, không cần ROS, chạy thử được trên laptop.
#
# Quy ước ô (giống UCCI, ví dụ "h2e2" = pháo đỏ ở h2 đi tới e2):
#   cột a..i từ TRÁI sang PHẢI nhìn từ phía quân ĐỎ; hàng 0..9 từ phía đỏ (hàng 0) lên phía đen (hàng 9).
#
#   hàng 9   a9 b9 c9 d9 e9 f9 g9 h9 i9     (phía đen)
#   ...
#   hàng 0   a0 b0 c0 d0 e0 f0 g0 h0 i0     (phía đỏ)
#
# Cách ánh xạ: dạy 4 góc a0, i0, a9, i9 (mỗi góc gồm vị trí thanh trượt + x + y của tay máy), rồi nội suy song tuyến
# cho cả ba giá trị. Vị trí robot là hàm tuyến tính của (thanh trượt, x, y) nên cách này đúng khi bàn cờ là hình chữ nhật
# và thanh trượt chạy thẳng. Không cần biết bước/mm hay hướng trượt.

FILES = 'abcdefghi'
NF, NR = 9, 10
CORNERS = ('a0', 'i0', 'a9', 'i9')


def parse_square(s):
    s = s.strip().lower()
    if len(s) != 2 or s[0] not in FILES or not s[1].isdigit():
        raise ValueError('Ô "%s" không hợp lệ (cột a-i, hàng 0-9, ví dụ h2)' % s)
    return FILES.index(s[0]), int(s[1])


def square_name(f, r):
    return '%s%d' % (FILES[f], r)


class Board:
    def __init__(self, corners, z_pick, z_safe=50.0, z_place=None):
        miss = [c for c in CORNERS if c not in corners or any(v is None for v in corners[c])]
        if miss:
            raise ValueError('Chưa điền góc: %s (slider, x, y)' % ', '.join(miss))
        if z_pick is None:
            raise ValueError('Chưa điền z_pick')
        self.c = {k: tuple(float(v) for v in corners[k]) for k in CORNERS}
        self.z_pick = float(z_pick)
        self.z_safe = float(z_safe)
        self.z_place = float(z_place) if z_place is not None else self.z_pick + 3.0

    @classmethod
    def load(cls, path):
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        corners = {k: (v.get('slider'), v.get('x'), v.get('y')) for k, v in (d.get('corners') or {}).items()}
        return cls(corners, d.get('z_pick'), d.get('z_safe', 50.0), d.get('z_place'))

    def position(self, sq):
        """Ô cờ (tên 'h2' hoặc (cột, hàng)) -> (slider:int, x, y) tại tâm giao điểm."""
        f, r = parse_square(sq) if isinstance(sq, str) else sq
        u, v = f / float(NF - 1), r / float(NR - 1)
        w = ((1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v)
        p = [sum(w[i] * self.c[CORNERS[i]][k] for i in range(4)) for k in range(3)]
        return int(round(p[0])), round(p[1], 1), round(p[2], 1)
