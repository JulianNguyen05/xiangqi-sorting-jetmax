#!/usr/bin/env python3
# Hình học bàn cờ: toạ độ mm của 90 giao điểm trong hệ BÀN CỜ.
# Gốc (0, 0) tại a0; u tăng theo cột a -> i, v tăng theo hàng 0 -> 9 (Đỏ ở hàng 0..4, Đen ở 5..9).
# Tương thích Python 3.6 (Jetson Nano của JetMax).

import os
import yaml

NF, NR = 9, 10                     # số cột (file), số hàng (rank)
FILES = 'abcdefghi'


def square_name(f, r):
    return '%s%d' % (FILES[f], r)


def parse_square(name):
    """'e0' -> (4, 0)"""
    name = name.strip().lower()
    return FILES.index(name[0]), int(name[1:])


class BoardGeometry(object):
    def __init__(self, cfg):
        g = cfg['grid']
        self.width = float(g['width'])
        self.length = float(g['length'])
        self.river = float(g['river'])
        self.df = self.width / (NF - 1)                          # khoảng cách 2 cột
        self.dr = (self.length - self.river) / (NR - 2)          # khoảng cách 2 hàng (không tính sông)
        self.border_offset = float(cfg.get('border', {}).get('offset', 10.0))
        p = cfg.get('piece', {})
        self.piece_d = float(p.get('diameter', 25.0))
        self.piece_h = float(p.get('height', 10.0))

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..', 'config', 'board.yaml')
        with open(path) as fh:
            return cls(yaml.safe_load(fh))

    def square_mm(self, f, r):
        u = f * self.df
        v = r * self.dr if r <= 4 else 4 * self.dr + self.river + (r - 5) * self.dr
        return u, v

    def all_squares(self):
        """[((f, r), (u, v)), ...] đủ 90 giao điểm."""
        return [((f, r), self.square_mm(f, r)) for r in range(NR) for f in range(NF)]

    def file_lines(self):
        return [f * self.df for f in range(NF)]

    def rank_lines(self):
        return [self.square_mm(0, r)[1] for r in range(NR)]

    def border_outer_mm(self):
        """4 góc mép ngoài khung viền đỏ: (trái-hàng0, phải-hàng0, phải-hàng9, trái-hàng9)."""
        o = self.border_offset
        return [(-o, -o), (self.width + o, -o), (self.width + o, self.length + o), (-o, self.length + o)]

    def gap_report(self):
        """Khe hở giữa 2 quân đứng cạnh nhau (mm) theo cột và theo hàng."""
        return self.df - self.piece_d, self.dr - self.piece_d


if __name__ == '__main__':
    g = BoardGeometry.load()
    gf, gr = g.gap_report()
    print('Khoảng cách cột %.2f mm, hàng %.2f mm, sông %.1f mm' % (g.df, g.dr, g.river))
    print('Khe hở giữa 2 quân cạnh nhau: theo hàng ngang %.2f mm, theo cột dọc %.2f mm' % (gf, gr))
    for name in ('a0', 'e0', 'i0', 'a4', 'a5', 'i9'):
        print(name, g.square_mm(*parse_square(name)))
