#!/usr/bin/env python3
# JetMaxExecutor: biến nước đi của robot (MoveInfo từ game.py) thành chuỗi gắp - đặt bằng Mover + Board.
# Thuần Python, KHÔNG import rospy/motion, nên chạy thử được trên laptop với FakeMover (robot/motion_fake.py).
# Trên robot thật truyền vào motion.Mover(), và limits=motion.in_limits.
#
# Với mỗi nước đi:
#   1. Nếu ăn quân: gắp quân bị ăn ở ô đích -> thả vào ô tiếp theo của khay.
#   2. Gắp quân ở ô đi -> đặt xuống ô đến.
# Mọi vị trí được kiểm tra (tầm với, giới hạn an toàn, khay còn chỗ) TRƯỚC khi tay máy cử động, để lỗi vị trí
# không bao giờ xảy ra giữa chừng. Nếu lỗi giữa chừng (mất hút, tay chưa tới đích...), tắt hút, nâng tay, đánh dấu
# `broken` và ném RobotError: game.py giữ nguyên thế cờ, và executor từ chối đi tiếp cho tới khi gọi reset() (sau khi
# bạn đã tự xếp lại bàn cờ cho khớp với game.pos).
#
# Cấu hình khay trong configs/board.yaml (Board.load bỏ qua khoá này, Tray.load đọc nó):
#   tray:
#     slider: 0            # vị trí thanh trượt khi thả quân vào khay
#     x: 80.0              # x, y của quân đầu tiên (mm)
#     y: -110.0
#     col_step: [0.0, -20.0]   # bước giữa hai quân liền nhau trong một hàng (mm)
#     row_step: [-20.0, 0.0]   # bước sang hàng khay tiếp theo (mm)
#     cols: 8
#     rows: 4              # sức chứa = cols * rows (cần >= 30 để chứa mọi quân có thể bị ăn)
#     z: null              # z khi thả vào khay; null = z_place của bàn cờ

from game import RobotExecutor, RobotError
from board import Board  # noqa: F401  (cho tiện import từ nơi khác)


class Tray(object):
    def __init__(self, slider, x, y, col_step=(0.0, -20.0), row_step=(-20.0, 0.0), cols=8, rows=4, z=None):
        self.slider, self.x, self.y = int(slider), float(x), float(y)
        self.col_step = tuple(float(v) for v in col_step)
        self.row_step = tuple(float(v) for v in row_step)
        self.cols, self.rows = int(cols), int(rows)
        self.z = None if z is None else float(z)

    @classmethod
    def load(cls, path):
        import yaml
        with open(path) as f:
            d = (yaml.safe_load(f) or {}).get('tray') or {}
        need = [k for k in ('slider', 'x', 'y') if d.get(k) is None]
        if need:
            raise ValueError('Chưa điền khay trong board.yaml: %s' % ', '.join('tray.' + k for k in need))
        return cls(d['slider'], d['x'], d['y'], d.get('col_step', (0.0, -20.0)), d.get('row_step', (-20.0, 0.0)),
                   d.get('cols', 8), d.get('rows', 4), d.get('z'))

    @property
    def capacity(self):
        return self.cols * self.rows

    def position(self, i):
        """Vị trí (slider, x, y) của ô khay thứ i (0, 1, 2...)."""
        if not 0 <= i < self.capacity:
            raise IndexError('Khay đầy (%d chỗ)' % self.capacity)
        c, r = i % self.cols, i // self.cols
        return (self.slider,
                round(self.x + c * self.col_step[0] + r * self.row_step[0], 1),
                round(self.y + c * self.col_step[1] + r * self.row_step[1], 1))


def make_resolver(board, tray=None):
    """Hàm đổi vị trí (slider, x, y) về tên ô ('h2') hoặc ô khay ('khay0'). Dùng để ghi log và cho FakeMover."""
    table = {}
    for f in range(9):
        for r in range(10):
            sq = '%s%d' % ('abcdefghi'[f], r)
            table[board.position(sq)] = sq
    if tray is not None:
        for i in range(tray.capacity):
            table[tray.position(i)] = 'khay%d' % i
    return lambda pos: table.get(tuple(pos))


class JetMaxExecutor(RobotExecutor):
    def __init__(self, mover, board, tray, limits=None, slider_range=(0, 5200), log=print):
        """limits: hàm (x, y, z) -> bool (motion.in_limits); None = không kiểm tra thêm."""
        self.mover, self.board, self.tray = mover, board, tray
        self.limits, self.slider_range, self.log = limits, slider_range, log
        self.tray_used = 0
        self.broken = None

    # ------------------------------------------------------------------ kế hoạch
    def plan(self, info):
        """Danh sách bước [(loại, nhãn, vị trí, z)] cho một nước đi."""
        b, steps = self.board, []
        if info.captured:
            steps.append(('pick', info.to, b.position(info.to), b.z_pick))
            n = self.tray_used
            if n >= self.tray.capacity:
                raise RobotError('Khay quân bị ăn đã đầy (%d quân), hãy dọn khay rồi gọi reset_tray()' % n)
            z = self.tray.z if self.tray.z is not None else b.z_place
            steps.append(('place', 'khay%d' % n, self.tray.position(n), z))
        steps.append(('pick', info.frm, b.position(info.frm), b.z_pick))
        steps.append(('place', info.to, b.position(info.to), b.z_place))
        return steps

    def _check(self, label, pos, z):
        s, x, y = pos
        lo, hi = self.slider_range
        if not lo <= s <= hi:
            raise RobotError('%s: thanh trượt %d ngoài khoảng %d..%d' % (label, s, lo, hi))
        if self.limits is not None:
            for zz in {z, self.board.z_safe}:
                if not self.limits((x, y, zz)):
                    raise RobotError('%s: điểm (%.1f, %.1f, %.1f) ngoài giới hạn an toàn hoặc ngoài tầm với' % (
                        label, x, y, zz))

    def unreachable(self):
        """Kiểm tra trước ván cờ: trả về danh sách ô bàn cờ và ô khay không dùng được (rỗng là tốt)."""
        bad = []
        for f in range(9):
            for r in range(10):
                sq = '%s%d' % ('abcdefghi'[f], r)
                for z in (self.board.z_pick, self.board.z_place):
                    try:
                        self._check(sq, self.board.position(sq), z)
                    except RobotError:
                        bad.append(sq)
                        break
        for i in range(self.tray.capacity):
            z = self.tray.z if self.tray.z is not None else self.board.z_place
            try:
                self._check('khay%d' % i, self.tray.position(i), z)
            except RobotError:
                bad.append('khay%d' % i)
        return bad

    # ------------------------------------------------------------------ thực hiện
    def execute(self, info):
        if self.broken:
            raise RobotError('Robot đang ở trạng thái lỗi (%s). Xếp lại bàn cờ cho khớp rồi gọi reset().' % self.broken)
        steps = self.plan(info)
        for kind, label, pos, z in steps:               # kiểm tra hết TRƯỚC khi động tới tay máy
            self._check(label, pos, z)
        zs = self.board.z_safe
        for i, (kind, label, pos, z) in enumerate(steps):
            self.log('    [robot] %s %s' % ('gắp ở' if kind == 'pick' else 'đặt xuống', label))
            try:
                if kind == 'pick':
                    self.mover.pick(pos, z, zs)
                else:
                    self.mover.place(pos, z, zs)
            except Exception as e:
                self._abort(zs)
                self.broken = 'hỏng ở bước %d/%d (%s %s): %s' % (i + 1, len(steps), kind, label, e)
                raise RobotError(self.broken)
            except KeyboardInterrupt:
                self._abort(zs)
                self.broken = 'bị ngắt ở bước %d/%d (%s %s)' % (i + 1, len(steps), kind, label)
                raise
            if kind == 'place' and label.startswith('khay'):
                self.tray_used += 1                      # quân bị ăn đã nằm trong khay, dù bước sau có hỏng

    def _abort(self, zs):
        try:
            self.mover.safe_stop(zs)
        except Exception as e:
            self.log('Không nâng được tay máy (%s), hãy kiểm tra thủ công' % e)

    def reset(self):
        """Gọi sau khi đã tự xếp lại bàn cờ vật lý cho khớp với game.pos."""
        self.broken = None

    def reset_tray(self):
        """Gọi sau khi đã dọn hết quân trong khay."""
        self.tray_used = 0
