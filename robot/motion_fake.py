#!/usr/bin/env python3
# FakeMover: thay motion.Mover khi chưa có robot. Cùng giao diện hover/pick/place/safe_stop, không cần rospy.
# Nếu truyền `physical` (dict nhãn ô -> quân) và `resolve` (executor.make_resolver) thì nó mô phỏng cả bàn cờ vật lý:
# gắp ở ô trống hoặc đặt vào ô đã có quân sẽ ném RuntimeError giống lỗi thật, để thử đường xử lý lỗi.

class FakeMover(object):
    def __init__(self, resolve=None, physical=None, fail_at=None):
        """fail_at: số thứ tự (từ 1) của lệnh pick/place sẽ bị ném RuntimeError, để thử xử lý lỗi giữa chừng."""
        self.resolve = resolve or (lambda pos: None)
        self.physical = physical if physical is not None else {}
        self.fail_at = fail_at
        self.calls, self.held, self.dropped, self.safe_stops, self._n = [], None, [], 0, 0

    def _count(self, kind, pos):
        self._n += 1
        self.calls.append((kind, self.resolve(pos), tuple(pos)))
        if self.fail_at is not None and self._n == self.fail_at:
            raise RuntimeError('lỗi giả lập ở lệnh %d (%s)' % (self._n, kind))

    def hover(self, sq_pos, zsafe):
        self.calls.append(('hover', self.resolve(sq_pos), tuple(sq_pos)))

    def pick(self, sq_pos, zpick, zsafe):
        self._count('pick', sq_pos)
        label = self.resolve(sq_pos)
        if self.held is not None:
            raise RuntimeError('đang cầm quân, không gắp thêm')
        if label not in self.physical:
            raise RuntimeError('không có quân để gắp ở %s' % label)
        self.held = self.physical.pop(label)

    def place(self, sq_pos, zplace, zsafe):
        self._count('place', sq_pos)
        label = self.resolve(sq_pos)
        if self.held is None:
            raise RuntimeError('không cầm quân nào để đặt')
        if label in self.physical:
            raise RuntimeError('ô %s đã có quân' % label)
        self.physical[label] = self.held
        self.held = None

    def safe_stop(self, zsafe):
        self.safe_stops += 1
        if self.held is not None:                 # tắt hút -> quân rơi, không còn nằm ở ô nào
            self.dropped.append(self.held)
            self.held = None

    def sequence(self):
        """Chuỗi (loại, nhãn) của các lệnh pick/place, tiện so sánh trong test."""
        return [(k, lab) for k, lab, _ in self.calls if k in ('pick', 'place')]
