#!/usr/bin/env python3
# Điều khiển tay máy bằng bàn phím (dùng chung cho tools/teleop.py và tools/calibrate.py).
# Phím:  d/a = x +/-   w/s = y +/-   r/f = z +/-   x/z = thanh trượt tiến/lùi   v = bật/tắt hút
#        t = đổi mức bước   CHỮ HOA = bước x4   p = in trạng thái   Enter = ghi điểm (khi được hỏi)   q = thoát
# Chạy trong terminal của robot (SSH / VNC). Mọi lệnh đi qua robot.motion.Robot nên vẫn kiểm tra giới hạn an toàn.

import select
import sys
import termios
import tty

MM_STEPS = [1, 5, 10, 20]                 # mm (bước 1 mm để căn tâm quân khi hiệu chỉnh)
SLIDER_STEPS = [100, 500, 1000, 2000]

HELP = """  d/a: x+/-   w/s: y+/-   r/f: z+/-   x/z: trượt tiến/lùi   v: hút bật/tắt
  t: đổi mức bước   CHỮ HOA: bước x4   p: in vị trí   Enter: GHI ĐIỂM   q: thoát/bỏ qua"""


class Jogger(object):
    def __init__(self, robot, log=print):
        self.robot, self.log = robot, log
        self.level = 1
        self.suck = False

    def show(self):
        x, y, z = self.robot.status()
        self.log('  robot xyz = (%.1f, %.1f, %.1f)  trượt = %d  bước = %d mm / %d  hút = %s' % (
            x, y, z, self.robot.s, MM_STEPS[self.level], SLIDER_STEPS[self.level], 'BẬT' if self.suck else 'tắt'))

    def _move(self, dx, dy, dz):
        x, y, z = self.robot.status()
        p = (x + dx, y + dy, z + dz)
        if not self.robot.in_limits(p):
            self.log('  Ngoài giới hạn an toàn, không đi')
            return
        try:
            self.robot.goto(p, speed=150.0)
        except Exception as e:
            self.log('  %s' % e)
        self.show()

    def loop(self, prompt='Enter = ghi điểm, q = bỏ qua'):
        """Chạy tới khi bấm Enter (trả về 'enter') hoặc q (trả về 'quit')."""
        self.log(HELP)
        self.log('  ' + prompt)
        self.show()
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                raw = sys.stdin.read(1)
                big = 4 if raw.isupper() else 1
                k = raw.lower()
                mm = MM_STEPS[self.level] * big
                sl = SLIDER_STEPS[self.level] * big
                if raw in ('\n', '\r'):
                    return 'enter'
                if k == 'q':
                    return 'quit'
                if k == 'd':
                    self._move(mm, 0, 0)
                elif k == 'a':
                    self._move(-mm, 0, 0)
                elif k == 'w':
                    self._move(0, mm, 0)
                elif k == 's':
                    self._move(0, -mm, 0)
                elif k == 'r':
                    self._move(0, 0, mm)
                elif k == 'f':
                    self._move(0, 0, -mm)
                elif k in ('x', 'z'):
                    try:
                        self.robot.slide_to(self.robot.s + (sl if k == 'x' else -sl))
                    except Exception as e:
                        self.log('  %s' % e)
                    self.show()
                elif k == 'v':
                    self.suck = not self.suck
                    self.robot.b.sucker(self.suck)
                    self.show()
                elif k == 't':
                    self.level = (self.level + 1) % len(MM_STEPS)
                    self.show()
                elif k == 'p':
                    self.show()
                termios.tcflush(fd, termios.TCIFLUSH)          # bỏ phím bấm dồn khi robot đang chạy
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
