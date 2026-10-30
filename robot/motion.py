#!/usr/bin/env python3
# Điều khiển JetMax: tay máy (ROS topic), đầu hút, thanh trượt (hiwonder.Stepper).
# Lớp Robot làm việc trong hệ THẾ GIỚI (world = hệ robot khi thanh trượt ở 0) và tự chọn vị trí thanh trượt.
# Hai backend: RosBackend (robot thật, cần node jetmax_control đang chạy) và FakeBackend (laptop, chỉ in ra).
# Tương thích Python 3.6. KHÔNG tạo hiwonder.JetMax() vì node jetmax_control đã giữ bus servo.

import math
import time


class MoveError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------- backends
class RosBackend(object):
    def __init__(self, cfg, use_slider):
        import rospy
        from std_msgs.msg import Bool
        from jetmax_control.msg import SetJetMax, JetMax
        self.rospy, self.Bool, self.SetJetMax, self.JetMax = rospy, Bool, SetJetMax, JetMax
        r = cfg['ros']
        self.status_topic = r['status']
        self.cmd = rospy.Publisher(r['command'], SetJetMax, queue_size=1)
        self.suck = rospy.Publisher(r['sucker'], Bool, queue_size=1)
        self.stepper = None
        if use_slider:
            import hiwonder
            self.stepper = hiwonder.Stepper(1)
        rospy.sleep(1.0)

    def status(self):
        m = self.rospy.wait_for_message(self.status_topic, self.JetMax, timeout=5)
        return m.x, m.y, m.z

    def move(self, p, duration):
        self.cmd.publish(self.SetJetMax(x=p[0], y=p[1], z=p[2], duration=duration))

    def sleep(self, t):
        self.rospy.sleep(t)

    def sucker(self, on):
        self.suck.publish(self.Bool(data=bool(on)))

    def slider_home(self):
        st = self.stepper
        st.go_home(True)                    # chặn tới khi về home, không timeout
        st.set_mode(st.EN)
        st.set_div(st.DIV_1_4)

    def slider_enable(self):
        self.stepper.set_mode(self.stepper.SLEEP)

    def slider_disable(self):
        self.stepper.set_mode(self.stepper.EN)

    def slider_step(self, d):
        self.stepper.goto(int(d))           # bước TƯƠNG ĐỐI


class FakeBackend(object):
    """Giả lập để chạy thử trên laptop: in lệnh, không chờ thật (trừ khi realtime=True)."""

    def __init__(self, cfg, use_slider, start=(0.0, -160.0, 150.0), log=print, realtime=False):
        self.p = tuple(start)
        self.log = log
        self.realtime = realtime
        self.stepper = object() if use_slider else None
        self.suck_on = False
        self.trace = []                     # lịch sử (loại, dữ liệu) để kiểm tra

    def status(self):
        return self.p

    def move(self, p, duration):
        self.trace.append(('move', tuple(p)))
        self.p = tuple(p)

    def sleep(self, t):
        if self.realtime:
            time.sleep(t)

    def sucker(self, on):
        self.trace.append(('suck', bool(on)))
        self.suck_on = bool(on)

    def slider_home(self):
        self.trace.append(('slider_home', 0))

    def slider_enable(self):
        pass

    def slider_disable(self):
        pass

    def slider_step(self, d):
        self.trace.append(('slider_step', int(d)))


# ------------------------------------------------------------------------------------------- Robot
class Robot(object):
    def __init__(self, cfg, calib=None, backend='ros', log=print, **kw):
        self.cfg, self.log = cfg, log
        sl = cfg['slider']
        self.use_slider = bool(sl.get('enabled', False))
        self.b = RosBackend(cfg, self.use_slider) if backend == 'ros' else FakeBackend(cfg, self.use_slider, log=log, **kw)
        lim = cfg['limits']
        self.xl, self.yl, self.zl = lim['x'], lim['y'], lim['z']
        self.r_min, self.r_max = float(lim['r_min']), float(lim.get('r_max', 1e9))
        sp = cfg['speed']
        self.fast, self.slow, self.settle, self.tol = sp['fast'], sp['slow'], sp['settle'], sp['tol']
        h = cfg['heights']
        self.z_safe = float(h['z_safe'])
        self.z_pick = float((calib or {}).get('z_pick', h['z_pick']))
        self.z_place = self.z_pick + float(h['place_gap'])
        self.suck_wait, self.release_wait = float(h['suck_wait']), float(h['release_wait'])
        self.s_min, self.s_max = int(sl['min']), int(sl['max'])
        self.s_speed, self.s_min_z = float(sl['speed']), float(sl['min_z'])
        self.s_step = int(sl.get('step_choice', 100))
        vec = (calib or {}).get('slider_vec', sl.get('vec', [0.0, 0.0]))
        self.vec = (float(vec[0]), float(vec[1]))
        if self.use_slider and abs(self.vec[0]) + abs(self.vec[1]) < 1e-6:
            self.log('CẢNH BÁO: chưa đo slider_vec (mm/bước): coi như thanh trượt không làm dịch toạ độ. '
                     'Chạy tools/calibrate.py --slider trước khi gắp thật.')
        self.s = 0
        self.s_enabled = False
        self.start = self.b.status()
        self.b.sucker(False)

    # ------------------------------------------------------------------ giới hạn & quy đổi
    def in_limits(self, p, margin=0.0):
        x, y, z = p
        r = math.hypot(x, y)
        return (self.xl[0] + margin <= x <= self.xl[1] - margin and self.yl[0] + margin <= y <= self.yl[1] - margin
                and self.zl[0] <= z <= self.zl[1] and self.r_min + margin <= r <= self.r_max - margin)

    def to_robot(self, wx, wy, s=None):
        s = self.s if s is None else s
        return wx - self.vec[0] * s, wy - self.vec[1] * s

    def to_world(self, rx, ry, s=None):
        s = self.s if s is None else s
        return rx + self.vec[0] * s, ry + self.vec[1] * s

    def choose_slider(self, wx, wy, margin=5.0):
        """Vị trí thanh trượt gần vị trí hiện tại nhất mà điểm (wx, wy) nằm trong vùng với ở cả z_safe và z_pick."""
        cands = [self.s] if not self.use_slider else \
            sorted(range(self.s_min, self.s_max + 1, self.s_step), key=lambda s: abs(s - self.s))
        for s in cands:
            rx, ry = self.to_robot(wx, wy, s)
            if self.in_limits((rx, ry, self.z_safe), margin) and self.in_limits((rx, ry, self.z_pick), margin):
                return s
        return None

    def reachable(self, wx, wy):
        return self.choose_slider(wx, wy) is not None

    # ----------------------------------------------------------------------- cấp thấp
    def status(self):
        return self.b.status()

    def goto(self, p, speed=None):
        p = tuple(float(v) for v in p)
        if not self.in_limits(p):
            raise MoveError('Điểm robot (%.1f, %.1f, %.1f) ngoài giới hạn an toàn' % p)
        cur = self.b.status()
        dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
        t = max(0.4, dist / (speed or self.fast))
        self.b.move(p, t)
        self.b.sleep(t + self.settle)
        now = self.b.status()
        err = max(abs(now[i] - p[i]) for i in range(3))
        if err > self.tol:
            raise MoveError('Tay máy chưa tới đích (lệch %.1f mm), có thể bị chặn bởi giới hạn động học' % err)

    def lift(self, z=None):
        z = self.z_safe if z is None else z
        x, y, cz = self.b.status()
        if cz < z - 1:
            self.goto((x, y, z), self.slow)

    def home_slider(self):
        if not self.use_slider:
            return
        if self.b.status()[2] < self.s_min_z:
            raise MoveError('Tay máy đang thấp hơn z=%d, hãy nâng lên trước khi về HOME' % self.s_min_z)
        self.b.slider_home()
        self.s, self.s_enabled = 0, False

    def slide_to(self, target):
        if not self.use_slider:
            return
        target = int(max(self.s_min, min(self.s_max, target)))
        d = target - self.s
        if d == 0:
            return
        self.lift(max(self.z_safe, self.s_min_z))
        if not self.s_enabled:
            self.b.slider_enable()
            self.s_enabled = True
        self.b.slider_step(d)
        self.b.sleep(abs(d) / self.s_speed + 0.3)
        self.s = target

    # --------------------------------------------------------------------- cấp cao (hệ thế giới)
    def move_above(self, wx, wy):
        s = self.choose_slider(wx, wy)
        if s is None:
            raise MoveError('Điểm (%.1f, %.1f) ngoài tầm với (kể cả khi trượt)' % (wx, wy))
        if s != self.s:
            self.slide_to(s)
        rx, ry = self.to_robot(wx, wy)
        self.lift()
        self.goto((rx, ry, self.z_safe))
        return rx, ry

    def pick(self, wx, wy):
        rx, ry = self.move_above(wx, wy)
        self.b.sucker(True)
        self.b.sleep(0.2)
        self.goto((rx, ry, self.z_pick), self.slow)
        self.b.sleep(self.suck_wait)
        self.goto((rx, ry, self.z_safe), self.slow)

    def place(self, wx, wy):
        rx, ry = self.move_above(wx, wy)
        self.goto((rx, ry, self.z_place), self.slow)
        self.b.sucker(False)
        self.b.sleep(self.release_wait)
        self.goto((rx, ry, self.z_safe), self.slow)

    def hover_check(self, wx, wy, dz=15.0):
        """Chỉ bay tới trên điểm, hạ xuống cách mặt quân dz mm rồi nâng lên (để kiểm tra bằng mắt, không hút)."""
        rx, ry = self.move_above(wx, wy)
        self.goto((rx, ry, self.z_pick + dz), self.slow)
        self.b.sleep(0.5)
        self.goto((rx, ry, self.z_safe), self.slow)

    def observe(self, view):
        """Đưa camera tới tư thế quan sát: nâng lên, trượt, rồi tới vị trí tay."""
        self.lift(max(self.z_safe, self.s_min_z))
        self.slide_to(int(view.get('slider', 0)))
        self.goto(tuple(view['arm']))
        self.b.sleep(0.4)

    def safe_stop(self):
        self.b.sucker(False)
        try:
            x, y, z = self.b.status()
            if z < self.z_safe:
                self.b.move((x, y, self.z_safe), 1.0)
                self.b.sleep(1.4)
        except Exception as e:
            self.log('Không nâng được tay máy (%s), kiểm tra thủ công' % e)

    def park(self):
        """Về vị trí lúc bắt đầu, thanh trượt về 0."""
        self.b.sucker(False)
        try:
            self.lift(max(self.z_safe, self.s_min_z))
            self.slide_to(0)
            self.goto(self.start)
        except Exception as e:
            self.log('Không tự về được vị trí ban đầu (%s), kiểm tra thủ công' % e)
        if self.use_slider and self.b.stepper is not None:
            self.b.slider_disable()
