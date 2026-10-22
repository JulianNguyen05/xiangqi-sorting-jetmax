#!/usr/bin/env python3
# Điều khiển tay máy (ROS topic) + thanh trượt (hiwonder.Stepper) + đầu hút. Tách từ tests/pick_place.py để dùng chung.
# Cần node jetmax_control đang chạy và rospy.init_node() đã được gọi trước khi tạo Mover.
# KHÔNG tạo hiwonder.JetMax() ở đây vì node jetmax_control đã giữ bus servo.

import math, time
import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0   # giống các script trước
TOL = 5.0                       # mm, sai lệch cho phép giữa đích và vị trí thực
FAST, SLOW = 100.0, 40.0        # mm/giây: di chuyển bình thường / khi hạ xuống chạm quân cờ
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0            # bước/giây, dùng ước lượng thời gian chờ (hãng ghi 1000; để thấp cho chắc)
SLIDER_MIN_Z = 40.0             # chỉ trượt khi tay máy nâng đủ cao


def in_limits(p):
    x, y, z = p
    return (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN)


class Mover:
    def __init__(self, use_slider=True):
        self.cmd = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
        self.sucker = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
        self.stepper, self.pos, self.enabled = None, 0, False
        if use_slider:
            import hiwonder
            self.stepper = hiwonder.Stepper(1)
        rospy.sleep(1.0)
        self.start = self.status()
        self.sucker.publish(Bool(data=False))

    # ---------------------------------------------------------------- tay máy
    def status(self):
        m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
        return m.x, m.y, m.z

    def goto(self, p, speed=FAST):
        if not in_limits(p):
            raise ValueError('Điểm (%.1f, %.1f, %.1f) ngoài giới hạn an toàn' % tuple(p))
        cur = self.status()
        dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
        t = max(0.5, dist / speed)
        self.cmd.publish(SetJetMax(x=p[0], y=p[1], z=p[2], duration=t))
        rospy.sleep(t + 0.3)
        now = self.status()
        err = max(abs(now[i] - p[i]) for i in range(3))
        if err > TOL:
            raise RuntimeError('Tay máy chưa tới đích (lệch %.1f mm): lệnh có thể bị chặn bởi giới hạn' % err)

    # --------------------------------------------------------------- thanh trượt
    def home_slider(self):
        if self.stepper is None:
            return
        if self.status()[2] < SLIDER_MIN_Z:
            raise RuntimeError('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
        st = self.stepper
        st.go_home(True)                         # chặn cho tới khi về home, không có timeout, canh nút nguồn
        st.set_mode(st.EN)
        st.set_div(st.DIV_1_4)
        self.pos, self.enabled = 0, False

    def slide_to(self, target):
        if self.stepper is None:
            return
        target = int(max(SLIDER_MIN, min(SLIDER_MAX, target)))
        d = target - self.pos
        if d == 0:
            return
        if self.status()[2] < SLIDER_MIN_Z:
            raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
        if not self.enabled:
            self.stepper.set_mode(self.stepper.SLEEP)   # chỉ gửi một lần
            self.enabled = True
        print('  thanh trượt %d -> %d' % (self.pos, target))
        self.stepper.goto(d)                     # bước TƯƠNG ĐỐI
        time.sleep(abs(d) / SLIDER_SPEED + 0.3)
        self.pos = target

    # ------------------------------------------------------------ thao tác cấp cao
    # sq_pos = (slider, x, y) lấy từ Board.position()
    def hover(self, sq_pos, zsafe):
        s, x, y = sq_pos
        self.slide_to(s)
        self.goto((x, y, zsafe))

    def pick(self, sq_pos, zpick, zsafe):
        self.hover(sq_pos, zsafe)
        _, x, y = sq_pos
        self.sucker.publish(Bool(data=True))
        rospy.sleep(0.3)
        self.goto((x, y, zpick), SLOW)
        rospy.sleep(0.8)                         # chờ hút bám chắc
        self.goto((x, y, zsafe), SLOW)

    def place(self, sq_pos, zplace, zsafe):
        self.hover(sq_pos, zsafe)
        _, x, y = sq_pos
        self.goto((x, y, zplace), SLOW)
        self.sucker.publish(Bool(data=False))    # thả, mở van xả khí
        rospy.sleep(1.0)
        self.goto((x, y, zsafe), SLOW)

    # ------------------------------------------------------------------ an toàn
    def safe_stop(self, zsafe):
        """Tắt hút và nâng tay máy lên z an toàn (dùng khi có lỗi giữa chừng)."""
        self.sucker.publish(Bool(data=False))
        try:
            cur = self.status()
            if cur[2] < zsafe:
                self.cmd.publish(SetJetMax(x=cur[0], y=cur[1], z=zsafe, duration=1.0))
                rospy.sleep(1.4)
        except Exception as e:
            print('Không nâng được tay máy (%s), hãy kiểm tra thủ công' % e)

    def restore(self, zsafe=50.0):
        """Về trạng thái ban đầu: tắt hút, nâng lên, thanh trượt về 0, tay máy về vị trí lúc bắt đầu."""
        self.sucker.publish(Bool(data=False))
        rospy.sleep(0.6)
        try:
            if self.stepper is not None and self.pos != 0:
                self.safe_stop(max(zsafe, SLIDER_MIN_Z))
                self.slide_to(0)
            self.cmd.publish(SetJetMax(x=self.start[0], y=self.start[1], z=self.start[2], duration=2.0))
            rospy.sleep(2.5)
        except Exception as e:
            print('Không tự về được vị trí ban đầu (%s), hãy kiểm tra thủ công' % e)
        if self.stepper is not None:
            self.stepper.set_mode(self.stepper.EN)
        print('Đã về vị trí ban đầu')
