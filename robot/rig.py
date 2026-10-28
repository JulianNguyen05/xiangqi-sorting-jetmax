#!/usr/bin/env python3
# HardwareRig: phần dính với robot thật, dùng chung cho scan_board.py, play_robot.py (và các công cụ sau này).
# Gồm: nạp mô hình, node ROS, camera, Mover (tay máy + thanh trượt + đầu hút), di chuyển tới một view, về vị trí chờ.
# Chỉ chạy trên robot (cần rospy, hiwonder, node jetmax_control và usb_cam); trên laptop dùng FakeWorld / --images.
# Import nặng (rospy, cv2, motion) đều làm trong __init__ nên import file này trên laptop không lỗi.

import time


class HardwareRig(object):
    TOPIC = '/usb_cam/image_rect_color'

    def __init__(self, model=None, conf=0.5, iou=0.45, z_safe=50.0, node='rig', log=print, ask=None, home=True):
        """model: đường dẫn best.pt/.onnx/.engine (None = chỉ chụp ảnh, không nhận diện).
        ask(text): hỏi người dùng trước các thao tác cần chú ý (nếu None thì không hỏi).
        home: đưa thanh trượt về HOME ngay khi khởi tạo."""
        import os
        import numpy as np
        import cv2
        import rospy
        from sensor_msgs.msg import Image
        import motion
        self.np, self.cv2, self.rospy, self.motion, self.log = np, cv2, rospy, motion, log
        self.detector = None
        if model:
            from detect_view import load_backend
            self.detector = load_backend(os.path.expanduser(model), conf, iou)        # nạp mô hình trước khi động tới robot
        rospy.init_node(node, anonymous=True)
        self._state = {'img': None, 'n': 0}
        rospy.Subscriber(self.TOPIC, Image, self._on_image, queue_size=1)
        self.mv = motion.Mover()
        self.lift = max(float(z_safe), motion.SLIDER_MIN_Z + 10.0)
        self.cur_view = '-'
        self.saver = None                      # hàm saver(tên_view, ảnh_bgr) gọi sau mỗi lần chụp, nếu có
        self.start = self.mv.start
        log('Vị trí ban đầu của tay máy: %s' % (self.start,))
        if home:
            if ask:
                ask('Đưa thanh trượt về HOME (tay máy phải đang nâng cao, không vướng)')
            self.mv.home_slider()

    # ---------------------------------------------------------------- camera
    def _on_image(self, msg):
        self._state['img'] = self.np.frombuffer(msg.data, self.np.uint8).reshape(msg.height, msg.width, 3).copy()   # RGB
        self._state['n'] += 1

    def grab(self):
        """Ảnh BGR mới nhất, đợi tới khi có khung hình mới sau khi tay máy đã dừng hẳn."""
        t0, n0 = time.time(), self._state['n']
        while self._state['n'] < n0 + 2:
            if time.time() - t0 > 3:
                raise RuntimeError('Không nhận được ảnh từ ' + self.TOPIC)
            time.sleep(0.02)
        bgr = self.cv2.cvtColor(self._state['img'], self.cv2.COLOR_RGB2BGR)       # mô hình train trên ảnh BGR
        if self.saver:
            self.saver(self.cur_view, bgr)
        return bgr

    # ---------------------------------------------------------------- di chuyển
    def raise_arm(self):
        p = self.mv.status()
        if p[2] < self.lift:
            self.mv.goto((p[0], p[1], self.lift))

    def move_to(self, view):
        """Đưa camera tới một view (có thuộc tính name, slider, arm). Chỉ trượt khi tay máy đã nâng đủ cao."""
        self.raise_arm()
        self.mv.slide_to(view.slider)
        self.mv.goto(tuple(view.arm))
        self.rospy.sleep(0.3)
        self.cur_view = view.name

    def park(self):
        """Về vị trí chờ (thanh trượt về 0, tay máy về chỗ lúc bắt đầu), tránh vướng tay người."""
        self.raise_arm()
        self.mv.slide_to(0)
        self.mv.goto(self.start)

    def cleanup(self, z_safe=50.0):
        self.mv.restore(z_safe)
