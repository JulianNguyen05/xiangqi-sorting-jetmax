#!/usr/bin/env python3
# Lấy ảnh camera. RosCamera: robot thật (topic usb_cam). FileCamera: laptop, đọc ảnh có sẵn theo tên view.
import glob
import os
import time

import cv2
import numpy as np


class RosCamera(object):
    def __init__(self, cfg):
        import rospy
        from sensor_msgs.msg import Image
        self.rospy = rospy
        self.topic = cfg['ros']['image']
        self.rgb = bool(cfg['ros'].get('image_is_rgb', True))
        self._img, self._n = None, 0
        rospy.Subscriber(self.topic, Image, self._cb, queue_size=1)

    def _cb(self, msg):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3].copy()
        self._img = a
        self._n += 1

    def grab(self, view_name=None, fresh=3, timeout=4.0):
        """Ảnh BGR mới nhất, chụp SAU khi tay máy đã dừng (bỏ qua `fresh` khung hình đầu)."""
        t0, n0 = time.time(), self._n
        while self._n < n0 + fresh:
            if time.time() - t0 > timeout:
                raise RuntimeError('Không nhận được ảnh từ ' + self.topic + ' (usb_cam đã chạy chưa?)')
            time.sleep(0.02)
        img = self._img
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.rgb else img.copy()


class FileCamera(object):
    """Đọc ảnh từ thư mục: <dir>/<view>.jpg|png. Nếu chỉ có một ảnh thì dùng cho mọi view."""

    def __init__(self, src):
        self.src = src

    def grab(self, view_name=None, **kw):
        if os.path.isfile(self.src):
            p = self.src
        else:
            cands = glob.glob(os.path.join(self.src, '%s.*' % view_name)) if view_name else []
            if not cands:
                cands = sorted(glob.glob(os.path.join(self.src, '*.jpg')) + glob.glob(os.path.join(self.src, '*.png')))
            if not cands:
                raise RuntimeError('Không có ảnh trong ' + self.src)
            p = cands[0]
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError('Không đọc được ' + p)
        return img
