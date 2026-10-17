#!/usr/bin/env python3
# Kiểm tra camera JetMax khi đang chạy (chạy trên JetMax, cần node camera /usb_cam đang bật):
#   python3 camera_check.py            # đo 10 giây
#   python3 camera_check.py --secs 20
#
# Đo: độ phân giải, FPS thực, khoảng cách lớn nhất giữa hai khung (giật hình), độ trôi độ sáng
# (tự động phơi sáng/đèn nhấp nháy), nhiễu theo thời gian (độ lệch chuẩn từng điểm ảnh qua các khung).
# Hãy để CẢNH TĨNH (băng chuyền dừng, không có vật chuyển động) trong lúc đo để số liệu có nghĩa.

import argparse, time
import numpy as np
import rospy
from sensor_msgs.msg import Image

TOPIC = '/usb_cam/image_rect_color'
frames, stamps, means, arrival = [], [], [], []
info = {}


def on_image(msg):
    info['w'], info['h'], info['enc'] = msg.width, msg.height, msg.encoding
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    gray = img.mean(axis=2)
    means.append(float(gray.mean()))
    stamps.append(msg.header.stamp.to_sec())
    arrival.append(time.time())
    if len(frames) < 30 and len(means) % 3 == 0:
        frames.append(gray.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secs', type=float, default=10)
    a = ap.parse_args()
    rospy.init_node('camera_check')
    rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
    print('Đang đo %.0f giây từ %s (giữ cảnh tĩnh)...' % (a.secs, TOPIC))
    t0 = time.time()
    while time.time() - t0 < a.secs and not rospy.is_shutdown():
        time.sleep(0.05)
    if len(arrival) < 5:
        print('Gần như không nhận được ảnh. Kiểm tra: rostopic list | grep usb_cam')
        return

    dt = np.diff(arrival)
    fps = (len(arrival) - 1) / (arrival[-1] - arrival[0])
    print('Độ phân giải: %dx%d, mã hóa: %s' % (info['w'], info['h'], info['enc']))
    print('FPS thực nhận được: %.1f   (khoảng cách khung lớn nhất: %.0f ms)' % (fps, dt.max() * 1000))
    m = np.array(means)
    print('Độ sáng trung bình: %.1f, độ lệch chuẩn theo thời gian: %.2f, dao động max-min: %.1f' % (
        m.mean(), m.std(), m.max() - m.min()))
    if m.max() - m.min() > 6:
        print('  -> Độ sáng dao động đáng kể: có thể do tự động phơi sáng hoặc đèn nhấp nháy.')
    if len(frames) >= 5:
        tn = float(np.mean(np.std(np.stack(frames), axis=0)))
        print('Nhiễu theo thời gian (độ lệch chuẩn mỗi điểm ảnh): %.2f mức xám' % tn)
        if tn > 4:
            print('  -> Nhiễu cao: kiểm tra ánh sáng, độ lợi (gain) của camera.')
    if fps < 15:
        print('  -> FPS thấp: băng chuyền nhanh sẽ khó bám vật; cân nhắc giảm độ phân giải hoặc tắt hiển thị.')


if __name__ == '__main__':
    main()
