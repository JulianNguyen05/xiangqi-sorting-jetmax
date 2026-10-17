#!/usr/bin/env python3
# Điều khiển tay máy JetMax bằng phím: tịnh tiến x/y/z và xoay TỪNG KHỚP (joint 1/2/3),
# xoay servo cổ tay (PWM servo 1), bật/tắt đầu hút. Điều khiển qua ROS topic như test_move.py
# nên cần node jetmax_control đang chạy. Thanh trượt vẫn dùng stepper_teleop.py.
#
# Mọi bước đều là bước nhỏ, kiểm tra giới hạn trước/sau khi chạy. Bấm q để về vị trí ban đầu và thoát.

import sys, select, termios, tty
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from jetmax_control.msg import SetJetMax, SetJoint, SetServo, JetMax

DUR = 0.5                       # giây cho mỗi lần bấm
MM_STEPS = [5, 10, 20]          # mm
DEG_STEPS = [2, 5, 10]          # độ (cho khớp và servo cổ tay)
X_LIM, Y_LIM, Z_LIM = (-120, 120), (-230, -100), (-12, 225)   # giống scan_capture.py
R_MIN = 50.0                    # bán kính xy tối thiểu (jetmax.py bỏ qua lệnh nếu nhỏ hơn)
PWM_MIN, PWM_MAX = 0, 180
JOINT_TOL = 0.5                 # độ, sai lệch cho phép giữa lệnh và trạng thái

rospy.init_node('joint_teleop')
abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
joint_pubs = [rospy.Publisher('/jetmax/joint%d/command' % i, SetJoint, queue_size=1) for i in (1, 2, 3)]
pwm_pub = rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1)
sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
rospy.sleep(1.0)


def status():
    return rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)


def joints_of(m):
    return [m.joint1, m.joint2, m.joint3]


def in_limits(x, y, z):
    return (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN)


def show():
    m = status()
    print('  xyz = (%.1f, %.1f, %.1f) | joint = (%.1f, %.1f, %.1f) | cổ tay pwm1 = %.0f | hút = %s' % (
        m.x, m.y, m.z, m.joint1, m.joint2, m.joint3, m.pwm1, m.sucker))


def move_xyz(dx, dy, dz):
    m = status()
    t = (m.x + dx, m.y + dy, m.z + dz)
    if not in_limits(*t):
        print('  Chạm giới hạn an toàn, không đi: (%.1f, %.1f, %.1f)' % t)
        return
    abs_pub.publish(SetJetMax(x=t[0], y=t[1], z=t[2], duration=DUR))
    rospy.sleep(DUR + 0.2)
    show()


def move_joint(i, delta):
    m = status()
    cur = joints_of(m)[i]
    new = cur + delta
    joint_pubs[i].publish(SetJoint(data=new, duration=DUR))
    rospy.sleep(DUR + 0.2)
    m2 = status()
    if abs(joints_of(m2)[i] - new) > JOINT_TOL:
        print('  Lệnh bị từ chối (ngoài giới hạn servo/động học), khớp %d giữ nguyên' % (i + 1))
    elif not in_limits(m2.x, m2.y, m2.z):
        print('  Ra ngoài vùng an toàn, hoàn lại khớp %d' % (i + 1))
        joint_pubs[i].publish(SetJoint(data=cur, duration=DUR))
        rospy.sleep(DUR + 0.2)
    show()


def move_wrist(delta):
    cur = status().pwm1
    new = int(max(PWM_MIN, min(PWM_MAX, cur + delta)))
    pwm_pub.publish(SetServo(data=new, duration=0.3))
    rospy.sleep(0.5)
    show()


def toggle_sucker():
    sucker_pub.publish(Bool(data=not status().sucker))
    rospy.sleep(0.3)
    show()


HELP = """
Tịnh tiến (mm):   d/a = x +/-    w/s = y +/-    r/f = z +/-
Xoay khớp (độ):   u/j = joint1 +/-   i/k = joint2 +/-   o/l = joint3 +/-
Cổ tay (độ):      n/m = pwm servo1 +/-
t : đổi mức bước (mm và độ)     v : bật/tắt đầu hút
h : về home (service /jetmax/go_home)     p : in trạng thái     q : thoát (về vị trí ban đầu)
"""

start = status()
print('Vị trí ban đầu:', (start.x, start.y, start.z), 'pwm1 =', start.pwm1)
print(HELP)
level = 0
print('Bước = %d mm / %d độ' % (MM_STEPS[level], DEG_STEPS[level]))
show()

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while not rospy.is_shutdown():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        k = sys.stdin.read(1).lower()
        mm, deg = MM_STEPS[level], DEG_STEPS[level]
        if k == 'd':
            move_xyz(mm, 0, 0)
        elif k == 'a':
            move_xyz(-mm, 0, 0)
        elif k == 'w':
            move_xyz(0, mm, 0)
        elif k == 's':
            move_xyz(0, -mm, 0)
        elif k == 'r':
            move_xyz(0, 0, mm)
        elif k == 'f':
            move_xyz(0, 0, -mm)
        elif k == 'u':
            move_joint(0, deg)
        elif k == 'j':
            move_joint(0, -deg)
        elif k == 'i':
            move_joint(1, deg)
        elif k == 'k':
            move_joint(1, -deg)
        elif k == 'o':
            move_joint(2, deg)
        elif k == 'l':
            move_joint(2, -deg)
        elif k == 'n':
            move_wrist(deg)
        elif k == 'm':
            move_wrist(-deg)
        elif k == 'v':
            toggle_sucker()
        elif k == 't':
            level = (level + 1) % len(MM_STEPS)
            print('Bước = %d mm / %d độ' % (MM_STEPS[level], DEG_STEPS[level]))
        elif k == 'h':
            rospy.ServiceProxy('/jetmax/go_home', Empty)()
            rospy.sleep(2.5)
            show()
        elif k == 'p':
            show()
        elif k == 'q':
            break
        termios.tcflush(fd, termios.TCIFLUSH)   # bỏ phím bấm dồn trong lúc robot đang chạy
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    sucker_pub.publish(Bool(data=False))
    abs_pub.publish(SetJetMax(x=start.x, y=start.y, z=start.z, duration=2.0))
    pwm_pub.publish(SetServo(data=int(start.pwm1), duration=1.0))
    rospy.sleep(2.5)
    print('Đã về vị trí ban đầu')
