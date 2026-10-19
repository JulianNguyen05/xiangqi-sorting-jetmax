#!/usr/bin/env python3
# Điều khiển JetMax bằng phím: tịnh tiến x/y/z, xoay TỪNG KHỚP, servo cổ tay, đầu hút và THANH TRƯỢT.
# Tay máy điều khiển qua ROS topic (cần node jetmax_control đang chạy); thanh trượt qua hiwonder.Stepper.
#
#   python3 joint_teleop.py            # thoát: tay máy và thanh trượt về chỗ ban đầu
#   python3 joint_teleop.py --stay     # thoát: để nguyên tại chỗ
#
# Muốn đi XA hơn: giữ Shift (gõ chữ HOA) = bước x4, hoặc bấm t để đổi mức bước.
# Tốc độ mỗi lần bấm tự tỉ lệ theo quãng đường nên bước nhỏ không còn chờ lâu.

import sys, time, select, termios, tty, argparse
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from jetmax_control.msg import SetJetMax, SetJoint, SetServo, JetMax

MM_STEPS = [5, 10, 20, 40]               # mm
DEG_STEPS = [2, 5, 10, 20]               # độ (khớp và cổ tay)
SLIDER_STEPS = [500, 1000, 2000, 3000]   # bước thanh trượt (bước nhỏ làm động cơ dừng-chạy liên tục nên giật)
BIG = 4                                  # hệ số khi giữ Shift
SPEED_MM = 150.0                         # mm/giây
SPEED_DEG = 120.0                        # độ/giây
MIN_DUR = 0.2                            # giây, thời gian tối thiểu mỗi lệnh
X_LIM, Y_LIM, Z_LIM = (-120, 120), (-230, -100), (-12, 225)   # giống các script trước
R_MIN = 50.0                             # bán kính xy tối thiểu (jetmax.py bỏ qua lệnh nếu nhỏ hơn)
PWM_MIN, PWM_MAX = 0, 180
JOINT_TOL = 0.5                          # độ
SLIDER_MIN, SLIDER_MAX = 0, 5200         # 5200 là điểm xa nhất trong ví dụ hãng, chưa phải hành trình đo được
SLIDER_SPEED = 800.0                     # bước/giây dùng để ước lượng thời gian chờ (hãng ghi 1000; để thấp hơn cho chắc)
SLIDER_MIN_Z = 40.0                      # chỉ cho trượt khi tay máy nâng đủ cao (ví dụ hãng nâng lên z=50 trước)

ap = argparse.ArgumentParser()
ap.add_argument('--stay', action='store_true', help='khi thoát, không đưa tay máy/thanh trượt về chỗ cũ')
args = ap.parse_args()

rospy.init_node('joint_teleop')
abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
joint_pubs = [rospy.Publisher('/jetmax/joint%d/command' % i, SetJoint, queue_size=1) for i in (1, 2, 3)]
pwm_pub = rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1)
sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)

stepper = None
try:
    import hiwonder
    stepper = hiwonder.Stepper(1)
except Exception as e:
    print('Không khởi tạo được thanh trượt (%s): các phím z/x/c bị tắt.' % e)
slider_enabled = False                   # đã gửi chế độ SLEEP (giữ động cơ) chưa
slider_pos = 0                           # không đọc lại được từ phần cứng nên tự theo dõi, coi lúc mở là 0
rospy.sleep(1.0)


def status():
    return rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)


def joints_of(m):
    return [m.joint1, m.joint2, m.joint3]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def in_limits(x, y, z):
    return (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN)


def show():
    m = status()
    print('  xyz = (%.1f, %.1f, %.1f) | joint = (%.1f, %.1f, %.1f) | cổ tay = %.0f | hút = %s | trượt = %d' % (
        m.x, m.y, m.z, m.joint1, m.joint2, m.joint3, m.pwm1, m.sucker, slider_pos))


def move_xyz(dx, dy, dz):
    m = status()
    cur = (m.x, m.y, m.z)
    t = (clamp(m.x + dx, *X_LIM), clamp(m.y + dy, *Y_LIM), clamp(m.z + dz, *Z_LIM))
    if (t[0] ** 2 + t[1] ** 2) ** 0.5 < R_MIN:
        print('  Quá gần trục (bán kính < %d mm), không đi' % R_MIN)
        return
    dist = sum((t[i] - cur[i]) ** 2 for i in range(3)) ** 0.5
    if dist < 0.1:
        print('  Đã ở biên giới hạn an toàn')
        return
    if abs(dx) + abs(dy) + abs(dz) - dist > 0.5:
        print('  (cắt bớt cho vừa giới hạn an toàn)')
    dur = max(MIN_DUR, dist / SPEED_MM)
    abs_pub.publish(SetJetMax(x=t[0], y=t[1], z=t[2], duration=dur))
    rospy.sleep(dur + 0.1)
    show()


def move_joint(i, delta):
    m = status()
    cur = joints_of(m)[i]
    new = cur + delta
    dur = max(MIN_DUR, abs(delta) / SPEED_DEG)
    joint_pubs[i].publish(SetJoint(data=new, duration=dur))
    rospy.sleep(dur + 0.1)
    m2 = status()
    if abs(joints_of(m2)[i] - new) > JOINT_TOL:
        print('  Lệnh bị từ chối (ngoài giới hạn servo/động học), khớp %d giữ nguyên' % (i + 1))
    elif not in_limits(m2.x, m2.y, m2.z):
        print('  Ra ngoài vùng an toàn, hoàn lại khớp %d' % (i + 1))
        joint_pubs[i].publish(SetJoint(data=cur, duration=dur))
        rospy.sleep(dur + 0.1)
    show()


def move_wrist(delta):
    cur = status().pwm1
    new = int(clamp(cur + delta, PWM_MIN, PWM_MAX))
    pwm_pub.publish(SetServo(data=new, duration=0.2))
    rospy.sleep(0.3)
    show()


def toggle_sucker():
    sucker_pub.publish(Bool(data=not status().sucker))
    rospy.sleep(0.3)
    show()


def arm_high_enough():
    if status().z < SLIDER_MIN_Z:
        print('  Nâng tay máy lên z >= %d trước khi trượt (bấm r)' % SLIDER_MIN_Z)
        return False
    return True


def move_slider(d):
    global slider_pos, slider_enabled
    if stepper is None:
        print('  Thanh trượt không khả dụng')
        return
    if not arm_high_enough():
        return
    new = clamp(slider_pos + d, SLIDER_MIN, SLIDER_MAX)
    d = new - slider_pos
    if d == 0:
        print('  Thanh trượt ở biên (%d)' % slider_pos)
        return
    if not slider_enabled:
        stepper.set_mode(stepper.SLEEP)       # chỉ gửi một lần, không gửi trước mỗi lần bấm
        slider_enabled = True
    stepper.goto(d)
    time.sleep(abs(d) / SLIDER_SPEED + 0.2)
    slider_pos = new
    show()


def slider_home():
    global slider_pos, slider_enabled
    if stepper is None or not arm_high_enough():
        return
    print('  Thanh trượt về home (chờ đến khi xong)...')
    stepper.go_home(True)
    stepper.set_mode(stepper.EN)
    stepper.set_div(stepper.DIV_1_4)
    slider_pos = 0
    slider_enabled = False
    show()


HELP = """
Tịnh tiến (mm):  d/a = x +/-    w/s = y +/-    r/f = z +/-
Khớp (độ):       u/j = joint1   i/k = joint2   o/l = joint3   (n/m = cổ tay)
Thanh trượt:     x/z = tiến/lùi    c = về home thật (đặt lại vị trí 0)
Đi xa hơn:       gõ chữ HOA (giữ Shift) = bước x4, hoặc t để đổi mức bước
v = bật/tắt đầu hút    h = tay máy về home    p = in trạng thái    q = thoát
"""

start = status()
print('Vị trí ban đầu:', (start.x, start.y, start.z), 'pwm1 =', start.pwm1)
print(HELP)
print('Thanh trượt coi như đang ở 0 lúc mở; nếu không chắc, bấm c để về home thật.')
level = 0
print('Bước = %d mm / %d độ / %d bước trượt' % (MM_STEPS[level], DEG_STEPS[level], SLIDER_STEPS[level]))
show()

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while not rospy.is_shutdown():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        raw = sys.stdin.read(1)
        f = BIG if raw.isupper() else 1
        k = raw.lower()
        mm, deg, sl = MM_STEPS[level] * f, DEG_STEPS[level] * f, SLIDER_STEPS[level] * f
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
        elif k == 'x':
            move_slider(sl)
        elif k == 'z':
            move_slider(-sl)
        elif k == 'c':
            slider_home()
        elif k == 'v':
            toggle_sucker()
        elif k == 't':
            level = (level + 1) % len(MM_STEPS)
            print('Bước = %d mm / %d độ / %d bước trượt' % (MM_STEPS[level], DEG_STEPS[level], SLIDER_STEPS[level]))
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
    if not args.stay:
        abs_pub.publish(SetJetMax(x=start.x, y=start.y, z=start.z, duration=2.0))
        pwm_pub.publish(SetServo(data=int(start.pwm1), duration=1.0))
        rospy.sleep(2.5)
        if stepper is not None and slider_pos != 0:
            if not slider_enabled:
                stepper.set_mode(stepper.SLEEP)
            stepper.goto(-slider_pos)
            time.sleep(slider_pos / SLIDER_SPEED + 0.3)
    if stepper is not None:
        stepper.set_mode(stepper.EN)
    print('Đã thoát' + ('' if args.stay else ' (về vị trí ban đầu)'))
