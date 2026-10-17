#!/usr/bin/env python3
# Dieu khien JetMax bang ban phim trong terminal.
import sys, select, termios, tty, math
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from jetmax_control.msg import SetJetMax, JetMax

# Gioi han an toan (mm). Day la gia tri than trong, hay kiem tra lai bang mat.
X_MIN, X_MAX = -120.0, 120.0
Y_MIN, Y_MAX = -230.0, -90.0
Z_MIN, Z_MAX = 80.0, 215.0     # ha Z_MIN tu tu sau khi biet do cao mat ban
STEPS = [5.0, 10.0, 20.0]
SPEED = 60.0                   # mm/giay

rospy.init_node('teleop_keyboard')
cmd_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
go_home = rospy.ServiceProxy('/jetmax/go_home', Empty)
rospy.sleep(1.0)


def get_pos():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return [m.x, m.y, m.z]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


target = get_pos()
step_i = 1
sucker_on = False


def move(dx, dy, dz):
    global target
    new = [clamp(target[0] + dx, X_MIN, X_MAX),
           clamp(target[1] + dy, Y_MIN, Y_MAX),
           clamp(target[2] + dz, Z_MIN, Z_MAX)]
    if new == target:
        print('Cham gioi han, khong di nua')
        return
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(new, target)))
    dur = max(0.2, dist / SPEED)
    cmd_pub.publish(SetJetMax(x=new[0], y=new[1], z=new[2], duration=dur))
    target = new
    print('target x=%.1f y=%.1f z=%.1f' % tuple(target))


HELP = """
d/a : x + / -        w/s : y + / -        r/f : z len / xuong
e   : bat/tat dau hut                     h   : ve vi tri home
1/2/3 : buoc 5/10/20 mm                   p   : in vi tri hien tai
q   : thoat (tat dau hut)
"""
print(HELP)
print('Vi tri ban dau x=%.1f y=%.1f z=%.1f, buoc = %.0f mm' % (tuple(target) + (STEPS[step_i],)))

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while not rospy.is_shutdown():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        k = sys.stdin.read(1).lower()
        s = STEPS[step_i]
        if k == 'd':
            move(s, 0, 0)
        elif k == 'a':
            move(-s, 0, 0)
        elif k == 'w':
            move(0, s, 0)
        elif k == 's':
            move(0, -s, 0)
        elif k == 'r':
            move(0, 0, s)
        elif k == 'f':
            move(0, 0, -s)
        elif k in ('1', '2', '3'):
            step_i = int(k) - 1
            print('Buoc = %.0f mm' % STEPS[step_i])
        elif k == 'e':
            sucker_on = not sucker_on
            sucker_pub.publish(Bool(data=sucker_on))
            print('Dau hut:', 'BAT' if sucker_on else 'TAT')
        elif k == 'h':
            go_home()
            target = get_pos()
            print('Da ve home x=%.1f y=%.1f z=%.1f' % tuple(target))
        elif k == 'p':
            print('Vi tri thuc x=%.1f y=%.1f z=%.1f' % tuple(get_pos()))
        elif k == 'q':
            break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    sucker_pub.publish(Bool(data=False))
    print('Da thoat, dau hut tat')