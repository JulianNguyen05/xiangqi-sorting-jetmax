#!/usr/bin/env python3
# Dieu khien thanh truot (stepper) cua JetMax bang ban phim trong terminal.
# Dua theo cach dung hiwonder.Stepper trong waste_classification_stepper.py cua hang:
#   - goto(n) di chuyen TUONG DOI n buoc (goto(-n) de quay lai), thoi gian ~ n/1000 giay
#   - vi tri 0 = diem home, vi du cua hang di toi da khoang 5200 buoc tu home
import sys, select, termios, tty, time
import rospy
import hiwonder

POS_MIN, POS_MAX = 0, 5200      # gioi han phan mem (buoc), 0 = home
STEPS = [200, 500, 1000]        # cac muc buoc
step_i = 0                      # bat dau bang buoc nho nhat

rospy.init_node('stepper_teleop')
stepper = hiwonder.Stepper(1)

print('Se dua thanh truot ve HOME (vi tri 0). Hay dam bao canh tay dang nang cao va khong vuong vat can.')
input('Bam Enter de bat dau, Ctrl+C de huy: ')
stepper.go_home(True)
stepper.set_mode(stepper.EN)
stepper.set_div(stepper.DIV_1_4)
pos = 0
print('Da o HOME, vi tri = 0')


def move(delta):
    global pos
    new = max(POS_MIN, min(POS_MAX, pos + delta))
    d = new - pos
    if d == 0:
        print('Cham gioi han, khong di nua (vi tri = %d)' % pos)
        return
    stepper.set_mode(stepper.SLEEP)
    stepper.goto(d)
    time.sleep(abs(d) / 1000.0 + 0.3)
    pos = new
    print('vi tri = %d / %d' % (pos, POS_MAX))


def go_home():
    global pos
    stepper.go_home(True)
    stepper.set_mode(stepper.EN)
    stepper.set_div(stepper.DIV_1_4)
    pos = 0
    print('Da ve HOME, vi tri = 0')


HELP = """
l : di ra xa home (+)      j : ve gan home (-)
1/2/3 : buoc 200/500/1000  h : ve home
p : in vi tri (uoc luong theo phan mem)
q : thoat (nha khoa thanh truot)
"""
print(HELP)
print('Buoc = %d' % STEPS[step_i])

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while not rospy.is_shutdown():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        k = sys.stdin.read(1).lower()
        s = STEPS[step_i]
        if k == 'l':
            move(s)
        elif k == 'j':
            move(-s)
        elif k in ('1', '2', '3'):
            step_i = int(k) - 1
            print('Buoc = %d' % STEPS[step_i])
        elif k == 'h':
            go_home()
        elif k == 'p':
            print('vi tri = %d (uoc luong)' % pos)
        elif k == 'q':
            break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    stepper.set_mode(stepper.EN)
    print('Da thoat')
