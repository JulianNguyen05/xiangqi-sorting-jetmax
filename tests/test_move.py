#!/usr/bin/env python3
# Chỉ đi ±20 mm theo x, y, z từ vị trí ban đầu (z đi xuống 20 mm rồi lên lại).
# Khi kết thúc hoặc bấm q, robot tự về chỗ cũ và tắt đầu hút.

import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

STEP = 20.0   # mm mỗi bước
DUR = 2.0     # giây mỗi bước

rospy.init_node('test_move')
abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
rel_pub = rospy.Publisher('/jetmax/relative_command', SetJetMax, queue_size=1)
sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
rospy.sleep(1.0)

def get_pos():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z

def ask(text):
    return input(text + ' [Enter = chạy, q = dừng] ').strip().lower() != 'q'

start = get_pos()
print('Vị trí ban đầu x,y,z =', start)

steps = [
    ('x +', (STEP, 0, 0)), ('x - (về)', (-STEP, 0, 0)),
    ('y +', (0, STEP, 0)), ('y - (về)', (0, -STEP, 0)),
    ('z - (xuống)', (0, 0, -STEP)), ('z + (về)', (0, 0, STEP)),
]
try:
    for name, (dx, dy, dz) in steps:
        if not ask('Bước: ' + name):
            break
        rel_pub.publish(SetJetMax(x=dx, y=dy, z=dz, duration=DUR))
        rospy.sleep(DUR + 0.5)
        print('  ->', get_pos())

    if ask('Test đầu hút (bật 2 giây)'):
        sucker_pub.publish(Bool(data=True))
        rospy.sleep(2)
finally:
    sucker_pub.publish(Bool(data=False))
    abs_pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
    rospy.sleep(2.5)
    print('Đã về vị trí ban đầu')