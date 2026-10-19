#!/usr/bin/env python3
# Gắp ở điểm A, đặt ở điểm B bằng đầu hút, CÓ dùng thanh trượt (chưa dùng camera).
# Tay máy điều khiển qua ROS topic (cần node jetmax_control đang chạy), thanh trượt qua hiwonder.Stepper.
#
# Bước 1 (chỉ bay trên A và B, KHÔNG hạ xuống, KHÔNG hút; thanh trượt vẫn chạy thật):
#   python3 pick_place.py --a 0 -210.4 --b 0 -200.4 --zpick -8.8 --zplace -5.4 --sa 1500 --sb 5000 --hover
# Bước 2 (gắp - đặt thật, hỏi Enter trước mỗi pha):
#   python3 pick_place.py --a 0 -210.4 --b 0 -200.4 --zpick -8.8 --zplace -5.4 --sa 1500 --sb 5000
# Khi chắc chắn, thêm --yes để chạy liền không hỏi.
#
# Tọa độ A, B và số bước trượt lấy từ joint_teleop.py (bấm p). Số bước trượt là tính từ HOME của thanh trượt,
# nên script sẽ đưa thanh trượt về HOME trước khi bắt đầu (bỏ qua bằng --no-home nếu chắc chắn nó đang ở home).

import argparse, math, time
import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0   # giống các script trước
TOL = 5.0                      # mm, sai lệch cho phép giữa đích và vị trí thực
FAST, SLOW = 100.0, 40.0       # mm/giây: di chuyển bình thường / khi hạ xuống chạm quân cờ
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0           # bước/giây, dùng để ước lượng thời gian chờ (hãng ghi 1000; để thấp cho chắc)
SLIDER_MIN_Z = 40.0            # chỉ trượt khi tay máy nâng đủ cao

S = {'stepper': None, 'pos': 0, 'enabled': False}


def status():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z


def check(p):
    x, y, z = p
    if not (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN):
        raise ValueError('Điểm (%.1f, %.1f, %.1f) ngoài giới hạn an toàn' % p)


def goto(pub, p, speed=FAST):
    check(p)
    cur = status()
    dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
    t = max(0.5, dist / speed)
    pub.publish(SetJetMax(x=p[0], y=p[1], z=p[2], duration=t))
    rospy.sleep(t + 0.3)
    now = status()
    err = max(abs(now[i] - p[i]) for i in range(3))
    if err > TOL:
        raise RuntimeError('Tay máy chưa tới đích (lệch %.1f mm): lệnh có thể bị chặn bởi giới hạn' % err)


def slide_to(target):
    st = S['stepper']
    if st is None:
        raise RuntimeError('Không có thanh trượt')
    target = int(max(SLIDER_MIN, min(SLIDER_MAX, target)))
    d = target - S['pos']
    if d == 0:
        return
    if status()[2] < SLIDER_MIN_Z:
        raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
    if not S['enabled']:
        st.set_mode(st.SLEEP)                    # chỉ gửi một lần
        S['enabled'] = True
    print('  thanh trượt %d -> %d' % (S['pos'], target))
    st.goto(d)
    time.sleep(abs(d) / SLIDER_SPEED + 0.3)
    S['pos'] = target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, nargs=2, required=True, metavar=('X', 'Y'), help='điểm gắp')
    ap.add_argument('--b', type=float, nargs=2, required=True, metavar=('X', 'Y'), help='điểm đặt')
    ap.add_argument('--zpick', type=float, required=True, help='z khi đầu hút chạm quân cờ ở A')
    ap.add_argument('--zplace', type=float, default=None, help='z khi thả ở B (mặc định = zpick + 3)')
    ap.add_argument('--zsafe', type=float, default=50.0, help='z khi di chuyển ngang')
    ap.add_argument('--sa', type=int, default=0, help='vị trí thanh trượt khi gắp ở A (bước từ home)')
    ap.add_argument('--sb', type=int, default=0, help='vị trí thanh trượt khi đặt ở B (bước từ home)')
    ap.add_argument('--no-home', action='store_true', help='không đưa thanh trượt về home lúc bắt đầu')
    ap.add_argument('--hover', action='store_true', help='chỉ bay trên A và B, không hạ, không hút')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước mỗi pha')
    a = ap.parse_args()
    zplace = a.zplace if a.zplace is not None else a.zpick + 3
    A, B = a.a, a.b
    for s in (a.sa, a.sb):
        if not SLIDER_MIN <= s <= SLIDER_MAX:
            raise SystemExit('Vị trí thanh trượt %d ngoài khoảng %d..%d' % (s, SLIDER_MIN, SLIDER_MAX))

    def ask(text):
        if not a.yes and input(text + ' [Enter = chạy, q = dừng] ').strip().lower() == 'q':
            raise KeyboardInterrupt

    rospy.init_node('pick_place')
    abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    sucker = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
    need_slider = a.sa != 0 or a.sb != 0
    if need_slider:
        try:
            import hiwonder
            S['stepper'] = hiwonder.Stepper(1)
        except Exception as e:
            raise SystemExit('Không khởi tạo được thanh trượt: %s' % e)
    rospy.sleep(1.0)

    start = status()
    print('Vị trí ban đầu:', start)
    print('A=%s (trượt %d)  B=%s (trượt %d)  zpick=%.1f zplace=%.1f zsafe=%.1f %s' % (
        A, a.sa, B, a.sb, a.zpick, zplace, a.zsafe, '[CHỈ BAY, KHÔNG HẠ]' if a.hover else ''))
    for p in [(A[0], A[1], a.zsafe), (B[0], B[1], a.zsafe), (A[0], A[1], a.zpick), (B[0], B[1], zplace)]:
        check(p)                                 # kiểm tra mọi điểm trước khi di chuyển
    try:
        sucker.publish(Bool(data=False))
        if need_slider and not a.no_home:
            ask('Đưa thanh trượt về HOME (vị trí 0). Tay máy phải đang nâng cao và không vướng')
            if start[2] < SLIDER_MIN_Z:
                raise RuntimeError('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
            st = S['stepper']
            st.go_home(True)                     # chặn cho tới khi về home, không có timeout
            st.set_mode(st.EN)
            st.set_div(st.DIV_1_4)
            S['pos'], S['enabled'] = 0, False
        ask('Trượt tới %d và bay tới trên A' % a.sa)
        slide_to(a.sa)
        goto(abs_pub, (A[0], A[1], a.zsafe))
        if a.hover:
            ask('Trượt tới %d và bay tới trên B' % a.sb)
            slide_to(a.sb)
            goto(abs_pub, (B[0], B[1], a.zsafe))
        else:
            ask('Hạ xuống A và hút')
            sucker.publish(Bool(data=True))
            rospy.sleep(0.3)
            goto(abs_pub, (A[0], A[1], a.zpick), SLOW)
            rospy.sleep(0.8)                     # chờ hút bám chắc
            goto(abs_pub, (A[0], A[1], a.zsafe), SLOW)
            ask('Mang tới B (trượt tới %d)' % a.sb)
            slide_to(a.sb)
            goto(abs_pub, (B[0], B[1], a.zsafe))
            ask('Hạ xuống B và thả')
            goto(abs_pub, (B[0], B[1], zplace), SLOW)
            sucker.publish(Bool(data=False))     # thả, mở van xả khí
            rospy.sleep(1.0)
            goto(abs_pub, (B[0], B[1], a.zsafe), SLOW)
        print('Xong')
    except KeyboardInterrupt:
        print('Dừng theo yêu cầu')
    except (ValueError, RuntimeError) as e:
        print('LỖI:', e)
    finally:
        sucker.publish(Bool(data=False))
        rospy.sleep(0.6)
        try:
            cur = status()
            if S['stepper'] is not None and S['pos'] != 0 and cur[2] < SLIDER_MIN_Z:
                abs_pub.publish(SetJetMax(x=cur[0], y=cur[1], z=a.zsafe, duration=1.0))   # nâng lên trước khi trượt về
                rospy.sleep(1.4)
            if S['stepper'] is not None and S['pos'] != 0:
                slide_to(0)
            abs_pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
            rospy.sleep(2.5)
        except Exception as e:
            print('Không tự về được vị trí ban đầu (%s), hãy kiểm tra thủ công' % e)
        if S['stepper'] is not None:
            S['stepper'].set_mode(S['stepper'].EN)
        print('Đã về vị trí ban đầu')


if __name__ == '__main__':
    main()
