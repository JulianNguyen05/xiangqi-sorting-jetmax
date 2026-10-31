#!/usr/bin/env python3
# Lái tay máy bằng bàn phím, hoặc TỰ DÒ vùng với (--scan).
#
#   python3 tools/teleop.py                  # lái bằng phím. Thoát: về chỗ cũ
#   python3 tools/teleop.py --stay           # thoát: để nguyên
#   python3 tools/teleop.py --no-home        # không đưa thanh trượt về HOME lúc đầu
#   python3 tools/teleop.py --wide           # nới giới hạn an toàn khi lái tay
#
#   python3 tools/teleop.py --scan           # TỰ DÒ vùng với ở nhiều độ cao, in bảng + cách đặt bàn cờ
#   python3 tools/teleop.py --scan --save    # ... và ghi giới hạn đo được vào config/robot.local.yaml
#   python3 tools/teleop.py --scan --heights 10 40 80     # độ cao (mm, tính TỪ MẶT BÀN) muốn dò
#
# --scan: đầu tiên bạn lái đầu hút CHẠM NHẸ MẶT BÀN rồi Enter (để biết z mặt bàn, không bao giờ hạ thấp hơn).
# Dọn trống vùng trước robot (bỏ bàn cờ, quân ra chỗ khác). Tay sẽ duỗi ra xa nhất có thể. Ctrl+C để dừng.
# Thanh trượt đứng yên khi dò; trục y (vuông góc thanh trượt) là chiều cần quan tâm.
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from robot import config                                         # noqa: E402
from robot.jog import Jogger                                     # noqa: E402
from robot.motion import Robot                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--stay', action='store_true')
ap.add_argument('--no-home', action='store_true')
ap.add_argument('--wide', action='store_true', help='nới giới hạn an toàn (x ±200, y -300..200) khi lái tay')
ap.add_argument('--scan', action='store_true', help='tự dò vùng với')
ap.add_argument('--heights', type=float, nargs='+', default=[10, 25, 50, 80, 120],
                help='độ cao dò, mm TÍNH TỪ MẶT BÀN (10 = mặt trên quân cờ)')
ap.add_argument('--save', action='store_true', help='ghi giới hạn vào config/robot.local.yaml')
ap.add_argument('--step', type=float, default=10.0, help='bước dò thô (mm)')
a = ap.parse_args()

import rospy                                                     # noqa: E402
rospy.init_node('xq_teleop', anonymous=True)
cfg = config.robot_cfg()
if a.wide or a.scan:
    cfg['limits'].update({'x': [-300, 300], 'y': [-400, 400], 'z': [-200, 300], 'r_min': 0.0, 'r_max': 1e9})
    print('GIỚI HẠN NỚI RỘNG: quan sát kỹ, tay máy tự từ chối điểm ngoài tầm động học.')
rb = Robot(cfg, config.calib(cfg), backend='ros')
os.makedirs(config.path('captures'), exist_ok=True)
TOL = 3.0                       # mm: lệch hơn mức này coi như KHÔNG tới được


# ----------------------------------------------------------------------------------- dò tự động
def try_move(p, dur=None):
    """Ra lệnh tới p, chờ, đọc lại vị trí. True nếu tới đúng (lệch <= TOL)."""
    cur = rb.status()
    dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
    t = dur or max(0.35, dist / 60.0)
    rb.b.move(p, t)
    rospy.sleep(t + 0.25)
    now = rb.status()
    return max(abs(now[i] - p[i]) for i in range(3)) <= TOL


def go_safe(p):
    """Về điểm p theo kiểu: lên cao trước (nếu đang thấp hơn), rồi ngang, rồi xuống."""
    cur = rb.status()
    zt = max(cur[2], p[2])
    if cur[2] < zt - 1:
        try_move((cur[0], cur[1], zt))
    try_move((p[0], p[1], zt))
    if zt > p[2] + 1:
        try_move(p)
    return max(abs(rb.status()[i] - p[i]) for i in range(3)) <= TOL


def edge(start, d, step, max_dist, guard=None):
    """Từ điểm start (tới được), đi theo hướng đơn vị d tới khi không tới được. Tinh chỉnh 2 mm.
    guard(p) -> False thì dừng (giới hạn an toàn tự đặt). Trả về khoảng cách xa nhất tới được."""
    good = 0.0
    s = step
    while good + s <= max_dist:
        p = tuple(start[i] + d[i] * (good + s) for i in range(3))
        if guard and not guard(p):
            break
        if try_move(p):
            good += s
        elif s > 2.0:
            try_move(tuple(start[i] + d[i] * good for i in range(3)))      # lùi về điểm tốt
            s = 2.0
        else:
            break
    try_move(tuple(start[i] + d[i] * good for i in range(3)))
    return good


def find_start(z):
    for y in (-160, -140, -180, -120, -200):
        if go_safe((0.0, y, z)):
            return (0.0, float(y), z)
    return None


def scan():
    print('\n1) Lái đầu hút xuống CHẠM NHẸ MẶT BÀN (bàn trống), bấm Enter. q = thoát.')
    jog = Jogger(rb)
    z0 = rb.status()[2]
    while True:
        r = jog.loop('Hạ bằng f (F = bước x4), chạm nhẹ MẶT BÀN rồi Enter')
        if r != 'enter':
            raise KeyboardInterrupt
        z_table = rb.status()[2]
        if z_table > z0 - 60 or z_table > 120:
            print('   z = %.1f: tay gần như CHƯA hạ xuống (lúc đầu z = %.1f). Hãy hạ tới khi chạm mặt bàn.' % (z_table, z0))
            continue
        if input('   z mặt bàn = %.1f mm. Đầu hút đang CHẠM mặt bàn đúng không? (y/N) ' % z_table).strip().lower() == 'y':
            break
    print('   z mặt bàn = %.1f mm. Dò ở các độ cao (so với mặt bàn): %s' % (z_table, a.heights))
    try_move(tuple(rb.status()[:2]) + (z_table + 60,))
    rows = []
    for hz in sorted(a.heights, reverse=True):                 # từ cao xuống thấp cho an toàn
        z = z_table + hz
        st = find_start(z)
        if st is None:
            print('   z=%.0f (mặt bàn +%.0f): không tìm được điểm xuất phát, bỏ qua' % (z, hz))
            continue
        t0 = time.time()
        # trục y: ra xa (y âm hơn) và vào gần (y tăng, dừng khi bán kính < 40 mm hoặc y > -30)
        far = edge(st, (0, -1, 0), a.step, 300)
        go_safe(st)
        near = edge(st, (0, 1, 0), a.step, 200, guard=lambda p: p[1] <= -30 and math.hypot(p[0], p[1]) >= 40)
        y_far, y_near = st[1] - far, st[1] + near
        y_mid = (y_far + y_near) / 2.0
        go_safe((0.0, y_mid, z))
        xr = edge((0.0, y_mid, z), (1, 0, 0), a.step, 300)
        go_safe((0.0, y_mid, z))
        xl = edge((0.0, y_mid, z), (-1, 0, 0), a.step, 300)
        go_safe((0.0, y_mid, z))
        row = {'h': hz, 'z': z, 'y_far': y_far, 'y_near': y_near, 'band': y_near - y_far,
               'x_left': -xl, 'x_right': xr}
        rows.append(row)
        print('   +%3.0f mm (z=%6.1f): y từ %7.1f tới %7.1f  -> rộng %5.1f mm | x tại giữa dải: %6.1f .. %6.1f  (%.0fs)' % (
            hz, z, y_far, y_near, row['band'], -xl, xr, time.time() - t0))
    return z_table, rows


def report(z_table, rows):
    from board.geometry import BoardGeometry
    g = BoardGeometry.load()
    need = g.width + 2 * 5.0          # tâm cột a .. tâm cột i, cộng 5 mm dư mỗi bên
    lines = ['', '=' * 78, 'VÙNG VỚI (thanh trượt đứng yên, x = 0). Cần dải y rộng >= %.0f mm để phủ 9 cột (%.0f mm lưới).' % (
        need, g.width), 'z mặt bàn = %.1f' % z_table, '']
    lines.append('  cao trên bàn |   y xa  |  y gần  | rộng dải |  x trái .. phải  | đủ 9 cột?')
    for r in rows:
        lines.append('  %8.0f mm  | %7.1f | %7.1f | %6.1f mm | %6.1f .. %6.1f | %s' % (
            r['h'], r['y_far'], r['y_near'], r['band'], r['x_left'], r['x_right'],
            'ĐỦ' if r['band'] >= need else 'thiếu %.0f mm' % (need - r['band'])))
    pick = [r for r in rows if r['h'] <= 15]
    safe = [r for r in rows if 40 <= r['h'] <= 60]
    lines.append('')
    if pick:
        p = pick[0]
        lo = max(p['y_far'], safe[0]['y_far'] if safe else p['y_far'])      # phải tới được ở cả độ cao gắp và di chuyển
        hi = min(p['y_near'], safe[0]['y_near'] if safe else p['y_near'])
        band = hi - lo
        if band >= need:
            y_i = hi - 5.0
            y_a = y_i - g.width
            lines.append('=> ĐẶT ĐƯỢC TRÊN MẶT BÀN. Đặt bàn cờ sao cho (toạ độ robot, thanh trượt ở chỗ hiện tại):')
            lines.append('   đường cột GẦN robot ở y = %.0f mm, cột XA ở y = %.0f mm (dư %.0f mm).' % (y_i, y_a, band - need))
            lines.append('   Kiểm tra: lái đầu hút tới (0, %.0f, %.0f) phải nằm trên đường cột gần.' % (y_i, z_table + 25))
        else:
            better = [r for r in rows if r['band'] >= need]
            lines.append('=> Ở độ cao gắp trên mặt bàn, dải với chỉ %.0f mm < %.0f mm: KHÔNG phủ hết 9 cột.' % (band, need))
            if better:
                b = min(better, key=lambda r: r['h'])
                lines.append('   Ở +%.0f mm thì đủ (%.0f mm). Kê bàn cờ cao lên ~%.0f mm (hộp / sách phẳng) rồi dò lại:' % (
                    b['h'], b['band'], b['h'] - 10))
                lines.append('   python3 tools/teleop.py --scan --heights 10 25 50 80   (chạm mặt bàn CỜ đã kê cao)')
            else:
                lines.append('   Ở mọi độ cao đã dò đều thiếu. Xem phương án thu hẹp (gửi bảng này cho Claude).')
    lines.append('=' * 78)
    txt = '\n'.join(lines)
    print(txt)
    with open(config.path('captures/reach_scan.txt'), 'a') as f:
        f.write(time.strftime('\n# %Y-%m-%d %H:%M:%S\n') + txt + '\n')
    print('Đã lưu captures/reach_scan.txt')
    return pick[0] if pick else None, safe[0] if safe else None


def save_limits(z_table, pick, safe):
    import yaml
    if not pick:
        print('Không có số đo ở độ cao gắp (+10 mm), không ghi.')
        return
    rs = [r for r in (pick, safe) if r]
    m = 5.0                                                     # thu vào 5 mm cho an toàn
    lim = {'x': [round(max(r['x_left'] for r in rs) + m, 1), round(min(r['x_right'] for r in rs) - m, 1)],
           'y': [round(max(r['y_far'] for r in rs) + m, 1), round(min(r['y_near'] for r in rs) - m, 1)],
           'z': [round(z_table + 5.0, 1), 225.0]}
    lim['r_max'] = round(max(abs(lim['y'][0]), abs(lim['x'][0]), abs(lim['x'][1])) + 5.0, 1)
    p = config.path(config.LOCAL)
    cur = {}
    if os.path.exists(p):
        with open(p) as f:
            cur = yaml.safe_load(f) or {}
        os.rename(p, p + '.bak')
    cur.setdefault('limits', {}).update(lim)
    cur.setdefault('heights', {})['z_pick'] = round(z_table + 10.0 - 0.5, 1)
    config.save(config.LOCAL, cur, header='Số đo trên robot (ghi đè config/robot.yaml). '
                                          'tools/teleop.py --scan --save đã cập nhật limits, heights.z_pick.')
    print('Đã ghi limits %s, z_pick %.1f vào %s (bản cũ: .bak)' % (lim, cur['heights']['z_pick'], p))


# ----------------------------------------------------------------------------------------- main
if rb.use_slider and not a.no_home:
    input('Đưa thanh trượt về HOME. Tay máy phải đang cao và không vướng. Enter để chạy...')
    rb.home_slider()

log = open(config.path('captures/teleop_points.txt'), 'a')
try:
    if a.scan:
        input('DÒ TỰ ĐỘNG: dọn trống vùng trước robot (không để bàn cờ / quân / vật cản). Enter để bắt đầu...')
        z_table, rows = scan()
        pick, safe = report(z_table, rows)
        if a.save:
            save_limits(z_table, pick, safe)
    else:
        jog = Jogger(rb)
        while True:
            r = jog.loop('Enter = ghi điểm hiện tại, q = thoát')
            if r == 'quit':
                break
            x, y, z = rb.status()
            line = '%s  robot=(%.1f, %.1f, %.1f)  slider=%d' % (time.strftime('%H:%M:%S'), x, y, z, rb.s)
            print('  GHI: ' + line)
            log.write(line + '\n')
            log.flush()
except KeyboardInterrupt:
    print('\nDừng (Ctrl+C).')
finally:
    rb.b.sucker(False)
    log.close()
    if not a.stay:
        rb.park()
