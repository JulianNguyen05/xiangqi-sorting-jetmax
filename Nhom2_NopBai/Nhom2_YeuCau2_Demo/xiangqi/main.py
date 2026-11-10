#!/usr/bin/env python3
# CHƯƠNG TRÌNH CHÍNH: nhìn bàn cờ + vùng chờ -> nhận diện quân -> lập kế hoạch -> gắp đặt -> kiểm tra lại.
#
# Trên robot:
#   python3 main.py                          # xếp thế khai cuộc
#   python3 main.py --fen "<FEN>"            # xếp thế cờ bất kỳ
#   python3 main.py --hover                  # CHẠY THỬ: chỉ bay tới trên quân/ô, không hạ, không hút
#   python3 main.py --max-moves 3            # chỉ làm 3 nước đầu
#   python3 main.py --scan-only              # chỉ nhìn + in kế hoạch, không di chuyển gắp đặt
# Trên laptop (không robot, robot giả lập):
#   python3 main.py --images anh.jpg --photo-calib --scan-only
#
# Kết quả mỗi lần chạy lưu ở runs/<thời gian>/ (ảnh ghép có nhãn, kế hoạch, log).
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import fen as fenmod                                       # noqa: E402
from board.geometry import BoardGeometry, square_name                 # noqa: E402
from planner import assign                                            # noqa: E402
from robot import config                                              # noqa: E402
from robot.motion import Robot, MoveError                             # noqa: E402
from vision import pieces as pc, world                                # noqa: E402


class Log(object):
    def __init__(self, path):
        self.f = open(path, 'a')

    def __call__(self, msg=''):
        print(msg)
        self.f.write(time.strftime('%H:%M:%S ') + str(msg) + '\n')
        self.f.flush()


# ------------------------------------------------------------------------------------------ nhìn
class Scene(object):
    def __init__(self):
        self.shots, self.pieces, self.board, self.mosaic, self.T = [], [], None, None, None


def scan(rb, cam, det, cfg, cal, geom, log, move=True, disc_fallback=False):
    sc = Scene()
    allp = []
    for v in cfg['views']:
        if move:
            rb.observe(v)
        img = cam.grab(v['name'])
        pose = cal.pose_for(v['arm'])
        vm = world.ViewMap(pose, int(v.get('slider', 0)), cal.slider_vec if any(cal.slider_vec) else rb.vec)
        sc.shots.append((img, vm))
        r = world.piece_radius_px(vm, img.shape, geom.piece_d * 1.04)
        if det is not None:
            ps = det.detect(img, expect_r=r)
        else:                                       # không có model: chỉ biết vị trí + màu
            ps = []
            for x, y, rr, c, cf in pc.find_discs(img, r):
                pr = np.zeros(14)
                pr[pc.RED_IDX if c == 'red' else pc.BLACK_IDX] = 1.0 / 7
                ps.append(pc.Piece(x, y, rr, pr))
        pc.apply_color(img, ps)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for p in ps:
            p.x, p.y, p.r = pc.refine_center(gray, p.x, p.y, p.r)
            p.world = tuple(vm.top(p.x, p.y))
        log('  view %s: %d quân' % (v['name'], len(ps)))
        allp += ps
    dbg = {}
    sc.board = world.locate_board(sc.shots, geom, cfg['board']['red_near_world'], dbg,
                                  mirror=bool(cfg['board'].get('mirror', False)))
    sc.mosaic, sc.T = dbg.get('mosaic'), dbg.get('T')
    if sc.board is None:
        raise RuntimeError('Không định vị được bàn cờ (xem runs/.../mosaic.jpg). Bàn có nằm trong các view không?')
    sc.pieces = world.merge_views(allp)
    world.classify_where(sc.pieces, sc.board, float(cfg['board'].get('staging_margin', 12.0)))
    changed = assign.correct_labels(sc.pieces)
    for p, old, new in changed:
        log('  sửa nhãn theo số lượng quân: %s -> %s tại (%.0f, %.0f)' % (old, new, p.world[0], p.world[1]))
    return sc


def draw_scene(sc, geom, plan=None):
    img = sc.mosaic.copy()
    T = sc.T
    for (f, r), (wx, wy) in sc.board.squares.items():
        x, y = world.apply_h(T, wx, wy)
        cv2.circle(img, (int(x), int(y)), 3, (0, 200, 0), -1)
    for i, p in enumerate(sc.pieces):
        x, y = world.apply_h(T, *p.world)
        rr = geom.piece_d / 2 * world.MOSAIC_S
        code = p.final or '?'
        col = (0, 0, 230) if code.isupper() else (200, 80, 0)
        cv2.circle(img, (int(x), int(y)), int(rr), col, 2)
        cv2.putText(img, '#%d %s' % (i + 1, code), (int(x - rr), int(y - rr - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    if plan:
        for k, m in enumerate(plan.moves):
            a = tuple(int(v) for v in world.apply_h(T, *m.pick_xy))
            b = tuple(int(v) for v in world.apply_h(T, *m.place_xy))
            cv2.arrowedLine(img, a, b, (0, 160, 255), 1, tipLength=0.03)
            cv2.putText(img, str(k + 1), b, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 2)
    return img


def piece_table(sc):
    rows = []
    for i, p in enumerate(sc.pieces):
        where = 'vùng chờ' if p.where[0] == 'staging' else 'trên bàn %s' % square_name(*p.where[1])
        rows.append('  #%-2d %-3s %-14s tin cậy %3.0f%%  màu %-5s  (%.0f, %.0f) %s' % (
            i + 1, p.final or '-', fenmod.piece_label(p.final) if p.final else 'THỪA/bỏ qua', 100 * p.conf,
            p.color or '?', p.world[0], p.world[1], where))
    return '\n'.join(rows)


# ------------------------------------------------------------------------------------- laptop
def photo_calib(img, geom, cfg):
    """Hiệu chỉnh GIẢ từ chính ảnh chụp (để chạy thử trên laptop): thế giới = hệ bàn cờ dời gốc về phía Đỏ."""
    from vision import board_detect
    h, w = img.shape[:2]
    fix = board_detect.locate(img, geom, red_hint=cfg.get('_hint') or (w / 2.0, float(h)))
    if fix is None:
        raise RuntimeError('Ảnh không thấy bàn cờ, không tạo được hiệu chỉnh giả')
    D = np.array([[1, 0, -geom.width / 2], [0, 1, 80.0], [0, 0, 1]])      # robot giả đứng giữa, cách hàng 0 80 mm
    H = D.dot(fix.Hinv)
    v = cfg['views'][0]
    return world.Calib({'poses': [{'arm': v['arm'], 'H_top': H.tolist(), 'H_table': H.tolist()}],
                        'z_pick': -8.0, 'slider_vec': [0.0, 0.0]})


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fen', default=fenmod.START)
    ap.add_argument('--hover', action='store_true', help='chỉ bay tới trên, không hạ không hút')
    ap.add_argument('--scan-only', action='store_true')
    ap.add_argument('--max-moves', type=int, default=0)
    ap.add_argument('--yes', action='store_true', help='không hỏi xác nhận')
    ap.add_argument('--no-rescan', action='store_true', help='không chụp lại trước pha 2 (cột b d f h hàng cuối)')
    ap.add_argument('--no-model', action='store_true', help='không dùng YOLO (chỉ tìm vị trí + màu, để thử)')
    ap.add_argument('--images', default=None, help='LAPTOP: ảnh / thư mục ảnh thay cho camera, robot giả lập')
    ap.add_argument('--photo-calib', action='store_true', help='LAPTOP: tạo hiệu chỉnh giả từ ảnh')
    ap.add_argument('--hint', type=float, nargs=2, default=None, help='LAPTOP: điểm ảnh gần phía Đỏ')
    ap.add_argument('--no-home', action='store_true')
    a = ap.parse_args()

    cfg = config.robot_cfg()
    geom = BoardGeometry.load()
    target = fenmod.parse(a.fen)
    run_dir = config.path(os.path.join('runs', time.strftime('%Y%m%d_%H%M%S')))
    os.makedirs(run_dir)
    log = Log(os.path.join(run_dir, 'log.txt'))
    log('Thế cờ đích: %s (%d quân)' % (a.fen.split()[0], len(target)))

    laptop = a.images is not None
    if laptop:
        from robot.camera import FileCamera
        cam = FileCamera(a.images)
        cfg['slider']['enabled'] = False
        cfg['limits'].update({'x': [-1e4, 1e4], 'y': [-1e4, 1e4], 'z': [-100, 400], 'r_min': 0.0, 'r_max': 1e9})
        if a.hint:
            cfg['_hint'] = tuple(a.hint)
        cal = photo_calib(cam.grab(cfg['views'][0]['name']), geom, cfg) if a.photo_calib else world.Calib(config.calib(cfg))
        rb = Robot(cfg, cal.data, backend='fake', log=log)
    else:
        import rospy
        from robot.camera import RosCamera
        rospy.init_node('xq_main', anonymous=True)
        cal = world.Calib(config.calib(cfg))
        missing = cal.ready_for(cfg['views'])
        if missing:
            sys.exit('Chưa hiệu chỉnh camera cho view: %s. Chạy tools/calibrate.py (RUNBOOK mục 4).' % missing)
        cam = RosCamera(cfg)
        rb = Robot(cfg, cal.data, backend='ros', log=log)

    det = None
    if not a.no_model:
        d = cfg['detector']
        det = pc.PieceDetector(config.path(d['model']), d['imgsz'], d['conf'], d['iou'], d['tta_rotations'])
        log('Model: %s (%s)' % (d['model'], det.net.name))

    try:
        if rb.use_slider and not a.no_home:
            if not a.yes:
                input('Đưa thanh trượt về HOME (tay máy cao, không vướng). Enter...')
            rb.home_slider()
        # ---------------------------------------------------------------- nhìn + lập kế hoạch
        while True:
            log('Quan sát...')
            sc = scan(rb, cam, det, cfg, cal, geom, log, move=not laptop)
            P = assign.plan(target, sc.board, sc.pieces, reachable=None if laptop else rb.reachable)
            log(piece_table(sc))
            log(assign.describe(P))
            cv2.imwrite(os.path.join(run_dir, 'scene.jpg'), draw_scene(sc, geom, P))
            for i, (img, vm) in enumerate(sc.shots):
                cv2.imwrite(os.path.join(run_dir, 'view%d.jpg' % i), img)
            log('Ảnh tổng hợp: %s' % os.path.join(run_dir, 'scene.jpg'))
            if a.scan_only:
                return
            if a.yes:
                if not P.ok:
                    log('Kế hoạch có vấn đề (xem trên), dừng.')
                    return
                break
            ans = input('\nEnter = CHẠY | "#5=N" sửa nhãn quân #5 thành N (mã FEN) | r = nhìn lại | q = thoát: ').strip()
            if ans == '':
                if not P.ok and input('Kế hoạch còn vấn đề, vẫn chạy phần làm được? (y/N) ').strip().lower() != 'y':
                    continue
                break
            if ans.lower() == 'q':
                return
            if ans.lower() == 'r':
                continue
            if ans.startswith('#') and '=' in ans:
                try:
                    i, code = ans[1:].split('=')
                    sc.pieces[int(i) - 1].final = code.strip()
                    P = assign.plan(target, sc.board, sc.pieces, reachable=None if laptop else rb.reachable)
                    log(assign.describe(P))
                except Exception as e:
                    log('Không hiểu lệnh: %s' % e)
                # vòng sau sẽ nhìn lại; muốn giữ nhãn đã sửa thì dùng kế hoạch hiện tại:
                if input('Chạy với kế hoạch vừa sửa? (Y/n) ').strip().lower() != 'n':
                    break
        # ---------------------------------------------------------------- thực thi
        moves = P.moves[:a.max_moves] if a.max_moves else P.moves
        phase1 = [m for m in moves if m.phase == 1]
        done = 0
        for k, m in enumerate(moves):
            if m.phase == 1 and phase1 and m is phase1[0] and not a.no_rescan and not a.hover and not laptop:
                log('Chụp lại để căn quân hàng cuối vào giữa 2 quân đã đặt...')
                refine_targets(rb, cam, det, cfg, cal, geom, sc.board, phase1, log)
            log('[%d/%d] %s' % (k + 1, len(moves), m))
            if a.hover:
                rb.hover_check(*m.pick_xy)
                rb.hover_check(*m.place_xy)
            else:
                rb.pick(*m.pick_xy)
                rb.place(*m.place_xy)
            done += 1
        log('Xong %d nước.' % done)
        # ---------------------------------------------------------------- kiểm tra lại
        if not a.hover and done and not laptop:
            log('Kiểm tra lại bàn cờ...')
            sc2 = scan(rb, cam, det, cfg, cal, geom, log)
            verify(sc2, target, log)
            cv2.imwrite(os.path.join(run_dir, 'after.jpg'), draw_scene(sc2, geom))
    except (MoveError, KeyboardInterrupt) as e:
        log('DỪNG: %s' % (e or 'Ctrl+C'))
        rb.safe_stop()
    finally:
        rb.park()


def refine_targets(rb, cam, det, cfg, cal, geom, board, phase1, log):
    """Pha 2: ô ở giữa 2 quân đã đặt -> đặt vào trung điểm tâm ĐO ĐƯỢC của 2 quân đó (khe chỉ 2.5 mm)."""
    sc = scan(rb, cam, det, cfg, cal, geom, log)
    placed = {p.where[1]: p for p in sc.pieces if p.where[0] == 'board'}
    for m in phase1:
        f, r = m.square
        a, b = placed.get((f - 1, r)), placed.get((f + 1, r))
        if a is not None and b is not None:
            mid = ((a.world[0] + b.world[0]) / 2.0, (a.world[1] + b.world[1]) / 2.0)
            d = np.hypot(mid[0] - m.place_xy[0], mid[1] - m.place_xy[1])
            if d < 4.0:
                m.place_xy = mid
                log('  %s: dời đích %.1f mm theo 2 quân bên cạnh' % (square_name(f, r), d))


def verify(sc, target, log):
    on = {p.where[1]: p for p in sc.pieces if p.where[0] == 'board'}
    wrong, miss, offs = [], [], []
    for sq, c in target.items():
        p = on.get(sq)
        if p is None:
            miss.append(square_name(*sq))
        elif p.final != c:
            wrong.append('%s(%s thay vì %s)' % (square_name(*sq), p.final, c))
        else:
            offs.append(p.where[2])
    log('KIỂM TRA: đúng %d/%d ô. Thiếu: %s. Sai quân: %s. Lệch tâm TB %.1f mm, max %.1f mm' % (
        len(offs), len(target), ', '.join(miss) or 'không', ', '.join(wrong) or 'không',
        np.mean(offs) if offs else 0, np.max(offs) if offs else 0))


if __name__ == '__main__':
    main()
