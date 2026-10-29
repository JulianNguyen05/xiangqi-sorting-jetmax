#!/usr/bin/env python3
# check_view_repeat: đo ĐỘ LẶP LẠI của các view. Robot đi qua các view nhiều vòng (thanh trượt chạy qua lại như khi chơi),
# mỗi vòng chụp một ảnh ở mỗi view; sau đó đo ảnh dịch bao nhiêu pixel giữa các lần quay lại cùng một view
# (tương quan pha, độ chính xác khoảng 0,2 px). Không cần mô hình nhận diện. Hãy xếp sẵn quân trên bàn (càng nhiều chi tiết
# trong ảnh càng đo chính xác), KHÔNG ai động vào bàn và ánh sáng không đổi trong lúc đo.
#
# Vì sao quan trọng: điểm dạy trong views.yaml chỉ đúng nếu mỗi lần quay lại view, ảnh nằm đúng chỗ cũ. Thanh trượt chạy
# vòng hở (không có cảm biến vị trí), nên mất bước hoặc rơ cơ khí sẽ làm lệch ảnh và gán sai ô cờ.
#
#   ROBOT THẬT (chạy từ thư mục gốc repo; jetmax_control và usb_cam đang chạy; views.yaml đã điền slider và arm):
#     python3 tests/check_view_repeat.py                      # 10 vòng qua tất cả view
#     python3 tests/check_view_repeat.py --rounds 20 --park   # về vị trí chờ giữa các vòng (giống lúc chơi, sau mỗi nước robot)
#     python3 tests/check_view_repeat.py --settle 1.0         # chờ thêm 1 giây sau khi tới view rồi mới chụp (thử xem có phải do rung)
#     python3 tests/check_view_repeat.py --view trai          # chỉ một view
#
#   OFFLINE (phân tích ảnh đã lưu, kể cả ảnh do scan_board.py --rounds N lưu):
#     python tests/check_view_repeat.py --images runs/repeat_20261030_101500
#
#   THỬ KHÔNG CẦN ROBOT:
#     python tests/check_view_repeat.py --fake --fake-jitter 1.0 --rounds 12
#     python tests/check_view_repeat.py --fake --fake-jitter 0.3 --fake-drift 0.6   # giả lập trôi dần
#
# Kết quả mỗi view: độ lệch lớn nhất quanh vị trí trung bình (pixel; và quy ra ô cờ nếu views.yaml đã có điểm dạy), độ lệch
# hiệu dụng, độ trôi dần, độ biến dạng giữa các góc ảnh, kèm kết luận OK / CẢNH BÁO / LỖI. Lưu ảnh gốc
# (scanNNNN_<view>.jpg) và repeat.csv vào runs/repeat_<thời điểm>/.
#
# Ngưỡng mặc định: OK khi lệch <= 3 px (hoặc <= 0,10 ô nếu quy ra được), LỖI khi > 2 lần ngưỡng.

import os, sys, re, csv, glob, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'robot')]

import numpy as np
import scan_board as sb
import preflight
from vision_multi import _apply

MIN_RESPONSE = 0.1          # tương quan pha thấp hơn mức này: ảnh quá khác nhau (có thứ gì đó di chuyển?) nên số đo không tin cậy


# ------------------------------------------------------------------ đo độ dịch -------------------
def to_gray(cv2, img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def phase_shift(cv2, ref, img, window=None):
    """Nội dung ảnh `img` dịch bao nhiêu so với `ref`: (dx, dy, độ tin cậy). dx > 0 là nội dung dịch sang phải."""
    h, w = ref.shape
    if window is None:
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref.astype(np.float32), img.astype(np.float32), window)
    return float(dx), float(dy), float(resp)


def frame_shifts(cv2, frames):
    """Độ dịch của từng ảnh so với ảnh đầu: [dict(dx, dy, resp, patch)].
    patch: chênh lớn nhất giữa độ dịch của 4 góc ảnh và độ dịch chung (khác 0 nghĩa là ảnh bị xoay hoặc đổi tỉ lệ)."""
    ref = frames[0]
    h, w = ref.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    hh, hw = h // 2, w // 2
    pwin = cv2.createHanningWindow((hw, hh), cv2.CV_32F)
    quads = [(0, 0), (0, hw), (hh, 0), (hh, hw)]
    out = []
    for img in frames:
        dx, dy, resp = phase_shift(cv2, ref, img, win)
        patch = 0.0
        for y0, x0 in quads:
            pdx, pdy, presp = phase_shift(cv2, ref[y0:y0 + hh, x0:x0 + hw], img[y0:y0 + hh, x0:x0 + hw], pwin)
            if presp >= MIN_RESPONSE:
                patch = max(patch, float(np.hypot(pdx - dx, pdy - dy)))
        out.append(dict(dx=dx, dy=dy, resp=resp, patch=patch))
    return out


def cell_error(view, dx, dy):
    """Độ dịch (dx, dy) pixel tại tâm ảnh tương đương bao nhiêu ô cờ (dùng phép biến đổi đã dạy của view)."""
    cx, cy = view.w / 2.0, view.h / 2.0
    f0, r0 = _apply(view.H, cx, cy)
    f1, r1 = _apply(view.H, cx + dx, cy + dy)
    return float(np.hypot(f1 - f0, r1 - r0))


def summarize(name, shifts, view=None):
    n = len(shifts)
    s = np.array([[t['dx'], t['dy']] for t in shifts])
    mean = s.mean(axis=0)
    dev = s - mean
    mag = np.hypot(dev[:, 0], dev[:, 1])
    drift = 0.0
    if n >= 4:                                              # hồi quy tuyến tính theo số thứ tự vòng
        k = np.arange(n, dtype=np.float64)
        sx, sy = np.polyfit(k, s[:, 0], 1)[0], np.polyfit(k, s[:, 1], 1)[0]
        drift = float(np.hypot(sx, sy) * (n - 1))
    res = dict(view=name, n=n, mean=(float(mean[0]), float(mean[1])), max_dev=float(mag.max()),
               rms=float(np.sqrt((mag ** 2).mean())), drift=drift,
               first_max=float(np.hypot(s[:, 0], s[:, 1]).max()), resp_min=min(t['resp'] for t in shifts),
               patch=max(t['patch'] for t in shifts), cells=None)
    if view is not None:
        res['cells'] = max(cell_error(view, d[0], d[1]) for d in dev)
    return res


def verdict(r, tol_px=3.0, tol_cell=0.10):
    """('OK' | 'CẢNH BÁO' | 'LỖI', lý do)."""
    if r['resp_min'] < MIN_RESPONSE:
        return 'LỖI', 'ảnh giữa các vòng quá khác nhau (độ tin cậy %.2f), có vật di chuyển hoặc ánh sáng đổi' % r['resp_min']
    if r['cells'] is not None:
        value, tol, unit = r['cells'], tol_cell, 'ô'
    else:
        value, tol, unit = r['max_dev'], tol_px, 'px'
    why = 'lệch lớn nhất %.2f %s' % (value, unit)
    if value > 2 * tol:
        return 'LỖI', why + ' (> %.2f)' % (2 * tol)
    notes = []
    if value > tol:
        notes.append(why + ' (> %.2f)' % tol)
    if r['drift'] > max(2.0, 2 * r['rms']):
        notes.append('trôi dần %.1f px qua các vòng' % r['drift'])
    if notes:
        return 'CẢNH BÁO', '; '.join(notes)
    return 'OK', why


# ------------------------------------------------------------------ báo cáo -----------------------
HINTS = [
    'Chạy lại với --settle 1.0: nếu lệch giảm hẳn thì tay máy/thanh trượt chưa dừng hẳn khi chụp, hãy tăng thời gian chờ.',
    'So sánh có và không có --park: nếu chỉ lệch khi không park, hướng tiếp cận thanh trượt (tới từ phía nào) đang ảnh hưởng.',
    'Lệch tăng dần (trôi) thường do thanh trượt mất bước: giảm tốc độ SLIDER_SPEED/thêm thời gian chờ trong motion.py, kiểm tra dây đai và vặn chặt.',
    'Kiểm tra camera và giá đỡ có bị rơ không (lắc nhẹ xem ảnh có dịch); chỉ có biến dạng giữa các góc ảnh thì nghi camera lắc.',
]


def report(summaries, tol_px, tol_cell, log):
    worst = 'OK'
    rank = {'OK': 0, 'CẢNH BÁO': 1, 'LỖI': 2}
    for r in summaries:
        v, why = verdict(r, tol_px, tol_cell)
        if rank[v] > rank[worst]:
            worst = v
        cells = ' = %.2f ô' % r['cells'] if r['cells'] is not None else ''
        log('  view %-8s %2d lần | lệch lớn nhất %.2f px%s, hiệu dụng %.2f px | trôi %.1f px | biến dạng %.2f px | độ tin cậy %.2f | %s: %s' % (
            r['view'], r['n'], r['max_dev'], cells, r['rms'], r['drift'], r['patch'], r['resp_min'], v, why))
    log('')
    if worst == 'OK':
        log('KẾT LUẬN: độ lặp lại đạt. Điểm dạy trong views.yaml dùng ổn định được.')
    else:
        log('KẾT LUẬN: %s. Gợi ý:' % worst)
        for h in HINTS:
            log('  - ' + h)
    return worst


def save_csv(path, per_view_shifts):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['view', 'round', 'dx_px', 'dy_px', 'response', 'patch_px'])
        for name, shifts in per_view_shifts.items():
            for i, t in enumerate(shifts):
                w.writerow([name, i + 1, '%.3f' % t['dx'], '%.3f' % t['dy'], '%.3f' % t['resp'], '%.3f' % t['patch']])


# ------------------------------------------------------------------ thu ảnh -----------------------
def synthetic_texture(cv2, seed, w=640, h=480):
    """Ảnh giống bàn cờ (đường kẻ + vòng tròn ngẫu nhiên) để thử khi chưa có robot."""
    rng = np.random.RandomState(seed)
    base = np.full((h, w), 200, np.uint8)
    for i in range(70):
        cv2.circle(base, (int(rng.randint(20, w - 20)), int(rng.randint(20, h - 20))), int(rng.randint(5, 18)),
                   int(rng.randint(30, 120)), 2)
    for x in range(50, w, 100):
        cv2.line(base, (x, 0), (x, h), 60, 1)
    for y in range(40, h, 50):
        cv2.line(base, (0, y), (w, y), 60, 1)
    return cv2.GaussianBlur(base, (3, 3), 0)


class FakeCamera(object):
    """Mỗi lần chụp: ảnh gốc của view dịch ngẫu nhiên N(0, jitter) cộng trôi dần drift (px/vòng), thêm nhiễu cảm biến."""

    def __init__(self, cv2, names, jitter, drift, seed):
        self.cv2, self.jitter, self.drift = cv2, jitter, drift
        self.rng = np.random.RandomState(seed)
        self.base = dict((n, synthetic_texture(cv2, 100 + i)) for i, n in enumerate(names))
        self.visits = dict((n, 0) for n in names)

    def grab(self, name):
        k = self.visits[name]
        self.visits[name] += 1
        sx = self.rng.normal(0, self.jitter) + self.drift * k
        sy = self.rng.normal(0, self.jitter) + 0.5 * self.drift * k
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        img = self.cv2.warpAffine(self.base[name], M, (640, 480), borderMode=self.cv2.BORDER_REFLECT)
        img = img.astype(np.float32) + self.rng.normal(0, 2.0, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return self.cv2.cvtColor(img, self.cv2.COLOR_GRAY2BGR)


def collect(a, log):
    """Trả về (frames: {view: [ảnh xám]}, thứ tự view, ctx_dọn_dẹp, thư mục lưu, mapper|None)."""
    import cv2
    save_dir = None if a.no_save else (a.save_dir or os.path.join(ROOT, 'runs', 'repeat_' + time.strftime('%Y%m%d_%H%M%S')))
    mapper = None
    if a.images:
        frames, order, _ = load_images(cv2, a, log)
        return frames, order, None, None
    if a.fake:
        from fake_world import FakeWorld
        w = FakeWorld(seed=a.seed)
        names = [v.name for v in w.views if not a.view or v.name in a.view]
        cam = FakeCamera(cv2, names, a.fake_jitter, a.fake_drift, a.seed or 0)
        cur = {'name': None}
        views = [v for v in w.views if v.name in names]
        move_to = lambda v: cur.update(name=v.name)
        grab = lambda: cam.grab(cur['name'])
        park = lambda: None
        cleanup = None
        mapper = dict((v.name, v) for v in views)
    else:
        from rig import HardwareRig
        views = sb.load_pose_views(a.views, a.view, log)

        def ask(text):
            if not a.yes and input(text + ' [Enter = tiếp tục, q = dừng] ').strip().lower() == 'q':
                raise KeyboardInterrupt
        rig = HardwareRig(model=None, node='check_view_repeat', log=log, ask=ask)
        move_to, grab, park, cleanup = rig.move_to, rig.grab, rig.park, rig.cleanup
    on_frame = sb.counter_on_frame(save_dir)
    frames = dict((v.name, []) for v in views)
    try:
        for r in range(a.rounds):
            log('Vòng %d/%d' % (r + 1, a.rounds))
            for v in views:
                move_to(v)
                if a.settle:
                    time.sleep(a.settle)
                img = grab()
                on_frame(v, img)
                frames[v.name].append(to_gray(cv2, img))
            if a.park and r < a.rounds - 1:
                park()
    finally:
        if cleanup:
            cleanup()
    return frames, [v.name for v in views], save_dir, mapper


def load_images(cv2, a, log):
    rx = re.compile(r'^scan(\d+)_(.+)\.(jpg|jpeg|png)$', re.I)
    items = []
    for p in glob.glob(os.path.join(a.images, '*')):
        m = rx.match(os.path.basename(p))
        if m and not m.group(2).endswith('_det') and (not a.view or m.group(2) in a.view):
            items.append((int(m.group(1)), m.group(2), p))
    items.sort()
    frames, order = {}, []
    for _, name, p in items:
        img = cv2.imread(p)
        if img is None:
            raise SystemExit('Không đọc được ảnh %s' % p)
        if name not in frames:
            frames[name] = []
            order.append(name)
        frames[name].append(to_gray(cv2, img))
    if not frames:
        raise SystemExit('Không thấy ảnh scanNNNN_<view>.jpg trong %s' % a.images)
    log('Đọc %d ảnh của %d view từ %s' % (len(items), len(frames), a.images))
    return frames, order, None


def load_view_models(a):
    """Phép biến đổi đã dạy của từng view (nếu views.yaml đã có điểm dạy) để quy pixel ra ô cờ."""
    try:
        rep = preflight.Report()
        m = preflight.check_views(a.views, preflight.get_limits(), rep)
        return dict((v.name, v) for v in m.views) if m is not None else {}
    except Exception:
        return {}


def analyze(frames, order, mapper, a, log):
    import cv2
    summaries, shifts = [], {}
    for name in order:
        fr = frames[name]
        if len(fr) < 3:
            log('  view %s: chỉ có %d ảnh, cần >= 3 để đo' % (name, len(fr)))
            continue
        shifts[name] = frame_shifts(cv2, fr)
        summaries.append(summarize(name, shifts[name], (mapper or {}).get(name)))
    return summaries, shifts


def main(argv=None, log=print):
    a = parse_args(argv)
    frames, order, save_dir, mapper = collect(a, log)
    if mapper is None:
        mapper = load_view_models(a)                        # nếu views.yaml đã có điểm dạy thì quy pixel ra ô cờ
    log('\n=== Độ lặp lại của các view (%d ảnh mỗi view) ===' % max(len(f) for f in frames.values()))
    summaries, shifts = analyze(frames, order, mapper, a, log)
    if not summaries:
        raise SystemExit('Không đủ ảnh để đo')
    worst = report(summaries, a.tol_px, a.tol_cell, log)
    if save_dir and not a.images:
        save_csv(os.path.join(save_dir, 'repeat.csv'), shifts)
        log('\nẢnh gốc và repeat.csv đã lưu trong %s' % save_dir)
    return dict(summaries=summaries, shifts=shifts, worst=worst)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', default=os.path.join(ROOT, 'configs', 'views.yaml'))
    ap.add_argument('--rounds', type=int, default=10, help='số vòng đi qua các view (>= 3)')
    ap.add_argument('--view', nargs='+', help='chỉ đo các view này (theo tên)')
    ap.add_argument('--park', action='store_true', help='về vị trí chờ giữa các vòng (giống lúc chơi)')
    ap.add_argument('--settle', type=float, default=0.0, help='chờ thêm bấy nhiêu giây sau khi tới view rồi mới chụp')
    ap.add_argument('--tol-px', type=float, default=3.0, help='ngưỡng OK (pixel) khi không quy ra được ô')
    ap.add_argument('--tol-cell', type=float, default=0.10, help='ngưỡng OK (ô cờ) khi views.yaml đã có điểm dạy')
    ap.add_argument('--save-dir')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước các bước')
    ap.add_argument('--images', help='phân tích offline thư mục ảnh scanNNNN_<view>.jpg đã lưu')
    ap.add_argument('--fake', action='store_true', help='camera giả lập, không cần robot')
    ap.add_argument('--fake-jitter', type=float, default=1.0, help='(--fake) độ lệch chuẩn của độ dịch ngẫu nhiên, pixel')
    ap.add_argument('--fake-drift', type=float, default=0.0, help='(--fake) độ trôi dần mỗi vòng, pixel')
    ap.add_argument('--seed', type=int)
    a = ap.parse_args(argv)
    if a.rounds < 3 and not a.images:
        ap.error('--rounds phải >= 3')
    return a


if __name__ == '__main__':
    main()
