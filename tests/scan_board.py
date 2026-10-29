#!/usr/bin/env python3
# scan_board: quét các view của camera, nhận diện, gộp thành MỘT bàn cờ, in ra và so với thế cờ kỳ vọng, lưu ảnh chẩn đoán.
# Công cụ chính khi hiệu chuẩn views.yaml và đánh giá nhận diện. Không cần engine, không gắp quân, không cần board.yaml.
#
#   ROBOT THẬT (chạy từ thư mục gốc repo; node jetmax_control và usb_cam đang chạy):
#     python3 tests/scan_board.py --capture-only                         # bước 1: chụp ảnh từng view (kèm ảnh có thước pixel) để đọc toạ độ điền views.yaml
#     python3 tests/scan_board.py --model robot/models/best.onnx         # bước 2+: quét, gộp, so với thế cờ ban đầu
#     python3 tests/scan_board.py --model robot/models/best.onnx --rounds 10          # lặp 10 vòng: đo độ ổn định
#     python3 tests/scan_board.py --model robot/models/best.onnx --view trai --labels # chỉ một view, ghi tên mọi giao điểm
#     python3 tests/scan_board.py --model ... --expect "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"   # thế cờ khác
#     python3 tests/scan_board.py --model ... --expect none              # chỉ in những gì nhìn thấy, không so sánh
#
#   OFFLINE (laptop, chạy lại trên ảnh đã lưu ở lần quét trước; không cần robot):
#     python tests/scan_board.py --images runs/scan_20261030_101500 --model robot/models/best.onnx
#
#   THỬ KHÔNG CẦN ROBOT (thế giới giả lập):
#     python tests/scan_board.py --fake --rounds 3 --noise 0.03
#     python tests/scan_board.py --fake --fake-remove b0 h2              # giả vờ thiếu 2 quân để xem báo cáo lệch
#
# Kết quả lưu trong runs/scan_<thời điểm>/ (hoặc --save-dir):
#   scanNNNN_<view>.jpg      ảnh gốc từ camera (chạy lại offline được)
#   scanNNNN_<view>_det.jpg  ảnh chẩn đoán: giao điểm lưới chiếu lên ảnh (xanh = view này nhìn thấy, xám = không) và
#                            các khung nhận diện (xanh lá = đúng thế kỳ vọng, đỏ = sai, vàng = bị bỏ vì sát mép,
#                            cam = trùng ô, tím = lệch tâm ô, xám = ngoài lưới)
#   detections.csv           mọi khung phát hiện, trạng thái và ô được gán
# Nếu lưới xanh KHÔNG trùng giao điểm thật trong ảnh: các điểm dạy trong views.yaml sai hoặc robot dừng chưa lặp lại.
#
# Chú giải bàn cờ in ra:  chữ = đúng thế kỳ vọng,  X! = nhận sai quân,  X+ = quân thừa,  ? = thiếu quân (view có nhìn thấy ô đó),
#                         ~ = ô có quân nhưng không view nào nhìn thấy.

import os, sys, re, csv, glob, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'robot')]

from board import square_name, NF, NR
from xiangqi_rules import Position
from vision_multi import MultiViewMapper, _apply
import preflight

VN = {'k': 'tướng', 'a': 'sĩ', 'b': 'tượng', 'n': 'mã', 'r': 'xe', 'c': 'pháo', 'p': 'tốt'}
# BGR
COL = {'good': (0, 200, 0), 'bad': (0, 0, 255), 'edge': (0, 230, 255), 'dup': (0, 140, 255),
       'offcentre': (200, 0, 200), 'outside': (150, 150, 150), 'unknown': (255, 100, 0)}


def vn(piece):
    return '%s %s' % (VN[piece.lower()], 'đỏ' if piece.isupper() else 'đen')


class ViewScan(object):
    def __init__(self, view, frame, dets, recs, ms, stem):
        self.view, self.frame, self.dets, self.recs, self.ms, self.stem = view, frame, dets, recs, ms, stem

    def counts(self):
        c = {}
        for r in self.recs:
            c[r['status']] = c.get(r['status'], 0) + 1
        return c


class RoundResult(object):
    def __init__(self, index, scans, board, seen, warns, diff, seconds):
        self.index, self.scans, self.board, self.seen, self.warns, self.diff, self.seconds = \
            index, scans, board, seen, warns, diff, seconds

    @property
    def perfect(self):
        d = self.diff
        return d is not None and not (d['missing'] or d['wrong'] or d['extra'] or d['unseen'])


# ------------------------------------------------------------------ lõi (không cần ROS/OpenCV) ------
def scan_once(mapper, detector, move_to, grab, on_frame=None):
    """Quét một vòng qua mọi view, trả về [ViewScan]. on_frame(view, frame) -> stem tên file (lưu ảnh gốc nếu muốn)."""
    out = []
    for v in mapper.views:
        move_to(v)
        frame = grab()
        stem = on_frame(v, frame) if on_frame else None
        t0 = time.time()
        dets = detector(frame)
        ms = (time.time() - t0) * 1000.0
        out.append(ViewScan(v, frame, dets, v.explain(dets, detector.names), ms, stem))
    return out


def merge_scans(mapper, scans, names):
    return mapper.merge([(s.view, s.dets, names) for s in scans])


def compare(expected, observed, seen):
    """So bàn cờ nhìn thấy với thế kỳ vọng. Mỗi mục là danh sách ô (hoặc bộ ô, quân...)."""
    d = dict(ok=[], missing=[], wrong=[], extra=[], unseen=[])
    for cell, p in sorted(expected.items()):
        o = observed.get(cell)
        if o == p:
            d['ok'].append(cell)
        elif o is not None:
            d['wrong'].append((cell, o, p))
        elif cell not in seen:
            d['unseen'].append(cell)
        else:
            d['missing'].append((cell, p))
    for cell, o in sorted(observed.items()):
        if cell not in expected:
            d['extra'].append((cell, o))
    return d


def parse_expect(s):
    if s is None or s == 'none':
        return None
    if s == 'start':
        return Position.start().to_dict()
    return Position.from_fen(s if ' ' in s else s + ' w').to_dict()


def board_text(expected, observed, seen):
    lines = ['     ' + ' '.join(c + ' ' for c in 'abcdefghi')]
    for r in range(NR - 1, -1, -1):
        cells = []
        for f in range(NF):
            c = (f, r)
            e, o = (expected or {}).get(c), observed.get(c)
            if expected is None:
                cells.append((o or '.') + ' ')
            elif o == e:
                cells.append((o or '.') + ' ')
            elif o is not None and e is not None:
                cells.append(o + '!')
            elif o is not None:
                cells.append(o + '+')
            elif c not in seen:
                cells.append('~ ')
            else:
                cells.append('? ')
        lines.append('%d    %s' % (r, ' '.join(cells)))
        if r == 5:
            lines.append('     ~ ~ ~ ~ sông ~ ~ ~ ~')
    return '\n'.join(lines)


def diff_text(diff):
    out = []
    for cell, p in diff['missing']:
        out.append('  thiếu   %s: %s (không thấy)' % (square_name(*cell), vn(p)))
    for cell, o, p in diff['wrong']:
        out.append('  nhầm    %s: thấy %s, đúng ra là %s' % (square_name(*cell), vn(o), vn(p)))
    for cell, o in diff['extra']:
        out.append('  thừa    %s: thấy %s ở ô đáng lẽ trống' % (square_name(*cell), vn(o)))
    if diff['unseen']:
        out.append('  không view nào nhìn thấy: %s' % ' '.join(square_name(*c) for c in diff['unseen']))
    return out


def stability(results):
    """Thống kê qua nhiều vòng: ô nào lúc đúng lúc sai."""
    n = len(results)
    bad = {}
    for r in results:
        d = r.diff
        for item in d['missing'] + d['wrong'] + d['extra']:
            bad[item[0]] = bad.get(item[0], 0) + 1
        for c in d['unseen']:
            bad[c] = bad.get(c, 0) + 1
    return n, sorted(bad.items(), key=lambda kv: (-kv[1], kv[0]))


# ------------------------------------------------------------------ vẽ ảnh (cần OpenCV) ------------
def draw_view(cv2, scan, expected, labels=False):
    img = scan.frame.copy()
    v = scan.view
    h, w = img.shape[:2]
    for f in range(NF):
        for r in range(NR):
            x, y = _apply(v.Hinv, f, r)
            if not (0 <= x < w and 0 <= y < h):
                continue
            cov = v.covers(f, r)
            cv2.circle(img, (int(x), int(y)), 3, (0, 200, 0) if cov else (150, 150, 150), -1)
            if labels or (f in (0, 4, 8) and r in (0, 5, 9)):
                cv2.putText(img, square_name(f, r), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 0) if cov else (180, 180, 180), 1)
    for r in scan.recs:
        x1, y1, x2, y2 = [int(t) for t in r['box']]
        st = r['status']
        if st == 'ok':
            good = expected is None or expected.get(r['cell']) == r['piece']
            col = COL['good'] if good else COL['bad']
            label = '%s %s %.2f' % (r['piece'], square_name(*r['cell']), r['conf'])
        else:
            col = COL.get(st, COL['outside'])
            label = '%s %s %.2f' % (st, r['piece'] or '?', r['conf'])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    c = scan.counts()
    head = '%s slider=%s  %d khung, %d vao o  %.0f ms' % (v.name, v.slider, len(scan.recs), c.get('ok', 0), scan.ms)
    cv2.putText(img, head, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, head, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return img


def draw_ruler(cv2, img, step=50):
    """Phủ lưới pixel để đọc toạ độ giao điểm bằng mắt (dùng khi điền views.yaml)."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        major = x % 100 == 0
        cv2.line(out, (x, 0), (x, h), (0, 255, 255) if major else (170, 170, 170), 1)
        if major:
            cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(0, h, step):
        major = y % 100 == 0
        cv2.line(out, (0, y), (w, y), (0, 255, 255) if major else (170, 170, 170), 1)
        if major:
            cv2.putText(out, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


# ------------------------------------------------------------------ chạy và báo cáo ---------------
def run_scan(ctx, a, log=print):
    expected = parse_expect(a.expect)
    cv2 = None
    if ctx.save_dir:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            log('CẢNH BÁO: không có OpenCV, bỏ qua việc lưu ảnh chẩn đoán')
    csvf = csvw = None
    if ctx.save_dir:
        os.makedirs(ctx.save_dir, exist_ok=True)
        csvf = open(os.path.join(ctx.save_dir, 'detections.csv'), 'w', newline='', encoding='utf-8')
        csvw = csv.writer(csvf)
        csvw.writerow(['round', 'view', 'class', 'conf', 'x1', 'y1', 'x2', 'y2', 'status', 'cell', 'piece'])
    n_rounds = ctx.rounds if ctx.rounds is not None else a.rounds
    results = []
    try:
        for ri in range(n_rounds):
            if ctx.begin_round:
                ctx.begin_round(ri)
            t0 = time.time()
            scans = scan_once(ctx.mapper, ctx.detector, ctx.move_to, ctx.grab, ctx.on_frame)
            board, seen, warns = merge_scans(ctx.mapper, scans, ctx.detector.names)
            diff = compare(expected, board, seen) if expected is not None else None
            res = RoundResult(ri, scans, board, seen, warns, diff, time.time() - t0)
            results.append(res)
            report_round(res, expected, n_rounds, log)
            for s in scans:
                if csvw:
                    for r in s.recs:
                        csvw.writerow([ri + 1, s.view.name, r['name'], '%.3f' % r['conf']] +
                                      ['%.1f' % t for t in r['box']] +
                                      [r['status'], square_name(*r['cell']) if r['cell'] else '', r['piece'] or ''])
                if cv2 is not None and s.stem:
                    cv2.imwrite(os.path.join(ctx.save_dir, s.stem + '_det.jpg'), draw_view(cv2, s, expected, a.labels))
    finally:
        if csvf:
            csvf.close()
    if len(results) > 1 and expected is not None:
        report_stability(results, log)
    if ctx.save_dir:
        log('\nẢnh và detections.csv đã lưu trong %s' % ctx.save_dir)
    return results


def report_round(res, expected, n_rounds, log):
    log('\n=== Vòng %d/%d (%.1f giây) ===' % (res.index + 1, n_rounds, res.seconds))
    for s in res.scans:
        c = s.counts()
        extra = ', '.join('%d %s' % (c[k], {'edge': 'sát mép', 'dup': 'trùng ô', 'outside': 'ngoài lưới',
                                             'offcentre': 'lệch tâm', 'unknown': 'lớp lạ'}[k])
                          for k in ('edge', 'dup', 'outside', 'offcentre', 'unknown') if c.get(k))
        log('  view %-8s %2d khung, %2d vào ô%s, suy luận %.0f ms' % (
            s.view.name, len(s.recs), c.get('ok', 0), (', ' + extra) if extra else '', s.ms))
    for w in res.warns:
        log('  CẢNH BÁO: %s' % w)
    log(board_text(expected, res.board, res.seen))
    if expected is None:
        log('Nhìn thấy %d quân.' % len(res.board))
        return
    d = res.diff
    log('So với thế kỳ vọng: đúng %d/%d quân (%.1f%%)%s' % (
        len(d['ok']), len(expected), 100.0 * len(d['ok']) / max(1, len(expected)),
        '' if res.perfect else ', còn lệch:'))
    for line in diff_text(d):
        log(line)
    if res.perfect:
        log('  => KHỚP HOÀN TOÀN')


def report_stability(results, log):
    n, bad = stability(results)
    perfect = sum(1 for r in results if r.perfect)
    log('\n=== Tổng hợp %d vòng ===' % n)
    log('Khớp hoàn toàn: %d/%d vòng' % (perfect, n))
    acc = [100.0 * len(r.diff['ok']) / max(1, len(r.diff['ok']) + len(r.diff['missing']) + len(r.diff['wrong']) + len(r.diff['unseen']))
           for r in results]
    log('Tỉ lệ quân đúng: trung bình %.1f%%, thấp nhất %.1f%%' % (sum(acc) / n, min(acc)))
    if bad:
        log('Ô hay sai (số vòng sai / %d): %s' % (n, ', '.join('%s=%d' % (square_name(*c), k) for c, k in bad[:15])))
    else:
        log('Không ô nào sai trong mọi vòng.')


# ------------------------------------------------------------------ các chế độ -------------------
class Ctx(object):
    mapper = detector = move_to = grab = on_frame = begin_round = rounds = cleanup = save_dir = None


def counter_on_frame(save_dir):
    """on_frame lưu ảnh gốc scanNNNN_<view>.jpg và trả về tên gốc (stem) để đặt tên ảnh chẩn đoán."""
    state = {'n': 0}

    def on_frame(view, frame):
        stem = 'scan%04d_%s' % (state['n'], view.name)
        state['n'] += 1
        if save_dir:
            import cv2
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, stem + '.jpg'), frame)
        return stem
    return on_frame


def select_views(mapper, names, log):
    if not names:
        return mapper
    keep = [v for v in mapper.views if v.name in names]
    missing = [n for n in names if n not in [v.name for v in mapper.views]]
    if missing:
        raise SystemExit('Không có view hợp lệ tên: %s (có: %s)' % (', '.join(missing), ', '.join(v.name for v in mapper.views)))
    return MultiViewMapper(keep)


def default_save_dir(a):
    if a.no_save:
        return None
    return a.save_dir or os.path.join(ROOT, 'runs', 'scan_' + time.strftime('%Y%m%d_%H%M%S'))


def build_fake(a, log):
    import numpy as np
    from fake_world import FakeWorld
    w = FakeWorld(a.noise, a.seed)
    for sq in a.fake_remove or []:
        w.physical.pop(sq, None)
    blank = np.full((480, 640, 3), 200, np.uint8)
    ctx = Ctx()
    ctx.mapper = select_views(MultiViewMapper(w.views), a.view, log)
    ctx.detector, ctx.move_to = w, w.move_to
    ctx.grab = lambda: (w.grab(), blank.copy())[1]
    ctx.save_dir = default_save_dir(a)
    ctx.on_frame = counter_on_frame(ctx.save_dir)
    ctx.world = w
    return ctx


def load_mapper(a, log):
    rep = preflight.Report()
    mapper = preflight.check_views(a.views, preflight.get_limits(), rep)
    log(rep.text())
    if mapper is None:
        raise SystemExit('Chưa có view nào dùng được trong %s. Chạy --capture-only để chụp ảnh và đọc toạ độ điểm dạy.' % a.views)
    if rep.errors:
        log('(một số view còn lỗi và bị bỏ qua; quét với %d view hợp lệ)' % len(mapper.views))
    return select_views(mapper, a.view, log)


def build_images(a, log):
    import cv2
    from detect_view import load_backend
    if not a.model:
        raise SystemExit('Cần --model để nhận diện lại trên ảnh đã lưu')
    mapper = load_mapper(a, log)
    names = [v.name for v in mapper.views]
    rx = re.compile(r'^scan(\d+)_(.+)\.(jpg|jpeg|png)$', re.I)
    items = []
    for p in glob.glob(os.path.join(a.images, '*')):
        m = rx.match(os.path.basename(p))
        if m and not m.group(2).endswith('_det') and m.group(2) in names:
            items.append((int(m.group(1)), m.group(2), p))
    items.sort()
    rounds, cur = [], {}
    for _, view, path in items:
        if view in cur:
            cur = {}                                   # vòng dở dang, bỏ
        cur[view] = path
        if len(cur) == len(names):
            rounds.append(cur)
            cur = {}
    if not rounds:
        raise SystemExit('Không tìm thấy vòng quét đầy đủ (đủ ảnh của các view %s) trong %s' % (', '.join(names), a.images))
    log('Tìm thấy %d vòng quét trong %s' % (len(rounds), a.images))
    state = {'round': 0, 'view': None}
    ctx = Ctx()
    ctx.mapper = mapper
    ctx.detector = load_backend(os.path.expanduser(a.model), a.conf, a.iou)
    ctx.rounds = len(rounds)
    ctx.begin_round = lambda i: state.update(round=i)
    ctx.move_to = lambda v: state.update(view=v.name)

    def grab():
        p = rounds[state['round']][state['view']]
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError('Không đọc được ảnh %s' % p)
        state['path'] = p
        return img
    ctx.grab = grab
    ctx.on_frame = lambda v, f: os.path.splitext(os.path.basename(state['path']))[0]
    ctx.save_dir = None if a.no_save else (a.save_dir or os.path.join(a.images, 'annotated'))
    return ctx


def build_hardware(a, log):
    if not a.model:
        raise SystemExit('Cần --model (best.pt / best.onnx / best.engine)')
    from rig import HardwareRig
    mapper = load_mapper(a, log)

    def ask(text):
        if a.yes:
            return
        if input(text + ' [Enter = tiếp tục, q = dừng] ').strip().lower() == 'q':
            raise KeyboardInterrupt
    rig = HardwareRig(model=a.model, conf=a.conf, iou=a.iou, node='scan_board', log=log, ask=ask)
    ctx = Ctx()
    ctx.mapper, ctx.detector = mapper, rig.detector
    ctx.move_to, ctx.grab = rig.move_to, rig.grab
    ctx.save_dir = default_save_dir(a)
    ctx.on_frame = counter_on_frame(ctx.save_dir)
    ctx.cleanup = rig.cleanup
    return ctx


def load_pose_views(path, names, log):
    """Đọc các view chỉ cần slider và arm (chưa cần điểm dạy) từ views.yaml. Trả về [SimpleNamespace(name, slider, arm)]."""
    import yaml
    from types import SimpleNamespace
    with open(path, encoding='utf-8') as f:
        d = yaml.safe_load(f) or {}
    views = []
    for v in d.get('views') or []:
        if v.get('slider') is None or not v.get('arm') or any(c is None for c in v['arm']):
            log('Bỏ qua view "%s": chưa điền slider/arm' % v.get('name'))
            continue
        if names and v.get('name') not in names:
            continue
        if not preflight.get_limits()(tuple(v['arm'])):
            log('Bỏ qua view "%s": arm %s ngoài giới hạn an toàn' % (v.get('name'), v['arm']))
            continue
        views.append(SimpleNamespace(name=str(v['name']), slider=int(v['slider']), arm=tuple(v['arm'])))
    if not views:
        raise SystemExit('Chưa có view nào điền đủ slider và arm trong %s' % path)
    return views


def capture_only(a, log):
    """Chụp ảnh ở từng view trong views.yaml (chỉ cần slider và arm, chưa cần điểm dạy) để đọc toạ độ giao điểm."""
    from rig import HardwareRig
    import cv2
    views = load_pose_views(a.views, a.view, log)

    def ask(text):
        if not a.yes and input(text + ' [Enter = tiếp tục, q = dừng] ').strip().lower() == 'q':
            raise KeyboardInterrupt
    rig = HardwareRig(model=None, node='scan_board', log=log, ask=ask)
    out = default_save_dir(a) or os.path.join(ROOT, 'runs', 'capture')
    os.makedirs(out, exist_ok=True)
    try:
        for v in views:
            ask('Chụp view "%s" (slider %d, arm %s)' % (v.name, v.slider, list(v.arm)))
            rig.move_to(v)
            img = rig.grab()
            raw = os.path.join(out, 'view_%s.jpg' % v.name)
            cv2.imwrite(raw, img)
            cv2.imwrite(os.path.join(out, 'view_%s_ruler.jpg' % v.name), draw_ruler(cv2, img))
            log('  đã lưu %s (kích thước %dx%d)' % (raw, img.shape[1], img.shape[0]))
    finally:
        rig.cleanup()
    log('\nMở view_<tên>_ruler.jpg, đọc pixel (x, y) của >= 6 giao điểm rải đều (kèm tên ô, ví dụ a0, e5) rồi điền vào '
        'points của view đó trong views.yaml.\nẢnh kích thước phải đúng bằng image_size trong views.yaml. Sau đó chạy lại '
        'với --model để quét; lưới xanh trong ảnh _det.jpg phải trùng giao điểm thật.')
    return out


def parse_args(argv=None):
    cfg = os.path.join(ROOT, 'configs')
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', default=os.path.join(cfg, 'views.yaml'))
    ap.add_argument('--model', help='best.pt / best.onnx / best.engine')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--rounds', type=int, default=1, help='số vòng quét lặp lại')
    ap.add_argument('--view', nargs='+', help='chỉ quét các view này (theo tên)')
    ap.add_argument('--expect', default='start', help='start (mặc định) | none | chuỗi FEN của thế cờ kỳ vọng')
    ap.add_argument('--save-dir')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--labels', action='store_true', help='ghi tên mọi giao điểm lên ảnh chẩn đoán')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước mỗi bước')
    ap.add_argument('--capture-only', action='store_true', help='chỉ chụp ảnh từng view (kèm thước pixel), không nhận diện')
    ap.add_argument('--images', help='quét lại offline trên thư mục ảnh đã lưu (scanNNNN_<view>.jpg)')
    ap.add_argument('--fake', action='store_true', help='thế giới giả lập, không cần robot')
    ap.add_argument('--noise', type=float, default=0.0, help='(--fake) xác suất bỏ sót mỗi quân')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--fake-remove', nargs='+', help='(--fake) bỏ các quân ở những ô này để xem báo cáo lệch')
    return ap.parse_args(argv)


def main(argv=None, log=print):
    a = parse_args(argv)
    if a.capture_only:
        capture_only(a, log)
        return None
    ctx = build_fake(a, log) if a.fake else (build_images(a, log) if a.images else build_hardware(a, log))
    try:
        return run_scan(ctx, a, log)
    except KeyboardInterrupt:
        log('\nDừng theo yêu cầu')
    finally:
        if ctx.cleanup:
            ctx.cleanup()


if __name__ == '__main__':
    main()
