#!/usr/bin/env python3
# Chạy ván cờ người - robot: engine (Pikafish) + nhận diện nhiều góc nhìn + điều khiển JetMax.
#
#   LAPTOP, chưa có robot (toàn bộ ván chạy trên thế giới giả lập, dùng đúng code thật):
#     python tests/play_robot.py --fake --human red                       # bạn gõ nước đi (h2e2, ? = gợi ý)
#     python tests/play_robot.py --fake --human black --auto --noise 0.02 --frames 3
#     python tests/play_robot.py --fake --human red --engine Pikafish/Pikafish-Windows-x86-64-universal.exe --nnue Pikafish/pikafish.nnue
#
#   ROBOT THẬT (node jetmax_control và usb_cam đang chạy; chạy từ thư mục gốc repo):
#     python3 robot/preflight.py --model ~/models/best.onnx              # LUÔN chạy trước, phải hết LỖI
#     python3 tests/play_robot.py --hover --model ~/models/best.onnx     # lần đầu: chỉ bay trên các ô, KHÔNG gắp/hút
#     python3 tests/play_robot.py --human red --model ~/models/best.onnx --engine ~/pikafish/pikafish --nnue ~/pikafish/pikafish.nnue
#
# Quy trình một ván trên robot thật:
#   1. Preflight tự chạy; còn LỖI thì dừng, chưa động tới tay máy.
#   2. Bạn xếp 32 quân đúng thế ban đầu, nhấn Enter; robot quét và đối chiếu (lệch thì chỉ ô nào, hỏi quét lại).
#   3. Mỗi lượt của bạn: bạn đi quân, TAY RA KHỎI BÀN, nhấn Enter. Robot chỉ quét sau khi bạn nhấn Enter,
#      nên tay máy không bao giờ chuyển động khi tay bạn còn trên bàn cờ. Robot đi xong sẽ về vị trí chờ.
#   4. Quân bạn ăn thì bạn tự bỏ ra ngoài. Quân robot ăn được robot bỏ vào khay.
#   5. Robot lỗi giữa chừng: tay máy tắt hút, nâng lên và dừng; chương trình in thế cờ đúng, bạn xếp lại bàn cờ cho
#      khớp rồi nhấn Enter để robot thử lại.
# Ảnh gốc mỗi lần quét được lưu ở runs/<thời điểm>/ để chạy lại offline và để sửa lỗi nhận diện.
#
# --hover: dùng để thử đường đi trên robot thật mà không gắp. Vì quân không được di chuyển thật, bạn phải tự dời quân
#          của robot thay nó (theo log) thì lượt quét kế tiếp mới khớp.

import os, sys, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'robot')]

from engine import create_engine, LEVELS
from game import Game, BoardStabilizer, RobotError, play, board_distance
from executor import JetMaxExecutor
from vision_multi import ScanObserver, MultiViewMapper
from board import square_name
import preflight

TOPIC = '/usb_cam/image_rect_color'


class Ctx(object):
    """Những thứ khác nhau giữa chế độ giả lập và robot thật; phần còn lại của chương trình dùng chung."""
    observer = executor = wait_human = park = repair = cleanup = confirm_start = None
    poll = 0.0
    save_dir = None


# ------------------------------------------------------------------ phần dùng chung ---
class HumanGate(object):
    """Chặn việc quét bàn cờ cho tới khi người báo đã đi xong (an toàn: tay máy không chạy khi tay người còn trên bàn)."""

    def __init__(self, observe, wait, reset=None):
        self.observe, self.wait, self.reset, self.armed, self.game = observe, wait, reset, True, None

    def arm(self):
        self.armed = True

    def __call__(self):
        if self.armed:
            self.wait(self.game)
            self.armed = False
            if self.reset:
                self.reset()                              # bỏ các khung hình cũ (trước khi người đi) khỏi bộ đệm bỏ phiếu
        return self.observe()


class ArmingExecutor(object):
    """Sau mỗi nước robot: về vị trí chờ rồi bắt người xác nhận trước khi quét lượt sau."""

    def __init__(self, inner, gate, park, log):
        self.inner, self.gate, self.park, self.log = inner, gate, park, log

    def execute(self, info):
        self.inner.execute(info)
        try:
            self.park()
        except Exception as e:
            self.log('CẢNH BÁO: không về được vị trí chờ (%s)' % e)
        self.gate.arm()


class HoverOnlyMover(object):
    """Bọc Mover thật: pick/place chỉ bay tới trên ô, không hạ, không hút."""

    def __init__(self, mover):
        self.m = mover

    def pick(self, sq_pos, zpick, zsafe):
        self.m.hover(sq_pos, zsafe)

    def place(self, sq_pos, zplace, zsafe):
        self.m.hover(sq_pos, zsafe)

    def __getattr__(self, name):
        return getattr(self.m, name)


def verify_start(game, observe, confirm, log, tries=10):
    """Quét và đối chiếu với thế cờ ban đầu trước khi chơi."""
    want = game.pos.to_dict()
    for _ in range(tries):
        if confirm:
            confirm()
        obs = None
        for _ in range(10):
            obs = observe()
            if obs is not None:
                break
        if obs is None:
            raise RuntimeError('Không quét được bàn cờ')
        d = board_distance(want, obs)
        if d == 0:
            log('Thế cờ ban đầu khớp.')
            return
        miss = sorted(square_name(*k) + '=' + v for k, v in set(want.items()) - set(obs.items()))
        extra = sorted(square_name(*k) + '=' + v for k, v in set(obs.items()) - set(want.items()))
        log('Thế cờ ban đầu lệch %d: thiếu/sai [%s], thừa/sai [%s]' % (d, ' '.join(miss), ' '.join(extra)))
        if d <= game.tolerance:
            log('  (lệch trong mức chịu được của nhận diện, tiếp tục)')
            return
    raise RuntimeError('Thế cờ ban đầu vẫn lệch sau %d lần quét, hãy kiểm tra quân cờ và hiệu chuẩn' % tries)


def run_game(game, gate, ctx, log, max_failures=5):
    fails = 0
    while True:
        try:
            return play(game, gate, log=log, poll=ctx.poll)
        except RobotError as e:
            fails += 1
            log('LỖI ROBOT: %s' % e)
            if fails > max_failures:
                raise
            ctx.executor.reset(ctx.repair(game, ctx.executor))


def run(a, log=print):
    ctx = build_fake(a, log) if a.fake else build_hardware(a, log)
    engine = create_engine(a.engine, a.level, a.nnue, a.seed)
    gate = HumanGate(ctx.observer, ctx.wait_human, ctx.observer.stab.reset)
    game = Game(a.human, engine, ArmingExecutor(ctx.executor, gate, ctx.park, log),
                confirm=a.confirm, max_plies=a.max_plies)
    gate.game = game
    original = game.human_observed

    def human_observed(obs):
        r, info = original(obs)
        if r.status in ('nochange', 'invalid', 'ambiguous'):
            gate.arm()                                   # chưa thấy nước đi hợp lệ: dừng quét, đợi người xác nhận lại
        return r, info
    game.human_observed = human_observed

    log('Người cầm %s, robot cầm %s (%s). Engine: %s' % (
        'ĐỎ' if a.human == 'red' else 'ĐEN', 'ĐEN' if a.human == 'red' else 'ĐỎ',
        'người đi trước' if a.human == 'red' else 'robot đi trước', engine.name))
    try:
        verify_start(game, ctx.observer, ctx.confirm_start, log)
        res = run_game(game, gate, ctx, log)
        log('Kết thúc: %s' % (res,))
    except KeyboardInterrupt:
        log('\nDừng theo yêu cầu')
    finally:
        try:
            engine.close()
        except Exception:
            pass
        if ctx.cleanup:
            ctx.cleanup()
        if ctx.save_dir and game.history:
            with open(os.path.join(ctx.save_dir, 'moves.txt'), 'w') as f:
                f.write('\n'.join('%s %s' % (i.by, i.uci) for i in game.history) + '\n')
    return game, ctx


# ------------------------------------------------------------------ giả lập ----------
def build_fake(a, log):
    from fake_world import FakeWorld
    w = FakeWorld(a.noise, a.seed, a.fake_fail_at)
    ctx = Ctx()
    ctx.world = w
    ctx.executor = JetMaxExecutor(w.mover, w.board, w.tray, log=log)
    ctx.observer = ScanObserver(MultiViewMapper(w.views), w, w.move_to, w.grab, BoardStabilizer(a.frames))
    ctx.wait_human = lambda game: w.human_move(game, a.auto, log=log)
    ctx.park = lambda: None
    ctx.repair = lambda game, ex: w.repair(game)
    ctx.poll = 0
    return ctx


# ------------------------------------------------------------------ robot thật -------
def build_hardware(a, log):
    if not a.model:
        raise SystemExit('Cần --model (best.pt / best.onnx / best.engine)')
    rep, objs = preflight.check_all(a.board, a.views, model=a.model)
    log(rep.text())
    log('--> preflight: %s' % rep.summary())
    if rep.errors:
        raise SystemExit('Còn LỖI trong cấu hình, chưa chạy. Sửa rồi chạy lại.')
    board, tray, mapper = objs['board'], objs['tray'], objs['mapper']

    from rig import HardwareRig

    save_dir = None
    if not a.no_save:
        save_dir = a.save_dir or os.path.join(ROOT, 'runs', time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(save_dir, exist_ok=True)
        log('Lưu ảnh quét và nước đi vào %s' % save_dir)

    def ask(text):
        if input(text + ' [Enter = tiếp tục, q = dừng] ').strip().lower() == 'q':
            raise KeyboardInterrupt

    rig = HardwareRig(model=a.model, conf=a.conf, iou=a.iou, z_safe=board.z_safe, node='play_robot', log=log, ask=ask)
    if save_dir:
        saved = {'n': 0}

        def saver(view, bgr):
            rig.cv2.imwrite(os.path.join(save_dir, 'scan%04d_%s.jpg' % (saved['n'], view)), bgr)
            saved['n'] += 1
        rig.saver = saver
    mover = HoverOnlyMover(rig.mv) if a.hover else rig.mv

    def repair(game, ex):
        log('Hãy xếp lại bàn cờ cho KHỚP thế cờ dưới đây (nếu quân bị ăn đang nằm trong khay thì đặt lại về ô):')
        log(str(game.pos))
        s = input('Số quân đang nằm trong khay (Enter = giữ nguyên %d, q = dừng): ' % ex.tray_used).strip().lower()
        if s == 'q':
            raise KeyboardInterrupt
        return int(s) if s else None

    ctx = Ctx()
    ctx.executor = JetMaxExecutor(mover, board, tray, limits=rig.motion.in_limits, log=log)
    ctx.observer = ScanObserver(mapper, rig.detector, rig.move_to, rig.grab, BoardStabilizer(a.frames))
    ctx.wait_human = lambda game: ask('Bạn đi xong và đã rút tay khỏi bàn cờ?')
    ctx.confirm_start = lambda: ask('Xếp đủ quân đúng thế cờ ban đầu, rút tay khỏi bàn cờ?')
    ctx.park, ctx.repair, ctx.poll, ctx.save_dir = rig.park, repair, a.poll, save_dir
    ctx.cleanup = lambda: rig.cleanup(board.z_safe)
    return ctx


# ------------------------------------------------------------------ dòng lệnh -------
def parse_args(argv=None):
    cfg = os.path.join(ROOT, 'configs')
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', choices=['red', 'black'], default='red', help='màu quân của người chơi')
    ap.add_argument('--level', choices=sorted(LEVELS), default='easy')
    ap.add_argument('--engine', help='đường dẫn Pikafish (bỏ trống = engine tự viết)')
    ap.add_argument('--nnue')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--max-plies', type=int, default=400)
    ap.add_argument('--frames', type=int, default=1, help='số vòng quét gộp bỏ phiếu (1 hoặc 3 là hợp lý)')
    ap.add_argument('--confirm', type=int, default=2, help='số lần liên tiếp phải thấy cùng một nước mới chấp nhận')
    ap.add_argument('--fake', action='store_true', help='chạy trên thế giới giả lập, không cần robot')
    ap.add_argument('--auto', action='store_true', help='(--fake) người đi ngẫu nhiên')
    ap.add_argument('--noise', type=float, default=0.0, help='(--fake) xác suất YOLO bỏ sót mỗi quân')
    ap.add_argument('--fake-fail-at', type=int, default=None, help='(--fake) giả lập lỗi tay máy ở lệnh gắp/đặt thứ N')
    ap.add_argument('--hover', action='store_true', help='(robot) chỉ bay trên các ô, không gắp, không hút')
    ap.add_argument('--model', help='(robot) best.pt / best.onnx / best.engine')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--board', default=os.path.join(cfg, 'board.yaml'))
    ap.add_argument('--views', default=os.path.join(cfg, 'views.yaml'))
    ap.add_argument('--poll', type=float, default=0.5)
    ap.add_argument('--save-dir')
    ap.add_argument('--no-save', action='store_true')
    return ap.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
