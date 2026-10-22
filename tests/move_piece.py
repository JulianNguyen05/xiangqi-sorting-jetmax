#!/usr/bin/env python3
# Bước 1 của robot đánh cờ: gắp quân từ ô này sang ô kia theo lệnh gõ tay (CHƯA dùng camera, CHƯA dùng mô hình).
# Vị trí các ô lấy từ configs/board.yaml (4 góc đã dạy bằng joint_teleop.py), xem robot/board.py.
# Yêu cầu: node jetmax_control đang chạy. Chạy từ thư mục gốc của repo trên robot:
#
#   python3 tests/move_piece.py --hover          # LẦN ĐẦU: chỉ bay trên các ô, không hạ, không hút
#   python3 tests/move_piece.py                  # gắp - đặt thật (hỏi Enter trước mỗi nước)
#   python3 tests/move_piece.py --yes            # không hỏi Enter
#
# Lệnh gõ trong chương trình:
#   h2 e2          gắp quân ở h2, đặt xuống e2 (ô đích phải TRỐNG; cũng nhận dạng "h2e2")
#   hover a0 i0 i9 a9   bay trên các ô lần lượt (kiểm tra hiệu chuẩn 4 góc, không hạ)
#   touch e5       hạ xuống z_pick ở e5 KHÔNG hút, để kiểm tra độ cao chạm quân (cần có quân ở đó)
#   where e5       in vị trí (thanh trượt, x, y) của ô
#   q              thoát (tay máy về vị trí ban đầu, thanh trượt về 0)

import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]   # chạy được cả khi để chung một thư mục lẫn theo cấu trúc repo

import rospy
from board import Board, parse_square, square_name, NF, NR
from motion import Mover, in_limits, SLIDER_MIN, SLIDER_MAX


class Cancel(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    cands = [os.path.join(HERE, '..', 'configs', 'board.yaml'), os.path.join(HERE, 'board.yaml')]
    ap.add_argument('--config', default=next((c for c in cands if os.path.exists(c)), cands[0]),
                    help='file hiệu chuẩn (mặc định: ../configs/board.yaml, hoặc board.yaml cạnh script)')
    ap.add_argument('--hover', action='store_true', help='chỉ bay trên các ô, không hạ, không hút')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước mỗi nước đi')
    ap.add_argument('--no-home', action='store_true', help='không đưa thanh trượt về home lúc bắt đầu')
    a = ap.parse_args()

    board = Board.load(a.config)
    zs, zp, zl = board.z_safe, board.z_pick, board.z_place

    bad = set()                                  # kiểm tra cả 90 ô trước khi động tới robot
    for f in range(NF):
        for r in range(NR):
            sq = square_name(f, r)
            s, x, y = board.position(sq)
            if not (SLIDER_MIN <= s <= SLIDER_MAX and all(in_limits((x, y, z)) for z in (zs, zp, zl))):
                bad.add(sq)
    if len(bad) == NF * NR:
        raise SystemExit('Không ô nào nằm trong giới hạn an toàn, hãy kiểm tra lại 4 góc và z_pick trong board.yaml')
    if bad:
        print('CẢNH BÁO: %d ô ngoài giới hạn an toàn/tầm với, sẽ bị từ chối: %s' % (len(bad), ' '.join(sorted(bad))))

    def pos_of(text):
        sq = square_name(*parse_square(text))
        if sq in bad:
            raise ValueError('Ô %s ngoài giới hạn an toàn hoặc ngoài tầm với' % sq)
        return sq, board.position(sq)

    def ask(text):
        if not a.yes and input(text + ' [Enter = chạy, q = hủy] ').strip().lower() == 'q':
            raise Cancel()

    rospy.init_node('move_piece')
    mv = Mover()
    moving = {'on': False}                       # đang giữa một chuyển động -> lỗi thì phải tắt hút và nâng lên
    print('Vị trí ban đầu:', mv.start)
    print('z_safe=%.1f z_pick=%.1f z_place=%.1f %s' % (zs, zp, zl, '[CHỈ BAY, KHÔNG HẠ]' if a.hover else ''))

    try:
        if not a.no_home:
            ask('Đưa thanh trượt về HOME (vị trí 0). Tay máy phải đang nâng cao và không vướng')
            mv.home_slider()
        print('Lệnh: "h2 e2" | hover <ô>... | touch <ô> | where <ô> | q')
        while True:
            try:
                line = input('> ').strip().lower()
            except EOFError:
                break
            if not line:
                continue
            if line in ('q', 'quit', 'exit'):
                break
            tok = line.replace(',', ' ').split()
            try:
                if tok[0] == 'where':
                    for t in tok[1:]:
                        sq, p = pos_of(t)
                        print('  %s: thanh trượt %d, x %.1f, y %.1f' % ((sq,) + p))
                elif tok[0] == 'hover':
                    plan = [pos_of(t) for t in tok[1:]]
                    if not plan:
                        raise ValueError('hover cần ít nhất một ô')
                    ask('Bay trên: %s' % ' '.join(sq for sq, _ in plan))
                    moving['on'] = True
                    for sq, p in plan:
                        print('  -> %s' % sq)
                        mv.hover(p, zs)
                elif tok[0] == 'touch':
                    if a.hover:
                        raise ValueError('Đang ở chế độ --hover, không hạ xuống')
                    if len(tok) != 2:
                        raise ValueError('Dùng: touch <ô>')
                    sq, p = pos_of(tok[1])
                    ask('Bay tới %s rồi hạ xuống z_pick=%.1f (không hút)' % (sq, zp))
                    moving['on'] = True
                    mv.hover(p, zs)
                    mv.goto((p[1], p[2], zp), 40.0)
                    input('  Đang chạm. Kiểm tra độ cao/vị trí, Enter để nâng lên ')
                    mv.goto((p[1], p[2], zs), 40.0)
                else:
                    if len(tok) == 1 and len(tok[0]) == 4:
                        tok = [tok[0][:2], tok[0][2:]]
                    if len(tok) != 2:
                        raise ValueError('Dùng: <ô đi> <ô đến>, ví dụ h2 e2')
                    (sa, A), (sb, B) = pos_of(tok[0]), pos_of(tok[1])
                    if sa == sb:
                        raise ValueError('Ô đi và ô đến trùng nhau')
                    ask('%s -> %s  (%s: trượt %d x %.1f y %.1f | %s: trượt %d x %.1f y %.1f)%s' % (
                        sa, sb, sa, A[0], A[1], A[2], sb, B[0], B[1], B[2],
                        '' if a.hover else '. Ô đích phải trống'))
                    moving['on'] = True
                    if a.hover:
                        mv.hover(A, zs)
                        mv.hover(B, zs)
                    else:
                        mv.pick(A, zp, zs)
                        mv.place(B, zl, zs)
                moving['on'] = False
                print('  xong')
            except Cancel:
                print('  đã hủy')
            except (ValueError, RuntimeError) as e:
                print('LỖI:', e)
                if moving['on']:
                    mv.safe_stop(zs)
                    moving['on'] = False
                    print('Đã tắt hút và nâng tay máy. Quân cờ có thể đã rơi hoặc lệch, hãy kiểm tra bàn cờ.')
    except Cancel:
        print('Dừng theo yêu cầu')
    except KeyboardInterrupt:
        print('\nDừng bằng Ctrl+C')
    except (ValueError, RuntimeError) as e:
        print('LỖI:', e)
    finally:
        mv.restore(zs)


if __name__ == '__main__':
    main()
