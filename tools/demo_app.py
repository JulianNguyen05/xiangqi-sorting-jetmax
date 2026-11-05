#!/usr/bin/env python3
# GIAO DIỆN DEMO thư viện XiangqiBrain (chạy trên laptop Windows / Linux / Colab, mở bằng trình duyệt).
#   py -3 tools\demo_app.py           (Windows; dùng -3 để py bỏ qua dòng #! ở trên)      -> tự mở http://127.0.0.1:7860
#   python3 tools/demo_app.py --share (Colab)        -> in ra đường link công khai
# Hai thẻ:
#   1. Phân tích 1 ảnh : ảnh (+ FEN lượt trước) -> thế cờ, nước tiếp theo
#   2. Chơi cả ván     : mỗi lượt tải ảnh sau khi người đi -> máy trả lời nước đi (bạn đi giúp máy trên bàn thật)
# Cần: pip install gradio matplotlib onnxruntime opencv-python numpy pyyaml  (xem DEMO_WINDOWS.md)
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
import matplotlib                                     # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
import gradio as gr                                   # noqa: E402

from board import rules as R                          # noqa: E402
from board.draw import draw_board                     # noqa: E402
from board.geometry import parse_square               # noqa: E402
from brain.xiangqi_brain import XiangqiBrain          # noqa: E402

NAME = {'k': 'Tướng', 'a': 'Sĩ', 'b': 'Tượng', 'n': 'Mã', 'r': 'Xe', 'c': 'Pháo', 'p': 'Tốt'}
STATUS_VN = {'ok': '✅ Có nước đi', 'retake': '📷 Cần chụp lại', 'invalid': '⚠️ Thế cờ vô lý',
             'game_over': '🏁 Hết ván', 'robot_move': '✅ Máy đã chọn nước', 'waiting': '⏳ Chưa thấy nước đi mới'}

ap = argparse.ArgumentParser()
ap.add_argument('--share', action='store_true', help='tạo link công khai (dùng trên Colab)')
ap.add_argument('--port', type=int, default=7860)
ap.add_argument('--depth', type=int, default=10)
args = ap.parse_args()

print('Đang nạp mô hình và engine...')
BRAIN = XiangqiBrain(depth=args.depth)
ENGINE_NAME = getattr(BRAIN.engine, 'name', 'engine')


# ------------------------------------------------------------------------------------------ vẽ
def fig_to_rgb(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def board_image(board, title='', move=None, wrong=None):
    fig, ax = plt.subplots(figsize=(4.6, 5.2), dpi=110)
    arrows = [tuple(parse_square(s) for s in (move[:2], move[2:]))] if move else None
    draw_board(ax, board or {}, title, arrows=arrows, wrong=wrong)
    plt.tight_layout()
    return fig_to_rgb(fig)


def annotate(rgb, read):
    out = rgb.copy()
    for p in (read or {}).get('pieces', []):
        col = (220, 40, 40) if p.code.isupper() else (30, 30, 30)
        cv2.circle(out, (int(p.x), int(p.y)), int(max(p.r, 6)), (0, 220, 90), 2)
        cv2.putText(out, p.code, (int(p.x) - 7, int(p.y) - int(max(p.r, 6)) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)
    return out


def piece_name(code):
    return '%s %s' % (NAME[code.lower()], 'Đỏ' if code.isupper() else 'Đen') if code else ''


def diff_squares(a, b):
    return [k for k in set(a or {}) | set(b or {}) if (a or {}).get(k) != (b or {}).get(k)]


# ------------------------------------------------------------------------------- thẻ 1: 1 ảnh
def analyze_one(rgb, side_label, prev_fen, depth):
    if rgb is None:
        return None, None, 'Hãy tải một ảnh bàn cờ lên.'
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    side = 'w' if side_label.startswith('Đỏ') else 'b'
    prev_fen = (prev_fen or '').strip() or None
    t0 = time.time()
    try:
        res = BRAIN.analyze(bgr, prev_fen=prev_fen, side=side, depth=int(depth))
    except ValueError as e:
        return None, None, '❌ FEN lượt trước không hợp lệ: %s' % e
    total = (time.time() - t0) * 1000
    rd = res['read']
    wrong = diff_squares(rd['board'], res['board']) if prev_fen and res['board'] else None
    bimg = board_image(res['board'] or rd['board'], '', res['best_move'], wrong)   # ô tô đỏ = ô đã tự sửa
    lines = ['### %s' % STATUS_VN.get(res['status'], res['status'])]
    if res['best_move']:
        lines.append('**Nước tiếp theo: %s** — gắp ô **%s** đặt ô **%s**%s' % (
            res['best_move_vn'], res['best_move_from'], res['best_move_to'],
            ' (ăn %s)' % piece_name(res['captures']) if res['captures'] else ''))
        sc = ('chiếu bí sau %d nước' % res['mate']) if res['mate'] is not None else '%+d (1/100 tốt)' % (res['score_cp'] or 0)
        lines.append('Đánh giá cho bên đi: %s' % sc)
    if res['last_move']:
        lines.append('Nước vừa đi: **%s** (%s)' % (res['last_move_vn'], res['last_move']))
    if prev_fen and res.get('corrected') is not None:
        lines.append('Số ô nhận diện sai đã tự sửa: **%d**' % res['corrected'])
    if res['issues']:
        lines.append('Ghi chú: ' + '; '.join(res['issues']))
    lines = ['\n\n'.join(lines), '| | |\n|---|---|']
    lines.append('| Quân nhận diện | %d (trên bàn %d, định vị bằng `%s`) |' % (rd['n_pieces'], rd['n_on_board'], rd['method']))
    ei = rd.get('enhance') or {}
    if ei.get('stats'):
        st = ei['stats']
        lines.append('| Chất lượng ảnh | sáng L = %.0f, nhiễu = %.1f, ám màu = %.2f |' % (st['brightness'], st['noise'], st['cast']))
        lines.append('| Tăng cường ảnh | %s |' % (', '.join({'denoise': 'lọc song phương', 'white_balance': 'cân bằng trắng Gray-World',
                                                              'gamma': 'gamma', 'clahe': 'CLAHE'}[k] for k in ei['steps'])
                                                     or 'không cần (ảnh tốt)'))
    lines.append('| Thời gian | tổng %.0f ms, engine %.0f ms |' % (total, res.get('think_ms') or 0))
    lines.append('| Engine | %s |' % ENGINE_NAME)
    if res['fen']:
        lines.append('\n**FEN** (dán vào ô "FEN lượt trước" cho ảnh tiếp theo):\n```\n%s\n```' % res['fen'])
    return annotate(rgb, rd), bimg, '\n'.join(lines)


# ---------------------------------------------------------------------------- thẻ 2: cả ván
def history_rows():
    rows, b = [], R.parse_fen(BRAIN.game.start_fen)[0]
    side = R.parse_fen(BRAIN.game.start_fen)[1]
    for i, u in enumerate(BRAIN.game.history):
        mv = R.uci_to_move(u)
        who = 'Máy' if side == BRAIN.game.robot_side else 'Người'
        rows.append([i + 1, '%s (%s)' % (who, 'Đỏ' if side == 'w' else 'Đen'), R.move_to_vn(b, mv), u])
        b, side = R.make_move(b, mv), R.other(side)
    return rows


def new_game(machine_label):
    side = 'b' if 'Đen' in machine_label else 'w'
    BRAIN.new_game(robot_side=side)
    msg = '### Ván mới — máy cầm %s\n' % ('Đen' if side == 'b' else 'Đỏ')
    move = None
    if side == 'w':                                   # máy đi trước: trả lời ngay, không cần ảnh
        res = BRAIN.step(None)
        m = res['robot_move']
        BRAIN.confirm(m['uci'])
        move = m['uci']
        msg += '**Máy đi: %s** — gắp **%s** đặt **%s**. Đi giúp máy trên bàn, rồi bạn đi, chụp ảnh và tải lên.' % (
            m['vn'], m['from'], m['to'])
    else:
        msg += 'Bạn (Đỏ) đi trước. Đi một nước trên bàn thật, chụp ảnh rồi tải lên.'
    return None, board_image(BRAIN.game.board, '', move), msg, history_rows()


def play_turn(rgb):
    if rgb is None:
        return None, board_image(BRAIN.game.board, ''), 'Hãy tải ảnh bàn cờ sau khi bạn đi.', history_rows()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    t0 = time.time()
    res = BRAIN.step(bgr)
    total = (time.time() - t0) * 1000
    read = res.get('read')
    st = res['status']
    lines = ['### %s' % STATUS_VN.get(st, st)]
    move = None
    if res.get('human_move'):
        h = res['human_move']
        lines.append('Bạn vừa đi: **%s**%s' % (h['vn'], ' (đã tự sửa %d ô nhận diện sai)' % h['corrected'] if h['corrected'] else ''))
    if st == 'robot_move':
        m = res['robot_move']
        BRAIN.confirm(m['uci'])                       # giả sử người đi giúp máy đúng nước này
        move = m['uci']
        lines.append('**Máy đi: %s** — gắp **%s** đặt **%s**%s' % (
            m['vn'], m['from'], m['to'], ' (ăn %s — bỏ quân này ra)' % piece_name(m['captures']) if m['captures'] else ''))
        lines.append('Hãy đi giúp máy trên bàn thật, sau đó bạn đi tiếp và tải ảnh mới.')
        if R.game_status(BRAIN.game.board, BRAIN.game.side) != 'ongoing':
            lines.append('🏁 **Bạn đã bị chiếu bí / hết nước đi — máy thắng.**')
    elif st == 'retake':
        lines.append('Ảnh chưa đủ chắc chắn (ảnh lệch nhiều ô so với mọi nước hợp lệ, hoặc không thấy bàn cờ). '
                     'Chụp lại rõ hơn: thẳng từ trên xuống, đủ sáng, tay ra khỏi khung hình.')
    elif st == 'waiting':
        lines.append('Ảnh giống hệt thế cờ trước — bạn chưa đi nước nào?')
    elif st == 'game_over':
        lines.append('🏁 Hết ván. Bấm **Ván mới** để chơi lại.')
    lines.append('Thời gian xử lý: %.0f ms · FEN: `%s`' % (total, BRAIN.game.fen))
    return (annotate(rgb, read) if read else rgb), board_image(BRAIN.game.board, '', move), \
        '\n\n'.join(lines), history_rows()


# ------------------------------------------------------------------------------------- giao diện
with gr.Blocks(title='Cờ tướng: ảnh → nước đi') as demo:
    gr.Markdown('# Nhận dạng thế cờ tướng từ ảnh và gợi ý nước đi\n'
                'YOLO11n nhận diện 14 loại quân · luật cờ kiểm tra và tự sửa lỗi nhận diện · %s chọn nước đi' % ENGINE_NAME)
    with gr.Tab('1 · Phân tích một ảnh'):
        with gr.Row():
            with gr.Column(scale=1):
                in_img = gr.Image(label='Ảnh bàn cờ', type='numpy', sources=['upload', 'clipboard'])
                side = gr.Radio(['Đỏ đi tiếp', 'Đen đi tiếp'], value='Đỏ đi tiếp', label='Bên đi tiếp (khi không có FEN lượt trước)')
                prev = gr.Textbox(label='FEN lượt trước (không bắt buộc) — có thì thư viện suy ra nước vừa đi và tự sửa lỗi',
                                  placeholder=R.START, lines=2)
                depth = gr.Slider(4, 16, value=args.depth, step=1, label='Độ sâu engine')
                btn = gr.Button('Phân tích', variant='primary')
            with gr.Column(scale=1):
                out_img = gr.Image(label='Quân nhận diện được', type='numpy')
            with gr.Column(scale=1):
                out_board = gr.Image(label='Thế cờ + nước tiếp theo (mũi tên xanh · ô đỏ = đã tự sửa)', type='numpy')
        out_md = gr.Markdown()
        btn.click(analyze_one, [in_img, side, prev, depth], [out_img, out_board, out_md])

    with gr.Tab('2 · Chơi cả ván'):
        gr.Markdown('Bàn cờ thật trên bàn. Mỗi lượt: **bạn đi → chụp ảnh → tải lên → máy trả lời → bạn đi giúp máy** trên bàn thật.')
        with gr.Row():
            machine = gr.Radio(['Máy cầm Đen', 'Máy cầm Đỏ (đi trước)'], value='Máy cầm Đen', label='Máy cầm')
            btn_new = gr.Button('Ván mới')
        with gr.Row():
            with gr.Column(scale=1):
                g_img = gr.Image(label='Ảnh sau khi bạn đi', type='numpy', sources=['upload', 'clipboard'])
                btn_turn = gr.Button('Gửi ảnh', variant='primary')
            with gr.Column(scale=1):
                g_ann = gr.Image(label='Quân nhận diện được', type='numpy')
            with gr.Column(scale=1):
                g_board = gr.Image(label='Thế cờ hiện tại (mũi tên = nước máy vừa chọn)', type='numpy', value=board_image(BRAIN.game.board, ''))
        g_md = gr.Markdown('Bấm **Ván mới** để bắt đầu.')
        g_hist = gr.Dataframe(headers=['#', 'Bên', 'Nước đi', 'UCI'], label='Biên bản ván cờ', interactive=False)
        btn_new.click(new_game, [machine], [g_ann, g_board, g_md, g_hist])
        btn_turn.click(play_turn, [g_img], [g_ann, g_board, g_md, g_hist])

if __name__ == '__main__':
    demo.launch(server_port=args.port, share=args.share, inbrowser=not args.share)
