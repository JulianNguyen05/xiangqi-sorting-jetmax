#!/usr/bin/env python3
# Vẽ bàn cờ bằng matplotlib (dùng trong notebook / báo cáo; robot không cần).
#   from board.draw import draw_board
#   draw_board(ax, board, 'Tiêu đề', moves=..., arrows=..., marks=...)


def draw_board(ax, board, title='', moves=None, arrows=None, marks=None, mark_color='#d81b60', wrong=None):
    """board: {(f, r): mã FEN}. moves: [(từ, đến)] -> chấm xanh / dấu x. arrows: [(từ, đến)].
    marks: [(f, r)] khoanh vòng nét đứt. wrong: [(f, r)] tô nền đỏ nhạt (ô nhận diện sai)."""
    import matplotlib.pyplot as plt
    from board.geometry import FILES

    ax.set_xlim(-0.8, 8.8)
    ax.set_ylim(-0.8, 9.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(plt.Rectangle((-0.6, -0.6), 9.2, 10.2, fc='#f3d9a4', ec='none', zorder=0))
    for r in range(10):
        ax.plot([0, 8], [r, r], c='#5d4037', lw=0.8, zorder=1)
    for f in range(9):
        if f in (0, 8):
            ax.plot([f, f], [0, 9], c='#5d4037', lw=0.8, zorder=1)
        else:
            ax.plot([f, f], [0, 4], c='#5d4037', lw=0.8, zorder=1)
            ax.plot([f, f], [5, 9], c='#5d4037', lw=0.8, zorder=1)
    for r0 in (0, 7):
        ax.plot([3, 5], [r0, r0 + 2], c='#5d4037', lw=0.8)
        ax.plot([5, 3], [r0, r0 + 2], c='#5d4037', lw=0.8)
    ax.text(4, 4.5, 'S Ô N G', ha='center', va='center', fontsize=8, color='#8d6e63', zorder=1)
    for f in range(9):
        ax.text(f, -0.75, FILES[f], ha='center', va='center', fontsize=7, color='#555')
    for r in range(10):
        ax.text(-0.75, r, str(r), ha='center', va='center', fontsize=7, color='#555')
    for (f, r) in wrong or []:
        ax.add_patch(plt.Rectangle((f - 0.5, r - 0.5), 1, 1, fc='#ef9a9a', ec='none', alpha=0.8, zorder=2))
    for (f, r), c in board.items():
        red = c.isupper()
        ax.add_patch(plt.Circle((f, r), 0.42, fc='#fff8e1', ec='#c62828' if red else '#212121', lw=1.6, zorder=3))
        ax.text(f, r, c, ha='center', va='center', fontsize=9, fontweight='bold',
                color='#c62828' if red else '#212121', zorder=4)
    for mv in moves or []:
        (f1, r1) = mv[1]
        if mv[1] in board:
            ax.plot(f1, r1, marker='x', ms=14, mew=2.5, c='#e53935', zorder=5)
        else:
            ax.plot(f1, r1, 'o', ms=7, c='#2e7d32', zorder=5)
    for (a, b) in arrows or []:
        ax.annotate('', xy=b, xytext=a, zorder=6, arrowprops=dict(arrowstyle='->', lw=2.2, color='#1565c0'))
    for (f, r) in marks or []:
        ax.add_patch(plt.Circle((f, r), 0.5, fc='none', ec=mark_color, lw=2.5, ls='--', zorder=6))
    ax.set_title(title, fontsize=10)
