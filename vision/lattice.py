#!/usr/bin/env python3
# Ảnh -> thế cờ KHÔNG cần biết bàn cờ trông thế nào: khớp lưới 9 x 10 trực tiếp từ TÂM CÁC QUÂN.
#
# Ý tưởng: quân cờ tướng luôn đứng trên giao điểm, nên tâm các quân (trong ảnh) nằm trên một lưới đều
# 9 cột x 10 hàng đã bị biến đổi phối cảnh. Ta tìm phép biến đổi đó (homography lưới -> ảnh):
#   B1. Ước lượng 2 hướng trục + khoảng cách ô từ các vectơ nối quân với quân lân cận.
#   B2. Gán toạ độ nguyên cho quân gần quân "hạt giống", khớp homography, rồi MỞ RỘNG dần bán kính
#       (chịu được phối cảnh vừa phải: ô xa camera nhỏ hơn ô gần camera).
#   B3. Chốt hướng + vị trí bàn (xoay 0/90/180/270°, dịch) bằng LUẬT CỜ: chọn cách đặt mà thế cờ
#       hợp lệ nhất (tướng/sĩ trong cung, tượng không qua sông, ...) và Đỏ ở hàng 0..4.
# Dùng cho ảnh của tập dữ liệu công khai (mỗi ảnh một kiểu bàn) và làm phương án dự phòng trên robot
# khi vision/board_detect.py không tìm được bàn.
# Giới hạn: giả sử ô vuông đều (sông rộng bằng 1 ô, đúng với hầu hết bàn cờ); cần >= 6 quân.
# Python 3.6, chỉ cần numpy + OpenCV.

import itertools
from collections import namedtuple

import cv2
import numpy as np

from board import rules as R
from board.geometry import NF, NR

LatticeFit = namedtuple('LatticeFit', 'H Hinv idx resid inlier cell_px')
# H: 3x3, toạ độ lưới (u, v) -> pixel ảnh. idx: (n, 2) toạ độ lưới nguyên (tương đối, chưa chốt hướng).
# resid: độ lệch tới giao điểm gần nhất, đơn vị = 1 ô. inlier: resid < tol. cell_px: cỡ ô trung bình (px).

Placement = namedtuple('Placement', 'rot of orr score ambiguous')
# rot: 0..7 (chỉ số phép đối xứng trong ROTS), of/orr: dịch cột/hàng; score: số lỗi luật (nhỏ = tốt).


def _apply(H, pts):
    p = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def _unit(lengths):
    """Khoảng cách ô từ các độ dài vectơ dọc 1 trục (độ dài thật = bội số nguyên của 1 ô)."""
    L = np.sort(np.asarray(lengths, np.float64))
    if len(L) == 0:
        return None
    lmin = np.percentile(L, 10)
    L = L[L <= lmin * 6.5]
    best, cands = [], np.linspace(lmin * 0.45, lmin * 1.08, 90)
    for c in cands:
        k = L / c
        err = k - np.round(k)
        best.append(np.mean(np.exp(-(err ** 2) / (2 * 0.09 ** 2))))
    best = np.asarray(best)
    ok = np.where(best >= 0.85 * best.max())[0]
    return float(cands[ok.max()])                         # ô lớn nhất vẫn giải thích được mọi độ dài


def _basis_candidates(pts, k=8, n_peaks=6):
    """Các cặp vectơ đơn vị lưới (pixel) có thể có, từ vectơ nối mỗi quân với k quân gần nhất.
    Trả về nhiều ứng viên; fit_lattice() thử hết và giữ ứng viên giải thích được nhiều quân nhất."""
    d = pts[None, :, :] - pts[:, None, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :min(k, len(pts) - 1)]
    vec = np.concatenate([d[i, nn[i]] for i in range(len(pts))])
    ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 180.0
    ln = np.linalg.norm(vec, axis=1)
    hist = np.zeros(180)
    for a, wi in zip(ang, 1.0 / np.maximum(ln, 1e-6)):    # vectơ ngắn (ô kề nhau) đáng tin hơn
        hist[int(a) % 180] += wi
    sm = np.zeros(180)
    for s in range(-3, 4):
        sm += np.roll(hist, s) * np.exp(-s * s / 4.0)
    peaks, h = [], sm.copy()
    for _ in range(n_peaks):                              # các hướng nổi bật (bỏ lân cận ±10°)
        t = int(np.argmax(h))
        if h[t] <= 0:
            break
        peaks.append(t)
        for s in range(-10, 11):
            h[(t + s) % 180] = 0

    def axis(t):
        diff = np.abs((ang - t + 90) % 180 - 90)
        sel = diff < 8
        if sel.sum() < 3:
            return None
        u = _unit(ln[sel])
        v = vec[sel]
        v = v * np.sign(v @ np.array([np.cos(np.radians(t)), np.sin(np.radians(t))]))[:, None]
        dv = v.mean(axis=0)
        return dv / np.linalg.norm(dv), u

    axes = {t: axis(t) for t in peaks}
    out = []
    for t1, t2 in itertools.combinations(peaks, 2):
        sep = (t2 - t1) % 180
        if not 50 <= sep <= 130 or axes[t1] is None or axes[t2] is None:
            continue
        (d1, u1), (d2, u2) = axes[t1], axes[t2]
        # Dọc một trục có thể không có 2 quân kề nhau -> ô bị ước lượng thành 2-3 ô. Ô gần vuông nên thử
        # thêm: dùng cỡ ô của trục này cho trục kia, và chia trục dài cho số nguyên gần nhất.
        cands = {(u1, u2), (u1, u1), (u2, u2)}
        hi, lo = max(u1, u2), min(u1, u2)
        kk = int(round(hi / lo))
        if kk >= 2:
            cands.add((u1 / kk, u2) if u1 > u2 else (u1, u2 / kk))
        for a, b in cands:
            out.append(np.array([d1 * a, d2 * b]).T)      # cột 0 = trục 1, cột 1 = trục 2
    return out


def _fit_from(pts, A):
    med = np.median(pts, axis=0)
    seed = pts[np.argmin(np.linalg.norm(pts - med, axis=1))]
    H = np.array([[A[0, 0], A[0, 1], seed[0]], [A[1, 0], A[1, 1], seed[1]], [0, 0, 1.0]])
    for radius in (2.5, 3.5, 5, 7, 10, 14, 14):
        q = _apply(np.linalg.inv(H), pts)
        sel = np.abs(q).max(axis=1) <= radius
        idx = np.round(q[sel])
        res = np.linalg.norm(q[sel] - idx, axis=1)
        good = res < 0.4
        src, dst = idx[good], pts[sel][good]
        if len(src) < 4 or np.linalg.matrix_rank(np.c_[src, np.ones(len(src))]) < 3:
            continue
        Hn = None
        if len({tuple(x) for x in src}) >= 6 and len(np.unique(src[:, 0])) >= 3 and len(np.unique(src[:, 1])) >= 3:
            Hn, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
        if Hn is None:                                    # chưa đủ điểm cho phối cảnh -> affine
            M, _, _, _ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
            Hn = np.array([[M[0, 0], M[1, 0], M[2, 0]], [M[0, 1], M[1, 1], M[2, 1]], [0, 0, 1.0]])
        H = Hn / Hn[2, 2]
    return H


def _best_window(idx, ok):
    """Khung 9 x 10 (hoặc 10 x 9) chứa được nhiều điểm nhất -> mặt nạ các điểm nằm trong khung."""
    pts = idx[ok]
    best, best_mask = -1, np.zeros(len(idx), bool)
    if len(pts) == 0:
        return best_mask
    for w, h in ((NF, NR), (NR, NF)):
        for x0 in range(pts[:, 0].min() - w + 1, pts[:, 0].max() + 1):
            inx = (idx[:, 0] >= x0) & (idx[:, 0] < x0 + w)
            for y0 in range(pts[:, 1].min() - h + 1, pts[:, 1].max() + 1):
                m = ok & inx & (idx[:, 1] >= y0) & (idx[:, 1] < y0 + h)
                n = int(m.sum())
                if n > best:
                    best, best_mask = n, m
    return best_mask


def fit_lattice(pts, tol=0.3, min_pts=6, return_all=False):
    """Khớp lưới đều qua các điểm (tâm quân). Trả về LatticeFit hoặc None.
    Thử mọi ứng viên trục; chọn lưới giải thích được nhiều quân nhất, hoà thì chọn ô LỚN nhất
    (lưới nửa ô cũng "khớp" mọi điểm nên phải ưu tiên ô lớn)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < min_pts:
        return [] if return_all else None
    results = []
    for A in _basis_candidates(pts):
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        ratio = np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1])
        if not 0.6 < ratio < 1.7:                         # ô gần vuông (cho phép phối cảnh vừa phải)
            continue
        try:
            H = _fit_from(pts, A)
            Hinv = np.linalg.inv(H)
        except (np.linalg.LinAlgError, cv2.error):
            continue
        q = _apply(Hinv, pts)
        idx = np.round(q).astype(int)
        resid = np.linalg.norm(q - idx, axis=1)
        inl = (resid < tol) & _best_window(idx, resid < tol)   # quân nằm ngoài khung 9 x 10 = để ngoài bàn
        if inl.sum() < min_pts:
            continue
        if len({tuple(x) for x in idx[inl]}) < inl.sum():
            continue                                      # 2 quân trùng 1 giao điểm -> lưới sai
        c = _apply(H, np.array([[0, 0], [1, 0], [0, 1]], np.float64))
        e1, e2 = c[1] - c[0], c[2] - c[0]
        area = abs(e1[0] * e2[1] - e1[1] * e2[0])
        results.append((int(inl.sum()), area, float(resid[inl].mean()), H, Hinv, idx, resid, inl, c))
    if not results:
        return [] if return_all else None
    best_n = max(r[0] for r in results)
    good = sorted([r for r in results if r[0] == best_n], key=lambda r: (-r[1], r[2]))
    fits = []
    for n, area, _, H, Hinv, idx, resid, inl, c in good:
        cell = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[0])) / 2
        if all(abs(cell - f.cell_px) > 0.05 * cell or np.abs(f.H - H).max() > 1e-3 * cell for f in fits):
            fits.append(LatticeFit(H, Hinv, idx, resid, inl, cell))
    return fits if return_all else fits[0]


# 8 phép đối xứng của lưới vuông (4 phép xoay + 4 phép lật). Trục (u, v) tìm được có thể ngược chiều,
# nên phải xét cả 8; place() chỉ giữ 4 phép cho ảnh đúng chiều (ảnh chụp thật không bị lật gương).
_R4 = [np.array([[1, 0], [0, 1]]), np.array([[0, -1], [1, 0]]), np.array([[-1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]])]
ROTS = _R4 + [m @ np.array([[1, 0], [0, -1]]) for m in _R4]


def _to_board(idx, rot, of, orr):
    """(u, v) lưới tương đối -> (f, r) tuyệt đối theo cách đặt (rot, of, orr)."""
    fr = np.asarray(idx) @ ROTS[rot].T
    return fr[:, 0] + of, fr[:, 1] + orr


def place(fit, codes, mirror_ok=False):
    """Chốt hướng + vị trí bàn bằng luật cờ. codes: mã FEN của từng điểm (cùng thứ tự với fit.idx).
    Chỉ xét các phép đặt có cùng 'chiều' với ảnh thật (ảnh chụp không thể bị lật gương)."""
    use = fit.inlier
    idx = fit.idx[use]
    cds = [c for c, u in zip(codes, use) if u]
    # chiều của hệ (u, v) trong ảnh: bàn thật nhìn từ phía Đỏ có (f -> phải, r -> lên) => định thức < 0 (y ảnh hướng xuống)
    J = fit.H[:2, :2] - np.outer(fit.H[:2, 2], fit.H[2, :2])
    cands = []
    for rot in range(len(ROTS)):
        Jr = J @ np.linalg.inv(ROTS[rot])
        if not mirror_ok and np.linalg.det(Jr) > 0:
            continue
        f, r = _to_board(idx, rot, 0, 0)
        for of in range(-f.max(), NF - f.min()):
            for orr in range(-r.max(), NR - r.min()):
                ff, rr = f + of, r + orr
                inside = (ff >= 0) & (ff < NF) & (rr >= 0) & (rr < NR)
                cands.append((int(inside.sum()), rot, of, orr))
    if not cands:
        return None
    best_n = max(c[0] for c in cands)
    scored = []
    for n, rot, of, orr in cands:
        if n < best_n:
            continue
        f, r = _to_board(idx, rot, of, orr)
        b = {}
        for fi, ri, c in zip(f, r, cds):
            if 0 <= fi < NF and 0 <= ri < NR:
                b[(int(fi), int(ri))] = c
        iss = [i for i in R.validate(b) if i[0] != 'side_in_check']
        red_r = [k[1] for k, c in b.items() if c.isupper()]
        blk_r = [k[1] for k, c in b.items() if c.islower()]
        balance = (np.mean(red_r) if red_r else 4.5) - (np.mean(blk_r) if blk_r else 4.5)
        scored.append((len(iss), balance, rot, of, orr))
    scored.sort()
    s0 = scored[0]
    amb = len(scored) > 1 and scored[1][:2] == s0[:2]
    return Placement(s0[2], s0[3], s0[4], s0[0], amb)


def detections_to_board(pts, codes, confs, fit, pl, tol=0.35):
    """Tâm quân (pixel) -> thế cờ {(f, r): mã} theo lưới fit và cách đặt pl.
    Hai quân rơi vào cùng 1 ô: giữ quân có độ tin cậy cao hơn. Trả về (board, chi tiết từng điểm)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    q = _apply(fit.Hinv, pts)
    idx = np.round(q).astype(int)
    res = np.linalg.norm(q - idx, axis=1)
    f, r = _to_board(idx, pl.rot, pl.of, pl.orr)
    board, conf_at, info = {}, {}, []
    for i in range(len(pts)):
        sq = (int(f[i]), int(r[i]))
        on = 0 <= sq[0] < NF and 0 <= sq[1] < NR and res[i] < tol
        info.append((sq if on else None, float(res[i])))
        if not on:
            continue
        if sq not in board or confs[i] > conf_at[sq]:
            board[sq] = codes[i]
            conf_at[sq] = confs[i]
    return board, info


def board_to_image(fit, pl, squares):
    """(f, r) -> pixel (để vẽ lưới / kiểm tra)."""
    sq = np.asarray(squares, np.float64).reshape(-1, 2)
    inv = np.linalg.inv(ROTS[pl.rot])
    uv = (sq - [pl.of, pl.orr]) @ inv.T
    return _apply(fit.H, uv)


def image_to_board(pts, codes, confs, tol=0.3, max_fits=4):
    """Gói gọn: tâm quân + mã + độ tin cậy -> (board, fit, placement); thất bại -> (None, fit, None).
    Thế cờ thưa có thể khớp nhiều lưới (vd. mọi quân cùng nằm hàng chẵn -> lưới 2 ô cũng khớp):
    khi đó thử từng lưới và chọn lưới cho thế cờ hợp lệ nhất theo luật."""
    fits = fit_lattice(pts, tol=tol, return_all=True)
    best = None
    for k, fit in enumerate(fits[:max_fits]):
        pl = place(fit, codes)
        if pl is None:
            continue
        key = (pl.score, k)                                 # cùng điểm luật -> giữ lưới ô lớn hơn (k nhỏ)
        if best is None or key < best[0]:
            best = (key, fit, pl)
    if best is None:
        return None, (fits[0] if fits else None), None
    _, fit, pl = best
    board, _ = detections_to_board(pts, codes, confs, fit, pl)
    return board, fit, pl
