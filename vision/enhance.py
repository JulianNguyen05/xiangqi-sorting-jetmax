# -*- coding: utf-8 -*-
"""Tăng cường chất lượng ảnh trước khi nhận dạng (tuỳ chọn, cấu hình ở mục brain.enhance của config/robot.yaml).

Bốn kỹ thuật, áp dụng theo thứ tự:
  1. Khử nhiễu       - lọc song phương (bilateral): làm mượt nhưng giữ biên đường kẻ / nét chữ
  2. Cân bằng trắng   - Gray-World: đưa trung bình 3 kênh về bằng nhau (bỏ ám màu đèn vàng)
  3. Gamma           - I_out = 255 * (I_in / 255) ^ gamma (bảng tra LUT)
  4. CLAHE           - cân bằng histogram thích nghi trên kênh L của LAB (không làm lệch màu đỏ / đen)

Chế độ 'auto' đo ảnh trước (độ sáng, mức nhiễu, độ ám màu) và chỉ bật kỹ thuật nào cần;
ảnh tốt thì đi qua nguyên vẹn. Ngưỡng mặc định chọn từ khảo sát trong training/image_preprocessing.ipynb:
ảnh mẫu của đề tài có L ~ 173, nhiễu ~ 0.5-0.8, độ ám ~ 1.02; định vị bàn cờ bắt đầu hỏng khi nhiễu ước lượng > ~10.

Chỉ dùng NumPy + OpenCV (chạy được trên robot, Python 3.6).
"""
import cv2
import numpy as np

DEFAULTS = {
    'mode': 'auto',          # 'off' | 'auto' (đo ảnh rồi tự bật) | 'manual' (bật đúng các mục ghi True / số)
    'apply_to': ['board'],   # 'board': ảnh cho bước định vị bàn cờ; 'pieces': ảnh cho YOLO (chưa đo đủ, mặc định không)
    'denoise': True,         # manual: True/False. auto: bật khi nhiễu > noise_max
    'white_balance': True,   # manual: True/False. auto: bật khi độ ám màu > cast_max
    'clahe': True,           # manual: True/False. auto: bật khi độ sáng L < dark_L
    'gamma': None,           # manual: số (vd. 0.6 làm sáng). auto: không dùng (CLAHE đã lo phần thiếu sáng)
    'noise_max': 4.0,        # ước lượng độ lệch chuẩn nhiễu (mức xám) - ảnh tốt ~ 1
    'cast_max': 1.25,        # tỉ lệ trung bình kênh lớn nhất / nhỏ nhất - ảnh tốt ~ 1.02
    'dark_L': 100,           # trung bình kênh L (0-255) - ảnh tốt ~ 170
    'clahe_clip': 2.0,
    'clahe_tile': 8,
}


def settings(cfg=None):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (cfg or {}).items() if v is not None or k == 'gamma'})
    return s


# ---------------------------------------------------------------------------- đo chất lượng ảnh
def estimate_noise(gray):
    """Độ lệch chuẩn nhiễu theo Immerkær (1996): tích chập với mặt nạ Laplace kép, lấy trung bình trị tuyệt đối."""
    g = gray.astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    s = np.abs(cv2.filter2D(g, -1, k))[1:-1, 1:-1].sum()
    return float(s * np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2)))


def measure(bgr):
    """Trả về {'brightness': trung bình L (0-255), 'noise': độ lệch chuẩn nhiễu, 'cast': max/min trung bình kênh}."""
    small = bgr if max(bgr.shape[:2]) <= 1280 else cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0]
    m = small.reshape(-1, 3).astype(np.float64).mean(0)
    return {'brightness': float(L.mean()),
            'noise': estimate_noise(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)),
            'cast': float(m.max() / max(m.min(), 1.0))}


# ---------------------------------------------------------------------------- từng kỹ thuật
def denoise(bgr, noise=None):
    """Lọc song phương; độ mạnh tăng theo mức nhiễu đo được."""
    sc = 60.0 if noise is None else float(np.clip(4.0 * noise, 40.0, 110.0))
    return cv2.bilateralFilter(bgr, 9, sc, 9)


def gray_world(bgr):
    f = bgr.astype(np.float32)
    m = f.reshape(-1, 3).mean(0)
    f *= m.mean() / np.maximum(m, 1.0)
    return np.clip(f, 0, 255).astype(np.uint8)


def gamma(bgr, g):
    lut = np.array([((i / 255.0) ** g) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def clahe(bgr, clip=2.0, tile=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------- ghép lại
def plan(stats, cfg=None):
    """Danh sách bước sẽ áp dụng cho ảnh có số đo stats, theo cấu hình cfg."""
    s = settings(cfg)
    mode = s['mode']
    if mode == 'off':
        return []
    steps = []
    if mode == 'manual':
        if s['denoise']:
            steps.append('denoise')
        if s['white_balance']:
            steps.append('white_balance')
        if s['gamma']:
            steps.append('gamma')
        if s['clahe']:
            steps.append('clahe')
        return steps
    # auto
    if s['denoise'] and stats['noise'] > s['noise_max']:
        steps.append('denoise')
    if s['white_balance'] and stats['cast'] > s['cast_max']:
        steps.append('white_balance')
    if s['clahe'] and stats['brightness'] < s['dark_L']:
        steps.append('clahe')
    return steps


def enhance(bgr, cfg=None, stats=None):
    """Ảnh BGR -> (ảnh đã tăng cường, info). info = {'stats': số đo, 'steps': các bước đã áp dụng}.
    Không bước nào cần thì trả lại chính ảnh đầu vào (không copy)."""
    s = settings(cfg)
    if s['mode'] == 'off':
        return bgr, {'stats': None, 'steps': []}
    st = stats or measure(bgr)
    steps = plan(st, s)
    out = bgr
    for k in steps:
        if k == 'denoise':
            out = denoise(out, st['noise'])
        elif k == 'white_balance':
            out = gray_world(out)
        elif k == 'gamma':
            out = gamma(out, float(s['gamma']))
        elif k == 'clahe':
            out = clahe(out, s['clahe_clip'], int(s['clahe_tile']))
    return out, {'stats': st, 'steps': steps}
