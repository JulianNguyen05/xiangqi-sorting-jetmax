#!/usr/bin/env python3
# Phân tích chất lượng ảnh chụp từ camera JetMax để quyết định có cần tăng cường/xử lý ảnh hay không.
# Chạy trên máy tính (cần: pip install opencv-python numpy):
#   python tests/analyze_images.py data/capture_scan
#   python tests/analyze_images.py data/capture_scan --imgsz 640
#
# Mỗi thư mục con được coi là một NHÓM (ví dụ: static, belt_slow, belt_mid, belt_fast, anh_sang_yeu...),
# ảnh nằm trực tiếp trong thư mục gốc là nhóm ".". Kết quả in ra màn hình và lưu quality_report.csv.
#
# Các ngưỡng cảnh báo là NGƯỠNG THAM KHẢO, phụ thuộc cảnh và băng chuyền của bạn. Hãy so sánh các nhóm
# với nhau (ví dụ độ nét khi băng chuyền chạy so với khi đứng yên) hơn là tin tuyệt đối vào một con số.

import os, sys, csv, argparse
import cv2
import numpy as np

EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
ANALYSIS_SIZE = 640          # cạnh dài khi tính độ nét/nhiễu, để so sánh được giữa các độ phân giải


def noise_sigma(gray):
    """Ước lượng độ lệch chuẩn nhiễu (phương pháp Immerkaer)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(np.mean(np.abs(conv)) * np.sqrt(0.5 * np.pi) / 6.0)


def piece_diameter(gray_small):
    """Ước lượng đường kính quân cờ (px, trên ảnh cạnh dài 640) bằng HoughCircles. Chỉ mang tính tham khảo."""
    blur = cv2.medianBlur(gray_small, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30, minRadius=10, maxRadius=60)
    if circles is None:
        return 0, float('nan')
    d = circles[0, :, 2] * 2
    return len(d), float(np.median(d))


def analyze(path, imgsz):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = ANALYSIS_SIZE / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    b, g, r = [float(c.mean()) for c in cv2.split(small)]
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    n_circ, d640 = piece_diameter(gray)
    return {
        'width': w, 'height': h,
        'sharp_lap': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'sharp_ten': float(np.mean(gx ** 2 + gy ** 2)),
        'brightness': float(gray.mean()),
        'contrast': float(gray.std()),
        'dark_clip_%': float((gray <= 5).mean() * 100),
        'bright_clip_%': float((gray >= 250).mean() * 100),
        'glare_%': float(((hsv[..., 2] >= 245) & (hsv[..., 1] < 40)).mean() * 100),
        'saturation': float(hsv[..., 1].mean()),
        'r_over_g': r / max(g, 1e-6), 'b_over_g': b / max(g, 1e-6),
        'noise_sigma': noise_sigma(gray),
        'n_circles': n_circ,
        'piece_px_640': d640,
        'piece_px_imgsz': d640 * imgsz / ANALYSIS_SIZE if d640 == d640 else float('nan'),
    }


def flags(m, imgsz):
    f = []
    if m['sharp_lap'] < 60:
        f.append('mờ')
    if m['brightness'] < 70:
        f.append('tối')
    if m['brightness'] > 190:
        f.append('quá sáng')
    if m['dark_clip_%'] > 2 or m['bright_clip_%'] > 2:
        f.append('mất chi tiết sáng/tối')
    if m['contrast'] < 35:
        f.append('tương phản thấp')
    if m['noise_sigma'] > 6:
        f.append('nhiễu')
    if m['glare_%'] > 1:
        f.append('chói/phản xạ')
    if m['piece_px_imgsz'] == m['piece_px_imgsz'] and m['piece_px_imgsz'] < 25:
        f.append('quân cờ nhỏ ở imgsz=%d' % imgsz)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--imgsz', type=int, default=640, help='kích thước đầu vào mô hình YOLO sẽ dùng')
    a = ap.parse_args()

    rows, groups = [], {}
    for root, _, files in os.walk(a.folder):
        group = os.path.relpath(root, a.folder)
        for fn in sorted(files):
            if not fn.lower().endswith(EXTS):
                continue
            m = analyze(os.path.join(root, fn), a.imgsz)
            if m is None:
                continue
            m['file'], m['group'] = fn, group
            m['flags'] = ';'.join(flags(m, a.imgsz))
            rows.append(m)
            groups.setdefault(group, []).append(m)
    if not rows:
        print('Không tìm thấy ảnh trong', a.folder)
        return

    keys = ['group', 'file', 'width', 'height', 'sharp_lap', 'sharp_ten', 'brightness', 'contrast',
            'dark_clip_%', 'bright_clip_%', 'glare_%', 'saturation', 'r_over_g', 'b_over_g',
            'noise_sigma', 'n_circles', 'piece_px_640', 'piece_px_imgsz', 'flags']
    out = os.path.join(a.folder, 'quality_report.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in rows:
            w.writerow({k: ('%.2f' % m[k] if isinstance(m[k], float) else m[k]) for k in keys})

    print('%-14s %4s %8s %7s %7s %7s %7s %7s %9s' % (
        'nhóm', 'ảnh', 'độ nét', 'sáng', 'tương', 'nhiễu', 'chói%', 'R/G', 'quân(px)'))
    for g, ms in sorted(groups.items()):
        avg = lambda k: float(np.nanmean([m[k] for m in ms])) if not all(m[k] != m[k] for m in ms) else float('nan')
        print('%-14s %4d %8.1f %7.1f %7.1f %7.2f %7.2f %7.2f %9.1f' % (
            g, len(ms), avg('sharp_lap'), avg('brightness'), avg('contrast'), avg('noise_sigma'),
            avg('glare_%'), avg('r_over_g'), avg('piece_px_imgsz')))
        bad = {}
        for m in ms:
            for fl in m['flags'].split(';'):
                if fl:
                    bad[fl] = bad.get(fl, 0) + 1
        if bad:
            print('   cảnh báo:', ', '.join('%s (%d/%d ảnh)' % (k, v, len(ms)) for k, v in bad.items()))
    print('\nĐã lưu chi tiết từng ảnh:', out)
    print('Độ phân giải ảnh đầu tiên: %dx%d' % (rows[0]['width'], rows[0]['height']))


if __name__ == '__main__':
    main()
