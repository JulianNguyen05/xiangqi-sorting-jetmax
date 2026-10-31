#!/usr/bin/env python3
# Nhận diện quân cờ: YOLO11 (ONNX / TensorRT) + chạy trên 4 góc xoay + xác định màu bằng HSV + tinh chỉnh tâm.
#
# Kết quả mỗi quân: Piece(x, y [px], r [px bán kính], color 'red'/'black', probs[14], votes)
#   probs: xác suất 14 lớp sau khi gộp 4 góc xoay và đã ép theo màu (dùng cho bước sửa nhãn theo số lượng quân).
# Tương thích Python 3.6; không cần ultralytics trên robot.

import math
import os

import cv2
import numpy as np

# đúng thứ tự id lúc train (training/README.md)
CLASSES = ['black_advisor', 'black_bishop', 'black_cannon', 'black_king', 'black_knight', 'black_pawn', 'black_rook',
           'red_advisor', 'red_bishop', 'red_cannon', 'red_king', 'red_knight', 'red_pawn', 'red_rook']
TYPE_CODE = {'advisor': 'a', 'bishop': 'b', 'cannon': 'c', 'king': 'k', 'knight': 'n', 'pawn': 'p', 'rook': 'r'}
CODES = [TYPE_CODE[c.split('_')[1]].upper() if c.startswith('red') else TYPE_CODE[c.split('_')[1]] for c in CLASSES]
RED_IDX = [i for i, c in enumerate(CLASSES) if c.startswith('red')]
BLACK_IDX = [i for i, c in enumerate(CLASSES) if c.startswith('black')]


class Piece(object):
    __slots__ = ('x', 'y', 'r', 'color', 'color_conf', 'probs', 'votes', 'world', 'where', 'final')

    def __init__(self, x, y, r, probs, votes=1):
        self.x, self.y, self.r = float(x), float(y), float(r)
        self.probs = np.asarray(probs, np.float64)
        self.votes = votes
        self.color, self.color_conf = None, 0.0
        self.world = None                     # (x, y) mm hệ thế giới, do vision/world.py điền
        self.where = None                     # ('board', ô, lệch mm) / ('staging', None, None)
        self.final = None                     # mã quân sau khi sửa theo số lượng (planner)

    @property
    def cls(self):
        return int(np.argmax(self.probs))

    @property
    def code(self):
        return CODES[self.cls]

    @property
    def conf(self):
        return float(self.probs.max())

    def __repr__(self):
        return 'Piece(%s %.2f @%.0f,%.0f)' % (self.code, self.conf, self.x, self.y)


# --------------------------------------------------------------------------------------- backends
class OrtBackend(object):
    def __init__(self, path):
        import onnxruntime as ort
        prov = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(path, providers=prov)
        self.inp = self.sess.get_inputs()[0].name
        self.name = 'onnxruntime(%s)' % ','.join(prov)

    def __call__(self, blob):
        return self.sess.run(None, {self.inp: blob})[0]


class CvDnnBackend(object):
    def __init__(self, path):
        self.net = cv2.dnn.readNetFromONNX(path)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass
        self.name = 'cv2.dnn'

    def __call__(self, blob):
        self.net.setInput(blob)
        return self.net.forward()


class _CudaRT(object):
    """Gọi thẳng libcudart (có sẵn trong JetPack) qua ctypes: KHÔNG cần pycuda, vì robot không có mạng để cài."""
    H2D, D2H = 1, 2

    def __init__(self):
        import ctypes
        import ctypes.util
        self.ct = ctypes
        names = ['libcudart.so', ctypes.util.find_library('cudart'), '/usr/local/cuda/lib64/libcudart.so']
        names += ['/usr/local/cuda/lib64/libcudart.so.%s' % v for v in ('10.2', '11.4', '10.0', '12')]
        self.lib, errs = None, []
        for n in names:
            if not n:
                continue
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError as e:
                errs.append(str(e))
        if self.lib is None:
            raise RuntimeError('Không nạp được libcudart: ' + '; '.join(errs))
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]

    def _ok(self, err, what):
        if err != 0:
            raise RuntimeError('CUDA %s lỗi mã %d' % (what, err))

    def malloc(self, nbytes):
        p = self.ct.c_void_p()
        self._ok(self.lib.cudaMalloc(self.ct.byref(p), nbytes), 'cudaMalloc')
        return p.value

    def h2d(self, dptr, arr):
        self._ok(self.lib.cudaMemcpy(dptr, arr.ctypes.data, arr.nbytes, self.H2D), 'memcpy H2D')

    def d2h(self, arr, dptr):
        self._ok(self.lib.cudaMemcpy(arr.ctypes.data, dptr, arr.nbytes, self.D2H), 'memcpy D2H')


class TrtBackend(object):
    """TensorRT engine (build trên chính Jetson: trtexec --onnx=best.onnx --saveEngine=best.engine --fp16).
    Dùng pycuda nếu có, không thì gọi libcudart trực tiếp."""

    def __init__(self, path):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, 'rb') as f:
            rt = trt.Runtime(logger)
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('Không đọc được engine (build trên máy khác / phiên bản TensorRT khác?)')
        self.ctx = self.engine.create_execution_context()
        self.cuda = _CudaRT()
        self.host, self.dev, self.shapes = [], [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            h = np.zeros(int(np.prod(shape)), dtype)
            self.host.append(h)
            self.dev.append(self.cuda.malloc(h.nbytes))
            self.shapes.append(shape)
        self.inp = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)][0]
        self.out = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)][0]
        self.name = 'tensorrt (ctypes cudart)'

    def __call__(self, blob):
        h = self.host[self.inp]
        np.copyto(h, blob.ravel().astype(h.dtype))
        self.cuda.h2d(self.dev[self.inp], h)
        if not self.ctx.execute_v2([int(d) for d in self.dev]):
            raise RuntimeError('TensorRT execute_v2 thất bại')
        o = self.host[self.out]
        self.cuda.d2h(o, self.dev[self.out])
        return o.reshape(self.shapes[self.out]).astype(np.float32)


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def engine_matches(onnx_path, eng_path):
    """tools/build_engine.py ghi md5 của file onnx vào <engine>.src. Không có file .src thì coi như khớp."""
    src = eng_path + '.src'
    if not os.path.exists(src):
        return True
    with open(src) as f:
        return f.read().strip() == file_md5(onnx_path)


def load_backend(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise RuntimeError('Không thấy file model: ' + path)
    if path.endswith('.engine') or path.endswith('.trt'):
        return TrtBackend(path)
    errs = []
    eng = os.path.splitext(path)[0] + '.engine'
    if os.path.exists(eng) and not engine_matches(path, eng):
        print('CẢNH BÁO: %s được build từ model KHÁC %s -> bỏ qua, chạy lại tools/build_engine.py' % (eng, path))
    elif os.path.exists(eng):                  # có engine cạnh file onnx -> ưu tiên (nhanh nhất trên Jetson)
        try:
            return TrtBackend(eng)
        except Exception as e:
            errs.append('TrtBackend(%s): %s' % (eng, e))
    for B in (OrtBackend, CvDnnBackend):
        try:
            b = B(path)
            b(np.zeros((1, 3, 640, 640), np.float32))      # chạy thử: OpenCV cũ nạp được nhưng chạy YOLO11 thì lỗi
            return b
        except Exception as e:                 # thiếu thư viện / OpenCV cũ không chạy được YOLO11
            errs.append('%s: %s' % (B.__name__, str(e).strip().splitlines()[-1][:160]))
    raise RuntimeError('Không nạp được model:\n  ' + '\n  '.join(errs) +
                       '\n  -> Trên Jetson: build engine bằng trtexec (RUNBOOK mục 2c)')


# --------------------------------------------------------------------------------------- YOLO
def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / float(h), size / float(w))
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob), r, px, py


def decode(out, r, px, py, conf, iou, nc=14):
    """Đầu ra YOLOv8/11: (1, 4+nc, N) -> [(cx, cy, w, h, scores[nc])] theo pixel ảnh gốc."""
    o = np.asarray(out)
    if o.ndim == 3:
        o = o[0]
    if o.shape[0] != 4 + nc and o.shape[1] == 4 + nc:
        o = o.T
    boxes, scores = o[:4].T, o[4:4 + nc].T
    best = scores.max(axis=1)
    keep = best >= conf
    boxes, scores, best = boxes[keep], scores[keep], best[keep]
    if len(boxes) == 0:
        return []
    cx = (boxes[:, 0] - px) / r
    cy = (boxes[:, 1] - py) / r
    bw, bh = boxes[:, 2] / r, boxes[:, 3] / r
    rects = [[float(x - w / 2), float(y - h / 2), float(w), float(h)] for x, y, w, h in zip(cx, cy, bw, bh)]
    idx = cv2.dnn.NMSBoxes(rects, best.astype(float).tolist(), conf, iou)       # không phân lớp: quân không chồng nhau
    idx = np.array(idx).reshape(-1) if len(idx) else []
    return [(cx[i], cy[i], bw[i], bh[i], scores[i]) for i in idx]


ROT_CODE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _unrotate(x, y, deg, w, h):
    """Điểm (x, y) trong ảnh đã xoay `deg` độ theo chiều kim đồng hồ -> ảnh gốc kích thước (w, h)."""
    if deg == 90:
        return y, h - 1 - x
    if deg == 180:
        return w - 1 - x, h - 1 - y
    if deg == 270:
        return w - 1 - y, x
    return x, y


class PieceDetector(object):
    def __init__(self, model, imgsz=640, conf=0.35, iou=0.5, rotations=(0, 90, 180, 270)):
        self.net = load_backend(model) if isinstance(model, str) else model
        self.imgsz, self.conf, self.iou = int(imgsz), float(conf), float(iou)
        self.rotations = tuple(rotations)

    def raw(self, bgr):
        blob, r, px, py = letterbox(bgr, self.imgsz)
        return decode(self.net(blob), r, px, py, self.conf, self.iou)

    def detect(self, bgr, expect_r=None):
        """Chạy trên các góc xoay rồi gộp. expect_r: bán kính quân dự kiến (px) nếu biết (lọc box vô lý)."""
        h, w = bgr.shape[:2]
        dets = []                                     # (x, y, r, scores, rot)
        for deg in self.rotations:
            img = bgr if deg == 0 else cv2.rotate(bgr, ROT_CODE[deg])
            for cx, cy, bw, bh, sc in self.raw(img):
                x, y = _unrotate(cx, cy, deg, w, h)
                rr = (bw + bh) / 4.0
                if expect_r and not (0.6 * expect_r < rr < 1.6 * expect_r):
                    continue
                dets.append((x, y, rr, sc, deg))
        return merge_rotations(dets, len(self.rotations))


def merge_rotations(dets, n_rot):
    """Gom các phát hiện cùng một quân ở các góc xoay (tâm cách nhau < 0.6 bán kính).
    Xác suất lớp = trung bình trên MỌI góc xoay (góc nào không thấy tính là 0) -> quân chỉ thấy ở 1 góc bị phạt."""
    dets = sorted(dets, key=lambda d: -float(np.max(d[3])))
    groups = []
    for d in dets:
        for g in groups:
            gx = np.mean([e[0] for e in g])
            gy = np.mean([e[1] for e in g])
            gr = np.mean([e[2] for e in g])
            if math.hypot(d[0] - gx, d[1] - gy) < 0.6 * gr and all(e[4] != d[4] for e in g):
                g.append(d)
                break
        else:
            groups.append([d])
    out = []
    need = max(1, int(math.ceil(n_rot / 2.0)))
    for g in groups:
        if len(g) < need:
            continue
        probs = np.sum([e[3] for e in g], axis=0) / float(n_rot)
        out.append(Piece(np.mean([e[0] for e in g]), np.mean([e[1] for e in g]),
                         np.median([e[2] for e in g]), probs, votes=len(g)))
    return out


# ------------------------------------------------------------------------------------ màu & tâm
def is_piece_surface(bgr, x, y, r):
    """Mặt quân màu kem (bão hoà S ~20..110) khác hẳn giấy trắng / gạch xám (S < 15)."""
    h, w = bgr.shape[:2]
    rr = max(4, int(r * 0.8))
    x0, y0, x1, y1 = max(0, int(x - rr)), max(0, int(y - rr)), min(w, int(x + rr) + 1), min(h, int(y + rr) + 1)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    s = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1]
    return float(np.median(s)) >= 18


def ink_color(bgr, x, y, r):
    """Màu chữ trên mặt quân: đếm điểm ảnh ĐỎ và điểm ảnh MỰC TỐI (xanh đậm/đen) trong hình tròn 0.8r.
    Trả về ('red' | 'black' | None, độ tin 0..1, (số đỏ, số tối))."""
    h, w = bgr.shape[:2]
    rr = max(4, int(r * 0.8))
    x0, y0, x1, y1 = int(x - rr), int(y - rr), int(x + rr) + 1, int(y + rr) + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0, (0, 0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    red = ((H <= 10) | (H >= 160)) & (S >= 80) & (V >= 40) & disk
    # mực quân Đen là xanh đậm (H 70..135). Không dùng "V thấp" vì mặt quân màu kem trong bóng râm cũng tối
    dark = (~red) & disk & (((H >= 70) & (H <= 135) & (S >= 40) & (V < 170)) | ((V < 50) & (S < 80)))
    nr, nd = int(red.sum()), int(dark.sum())
    tot = nr + nd
    if tot < 0.02 * disk.sum():
        return None, 0.0, (nr, nd)
    if nr > nd:
        return 'red', nr / float(tot), (nr, nd)
    return 'black', nd / float(tot), (nr, nd)


def apply_color(bgr, pieces, min_conf=0.75):
    """Ép xác suất lớp theo màu chữ đo được (nếu chắc chắn): quân chữ đỏ thì chỉ còn 7 lớp đỏ."""
    for p in pieces:
        c, cf, _ = ink_color(bgr, p.x, p.y, p.r)
        p.color, p.color_conf = c, cf
        if c is None or cf < min_conf:
            continue
        keep = RED_IDX if c == 'red' else BLACK_IDX
        other = BLACK_IDX if c == 'red' else RED_IDX
        # gộp xác suất cùng loại quân của màu kia sang màu đúng (model hay đúng loại nhưng nhầm màu)
        p.probs[keep] = p.probs[keep] + p.probs[other]
        p.probs[other] = 0.0
    return pieces


def refine_center(gray, x, y, r, search=0.35):
    """Tinh chỉnh tâm bằng đường tròn Hough trong vùng quanh (x, y). Trả về (x, y, r) hoặc giữ nguyên."""
    h, w = gray.shape[:2]
    m = int(r * (1 + search)) + 2
    x0, y0, x1, y1 = max(0, int(x) - m), max(0, int(y) - m), min(w, int(x) + m), min(h, int(y) + m)
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 1.2)
    if crop.size == 0:
        return x, y, r
    cs = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1, minDist=r,
                          param1=80, param2=18, minRadius=int(r * 0.8), maxRadius=int(r * 1.2) + 1)
    if cs is None:
        return x, y, r
    best = None
    for cx, cy, cr in cs[0]:
        d = math.hypot(cx + x0 - x, cy + y0 - y)
        if d < search * r and (best is None or d < best[0]):
            best = (d, cx + x0, cy + y0, cr)
    return (best[1], best[2], best[3]) if best else (x, y, r)


def find_discs(bgr, r_px, min_dist=None):
    """Tìm các quân (đĩa tròn) không cần model: Hough circle trên toàn ảnh. Dùng cho hiệu chỉnh camera
    và làm dự phòng kiểm tra số lượng. r_px: bán kính quân dự kiến (px)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    cs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist or 1.6 * r_px,
                          param1=90, param2=22, minRadius=int(r_px * 0.8), maxRadius=int(r_px * 1.2) + 1)
    if cs is None:
        return []
    out = []
    for x, y, r in cs[0]:
        c, cf, (nr, nd) = ink_color(bgr, x, y, r)
        if c is not None and is_piece_surface(bgr, x, y, r):   # mặt kem + có chữ bên trong -> là quân cờ
            out.append((float(x), float(y), float(r), c, cf))
    return out


def draw(bgr, pieces, label=True):
    img = bgr.copy()
    for p in pieces:
        col = (0, 0, 230) if p.code.isupper() else (200, 80, 0)
        cv2.circle(img, (int(p.x), int(p.y)), int(p.r), col, 2)
        cv2.circle(img, (int(p.x), int(p.y)), 2, (0, 255, 0), -1)
        if label:
            cv2.putText(img, '%s %.0f' % (p.code, 100 * p.conf), (int(p.x - p.r), int(p.y - p.r - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img
