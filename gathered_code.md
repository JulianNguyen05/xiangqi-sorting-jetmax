# GATHERED SOURCE CODE - XIANGQI SORTING JETMAX

- **Thư mục gốc:** `D:\2026_2027\Chuyen-de\xiangqi-sorting-jetmax`
- **Tổng số file đã gôm:** 20

## Mục lục

### Nhóm 1: Lõi hệ thống (bắt buộc)
- [robot\board.py](#robotboardpy)
- [robot\vision_board.py](#robotvisionboardpy)
- [robot\detect_view.py](#robotdetectviewpy)
- [robot\engine.py](#robotenginepy)
- [robot\game.py](#robotgamepy)
- [robot\motion.py](#robotmotionpy)
- [robot\xiangqi_rules.py](#robotxiangqirulespy)
- [configs\board.yaml](#configsboardyaml)
- [configs\vision.yaml](#configsvisionyaml)

### Nhóm 2: Cách bạn đang điều khiển tay và chạy mô phỏng
- [tests\play_sim.py](#testsplaysimpy)
- [tests\fake_uci_engine.py](#testsfakeucienginepy)
- [tests\scan_detect.py](#testsscandetectpy)
- [tests\move_piece.py](#testsmovepiecepy)
- [tests\pick_place.py](#testspickplacepy)
- [tests\joint_teleop.py](#testsjointteleoppy)

### Nhóm 3: Tham chiếu SDK của robot (để viết motion.py đúng API)
- [docs\robot-reference\waste_classification_stepper.py](#docsrobot-referencewasteclassificationstepperpy)
- [docs\robot-reference\jetmax_control\scripts\jetmax_control_main.py](#docsrobot-referencejetmaxcontrolscriptsjetmaxcontrolmainpy)

### Nhóm 4: Chỉ gửi nếu tôi hỏi thêm
- [tests\test_game.py](#teststestgamepy)
- [tests\test_rules.py](#teststestrulespy)
- [README.md](#readmemd)

---

## Nhóm 1: Lõi hệ thống (bắt buộc)

### `robot\board.py`

*Dung lượng: 2.65 KB | Số dòng: 58*

```python
#!/usr/bin/env python3
# Bàn cờ tướng: tên ô <-> chỉ số, và ánh xạ ô cờ -> (thanh trượt, x, y) của robot từ 4 góc đã dạy.
# Thuần Python, không cần ROS, chạy thử được trên laptop.
#
# Quy ước ô (giống UCCI, ví dụ "h2e2" = pháo đỏ ở h2 đi tới e2):
#   cột a..i từ TRÁI sang PHẢI nhìn từ phía quân ĐỎ; hàng 0..9 từ phía đỏ (hàng 0) lên phía đen (hàng 9).
#
#   hàng 9   a9 b9 c9 d9 e9 f9 g9 h9 i9     (phía đen)
#   ...
#   hàng 0   a0 b0 c0 d0 e0 f0 g0 h0 i0     (phía đỏ)
#
# Cách ánh xạ: dạy 4 góc a0, i0, a9, i9 (mỗi góc gồm vị trí thanh trượt + x + y của tay máy), rồi nội suy song tuyến
# cho cả ba giá trị. Vị trí robot là hàm tuyến tính của (thanh trượt, x, y) nên cách này đúng khi bàn cờ là hình chữ nhật
# và thanh trượt chạy thẳng. Không cần biết bước/mm hay hướng trượt.

FILES = 'abcdefghi'
NF, NR = 9, 10
CORNERS = ('a0', 'i0', 'a9', 'i9')


def parse_square(s):
    s = s.strip().lower()
    if len(s) != 2 or s[0] not in FILES or not s[1].isdigit():
        raise ValueError('Ô "%s" không hợp lệ (cột a-i, hàng 0-9, ví dụ h2)' % s)
    return FILES.index(s[0]), int(s[1])


def square_name(f, r):
    return '%s%d' % (FILES[f], r)


class Board:
    def __init__(self, corners, z_pick, z_safe=50.0, z_place=None):
        miss = [c for c in CORNERS if c not in corners or any(v is None for v in corners[c])]
        if miss:
            raise ValueError('Chưa điền góc: %s (slider, x, y)' % ', '.join(miss))
        if z_pick is None:
            raise ValueError('Chưa điền z_pick')
        self.c = {k: tuple(float(v) for v in corners[k]) for k in CORNERS}
        self.z_pick = float(z_pick)
        self.z_safe = float(z_safe)
        self.z_place = float(z_place) if z_place is not None else self.z_pick + 3.0

    @classmethod
    def load(cls, path):
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        corners = {k: (v.get('slider'), v.get('x'), v.get('y')) for k, v in (d.get('corners') or {}).items()}
        return cls(corners, d.get('z_pick'), d.get('z_safe', 50.0), d.get('z_place'))

    def position(self, sq):
        """Ô cờ (tên 'h2' hoặc (cột, hàng)) -> (slider:int, x, y) tại tâm giao điểm."""
        f, r = parse_square(sq) if isinstance(sq, str) else sq
        u, v = f / float(NF - 1), r / float(NR - 1)
        w = ((1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v)
        p = [sum(w[i] * self.c[CORNERS[i]][k] for i in range(4)) for k in range(3)]
        return int(round(p[0])), round(p[1], 1), round(p[2], 1)
```

---

### `robot\vision_board.py`

*Dung lượng: 5.08 KB | Số dòng: 105*

```python
#!/usr/bin/env python3
# Chuyển kết quả nhận diện YOLO (khung + lớp) thành bàn cờ {(cột, hàng): 'R'} để đưa vào game.py.
#
# Cách làm: dùng 4 góc bàn cờ trong ẢNH (pixel tại giao điểm a0, i0, a9, i9) để tính phép biến đổi phối cảnh sang lưới
# 9x10 ô; tâm mỗi khung quân cờ được đổi sang toạ độ lưới rồi làm tròn về giao điểm gần nhất.
# Cần điền configs/vision.yaml. Camera phải ở MỘT TƯ THẾ CỐ ĐỊNH nhìn thấy cả bàn cờ (nếu camera gắn trên tay máy thì
# phải đưa tay máy về đúng tư thế quan sát này mỗi lần chụp, vì 4 góc chỉ đúng cho tư thế đó).
#
# Định dạng đầu vào khớp với robot/detect_view.py: mỗi quân là (x1, y1, x2, y2, conf, cls_id) và `names[cls_id]` là
# 'red_rook', 'black_cannon', ... (đúng thứ tự 14 lớp trong training/build_dataset.ipynb).

import numpy as np

CORNER_KEYS = ('a0', 'i0', 'a9', 'i9')
PIECE_CODE = {'advisor': 'a', 'bishop': 'b', 'cannon': 'c', 'king': 'k', 'knight': 'n', 'pawn': 'p', 'rook': 'r'}


def class_to_piece(name):
    """'red_rook' -> 'R', 'black_cannon' -> 'c'. Trả về None nếu tên lạ."""
    try:
        color, kind = name.strip().lower().split('_', 1)
    except ValueError:
        return None
    code = PIECE_CODE.get(kind)
    if code is None or color not in ('red', 'black'):
        return None
    return code.upper() if color == 'red' else code


def _homography(src, dst):
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.solve(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)


class ImageBoardMapper(object):
    def __init__(self, image_corners, max_offset=0.35, anchor_y=0.5):
        """image_corners: {'a0': (x, y), 'i0': ..., 'a9': ..., 'i9': ...} (pixel).
        max_offset: quân lệch tâm giao điểm quá mức này (đơn vị ô) thì bỏ (đang bị cầm hoặc nằm giữa hai ô).
        anchor_y: vị trí lấy điểm trong khung theo chiều dọc, 0.5 = tâm khung, 1.0 = cạnh dưới."""
        miss = [k for k in CORNER_KEYS if k not in image_corners or image_corners[k] is None]
        if miss:
            raise ValueError('Chưa điền góc ảnh: %s (x, y pixel) trong configs/vision.yaml' % ', '.join(miss))
        src = [tuple(float(v) for v in image_corners[k]) for k in CORNER_KEYS]
        dst = [(0.0, 0.0), (8.0, 0.0), (0.0, 9.0), (8.0, 9.0)]
        self.H = _homography(src, dst)
        self.max_offset = float(max_offset)
        self.anchor_y = float(anchor_y)

    @classmethod
    def load(cls, path):
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(d.get('image_corners') or {}, d.get('max_offset', 0.35), d.get('anchor_y', 0.5))

    def to_grid(self, x, y):
        v = self.H.dot(np.array([x, y, 1.0]))
        return v[0] / v[2], v[1] / v[2]

    def board_from_detections(self, dets, names):
        """Trả về (bàn cờ {(f, r): ký tự}, danh sách cảnh báo)."""
        board, conf_at, warns = {}, {}, []
        for x1, y1, x2, y2, conf, cls in dets:
            k = int(cls)
            name = names[k] if k < len(names) else str(k)
            piece = class_to_piece(name)
            if piece is None:
                warns.append('lớp lạ "%s"' % name)
                continue
            gf, gr = self.to_grid((x1 + x2) / 2.0, y1 + (y2 - y1) * self.anchor_y)
            f, r = int(round(gf)), int(round(gr))
            if not (0 <= f < 9 and 0 <= r < 10):
                warns.append('%s ngoài bàn cờ' % name)
                continue
            if abs(gf - f) > self.max_offset or abs(gr - r) > self.max_offset:
                warns.append('%s lệch tâm ô (%.2f, %.2f)' % (name, gf - f, gr - r))
                continue
            if (f, r) in board:
                warns.append('hai quân cùng một ô (%d,%d), giữ quân tin cậy hơn' % (f, r))
                if conf <= conf_at[(f, r)]:
                    continue
            board[(f, r)] = piece
            conf_at[(f, r)] = conf
        return board, warns


class BoardObserver(object):
    """Ghép camera + YOLO + ánh xạ ô + gộp khung hình thành hàm observe() cho game.play().
      detector   : đối tượng từ detect_view.load_backend(...) (gọi detector(bgr) -> Nx6, có thuộc tính .names)
      grab_frame : hàm không tham số trả về ảnh BGR mới nhất (ví dụ dùng cách đọc topic /usb_cam trong scan_detect.py)
      stabilizer : game.BoardStabilizer"""

    def __init__(self, detector, grab_frame, mapper, stabilizer):
        self.detector, self.grab_frame, self.mapper, self.stab = detector, grab_frame, mapper, stabilizer
        self.last_warnings = []

    def __call__(self):
        dets = self.detector(self.grab_frame())
        board, self.last_warnings = self.mapper.board_from_detections(dets, self.detector.names)
        return self.stab.update(board)
```

---

### `robot\detect_view.py`

*Dung lượng: 11.09 KB | Số dòng: 249*

```python
#!/usr/bin/env python3
# Hiển thị nhận diện quân cờ (YOLO11) từ camera JetMax. CHƯA điều khiển tay máy, chỉ vẽ khung + tên + độ tin cậy.
# Cần: node usb_cam đang chạy (topic ảnh RGB). Không cần node jetmax_control.
#
# Ví dụ:
#   python3 detect_view.py --model ~/models/best.pt                 # dùng ultralytics (nếu cài được trên robot)
#   python3 detect_view.py --model ~/models/best.onnx               # onnxruntime nếu có, không thì cv2.dnn (OpenCV cũ không chạy được YOLO11)
#   python3 detect_view.py --model ~/models/best.engine             # TensorRT (tạo bằng trtexec trên robot)
#   python3 detect_view.py --model ~/models/best.onnx --image a.jpg # thử trên ảnh tĩnh, không cần ROS
#   python3 detect_view.py --model ~/models/best.onnx --conf 0.4 --save-dir ~/detect_out
#
# Phím: q / Esc = thoát, s = lưu khung hình đã vẽ vào --save-dir.

import os, time, argparse
import cv2
import numpy as np

TOPIC = '/usb_cam/image_rect_color'
IMGSZ = 640
# Thứ tự lớp theo build_dataset.ipynb (id 0-6 = black, 7-13 = red). Với .pt thì lấy tên ngay trong mô hình.
NAMES = ['%s_%s' % (c, p) for c in ('black', 'red')
         for p in ('advisor', 'bishop', 'cannon', 'king', 'knight', 'pawn', 'rook')]
WIN = 'detect_view (q/Esc = thoat, s = luu)'


# ---------------------------------------------------------------- backend ----
class TorchBackend:
    """Ultralytics + best.pt. Nhận ảnh BGR, trả về mảng Nx6: x1 y1 x2 y2 conf cls."""

    def __init__(self, path, conf, iou):
        from ultralytics import YOLO
        self.m, self.conf, self.iou = YOLO(path), conf, iou
        self.names = [self.m.names[i] for i in sorted(self.m.names)]

    def __call__(self, bgr):
        r = self.m.predict(bgr, imgsz=IMGSZ, conf=self.conf, iou=self.iou, verbose=False)[0]
        if len(r.boxes) == 0:
            return np.zeros((0, 6), np.float32)
        b = r.boxes
        return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                          b.cls.cpu().numpy()[:, None]]).astype(np.float32)


def preprocess(bgr):
    """Letterbox giống Ultralytics -> blob (1,3,640,640) RGB float32 + tham số để đổi toạ độ về ảnh gốc."""
    h, w = bgr.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True)   # BGR -> RGB
    return blob, (r, left, top, w, h)


def postprocess(out, meta, conf_thr, iou_thr):
    """Đầu ra YOLO11 (1, 4+nc, 8400) -> mảng Nx6: x1 y1 x2 y2 conf cls (toạ độ ảnh gốc), đã NMS theo lớp."""
    r, left, top, w, h = meta
    p = np.asarray(out).reshape(1, -1, out.shape[-1])[0].T if np.ndim(out) == 3 else np.asarray(out).T
    scores = p[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(p)), cls]
    keep = conf >= conf_thr
    p, cls, conf = p[keep], cls[keep], conf[keep]
    if len(p) == 0:
        return np.zeros((0, 6), np.float32)
    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    x1, y1 = (cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r
    x2, y2 = (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r
    boxes = np.stack([x1, y1, x2, y2], 1)
    off = cls[:, None] * 4096.0                             # NMS theo từng lớp
    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], 1) + np.hstack([off, off, 0 * off, 0 * off])
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), conf_thr, iou_thr)
    idx = np.array(idx).reshape(-1).astype(int)
    boxes = np.clip(boxes[idx], 0, [w, h, w, h])
    return np.hstack([boxes, conf[idx, None], cls[idx, None]]).astype(np.float32)


class OnnxBackend:
    """best.onnx qua onnxruntime (nếu có) hoặc cv2.dnn. Lưu ý: cv2.dnn của OpenCV cũ (4.5.x) không chạy được YOLO11."""

    def __init__(self, path, conf, iou):
        self.conf, self.iou, self.names = conf, iou, NAMES
        self.ort = None
        try:
            import onnxruntime as ort
            prov = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                    if p in ort.get_available_providers()]
            self.ort = ort.InferenceSession(path, providers=prov)
            self.inp = self.ort.get_inputs()[0].name
            print('Backend: onnxruntime', prov)
        except ImportError:
            self.net = cv2.dnn.readNetFromONNX(path)
            print('Backend: cv2.dnn (OpenCV %s)' % cv2.__version__)

    def __call__(self, bgr):
        blob, meta = preprocess(bgr)
        if self.ort is not None:
            out = self.ort.run(None, {self.inp: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        return postprocess(out, meta, self.conf, self.iou)


class TrtBackend:
    """TensorRT engine (.engine/.trt) tạo bằng trtexec ngay trên robot. Cần python3 có tensorrt và pycuda (Hiwonder đã cài)."""

    def __init__(self, path, conf, iou):
        import tensorrt as trt
        import pycuda.autoinit            # noqa: F401  (tạo CUDA context)
        import pycuda.driver as cuda
        self.cuda, self.conf, self.iou, self.names = cuda, conf, iou, NAMES
        with open(path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('Không nạp được engine (tạo trên robot này, cùng phiên bản TensorRT chưa?)')
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.h_in, self.d_in, self.h_out, self.d_out = [], None, None, None, None
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host = cuda.pagelocked_empty(int(trt.volume(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            if self.engine.binding_is_input(i):
                self.h_in, self.d_in = host, dev
            else:
                self.h_out, self.d_out, self.out_shape = host, dev, shape
        print('Backend: TensorRT, đầu ra', self.out_shape)

    def __call__(self, bgr):
        blob, meta = preprocess(bgr)
        self.h_in[:] = blob.ravel().astype(self.h_in.dtype)
        self.cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        out = self.h_out.reshape(self.out_shape).astype(np.float32)
        return postprocess(out, meta, self.conf, self.iou)


def load_backend(path, conf, iou):
    low = path.lower()
    if low.endswith('.pt'):
        return TorchBackend(path, conf, iou)
    if low.endswith('.engine') or low.endswith('.trt'):
        return TrtBackend(path, conf, iou)
    return OnnxBackend(path, conf, iou)


# ------------------------------------------------------------------- vẽ ------
def draw(bgr, dets, names, fps, ms):
    for x1, y1, x2, y2, c, k in dets:
        k = int(k)
        name = names[k] if k < len(names) else str(k)
        col = (0, 0, 255) if name.startswith('red') else (255, 255, 0)   # BGR: đỏ / vàng-xanh cho quân đen
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(bgr, p1, p2, col, 2)
        label = '%s %.2f' % (name, c)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(bgr, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), col, -1)
        cv2.putText(bgr, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(bgr, 'FPS %.1f  infer %.0f ms  quan %d' % (fps, ms, len(dets)), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return bgr


def save(img, d):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'det_%d.jpg' % int(time.time() * 1000))
    cv2.imwrite(p, img)
    print('đã lưu', p)


# ------------------------------------------------------------------ main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='đường dẫn best.pt hoặc best.onnx')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--image', help='chạy trên một ảnh tĩnh (BGR, jpg/png) thay vì camera')
    ap.add_argument('--save-dir', default=os.path.expanduser('~/detect_out'))
    a = ap.parse_args()

    det = load_backend(os.path.expanduser(a.model), a.conf, a.iou)

    if a.image:
        img = cv2.imread(os.path.expanduser(a.image))
        if img is None:
            raise SystemExit('Không đọc được ảnh: %s (kiểm tra đường dẫn đầy đủ)' % a.image)
        t0 = time.time()
        dets = det(img)
        ms = (time.time() - t0) * 1000
        for d in dets:
            print('%-14s %.2f  (%d,%d,%d,%d)' % (det.names[int(d[5])], d[4], *d[:4]))
        print('%d quân, suy luận %.0f ms' % (len(dets), ms))
        vis = draw(img, dets, det.names, 0, ms)
        os.makedirs(os.path.expanduser(a.save_dir), exist_ok=True)
        outp = os.path.join(os.path.expanduser(a.save_dir), 'det_' + os.path.basename(a.image))
        cv2.imwrite(outp, vis)
        print('Đã lưu ảnh kết quả:', outp)       # mở bằng trình xem ảnh nếu cửa sổ không hiện
        try:
            cv2.imshow(WIN, vis)
            cv2.waitKey(0)
        except cv2.error as e:
            print('Không mở được cửa sổ hiển thị (%s)' % e)
        return

    import rospy
    from sensor_msgs.msg import Image
    state = {'img': None, 'n': 0}

    def on_image(msg):
        state['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()  # RGB
        state['n'] += 1

    rospy.init_node('detect_view', anonymous=True)
    rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
    print('Chờ ảnh từ', TOPIC, '...')
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    last_n, fps, t_prev = 0, 0.0, time.time()
    while not rospy.is_shutdown():
        if state['img'] is None or state['n'] == last_n:
            if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
                break
            continue
        last_n = state['n']
        bgr = cv2.cvtColor(state['img'], cv2.COLOR_RGB2BGR)   # mô hình train trên ảnh BGR (cv2.imread)
        t0 = time.time()
        dets = det(bgr)
        ms = (time.time() - t0) * 1000
        now = time.time()
        fps = 1.0 / (now - t_prev) if fps == 0 else fps * 0.8 + 0.2 / (now - t_prev)
        t_prev = now
        vis = draw(bgr, dets, det.names, fps, ms)
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('s'):
            save(vis, a.save_dir)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

---

### `robot\engine.py`

*Dung lượng: 10.30 KB | Số dòng: 241*

```python
#!/usr/bin/env python3
# Engine chọn nước đi cho robot. Hai lựa chọn, cùng giao diện:   move = engine.choose(position)
#
#   UciEngine    : chạy Pikafish (engine cờ tướng mã nguồn mở, giao tiếp UCI) như một tiến trình con.
#                  Đã cấu hình NHẸ cho Jetson: 1 luồng, hash 16 MB, giới hạn độ sâu + thời gian suy nghĩ.
#                  Cần: file chạy `pikafish` (bản build cho aarch64/ARM64) và file mạng `pikafish.nnue`
#                  để cạnh file chạy (hoặc truyền --nnue).
#   SimpleEngine : tìm kiếm alpha-beta độ sâu 2-3 tự viết, chỉ tính điểm theo quân. Không cần cài gì, dùng để thử
#                  logic khi chưa có Pikafish, hoặc dự phòng nếu Pikafish lỗi.
#
# "Mức dễ" được làm bằng cách cho engine trả về nhiều nước tốt nhất (MultiPV), rồi với một xác suất nhất định chọn
# NGẪU NHIÊN trong các nước không tệ hơn nước tốt nhất quá `margin` (đơn vị centipawn, 100 ~ một tốt).
# Cách này không phụ thuộc phiên bản engine nên mức khó dễ chỉnh được ngay trong LEVELS bên dưới.

import os, re, sys, random, subprocess, threading, time
try:
    import queue
except ImportError:           # Python 2 (không dùng, để cho chắc)
    import Queue as queue

from xiangqi_rules import Position, RED, BLACK, has_crossed, parse_uci, move_uci

LEVELS = {
    #          độ sâu  thời gian(ms)  số nước ứng viên  chênh lệch tối đa  xác suất đi "lệch"
    'easy':   dict(depth=1, movetime=800, multipv=5, margin=200, p_random=0.40, simple_depth=2),
    'normal': dict(depth=3, movetime=1500, multipv=3, margin=100, p_random=0.20, simple_depth=2),
    'hard':   dict(depth=8, movetime=3000, multipv=1, margin=0, p_random=0.0, simple_depth=3),
}
MATE = 30000


def _pick(cands, cfg, rng):
    """cands: [(move, score)] đã sắp theo điểm giảm dần. Chọn nước theo mức độ."""
    best = cands[0][1]
    if len(cands) > 1 and cfg['p_random'] > 0 and rng.random() < cfg['p_random']:
        pool = [m for m, s in cands if s >= best - cfg['margin']]
        return rng.choice(pool)
    return cands[0][0]


# ------------------------------------------------------------------ Pikafish -----
class UciEngine(object):
    def __init__(self, path, level='easy', nnue=None, threads=1, hash_mb=16, seed=None, startup_timeout=30.0,
                 args=None):
        if level not in LEVELS:
            raise ValueError('level phải là một trong %s' % sorted(LEVELS))
        self.cfg = LEVELS[level]
        self.rng = random.Random(seed)
        self.name = 'pikafish(%s)' % level
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            raise IOError('Không thấy engine: %s' % path)
        self.proc = subprocess.Popen([path] + list(args or []), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1,
                                     cwd=os.path.dirname(path))
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        self._send('uci')
        lines = self._read_until(lambda l: l.strip() == 'uciok', startup_timeout)
        self.options = set()
        for l in lines:
            m = re.match(r'option name (.+?) type', l.strip())
            if m:
                self.options.add(m.group(1))
        self._set('Threads', threads)
        self._set('Hash', hash_mb)
        self._set('MultiPV', self.cfg['multipv'])
        if nnue:
            self._set('EvalFile', os.path.abspath(os.path.expanduser(nnue)))
        self._send('isready')
        self._read_until(lambda l: l.strip() == 'readyok', startup_timeout)
        self._send('ucinewgame')

    # ---- giao tiếp
    def _reader(self):
        for line in iter(self.proc.stdout.readline, ''):
            self.q.put(line)
        self.q.put(None)

    def _send(self, s):
        try:
            self.proc.stdin.write(s + '\n')
            self.proc.stdin.flush()
        except (IOError, OSError, ValueError):
            raise RuntimeError('Engine đã thoát (kiểm tra file nnue và bản build có chạy được trên robot không)')

    def _read_until(self, pred, timeout):
        end = time.time() + timeout
        lines = []
        while True:
            left = end - time.time()
            if left <= 0:
                raise RuntimeError('Engine không trả lời trong %.0f giây' % timeout)
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                raise RuntimeError('Engine không trả lời trong %.0f giây' % timeout)
            if line is None:
                raise RuntimeError('Engine đã thoát. Đầu ra cuối: %s' % ' | '.join(l.strip() for l in lines[-5:]))
            lines.append(line)
            if pred(line):
                return lines

    def _set(self, name, value):
        if name in self.options:
            self._send('setoption name %s value %s' % (name, value))
        else:
            print('CẢNH BÁO: engine không có tùy chọn "%s", bỏ qua' % name, file=sys.stderr)

    # ---- chọn nước
    def choose(self, pos):
        legal = pos.legal_moves()
        if not legal:
            raise ValueError('Không còn nước đi hợp lệ')
        c = self.cfg
        self._send('position fen ' + pos.fen())
        self._send('go depth %d movetime %d' % (c['depth'], c['movetime']))
        limit = c['movetime'] / 1000.0 + 10.0
        try:
            lines = self._read_until(lambda l: l.startswith('bestmove'), limit)
        except RuntimeError:
            self._send('stop')                       # engine chậm hơn dự kiến: ép dừng và lấy kết quả hiện có
            lines = self._read_until(lambda l: l.startswith('bestmove'), 5.0)

        best = {}                                    # multipv -> (độ sâu, điểm, nước)
        for l in lines:
            tok = l.split()
            if not tok or tok[0] != 'info' or 'pv' not in tok or 'score' not in tok:
                continue
            try:
                d = int(tok[tok.index('depth') + 1]) if 'depth' in tok else 0
                k = int(tok[tok.index('multipv') + 1]) if 'multipv' in tok else 1
                i = tok.index('score')
                v = int(tok[i + 2])
                score = v if tok[i + 1] == 'cp' else (MATE - abs(v)) * (1 if v > 0 else -1)
                mv = parse_uci(tok[tok.index('pv') + 1])
            except (ValueError, IndexError):
                continue
            if k not in best or d >= best[k][0]:
                best[k] = (d, score, mv)
        legal_set = set(legal)
        cands = sorted([(mv, s) for d, s, mv in best.values() if mv in legal_set], key=lambda x: -x[1])
        if not cands:                                # không đọc được info: dùng bestmove
            for l in lines:
                if l.startswith('bestmove'):
                    try:
                        mv = parse_uci(l.split()[1])
                        if mv in legal_set:
                            cands = [(mv, 0)]
                    except (ValueError, IndexError):
                        pass
        if not cands:
            return self.rng.choice(legal)
        return _pick(cands, c, self.rng)

    def close(self):
        try:
            self._send('quit')
            self.proc.wait()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


# ------------------------------------------------------------------ tự viết -----
VALUE = {'K': 0, 'A': 200, 'B': 200, 'N': 400, 'R': 900, 'C': 450, 'P': 100}


def evaluate(pos):
    """Điểm theo quân, từ góc nhìn của bên đang đến lượt."""
    s = 0
    for r in range(10):
        row = pos.b[r]
        for f in range(9):
            p = row[f]
            if p == '.':
                continue
            t = p.upper()
            v = VALUE[t]
            if t == 'P' and has_crossed(r, RED if p.isupper() else BLACK):
                v += 100
            s += v if p.isupper() else -v
    return s if pos.turn == RED else -s


class SimpleEngine(object):
    def __init__(self, level='easy', seed=None):
        if level not in LEVELS:
            raise ValueError('level phải là một trong %s' % sorted(LEVELS))
        self.cfg = LEVELS[level]
        self.depth = self.cfg['simple_depth']
        self.rng = random.Random(seed)
        self.name = 'simple(%s, độ sâu %d)' % (level, self.depth)

    def _negamax(self, pos, depth, alpha, beta, ply):
        if depth == 0:
            return evaluate(pos)
        moves = pos.legal_moves()
        if not moves:
            return -MATE + ply                       # bên đến lượt hết nước = thua
        moves.sort(key=lambda m: -VALUE.get(pos.b[m[3]][m[2]].upper(), 0))   # ăn quân trước để cắt tỉa tốt hơn
        best = -MATE - 1
        for m in moves:
            s = -self._negamax(pos.make(m), depth - 1, -beta, -alpha, ply + 1)
            if s > best:
                best = s
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def choose(self, pos):
        moves = pos.legal_moves()
        if not moves:
            raise ValueError('Không còn nước đi hợp lệ')
        self.rng.shuffle(moves)
        scored = [(m, -self._negamax(pos.make(m), self.depth - 1, -MATE - 1, MATE + 1, 1)) for m in moves]
        scored.sort(key=lambda x: -x[1])
        # nếu có nước chiếu bí thì luôn đi, không "đi lệch"
        if scored[0][1] >= MATE - 10:
            return scored[0][0]
        return _pick(scored, self.cfg, self.rng)

    def close(self):
        pass


def create_engine(path=None, level='easy', nnue=None, seed=None):
    """Có đường dẫn Pikafish thì dùng Pikafish; lỗi hoặc không có thì dùng SimpleEngine (có in cảnh báo)."""
    if path:
        try:
            return UciEngine(path, level=level, nnue=nnue, seed=seed)
        except Exception as e:
            print('CẢNH BÁO: không chạy được Pikafish (%s). Dùng engine tự viết thay thế.' % e, file=sys.stderr)
    return SimpleEngine(level=level, seed=seed)
```

---

### `robot\game.py`

*Dung lượng: 9.34 KB | Số dòng: 219*

```python
#!/usr/bin/env python3
# Logic ván cờ giữa người và robot (thuần Python, không cần ROS, không cần robot).
#
#   Người cầm ĐỎ  -> người đi trước, robot cầm ĐEN đi sau.
#   Người cầm ĐEN -> robot cầm ĐỎ và đi trước.
#
# Luồng chạy:
#   robot đến lượt : engine.choose(thế cờ) -> executor.execute(nước đi) -> cập nhật thế cờ
#   người đến lượt : observe() trả về bàn cờ nhìn thấy từ camera {(cột, hàng): 'R', ...}
#                    -> detect_move() so với thế cờ hiện tại để TÌM RA nước người vừa đi (và kiểm tra hợp lệ)
#
# Nhận diện nước đi bằng cách thử mọi nước hợp lệ rồi chọn nước cho bàn cờ giống ảnh nhất, nên chịu được việc YOLO
# bỏ sót/nhận nhầm 1 quân (tolerance) và tự loại các tư thế dở dang (tay che, quân đang cầm lên).
#
# Cách nối với robot thật: viết một lớp kế thừa RobotExecutor, trong execute(info) dùng
#   - info.captured khác None: gắp quân ở info.to đem bỏ ra khay (quân bị ăn) TRƯỚC,
#   - rồi gắp quân ở info.frm đặt xuống info.to.
# Ở bước đó có thể dùng Board.position() + Mover.pick()/place() đã có trong robot/board.py và robot/motion.py.
#
# Chưa cài: luật cấm chiếu liên tục/đuổi liên tục và hòa 60 nước không ăn quân; chỉ có giới hạn tổng số nước (max_plies).

import time
from collections import namedtuple

from xiangqi_rules import Position, RED, BLACK, other, move_uci, sq_name, piece_name

MoveInfo = namedtuple('MoveInfo', 'move uci frm to piece captured by')
DetectResult = namedtuple('DetectResult', 'status move distance candidates')
# status: 'nochange' (bàn cờ chưa đổi) | 'pending' (đã thấy một nước, chờ xác nhận) | 'move' (đã xác nhận)
#         'ambiguous' (nhiều nước cùng khớp) | 'invalid' (không nước hợp lệ nào khớp)


def make_info(pos, move, by):
    f1, r1, f2, r2 = move
    cap = pos.b[r2][f2]
    return MoveInfo(move, move_uci(move), sq_name(f1, r1), sq_name(f2, r2), pos.b[r1][f1],
                    None if cap == '.' else cap, by)


def describe(info):
    s = '%s: %s %s -> %s' % ('Robot' if info.by == 'robot' else 'Người', piece_name(info.piece), info.frm, info.to)
    if info.captured:
        s += ' (ăn %s)' % piece_name(info.captured)
    return s


# ------------------------------------------------------------------ robot --------
class RobotError(Exception):
    pass


class RobotExecutor(object):
    def execute(self, info):
        """Thực hiện nước đi bằng tay máy. Ném RobotError nếu thất bại (khi đó thế cờ KHÔNG được cập nhật)."""
        raise NotImplementedError


class DryRunExecutor(RobotExecutor):
    """Chưa có robot: chỉ in các bước sẽ làm."""

    def __init__(self, log=print):
        self.log = log

    def execute(self, info):
        if info.captured:
            self.log('    [robot] gắp %s ở %s -> bỏ ra khay quân bị ăn' % (piece_name(info.captured), info.to))
        self.log('    [robot] gắp %s ở %s -> đặt xuống %s' % (piece_name(info.piece), info.frm, info.to))


# ------------------------------------------------------------------ nhận diện nước đi
def board_distance(a, b):
    """Số ô khác nhau (ô sai quân tính 2, thiếu/thừa quân tính 1)."""
    return len(set(a.items()) ^ set(b.items()))


def detect_move(pos, observed, tolerance=1):
    """So bàn cờ nhìn thấy với thế cờ `pos` (người đang đến lượt) và tìm nước người vừa đi."""
    prev = pos.to_dict()
    d_prev = board_distance(prev, observed)
    if d_prev == 0:
        return DetectResult('nochange', None, 0, [])
    scored = []
    for m in pos.legal_moves():
        exp = dict(prev)
        exp[(m[2], m[3])] = exp.pop((m[0], m[1]))
        scored.append((board_distance(exp, observed), m))
    if not scored:
        return DetectResult('invalid', None, d_prev, [])
    scored.sort()
    best = scored[0][0]
    ties = [m for d, m in scored if d == best]
    if d_prev <= tolerance and d_prev < best:
        return DetectResult('nochange', None, d_prev, [])      # chỉ là nhiễu nhận diện, người chưa đi
    if best > tolerance:
        return DetectResult('invalid', None, best, [])
    if len(ties) > 1 or d_prev == best:
        return DetectResult('ambiguous', None, best, [move_uci(m) for m in ties])
    return DetectResult('move', ties[0], best, [move_uci(ties[0])])


class BoardStabilizer(object):
    """Gộp nhiều khung hình liên tiếp: ô nào có cùng một quân ở đa số khung hình thì mới tin.
    Trả về None cho tới khi đủ `frames` khung hình, nên tay đang che bàn cờ hay một khung hình hỏng không làm sai nước."""

    def __init__(self, frames=3):
        self.n = max(1, int(frames))
        self.buf = []

    def reset(self):
        self.buf = []

    def update(self, observed):
        self.buf.append(dict(observed))
        self.buf = self.buf[-self.n:]
        if len(self.buf) < self.n:
            return None
        need = self.n // 2 + 1
        count = {}
        for b in self.buf:
            for item in b.items():
                count[item] = count.get(item, 0) + 1
        return dict(k for k, c in count.items() if c >= need)


# ------------------------------------------------------------------ ván cờ ------
class Game(object):
    def __init__(self, human='red', engine=None, executor=None, start=None, tolerance=1, confirm=2, max_plies=400):
        h = str(human).lower()
        if h not in ('red', 'black', 'w', 'b'):
            raise ValueError('human phải là "red" hoặc "black"')
        self.human = RED if h in ('red', 'w') else BLACK
        self.robot = other(self.human)
        self.engine = engine
        self.executor = executor if executor is not None else DryRunExecutor()
        self.pos = start if start is not None else Position.start()
        self.tolerance = tolerance
        self.confirm = max(1, int(confirm))      # số lần liên tiếp phải thấy cùng một nước mới chấp nhận
        self.max_plies = max_plies
        self.history = []
        self._cand = (None, 0)

    def robot_to_move(self):
        return self.pos.turn == self.robot

    def result(self):
        """None nếu còn đang chơi, ngược lại (bên thắng hoặc None nếu hòa, lý do)."""
        if len(self.history) >= self.max_plies:
            return (None, 'max_plies')
        r = self.pos.result()
        if r == 'ongoing':
            return None
        return (other(self.pos.turn), r)

    def _apply(self, move, by):
        info = make_info(self.pos, move, by)
        self.pos = self.pos.make(move)
        self.history.append(info)
        self._cand = (None, 0)
        return info

    def robot_move(self):
        if not self.robot_to_move():
            raise RuntimeError('Chưa đến lượt robot')
        move = self.engine.choose(self.pos)
        if not self.pos.is_legal(move):
            raise RuntimeError('Engine trả nước không hợp lệ: %s' % move_uci(move))
        info = make_info(self.pos, move, 'robot')
        self.executor.execute(info)              # lỗi robot -> ngoại lệ, thế cờ giữ nguyên
        return self._apply(move, 'robot')

    def human_observed(self, observed):
        """Đưa bàn cờ vừa nhìn thấy vào. Trả về (DetectResult, MoveInfo hoặc None)."""
        if self.robot_to_move():
            raise RuntimeError('Đang đến lượt robot')
        r = detect_move(self.pos, observed, self.tolerance)
        if r.status != 'move':
            self._cand = (None, 0)
            return r, None
        cand, n = self._cand
        n = n + 1 if cand == r.move else 1
        self._cand = (r.move, n)
        if n < self.confirm:
            return DetectResult('pending', r.move, r.distance, r.candidates), None
        return r, self._apply(r.move, 'human')


def play(game, observe, log=print, poll=0.3):
    """Chạy cả ván. observe() trả về dict bàn cờ nhìn thấy, hoặc None nếu chưa có dữ liệu ổn định."""
    last = None
    while True:
        res = game.result()
        if res is not None:
            winner, why = res
            if winner is None:
                log('HÒA (quá %d nước)' % game.max_plies)
            else:
                log('%s THẮNG (%s)' % ('Robot' if winner == game.robot else 'Người', why))
            return res
        if game.robot_to_move():
            log('Robot đang nghĩ...')
            log(describe(game.robot_move()))
            log(str(game.pos))
            last = None
            continue
        obs = observe()
        info = None
        if obs is not None:
            r, info = game.human_observed(obs)
            if info is not None:
                log(describe(info))
                log(str(game.pos))
                last = None
            elif r.status in ('ambiguous', 'invalid') and r.status != last:
                log('CẢNH BÁO: %s' % ('nhiều nước cùng khớp %s' % r.candidates if r.status == 'ambiguous'
                                      else 'bàn cờ không khớp nước hợp lệ nào (độ lệch %d)' % r.distance))
            if r.status != 'nochange':
                last = r.status
        if info is None and poll:
            time.sleep(poll)
```

---

### `robot\motion.py`

*Dung lượng: 5.94 KB | Số dòng: 134*

```python
#!/usr/bin/env python3
# Điều khiển tay máy (ROS topic) + thanh trượt (hiwonder.Stepper) + đầu hút. Tách từ tests/pick_place.py để dùng chung.
# Cần node jetmax_control đang chạy và rospy.init_node() đã được gọi trước khi tạo Mover.
# KHÔNG tạo hiwonder.JetMax() ở đây vì node jetmax_control đã giữ bus servo.

import math, time
import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0   # giống các script trước
TOL = 5.0                       # mm, sai lệch cho phép giữa đích và vị trí thực
FAST, SLOW = 100.0, 40.0        # mm/giây: di chuyển bình thường / khi hạ xuống chạm quân cờ
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0            # bước/giây, dùng ước lượng thời gian chờ (hãng ghi 1000; để thấp cho chắc)
SLIDER_MIN_Z = 40.0             # chỉ trượt khi tay máy nâng đủ cao


def in_limits(p):
    x, y, z = p
    return (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN)


class Mover:
    def __init__(self, use_slider=True):
        self.cmd = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
        self.sucker = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
        self.stepper, self.pos, self.enabled = None, 0, False
        if use_slider:
            import hiwonder
            self.stepper = hiwonder.Stepper(1)
        rospy.sleep(1.0)
        self.start = self.status()
        self.sucker.publish(Bool(data=False))

    # ---------------------------------------------------------------- tay máy
    def status(self):
        m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
        return m.x, m.y, m.z

    def goto(self, p, speed=FAST):
        if not in_limits(p):
            raise ValueError('Điểm (%.1f, %.1f, %.1f) ngoài giới hạn an toàn' % tuple(p))
        cur = self.status()
        dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
        t = max(0.5, dist / speed)
        self.cmd.publish(SetJetMax(x=p[0], y=p[1], z=p[2], duration=t))
        rospy.sleep(t + 0.3)
        now = self.status()
        err = max(abs(now[i] - p[i]) for i in range(3))
        if err > TOL:
            raise RuntimeError('Tay máy chưa tới đích (lệch %.1f mm): lệnh có thể bị chặn bởi giới hạn' % err)

    # --------------------------------------------------------------- thanh trượt
    def home_slider(self):
        if self.stepper is None:
            return
        if self.status()[2] < SLIDER_MIN_Z:
            raise RuntimeError('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
        st = self.stepper
        st.go_home(True)                         # chặn cho tới khi về home, không có timeout, canh nút nguồn
        st.set_mode(st.EN)
        st.set_div(st.DIV_1_4)
        self.pos, self.enabled = 0, False

    def slide_to(self, target):
        if self.stepper is None:
            return
        target = int(max(SLIDER_MIN, min(SLIDER_MAX, target)))
        d = target - self.pos
        if d == 0:
            return
        if self.status()[2] < SLIDER_MIN_Z:
            raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
        if not self.enabled:
            self.stepper.set_mode(self.stepper.SLEEP)   # chỉ gửi một lần
            self.enabled = True
        print('  thanh trượt %d -> %d' % (self.pos, target))
        self.stepper.goto(d)                     # bước TƯƠNG ĐỐI
        time.sleep(abs(d) / SLIDER_SPEED + 0.3)
        self.pos = target

    # ------------------------------------------------------------ thao tác cấp cao
    # sq_pos = (slider, x, y) lấy từ Board.position()
    def hover(self, sq_pos, zsafe):
        s, x, y = sq_pos
        self.slide_to(s)
        self.goto((x, y, zsafe))

    def pick(self, sq_pos, zpick, zsafe):
        self.hover(sq_pos, zsafe)
        _, x, y = sq_pos
        self.sucker.publish(Bool(data=True))
        rospy.sleep(0.3)
        self.goto((x, y, zpick), SLOW)
        rospy.sleep(0.8)                         # chờ hút bám chắc
        self.goto((x, y, zsafe), SLOW)

    def place(self, sq_pos, zplace, zsafe):
        self.hover(sq_pos, zsafe)
        _, x, y = sq_pos
        self.goto((x, y, zplace), SLOW)
        self.sucker.publish(Bool(data=False))    # thả, mở van xả khí
        rospy.sleep(1.0)
        self.goto((x, y, zsafe), SLOW)

    # ------------------------------------------------------------------ an toàn
    def safe_stop(self, zsafe):
        """Tắt hút và nâng tay máy lên z an toàn (dùng khi có lỗi giữa chừng)."""
        self.sucker.publish(Bool(data=False))
        try:
            cur = self.status()
            if cur[2] < zsafe:
                self.cmd.publish(SetJetMax(x=cur[0], y=cur[1], z=zsafe, duration=1.0))
                rospy.sleep(1.4)
        except Exception as e:
            print('Không nâng được tay máy (%s), hãy kiểm tra thủ công' % e)

    def restore(self, zsafe=50.0):
        """Về trạng thái ban đầu: tắt hút, nâng lên, thanh trượt về 0, tay máy về vị trí lúc bắt đầu."""
        self.sucker.publish(Bool(data=False))
        rospy.sleep(0.6)
        try:
            if self.stepper is not None and self.pos != 0:
                self.safe_stop(max(zsafe, SLIDER_MIN_Z))
                self.slide_to(0)
            self.cmd.publish(SetJetMax(x=self.start[0], y=self.start[1], z=self.start[2], duration=2.0))
            rospy.sleep(2.5)
        except Exception as e:
            print('Không tự về được vị trí ban đầu (%s), hãy kiểm tra thủ công' % e)
        if self.stepper is not None:
            self.stepper.set_mode(self.stepper.EN)
        print('Đã về vị trí ban đầu')
```

---

### `robot\xiangqi_rules.py`

*Dung lượng: 12.19 KB | Số dòng: 349*

```python
#!/usr/bin/env python3
# Luật cờ tướng thuần Python (không cần ROS, không cần thư viện ngoài, chạy được trên Python 3.6 của Jetson).
#
# Quy ước TRÙNG với robot/board.py:
#   cột a..i = chỉ số 0..8 (trái -> phải nhìn từ phía quân ĐỎ), hàng 0..9 (hàng 0 = phía đỏ, hàng 9 = phía đen).
#   Nước đi dạng UCI/UCCI "h2e2" = quân ở h2 đi tới e2. Trong code một nước đi là tuple (f1, r1, f2, r2).
#
# Ký hiệu quân (giống FEN của Pikafish):  K tướng, A sĩ, B tượng, N mã, R xe, C pháo, P tốt
#   chữ HOA = quân ĐỎ (đi trước, 'w'), chữ thường = quân ĐEN ('b').
#
# Luật đã cài: mã bị cản chân, tượng bị chặn mắt và không qua sông, sĩ/tướng trong cung, tốt qua sông mới đi ngang,
# pháo ăn qua đúng một quân, hai tướng không được đối mặt (lộ mặt tướng), không được để tướng mình bị chiếu.
# Hết nước đi hợp lệ = THUA (bị chiếu bí hoặc hết nước đều thua trong cờ tướng).
# CHƯA cài: luật cấm chiếu liên tục / đuổi liên tục, luật hòa 60 nước không ăn quân (xem ghi chú trong game.py).

FILES = 'abcdefghi'
RED, BLACK = 'w', 'b'
EMPTY = '.'
START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'

ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT = ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))
PIECES = 'KABNRCPkabnrcp'


# ------------------------------------------------------------------ tiện ích ----
def sq_name(f, r):
    return '%s%d' % (FILES[f], r)


def parse_sq(s):
    s = s.strip().lower()
    if len(s) != 2 or s[0] not in FILES or not s[1].isdigit():
        raise ValueError('Ô "%s" không hợp lệ (cột a-i, hàng 0-9, ví dụ h2)' % s)
    return FILES.index(s[0]), int(s[1])


def move_uci(m):
    return sq_name(m[0], m[1]) + sq_name(m[2], m[3])


def parse_uci(s):
    s = s.strip().lower().replace(' ', '').replace('-', '')
    if len(s) != 4:
        raise ValueError('Nước đi "%s" không hợp lệ (dạng h2e2)' % s)
    f1, r1 = parse_sq(s[:2])
    f2, r2 = parse_sq(s[2:])
    return (f1, r1, f2, r2)


def side_of(p):
    return RED if p.isupper() else BLACK


def other(side):
    return BLACK if side == RED else RED


def in_palace(f, r, side):
    return 3 <= f <= 5 and (0 <= r <= 2 if side == RED else 7 <= r <= 9)


def has_crossed(r, side):
    """Quân đã qua sông chưa (r là hàng hiện tại)."""
    return r >= 5 if side == RED else r <= 4


def _on(f, r):
    return 0 <= f < 9 and 0 <= r < 10


# ------------------------------------------------------------------ thế cờ -----
class Position(object):
    """Thế cờ + bên đi. b[r][f] là ký tự quân hoặc '.'."""
    __slots__ = ('b', 'turn')

    def __init__(self, board=None, turn=RED):
        self.b = board if board is not None else [[EMPTY] * 9 for _ in range(10)]
        self.turn = turn

    # ---- tạo / chuyển đổi
    @classmethod
    def start(cls):
        return cls.from_fen(START_FEN)

    @classmethod
    def from_fen(cls, fen):
        parts = fen.split()
        if not parts:
            raise ValueError('FEN rỗng')
        rows = parts[0].split('/')
        if len(rows) != 10:
            raise ValueError('FEN phải có 10 hàng, nhận được %d' % len(rows))
        board = [[EMPTY] * 9 for _ in range(10)]
        for i, row in enumerate(rows):
            r, f = 9 - i, 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                elif ch in PIECES:
                    if f >= 9:
                        raise ValueError('Hàng FEN quá dài: %s' % row)
                    board[r][f] = ch
                    f += 1
                else:
                    raise ValueError('Ký tự lạ trong FEN: %s' % ch)
            if f != 9:
                raise ValueError('Hàng FEN không đủ 9 cột: %s' % row)
        turn = BLACK if len(parts) > 1 and parts[1].lower() == 'b' else RED
        return cls(board, turn)

    @classmethod
    def from_dict(cls, d, turn=RED):
        """d = {(f, r): 'R', ...}"""
        board = [[EMPTY] * 9 for _ in range(10)]
        for (f, r), p in d.items():
            board[r][f] = p
        return cls(board, turn)

    def to_dict(self):
        return dict(((f, r), self.b[r][f]) for r in range(10) for f in range(9) if self.b[r][f] != EMPTY)

    def fen(self):
        rows = []
        for r in range(9, -1, -1):
            s, n = '', 0
            for f in range(9):
                p = self.b[r][f]
                if p == EMPTY:
                    n += 1
                else:
                    if n:
                        s += str(n)
                        n = 0
                    s += p
            if n:
                s += str(n)
            rows.append(s)
        return '/'.join(rows) + ' %s - - 0 1' % self.turn

    def copy(self):
        return Position([row[:] for row in self.b], self.turn)

    def piece_at(self, f, r):
        p = self.b[r][f]
        return None if p == EMPTY else p

    def make(self, move):
        """Trả về thế cờ mới sau nước đi (KHÔNG kiểm tra hợp lệ; dùng is_legal trước nếu cần)."""
        f1, r1, f2, r2 = move
        n = self.copy()
        n.b[r2][f2] = n.b[r1][f1]
        n.b[r1][f1] = EMPTY
        n.turn = other(self.turn)
        return n

    def __str__(self):
        out = ['    a b c d e f g h i']
        for r in range(9, -1, -1):
            out.append('%d   %s' % (r, ' '.join(self.b[r])))
            if r == 5:
                out.append('    ~ ~ ~ ~ sông ~ ~ ~ ~')
        out.append('Lượt: %s' % ('ĐỎ (chữ hoa)' if self.turn == RED else 'ĐEN (chữ thường)'))
        return '\n'.join(out)

    # ---- sinh nước đi
    def _targets(self, f, r):
        """Các ô đến 'giả hợp lệ' (chưa xét bị chiếu) của quân tại (f, r)."""
        b = self.b
        p = b[r][f]
        side = side_of(p)
        t = p.upper()
        out = []

        def ok(f2, r2):
            q = b[r2][f2]
            return q == EMPTY or side_of(q) != side

        if t == 'R':
            for df, dr in ORTH:
                f2, r2 = f + df, r + dr
                while _on(f2, r2):
                    q = b[r2][f2]
                    if q == EMPTY:
                        out.append((f2, r2))
                    else:
                        if side_of(q) != side:
                            out.append((f2, r2))
                        break
                    f2 += df
                    r2 += dr
        elif t == 'C':
            for df, dr in ORTH:
                f2, r2 = f + df, r + dr
                while _on(f2, r2) and b[r2][f2] == EMPTY:
                    out.append((f2, r2))
                    f2 += df
                    r2 += dr
                f2 += df          # bỏ qua quân làm "ngòi"
                r2 += dr
                while _on(f2, r2):
                    q = b[r2][f2]
                    if q != EMPTY:
                        if side_of(q) != side:
                            out.append((f2, r2))
                        break
                    f2 += df
                    r2 += dr
        elif t == 'N':
            for df, dr in KNIGHT:
                f2, r2 = f + df, r + dr
                if not _on(f2, r2):
                    continue
                lf, lr = (f, r + dr // 2) if abs(dr) == 2 else (f + df // 2, r)   # chân mã
                if b[lr][lf] == EMPTY and ok(f2, r2):
                    out.append((f2, r2))
        elif t == 'B':
            for df, dr in DIAG:
                f2, r2 = f + 2 * df, r + 2 * dr
                if not _on(f2, r2):
                    continue
                if (r2 > 4) if side == RED else (r2 < 5):      # tượng không qua sông
                    continue
                if b[r + dr][f + df] == EMPTY and ok(f2, r2):  # mắt tượng
                    out.append((f2, r2))
        elif t == 'A':
            for df, dr in DIAG:
                f2, r2 = f + df, r + dr
                if _on(f2, r2) and in_palace(f2, r2, side) and ok(f2, r2):
                    out.append((f2, r2))
        elif t == 'K':
            for df, dr in ORTH:
                f2, r2 = f + df, r + dr
                if _on(f2, r2) and in_palace(f2, r2, side) and ok(f2, r2):
                    out.append((f2, r2))
        elif t == 'P':
            d = 1 if side == RED else -1
            cand = [(f, r + d)]
            if has_crossed(r, side):
                cand += [(f - 1, r), (f + 1, r)]
            for f2, r2 in cand:
                if _on(f2, r2) and ok(f2, r2):
                    out.append((f2, r2))
        return out

    def _king_sq(self, side):
        k = 'K' if side == RED else 'k'
        for r in range(10):
            for f in range(9):
                if self.b[r][f] == k:
                    return f, r
        return None

    def _king_attacked(self, side):
        """Tướng của `side` có đang bị chiếu không (kể cả lộ mặt tướng)."""
        ks = self._king_sq(side)
        if ks is None:
            return False
        kf, kr = ks
        b = self.b
        en = other(side)
        er, ec, eh, ep, ek = ('r', 'c', 'n', 'p', 'k') if side == RED else ('R', 'C', 'N', 'P', 'K')
        # xe, pháo, và tướng đối mặt theo đường thẳng
        for df, dr in ORTH:
            f2, r2 = kf + df, kr + dr
            while _on(f2, r2) and b[r2][f2] == EMPTY:
                f2 += df
                r2 += dr
            if not _on(f2, r2):
                continue
            q = b[r2][f2]
            if q == er or q == ek:
                return True
            f2 += df
            r2 += dr
            while _on(f2, r2) and b[r2][f2] == EMPTY:
                f2 += df
                r2 += dr
            if _on(f2, r2) and b[r2][f2] == ec:
                return True
        # mã (kiểm tra chân mã tính từ phía mã)
        for df, dr in KNIGHT:
            nf, nr = kf + df, kr + dr
            if not _on(nf, nr) or b[nr][nf] != eh:
                continue
            if abs(dr) == 2:
                lf, lr = nf, nr + (-dr) // 2
            else:
                lf, lr = nf + (-df) // 2, nr
            if b[lr][lf] == EMPTY:
                return True
        # tốt: đánh thẳng tới, hoặc đánh ngang nếu đã qua sông
        d_e = 1 if en == RED else -1
        if _on(kf, kr - d_e) and b[kr - d_e][kf] == ep:
            return True
        if has_crossed(kr, en):
            for sf in (kf - 1, kf + 1):
                if _on(sf, kr) and b[kr][sf] == ep:
                    return True
        return False

    def legal_moves(self):
        side = self.turn
        b = self.b
        res = []
        for r in range(10):
            for f in range(9):
                p = b[r][f]
                if p == EMPTY or side_of(p) != side:
                    continue
                for f2, r2 in self._targets(f, r):
                    cap = b[r2][f2]
                    b[r2][f2] = p
                    b[r][f] = EMPTY
                    bad = self._king_attacked(side)
                    b[r][f] = p
                    b[r2][f2] = cap
                    if not bad:
                        res.append((f, r, f2, r2))
        return res

    def is_legal(self, move):
        return tuple(move) in self.legal_moves()

    def in_check(self, side=None):
        return self._king_attacked(side or self.turn)

    def result(self):
        """'ongoing' | 'checkmate' | 'stalemate'. Cả hai kết quả cuối đều là bên đang đến lượt THUA."""
        if self.legal_moves():
            return 'ongoing'
        return 'checkmate' if self.in_check() else 'stalemate'


# ------------------------------------------------------------------ mô tả nước đi
PIECE_VN = {'K': 'tướng', 'A': 'sĩ', 'B': 'tượng', 'N': 'mã', 'R': 'xe', 'C': 'pháo', 'P': 'tốt'}


def piece_name(p):
    return '%s %s' % (PIECE_VN[p.upper()], 'đỏ' if p.isupper() else 'đen')


def perft(pos, depth):
    """Đếm số nút lá, dùng để kiểm tra luật (xem tests/test_rules.py)."""
    moves = pos.legal_moves()
    if depth <= 1:
        return len(moves)
    return sum(perft(pos.make(m), depth - 1) for m in moves)
```

---

### `configs\board.yaml`

*Dung lượng: 1.08 KB | Số dòng: 15*

```yaml
# Hiệu chuẩn bàn cờ. Đơn vị mm; slider tính bằng BƯỚC từ HOME của thanh trượt.
# Quy ước ô: cột a..i từ trái sang phải nhìn từ phía quân ĐỎ, hàng 0 (phía đỏ) .. 9 (phía đen).
#
# Cách lấy số: chạy joint_teleop.py, đưa đầu hút tới TÂM giao điểm ở mỗi góc (đúng tâm quân cờ đặt ở góc đó),
# bấm p để in slider/x/y rồi điền vào đây. Ở mỗi góc có thể dùng vị trí thanh trượt khác nhau,
# miễn tay máy ở vùng với thoải mái (không sát biên).
corners:
  a0: {slider: null, x: null, y: null}   # góc đỏ bên trái
  i0: {slider: null, x: null, y: null}   # góc đỏ bên phải
  a9: {slider: null, x: null, y: null}   # góc đen bên trái
  i9: {slider: null, x: null, y: null}   # góc đen bên phải

z_pick: null    # z khi đầu hút chạm mặt quân cờ (giống --zpick trong pick_place.py)
z_place: null   # z khi thả quân; để null thì = z_pick + 3
z_safe: 50.0    # z khi di chuyển ngang (phải >= 40 để thanh trượt được phép chạy)
```

---

### `configs\vision.yaml`

*Dung lượng: 1011 B | Số dòng: 13*

```yaml
# Hiệu chuẩn ảnh camera -> ô cờ (dùng cho robot/vision_board.py). Đơn vị: PIXEL trong ảnh camera.
# Đặt camera ở một tư thế cố định nhìn thấy toàn bộ bàn cờ, chụp một ảnh, rồi đọc toạ độ pixel (x, y) của TÂM
# giao điểm ở bốn góc (mở ảnh bằng trình xem có hiển thị toạ độ chuột, ví dụ Paint/GIMP).
# Quy ước ô giống configs/board.yaml: a0/i0 là hai góc phía quân ĐỎ, a9/i9 là hai góc phía quân ĐEN,
# cột a..i từ trái sang phải nhìn từ phía quân đỏ. Ảnh bị xoay/lật thế nào cũng được, vì ánh xạ dựa vào 4 góc này.
image_corners:
  a0: null     # ví dụ: [112, 438]
  i0: null
  a9: null
  i9: null

max_offset: 0.35   # quân lệch tâm giao điểm quá 0.35 ô thì bỏ qua (đang bị cầm lên, nằm giữa hai ô)
anchor_y: 0.5      # 0.5 = lấy tâm khung nhận diện; tăng về 1.0 nếu camera nhìn nghiêng nhiều (lấy gần chân quân hơn)
```

---

## Nhóm 2: Cách bạn đang điều khiển tay và chạy mô phỏng

### `tests\play_sim.py`

*Dung lượng: 4.82 KB | Số dòng: 102*

```python
#!/usr/bin/env python3
# Chơi thử cả ván với robot GIẢ LẬP, không cần robot/camera/ROS.
# Bàn cờ "vật lý" được mô phỏng: robot đi xong thì bàn cờ vật lý đổi theo, người đi (gõ tay hoặc ngẫu nhiên) thì bàn cờ
# vật lý đổi, rồi bàn cờ được đưa qua đúng đường nhận diện thật (có thể thêm nhiễu: bỏ sót quân như YOLO) để
# game.detect_move tìm ra nước người vừa đi.
#
#   python3 tests/play_sim.py --human red                       # bạn cầm ĐỎ đi trước, gõ nước dạng h2e2 (q = thoát)
#   python3 tests/play_sim.py --human black                     # bạn cầm ĐEN, robot cầm ĐỎ đi trước
#   python3 tests/play_sim.py --human red --auto --noise 0.03   # người đi ngẫu nhiên, 3% quân bị bỏ sót mỗi khung hình
#   python3 tests/play_sim.py --level normal --engine ~/pikafish/pikafish --nnue ~/pikafish/pikafish.nnue
# Gõ "?" khi đến lượt bạn để xem các nước hợp lệ.

import os, sys, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

from xiangqi_rules import Position, parse_uci, move_uci
from engine import create_engine, LEVELS
from game import Game, DryRunExecutor, play


class Sim(DryRunExecutor):
    """Vừa là 'robot' (execute), vừa là 'camera' (observe) của bàn cờ vật lý mô phỏng."""

    def __init__(self, noise=0.0, auto=False, rng=None, log=print):
        DryRunExecutor.__init__(self, log)
        self.noise, self.auto, self.rng = noise, auto, rng or random.Random()
        self.physical = Position.start()
        self.game = None
        self.calls = 0

    def execute(self, info):                      # robot đi: bàn cờ vật lý đổi theo
        DryRunExecutor.execute(self, info)
        self.physical = self.physical.make(info.move)

    def _human_move(self):
        legal = self.physical.legal_moves()
        if self.auto:
            return self.rng.choice(legal)
        while True:
            s = input('Nước đi của bạn (vd h2e2, ? = gợi ý, q = thoát): ').strip().lower()
            if s in ('q', 'quit', 'exit'):
                raise KeyboardInterrupt
            if s == '?':
                self.log(' '.join(sorted(move_uci(m) for m in legal)))
                continue
            try:
                m = parse_uci(s)
            except ValueError as e:
                self.log('LỖI: %s' % e)
                continue
            if m in legal:
                return m
            self.log('Nước %s không hợp lệ' % s)

    def observe(self):
        if self.game.pos.fen() == self.physical.fen():    # robot đã tin thế cờ này và người CHƯA đi -> người đi
            self.calls = 0
            self.physical = self.physical.make(self._human_move())
        self.calls += 1
        if self.calls > 500:
            raise RuntimeError('Nhận diện mãi không ra nước người vừa đi (nhiễu quá lớn?)')
        return dict((sq, p) for sq, p in self.physical.to_dict().items() if self.rng.random() >= self.noise)


def build(human, engine_path=None, level='easy', nnue=None, noise=0.0, auto=False, seed=None,
          max_plies=400, log=print):
    rng = random.Random(seed)
    sim = Sim(noise, auto, rng, log)
    eng = create_engine(engine_path, level, nnue, seed)
    game = Game(human, eng, sim, max_plies=max_plies)
    sim.game = game
    return game, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', choices=['red', 'black'], default='red', help='màu quân của người chơi')
    ap.add_argument('--level', choices=sorted(LEVELS), default='easy')
    ap.add_argument('--engine', help='đường dẫn file chạy Pikafish (bỏ trống = engine tự viết)')
    ap.add_argument('--nnue', help='đường dẫn pikafish.nnue (nếu không nằm cạnh file chạy)')
    ap.add_argument('--auto', action='store_true', help='người chơi đi ngẫu nhiên (để chạy thử tự động)')
    ap.add_argument('--noise', type=float, default=0.0, help='xác suất bỏ sót mỗi quân ở mỗi khung hình (0-1)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--max-plies', type=int, default=400)
    a = ap.parse_args()

    game, sim = build(a.human, a.engine, a.level, a.nnue, a.noise, a.auto, a.seed, a.max_plies)
    print('Người cầm %s, robot cầm %s (%s). Engine: %s' % (
        'ĐỎ' if a.human == 'red' else 'ĐEN', 'ĐEN' if a.human == 'red' else 'ĐỎ',
        'người đi trước' if a.human == 'red' else 'robot đi trước', game.engine.name))
    print(game.pos)
    try:
        play(game, sim.observe, poll=0)
    except KeyboardInterrupt:
        print('\nDừng theo yêu cầu')
    finally:
        game.engine.close()


if __name__ == '__main__':
    main()
```

---

### `tests\fake_uci_engine.py`

*Dung lượng: 1.99 KB | Số dòng: 46*

```python
#!/usr/bin/env python3
# Engine UCI GIẢ (không phải Pikafish) dùng để kiểm thử robot/engine.py::UciEngine mà không cần cài Pikafish.
# Nó in đúng dạng dòng "option", "info ... multipv ... score ... pv ...", "bestmove" như Pikafish: nước tốt nhất
# là nước đầu tiên trong danh sách hợp lệ, các nước sau điểm thấp dần.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'robot')]
from xiangqi_rules import Position, move_uci

pos, multipv = Position.start(), 1


def out(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


for line in sys.stdin:
    tok = line.split()
    if not tok:
        continue
    if tok[0] == 'uci':
        out('id name FakeFish')
        out('option name Threads type spin default 1 min 1 max 1024')
        out('option name Hash type spin default 16 min 1 max 33554432')
        out('option name MultiPV type spin default 1 min 1 max 500')
        out('option name EvalFile type string default pikafish.nnue')
        out('uciok')
    elif tok[0] == 'isready':
        out('readyok')
    elif tok[0] == 'setoption':
        if 'MultiPV' in tok:
            multipv = int(tok[-1])
    elif tok[0] == 'position' and tok[1] == 'fen':
        pos = Position.from_fen(' '.join(tok[2:]))
    elif tok[0] == 'go':
        moves = pos.legal_moves()
        out('info string fake engine')
        for k in range(min(multipv, len(moves))):
            out('info depth 1 seldepth 1 multipv %d score cp %d nodes 10 nps 1000 time 1 pv %s' % (
                k + 1, 10 - 100 * k, move_uci(moves[len(moves) - 1 - k])))        # depth 1: nước "sai" (đảo thứ tự)
            out('info depth 2 seldepth 3 multipv %d score cp %d nodes 99 nps 1000 time 2 pv %s %s' % (
                k + 1, 50 - 100 * k, move_uci(moves[k]), move_uci(moves[(k + 1) % len(moves)])))
        out('bestmove %s ponder %s' % (move_uci(moves[0]), move_uci(moves[1 % len(moves)])))
    elif tok[0] == 'quit':
        break
```

---

### `tests\scan_detect.py`

*Dung lượng: 8.53 KB | Số dòng: 203*

```python
#!/usr/bin/env python3
# Quét + nhận diện: thanh trượt -> tay máy -> chụp ảnh -> YOLO11 nhận diện, tại từng điểm dừng.
# Ghép từ scan_capture.py (điều khiển, giới hạn an toàn) và detect_view.py (mô hình).
# CHƯA gắp quân cờ; chỉ di chuyển, chụp, nhận diện, vẽ và ghi kết quả.
#
# Đặt scan_detect.py CÙNG THƯ MỤC với detect_view.py (file này import từ đó).
# Yêu cầu: node jetmax_control và usb_cam đang chạy.
#
# Ví dụ:
#   python3 scan_detect.py --model ~/models/best.onnx
#   python3 scan_detect.py --model ~/models/best.onnx --sliders 0 1500 3000 --conf 0.4
#   python3 scan_detect.py --model ~/models/best.onnx --yes          # không hỏi Enter mỗi điểm
#
# Kết quả trong ~/scan_detect/<thời điểm chạy>/ :
#   raw_*.jpg (ảnh gốc), det_*.jpg (ảnh đã vẽ khung), detections.csv (mỗi dòng 1 quân cờ).
# Kết thúc/lỗi/q/Ctrl+C: tay máy về chỗ cũ, thanh trượt về 0.

import os, csv, time, argparse
import cv2
import numpy as np
import rospy
import hiwonder
from sensor_msgs.msg import Image
from jetmax_control.msg import SetJetMax, JetMax
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]
from detect_view import load_backend, draw

TOPIC = '/usb_cam/image_rect_color'
OUT_ROOT = os.path.expanduser('~/scan_detect')

# ---- Kế hoạch quét (chỉnh ở đây hoặc bằng --sliders) ----
SLIDER_POS = [0, 1000, 2000]                 # bước từ home
ARM_OFFSETS = [(0, 0, 0), (20, 0, 0), (-20, 0, 0),
               (0, 20, 0), (0, -20, 0), (0, 0, -20)]   # mm, tính từ vị trí ban đầu
DUR = 2.0                                    # giây mỗi lần di chuyển tay máy

# ---- Giới hạn an toàn (giống scan_capture.py / pick_place.py) ----
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0                         # bước/giây, dùng ước lượng thời gian chờ (hãng ghi 1000)
SLIDER_MIN_Z = 40.0                          # chỉ trượt khi tay máy nâng đủ cao
X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0
TOL = 5.0                                    # mm


class Stop(Exception):
    pass


state = {'img': None, 'n': 0}


def on_image(msg):
    state['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()   # RGB
    state['n'] += 1


def get_pos():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z


def check_target(x, y, z):
    ok = (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
          and (x * x + y * y) ** 0.5 >= R_MIN)
    if not ok:
        raise ValueError('Đích (%.1f, %.1f, %.1f) ngoài giới hạn an toàn, bỏ qua' % (x, y, z))


def move_arm(pub, target):
    check_target(*target)
    pub.publish(SetJetMax(x=target[0], y=target[1], z=target[2], duration=DUR))
    rospy.sleep(DUR + 0.7)
    cur = get_pos()
    if max(abs(cur[i] - target[i]) for i in range(3)) > TOL:
        print('  CẢNH BÁO: tay máy lệch so với đích (lệnh có thể bị chặn bởi giới hạn)')
    return cur


def move_slider(stepper, pos, new):
    new = int(max(SLIDER_MIN, min(SLIDER_MAX, new)))
    d = new - pos
    if d == 0:
        return pos
    if get_pos()[2] < SLIDER_MIN_Z:
        raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
    stepper.set_mode(stepper.SLEEP)
    stepper.goto(d)                          # bước TƯƠNG ĐỐI
    time.sleep(abs(d) / SLIDER_SPEED + 0.5)
    return new


def grab():
    t0, n0 = time.time(), state['n']
    while state['n'] < n0 + 2:               # đợi khung hình mới sau khi dừng hẳn
        if time.time() - t0 > 3:
            raise RuntimeError('Không nhận được ảnh từ ' + TOPIC)
        time.sleep(0.02)
    return cv2.cvtColor(state['img'], cv2.COLOR_RGB2BGR)   # mô hình train trên ảnh BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='đường dẫn best.pt hoặc best.onnx')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--sliders', type=int, nargs='+', default=SLIDER_POS, help='các vị trí thanh trượt (bước từ home)')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước mỗi điểm dừng')
    ap.add_argument('--no-show', action='store_true', help='không mở cửa sổ hiển thị')
    a = ap.parse_args()
    for s in a.sliders:
        if not SLIDER_MIN <= s <= SLIDER_MAX:
            raise SystemExit('Vị trí thanh trượt %d ngoài khoảng %d..%d' % (s, SLIDER_MIN, SLIDER_MAX))

    def ask(text):
        if not a.yes and input(text + ' [Enter = chạy, q = dừng] ').strip().lower() == 'q':
            raise Stop()

    det = load_backend(os.path.expanduser(a.model), a.conf, a.iou)   # nạp mô hình trước khi động tới robot
    show = not a.no_show and bool(os.environ.get('DISPLAY'))

    out = os.path.join(OUT_ROOT, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)

    rospy.init_node('scan_detect')
    pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    rospy.Subscriber(TOPIC, Image, on_image, queue_size=1)
    rospy.sleep(1.0)

    start = get_pos()
    print('Vị trí tay máy ban đầu =', start)
    print('Thanh trượt sẽ về HOME. Hãy chắc tay máy đang nâng cao (z >= %d) và không vướng vật cản.' % SLIDER_MIN_Z)
    if start[2] < SLIDER_MIN_Z:
        raise SystemExit('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
    ask('Bắt đầu')

    stepper = hiwonder.Stepper(1)
    pos = 0
    f = open(os.path.join(out, 'detections.csv'), 'w', newline='')
    w = csv.writer(f)
    w.writerow(['image', 'slider', 'arm_x', 'arm_y', 'arm_z', 'class', 'conf', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy'])
    n = 0
    try:
        stepper.go_home(True)                # chặn cho tới khi về home (không có timeout, canh nút nguồn)
        stepper.set_mode(stepper.EN)
        stepper.set_div(stepper.DIV_1_4)

        for sp in a.sliders:
            ask('Thanh trượt -> %d' % sp)
            move_arm(pub, start)             # luôn về tư thế cao ban đầu trước khi trượt
            pos = move_slider(stepper, pos, sp)
            print('  thanh trượt = %d' % pos)

            for off in ARM_OFFSETS:
                target = (start[0] + off[0], start[1] + off[1], start[2] + off[2])
                ask('  Tay máy offset %s' % (off,))
                try:
                    cur = move_arm(pub, target)
                except ValueError as e:
                    print('  ', e)
                    continue
                img = grab()
                t0 = time.time()
                dets = det(img)
                ms = (time.time() - t0) * 1000
                tag = 's%d_%03d' % (pos, n)
                cv2.imwrite(os.path.join(out, 'raw_%s.jpg' % tag), img)
                vis = draw(img.copy(), dets, det.names, 0, ms)
                cv2.imwrite(os.path.join(out, 'det_%s.jpg' % tag), vis)
                for x1, y1, x2, y2, c, k in dets:
                    name = det.names[int(k)] if int(k) < len(det.names) else str(int(k))
                    w.writerow(['det_%s.jpg' % tag, pos, '%.1f' % cur[0], '%.1f' % cur[1], '%.1f' % cur[2],
                                name, '%.3f' % c, int(x1), int(y1), int(x2), int(y2),
                                int((x1 + x2) / 2), int((y1 + y2) / 2)])
                f.flush()
                print('  %s: %d quân (%.0f ms) %s' % (tag, len(dets), ms,
                                                       ', '.join(sorted(det.names[int(d[5])] for d in dets))))
                if show:
                    cv2.imshow('scan_detect', vis)
                    cv2.waitKey(1)
                n += 1
    except Stop:
        print('Dừng theo yêu cầu')
    except (ValueError, RuntimeError) as e:
        print('LỖI:', e)
    finally:
        f.close()
        try:
            pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
            rospy.sleep(2.5)
            if pos > 0:
                stepper.goto(-pos)           # quay lại home
                time.sleep(pos / SLIDER_SPEED + 0.5)
            stepper.set_mode(stepper.EN)
        finally:
            if show:
                cv2.destroyAllWindows()
            print('Đã về vị trí ban đầu. Đã xử lý %d ảnh, kết quả ở %s' % (n, out))


if __name__ == '__main__':
    main()
```

---

### `tests\move_piece.py`

*Dung lượng: 7.26 KB | Số dòng: 151*

```python
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
```

---

### `tests\pick_place.py`

*Dung lượng: 8.18 KB | Số dòng: 172*

```python
#!/usr/bin/env python3
# Gắp ở điểm A, đặt ở điểm B bằng đầu hút, CÓ dùng thanh trượt (chưa dùng camera).
# Tay máy điều khiển qua ROS topic (cần node jetmax_control đang chạy), thanh trượt qua hiwonder.Stepper.
#
# Bước 1 (chỉ bay trên A và B, KHÔNG hạ xuống, KHÔNG hút; thanh trượt vẫn chạy thật):
#   python3 pick_place.py --a 0 -210.4 --b 0 -200.4 --zpick -8.8 --zplace -5.4 --sa 1500 --sb 5000 --hover
# Bước 2 (gắp - đặt thật, hỏi Enter trước mỗi pha):
#   python3 pick_place.py --a 0 -210.4 --b 0 -200.4 --zpick -8.8 --zplace -5.4 --sa 1500 --sb 5000
# Khi chắc chắn, thêm --yes để chạy liền không hỏi.
#
# Tọa độ A, B và số bước trượt lấy từ joint_teleop.py (bấm p). Số bước trượt là tính từ HOME của thanh trượt,
# nên script sẽ đưa thanh trượt về HOME trước khi bắt đầu (bỏ qua bằng --no-home nếu chắc chắn nó đang ở home).

import argparse, math, time
import rospy
from std_msgs.msg import Bool
from jetmax_control.msg import SetJetMax, JetMax

X_LIM, Y_LIM, Z_LIM, R_MIN = (-120, 120), (-230, -100), (-12, 225), 50.0   # giống các script trước
TOL = 5.0                      # mm, sai lệch cho phép giữa đích và vị trí thực
FAST, SLOW = 100.0, 40.0       # mm/giây: di chuyển bình thường / khi hạ xuống chạm quân cờ
SLIDER_MIN, SLIDER_MAX = 0, 5200
SLIDER_SPEED = 800.0           # bước/giây, dùng để ước lượng thời gian chờ (hãng ghi 1000; để thấp cho chắc)
SLIDER_MIN_Z = 40.0            # chỉ trượt khi tay máy nâng đủ cao

S = {'stepper': None, 'pos': 0, 'enabled': False}


def status():
    m = rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)
    return m.x, m.y, m.z


def check(p):
    x, y, z = p
    if not (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN):
        raise ValueError('Điểm (%.1f, %.1f, %.1f) ngoài giới hạn an toàn' % p)


def goto(pub, p, speed=FAST):
    check(p)
    cur = status()
    dist = math.sqrt(sum((p[i] - cur[i]) ** 2 for i in range(3)))
    t = max(0.5, dist / speed)
    pub.publish(SetJetMax(x=p[0], y=p[1], z=p[2], duration=t))
    rospy.sleep(t + 0.3)
    now = status()
    err = max(abs(now[i] - p[i]) for i in range(3))
    if err > TOL:
        raise RuntimeError('Tay máy chưa tới đích (lệch %.1f mm): lệnh có thể bị chặn bởi giới hạn' % err)


def slide_to(target):
    st = S['stepper']
    if st is None:
        raise RuntimeError('Không có thanh trượt')
    target = int(max(SLIDER_MIN, min(SLIDER_MAX, target)))
    d = target - S['pos']
    if d == 0:
        return
    if status()[2] < SLIDER_MIN_Z:
        raise RuntimeError('Tay máy đang thấp hơn z=%d, không trượt' % SLIDER_MIN_Z)
    if not S['enabled']:
        st.set_mode(st.SLEEP)                    # chỉ gửi một lần
        S['enabled'] = True
    print('  thanh trượt %d -> %d' % (S['pos'], target))
    st.goto(d)
    time.sleep(abs(d) / SLIDER_SPEED + 0.3)
    S['pos'] = target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, nargs=2, required=True, metavar=('X', 'Y'), help='điểm gắp')
    ap.add_argument('--b', type=float, nargs=2, required=True, metavar=('X', 'Y'), help='điểm đặt')
    ap.add_argument('--zpick', type=float, required=True, help='z khi đầu hút chạm quân cờ ở A')
    ap.add_argument('--zplace', type=float, default=None, help='z khi thả ở B (mặc định = zpick + 3)')
    ap.add_argument('--zsafe', type=float, default=50.0, help='z khi di chuyển ngang')
    ap.add_argument('--sa', type=int, default=0, help='vị trí thanh trượt khi gắp ở A (bước từ home)')
    ap.add_argument('--sb', type=int, default=0, help='vị trí thanh trượt khi đặt ở B (bước từ home)')
    ap.add_argument('--no-home', action='store_true', help='không đưa thanh trượt về home lúc bắt đầu')
    ap.add_argument('--hover', action='store_true', help='chỉ bay trên A và B, không hạ, không hút')
    ap.add_argument('--yes', action='store_true', help='không hỏi Enter trước mỗi pha')
    a = ap.parse_args()
    zplace = a.zplace if a.zplace is not None else a.zpick + 3
    A, B = a.a, a.b
    for s in (a.sa, a.sb):
        if not SLIDER_MIN <= s <= SLIDER_MAX:
            raise SystemExit('Vị trí thanh trượt %d ngoài khoảng %d..%d' % (s, SLIDER_MIN, SLIDER_MAX))

    def ask(text):
        if not a.yes and input(text + ' [Enter = chạy, q = dừng] ').strip().lower() == 'q':
            raise KeyboardInterrupt

    rospy.init_node('pick_place')
    abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
    sucker = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)
    need_slider = a.sa != 0 or a.sb != 0
    if need_slider:
        try:
            import hiwonder
            S['stepper'] = hiwonder.Stepper(1)
        except Exception as e:
            raise SystemExit('Không khởi tạo được thanh trượt: %s' % e)
    rospy.sleep(1.0)

    start = status()
    print('Vị trí ban đầu:', start)
    print('A=%s (trượt %d)  B=%s (trượt %d)  zpick=%.1f zplace=%.1f zsafe=%.1f %s' % (
        A, a.sa, B, a.sb, a.zpick, zplace, a.zsafe, '[CHỈ BAY, KHÔNG HẠ]' if a.hover else ''))
    for p in [(A[0], A[1], a.zsafe), (B[0], B[1], a.zsafe), (A[0], A[1], a.zpick), (B[0], B[1], zplace)]:
        check(p)                                 # kiểm tra mọi điểm trước khi di chuyển
    try:
        sucker.publish(Bool(data=False))
        if need_slider and not a.no_home:
            ask('Đưa thanh trượt về HOME (vị trí 0). Tay máy phải đang nâng cao và không vướng')
            if start[2] < SLIDER_MIN_Z:
                raise RuntimeError('Tay máy đang thấp hơn z=%d, hãy nâng lên trước' % SLIDER_MIN_Z)
            st = S['stepper']
            st.go_home(True)                     # chặn cho tới khi về home, không có timeout
            st.set_mode(st.EN)
            st.set_div(st.DIV_1_4)
            S['pos'], S['enabled'] = 0, False
        ask('Trượt tới %d và bay tới trên A' % a.sa)
        slide_to(a.sa)
        goto(abs_pub, (A[0], A[1], a.zsafe))
        if a.hover:
            ask('Trượt tới %d và bay tới trên B' % a.sb)
            slide_to(a.sb)
            goto(abs_pub, (B[0], B[1], a.zsafe))
        else:
            ask('Hạ xuống A và hút')
            sucker.publish(Bool(data=True))
            rospy.sleep(0.3)
            goto(abs_pub, (A[0], A[1], a.zpick), SLOW)
            rospy.sleep(0.8)                     # chờ hút bám chắc
            goto(abs_pub, (A[0], A[1], a.zsafe), SLOW)
            ask('Mang tới B (trượt tới %d)' % a.sb)
            slide_to(a.sb)
            goto(abs_pub, (B[0], B[1], a.zsafe))
            ask('Hạ xuống B và thả')
            goto(abs_pub, (B[0], B[1], zplace), SLOW)
            sucker.publish(Bool(data=False))     # thả, mở van xả khí
            rospy.sleep(1.0)
            goto(abs_pub, (B[0], B[1], a.zsafe), SLOW)
        print('Xong')
    except KeyboardInterrupt:
        print('Dừng theo yêu cầu')
    except (ValueError, RuntimeError) as e:
        print('LỖI:', e)
    finally:
        sucker.publish(Bool(data=False))
        rospy.sleep(0.6)
        try:
            cur = status()
            if S['stepper'] is not None and S['pos'] != 0 and cur[2] < SLIDER_MIN_Z:
                abs_pub.publish(SetJetMax(x=cur[0], y=cur[1], z=a.zsafe, duration=1.0))   # nâng lên trước khi trượt về
                rospy.sleep(1.4)
            if S['stepper'] is not None and S['pos'] != 0:
                slide_to(0)
            abs_pub.publish(SetJetMax(x=start[0], y=start[1], z=start[2], duration=2.0))
            rospy.sleep(2.5)
        except Exception as e:
            print('Không tự về được vị trí ban đầu (%s), hãy kiểm tra thủ công' % e)
        if S['stepper'] is not None:
            S['stepper'].set_mode(S['stepper'].EN)
        print('Đã về vị trí ban đầu')


if __name__ == '__main__':
    main()
```

---

### `tests\joint_teleop.py`

*Dung lượng: 9.60 KB | Số dòng: 256*

```python
#!/usr/bin/env python3
# Điều khiển JetMax bằng phím: tịnh tiến x/y/z, xoay TỪNG KHỚP, servo cổ tay, đầu hút và THANH TRƯỢT.
# Tay máy điều khiển qua ROS topic (cần node jetmax_control đang chạy); thanh trượt qua hiwonder.Stepper.
#
#   python3 joint_teleop.py            # thoát: tay máy và thanh trượt về chỗ ban đầu
#   python3 joint_teleop.py --stay     # thoát: để nguyên tại chỗ
#
# Muốn đi XA hơn: giữ Shift (gõ chữ HOA) = bước x4, hoặc bấm t để đổi mức bước.
# Tốc độ mỗi lần bấm tự tỉ lệ theo quãng đường nên bước nhỏ không còn chờ lâu.

import sys, time, select, termios, tty, argparse
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from jetmax_control.msg import SetJetMax, SetJoint, SetServo, JetMax

MM_STEPS = [5, 10, 20, 40]               # mm
DEG_STEPS = [2, 5, 10, 20]               # độ (khớp và cổ tay)
SLIDER_STEPS = [500, 1000, 2000, 3000]   # bước thanh trượt (bước nhỏ làm động cơ dừng-chạy liên tục nên giật)
BIG = 4                                  # hệ số khi giữ Shift
SPEED_MM = 150.0                         # mm/giây
SPEED_DEG = 120.0                        # độ/giây
MIN_DUR = 0.2                            # giây, thời gian tối thiểu mỗi lệnh
X_LIM, Y_LIM, Z_LIM = (-120, 120), (-230, -100), (-12, 225)   # giống các script trước
R_MIN = 50.0                             # bán kính xy tối thiểu (jetmax.py bỏ qua lệnh nếu nhỏ hơn)
PWM_MIN, PWM_MAX = 0, 180
JOINT_TOL = 0.5                          # độ
SLIDER_MIN, SLIDER_MAX = 0, 5200         # 5200 là điểm xa nhất trong ví dụ hãng, chưa phải hành trình đo được
SLIDER_SPEED = 800.0                     # bước/giây dùng để ước lượng thời gian chờ (hãng ghi 1000; để thấp hơn cho chắc)
SLIDER_MIN_Z = 40.0                      # chỉ cho trượt khi tay máy nâng đủ cao (ví dụ hãng nâng lên z=50 trước)

ap = argparse.ArgumentParser()
ap.add_argument('--stay', action='store_true', help='khi thoát, không đưa tay máy/thanh trượt về chỗ cũ')
args = ap.parse_args()

rospy.init_node('joint_teleop')
abs_pub = rospy.Publisher('/jetmax/command', SetJetMax, queue_size=1)
joint_pubs = [rospy.Publisher('/jetmax/joint%d/command' % i, SetJoint, queue_size=1) for i in (1, 2, 3)]
pwm_pub = rospy.Publisher('/jetmax/end_effector/servo1/command', SetServo, queue_size=1)
sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/command', Bool, queue_size=1)

stepper = None
try:
    import hiwonder
    stepper = hiwonder.Stepper(1)
except Exception as e:
    print('Không khởi tạo được thanh trượt (%s): các phím z/x/c bị tắt.' % e)
slider_enabled = False                   # đã gửi chế độ SLEEP (giữ động cơ) chưa
slider_pos = 0                           # không đọc lại được từ phần cứng nên tự theo dõi, coi lúc mở là 0
rospy.sleep(1.0)


def status():
    return rospy.wait_for_message('/jetmax/status', JetMax, timeout=5)


def joints_of(m):
    return [m.joint1, m.joint2, m.joint3]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def in_limits(x, y, z):
    return (X_LIM[0] <= x <= X_LIM[1] and Y_LIM[0] <= y <= Y_LIM[1] and Z_LIM[0] <= z <= Z_LIM[1]
            and (x * x + y * y) ** 0.5 >= R_MIN)


def show():
    m = status()
    print('  xyz = (%.1f, %.1f, %.1f) | joint = (%.1f, %.1f, %.1f) | cổ tay = %.0f | hút = %s | trượt = %d' % (
        m.x, m.y, m.z, m.joint1, m.joint2, m.joint3, m.pwm1, m.sucker, slider_pos))


def move_xyz(dx, dy, dz):
    m = status()
    cur = (m.x, m.y, m.z)
    t = (clamp(m.x + dx, *X_LIM), clamp(m.y + dy, *Y_LIM), clamp(m.z + dz, *Z_LIM))
    if (t[0] ** 2 + t[1] ** 2) ** 0.5 < R_MIN:
        print('  Quá gần trục (bán kính < %d mm), không đi' % R_MIN)
        return
    dist = sum((t[i] - cur[i]) ** 2 for i in range(3)) ** 0.5
    if dist < 0.1:
        print('  Đã ở biên giới hạn an toàn')
        return
    if abs(dx) + abs(dy) + abs(dz) - dist > 0.5:
        print('  (cắt bớt cho vừa giới hạn an toàn)')
    dur = max(MIN_DUR, dist / SPEED_MM)
    abs_pub.publish(SetJetMax(x=t[0], y=t[1], z=t[2], duration=dur))
    rospy.sleep(dur + 0.1)
    show()


def move_joint(i, delta):
    m = status()
    cur = joints_of(m)[i]
    new = cur + delta
    dur = max(MIN_DUR, abs(delta) / SPEED_DEG)
    joint_pubs[i].publish(SetJoint(data=new, duration=dur))
    rospy.sleep(dur + 0.1)
    m2 = status()
    if abs(joints_of(m2)[i] - new) > JOINT_TOL:
        print('  Lệnh bị từ chối (ngoài giới hạn servo/động học), khớp %d giữ nguyên' % (i + 1))
    elif not in_limits(m2.x, m2.y, m2.z):
        print('  Ra ngoài vùng an toàn, hoàn lại khớp %d' % (i + 1))
        joint_pubs[i].publish(SetJoint(data=cur, duration=dur))
        rospy.sleep(dur + 0.1)
    show()


def move_wrist(delta):
    cur = status().pwm1
    new = int(clamp(cur + delta, PWM_MIN, PWM_MAX))
    pwm_pub.publish(SetServo(data=new, duration=0.2))
    rospy.sleep(0.3)
    show()


def toggle_sucker():
    sucker_pub.publish(Bool(data=not status().sucker))
    rospy.sleep(0.3)
    show()


def arm_high_enough():
    if status().z < SLIDER_MIN_Z:
        print('  Nâng tay máy lên z >= %d trước khi trượt (bấm r)' % SLIDER_MIN_Z)
        return False
    return True


def move_slider(d):
    global slider_pos, slider_enabled
    if stepper is None:
        print('  Thanh trượt không khả dụng')
        return
    if not arm_high_enough():
        return
    new = clamp(slider_pos + d, SLIDER_MIN, SLIDER_MAX)
    d = new - slider_pos
    if d == 0:
        print('  Thanh trượt ở biên (%d)' % slider_pos)
        return
    if not slider_enabled:
        stepper.set_mode(stepper.SLEEP)       # chỉ gửi một lần, không gửi trước mỗi lần bấm
        slider_enabled = True
    stepper.goto(d)
    time.sleep(abs(d) / SLIDER_SPEED + 0.2)
    slider_pos = new
    show()


def slider_home():
    global slider_pos, slider_enabled
    if stepper is None or not arm_high_enough():
        return
    print('  Thanh trượt về home (chờ đến khi xong)...')
    stepper.go_home(True)
    stepper.set_mode(stepper.EN)
    stepper.set_div(stepper.DIV_1_4)
    slider_pos = 0
    slider_enabled = False
    show()


HELP = """
Tịnh tiến (mm):  d/a = x +/-    w/s = y +/-    r/f = z +/-
Khớp (độ):       u/j = joint1   i/k = joint2   o/l = joint3   (n/m = cổ tay)
Thanh trượt:     x/z = tiến/lùi    c = về home thật (đặt lại vị trí 0)
Đi xa hơn:       gõ chữ HOA (giữ Shift) = bước x4, hoặc t để đổi mức bước
v = bật/tắt đầu hút    h = tay máy về home    p = in trạng thái    q = thoát
"""

start = status()
print('Vị trí ban đầu:', (start.x, start.y, start.z), 'pwm1 =', start.pwm1)
print(HELP)
print('Thanh trượt coi như đang ở 0 lúc mở; nếu không chắc, bấm c để về home thật.')
level = 0
print('Bước = %d mm / %d độ / %d bước trượt' % (MM_STEPS[level], DEG_STEPS[level], SLIDER_STEPS[level]))
show()

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while not rospy.is_shutdown():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        raw = sys.stdin.read(1)
        f = BIG if raw.isupper() else 1
        k = raw.lower()
        mm, deg, sl = MM_STEPS[level] * f, DEG_STEPS[level] * f, SLIDER_STEPS[level] * f
        if k == 'd':
            move_xyz(mm, 0, 0)
        elif k == 'a':
            move_xyz(-mm, 0, 0)
        elif k == 'w':
            move_xyz(0, mm, 0)
        elif k == 's':
            move_xyz(0, -mm, 0)
        elif k == 'r':
            move_xyz(0, 0, mm)
        elif k == 'f':
            move_xyz(0, 0, -mm)
        elif k == 'u':
            move_joint(0, deg)
        elif k == 'j':
            move_joint(0, -deg)
        elif k == 'i':
            move_joint(1, deg)
        elif k == 'k':
            move_joint(1, -deg)
        elif k == 'o':
            move_joint(2, deg)
        elif k == 'l':
            move_joint(2, -deg)
        elif k == 'n':
            move_wrist(deg)
        elif k == 'm':
            move_wrist(-deg)
        elif k == 'x':
            move_slider(sl)
        elif k == 'z':
            move_slider(-sl)
        elif k == 'c':
            slider_home()
        elif k == 'v':
            toggle_sucker()
        elif k == 't':
            level = (level + 1) % len(MM_STEPS)
            print('Bước = %d mm / %d độ / %d bước trượt' % (MM_STEPS[level], DEG_STEPS[level], SLIDER_STEPS[level]))
        elif k == 'h':
            rospy.ServiceProxy('/jetmax/go_home', Empty)()
            rospy.sleep(2.5)
            show()
        elif k == 'p':
            show()
        elif k == 'q':
            break
        termios.tcflush(fd, termios.TCIFLUSH)   # bỏ phím bấm dồn trong lúc robot đang chạy
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    sucker_pub.publish(Bool(data=False))
    if not args.stay:
        abs_pub.publish(SetJetMax(x=start.x, y=start.y, z=start.z, duration=2.0))
        pwm_pub.publish(SetServo(data=int(start.pwm1), duration=1.0))
        rospy.sleep(2.5)
        if stepper is not None and slider_pos != 0:
            if not slider_enabled:
                stepper.set_mode(stepper.SLEEP)
            stepper.goto(-slider_pos)
            time.sleep(slider_pos / SLIDER_SPEED + 0.3)
    if stepper is not None:
        stepper.set_mode(stepper.EN)
    print('Đã thoát' + ('' if args.stay else ' (về vị trí ban đầu)'))
```

---

## Nhóm 3: Tham chiếu SDK của robot (để viết motion.py đúng API)

### `docs\robot-reference\waste_classification_stepper.py`

*Dung lượng: 10.41 KB | Số dòng: 283*

```python
#!/usr/bin/env python3
import os
import sys
import cv2
import math
import time
import rospy
import numpy as np
import threading
import queue
import hiwonder
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from jetmax_control.msg import SetServo
from yolov5_tensorrt import Yolov5TensorRT

ROS_NODE_NAME = "waste_classification_stepper"

TRT_ENGINE_PATH = os.path.join(sys.path[0], "models/waste_v5_160.trt")
TRT_INPUT_SIZE = 160
TRT_CLASS_NAMES = ('Banana Peel', 'Broken Bones', 'Cigarette End', 'Disposable Chopsticks',
                   'Ketchup', 'Marker', 'Oral Liquid Bottle', 'Plate',
                   'Plastic Bottle', 'Storage Battery', 'Toothbrush', 'Umbrella')
TRT_NUM_CLASSES = 12
WASTE_CLASSES = {
    'food_waste': ('Banana Peel', 'Broken Bones', 'Ketchup'),
    'hazardous_waste': ('Marker', 'Oral Liquid Bottle', 'Storage Battery'),
    'recyclable_waste': ('Plastic Bottle', 'Toothbrush', 'Umbrella'),
    'residual_waste': ('Plate', 'Cigarette End', 'Disposable Chopsticks'),
}
COLORS = {
    'recyclable_waste': (0, 0, 255),
    'hazardous_waste': (255, 0, 0),
    'food_waste': (0, 255, 0),
    'residual_waste': (80, 80, 80)
}

TARGET_POSITION = {
    'recyclable_waste': (0, -180, 5, 5200),
    'hazardous_waste': (0, -180, 5, 4400),
    'food_waste': (0, -180, 5, 3600),
    'residual_waste': (0, -180, 5, 2800)
}

# 现实坐标转换函数
def camera_to_world(cam_mtx, r, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
    # invR * T
    inv_r = np.asmatrix(r_mat).I  # 3*3
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))  # 3*3 dot 3*1 = 3*1
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)  # 3*3 dot 3*1 = 3*1
        # [x,y,1] * invR
        worldPtPlane = np.dot(inv_r, worldPtCam)  # 3*3 dot 3*1 = 3*1
        # zc
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        # zc * [x,y,1] * invR
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        # [X,Y,Z]=zc*[x,y,1]*invR - invR*T
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt


class WasteClassification:
    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = False
        self.moving_box = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0
        self.camera_params = None
        self.fps_t0 = time.time()
        self.fps = 0
        self.K = None
        self.R = None
        self.T = None

    def reset(self):
        self.is_running = False
        self.moving_box = None
        self.image_sub = None
        self.heartbeat_timer = None
        self.runner = None
        self.count = 0

    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/card_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)

# 移动控制函数
def moving():
    try:
        c_x, c_y, waste_class_name = state.moving_box # 目标的画面坐标
        x, y, _ = camera_to_world(state.K, state.R, state.T, np.array([c_x, c_y]).reshape((1, 1, 2)))[0][0] #转换成现实距离
        cur_x, cur_y, cur_z = jetmax.position  # 当前的机械臂位置
        t = math.sqrt(x * x + y * y + 140 * 140) / 140 #计算移动的时间
        new_x, new_y = cur_x + x, cur_y + y #目标的现实坐标
        arm_angle = math.atan(new_y / new_x) * 180 / math.pi # 计算偏转角
        if arm_angle > 0:
            arm_angle = (90 - arm_angle)
        elif arm_angle < 0:
            arm_angle = (-90 - arm_angle)
        else:
            pass
        hiwonder.pwm_servo1.set_position(90 - arm_angle, 0.1) #角度补偿

        x, y, z, stepper_position = TARGET_POSITION[waste_class_name] # 读取放置坐标
        jetmax.set_position((new_x, new_y, 50), t) # 移动到目标上方
        rospy.sleep(t + 0.2)

        sucker.set_state(True) # 打开气泵
        jetmax.set_position((new_x, new_y, -12), 0.8) # 吸取目标
        rospy.sleep(0.8)
        jetmax.set_position((new_x, new_y, 30), 0.3)
        rospy.sleep(0.3)
        t = abs(new_x)/120.0 + abs(new_y-cur_y)/120.0
        hiwonder.pwm_servo1.set_position(90, 0.5) #角度回正
        jetmax.set_position((x, y, 50), t) # 移动到放置坐标上方
        rospy.sleep(t)

        stepper.set_mode(stepper.SLEEP) # 设置滑轨使能
        stepper.goto(stepper_position) # 驱动滑轨移动到放置位置
        stepper_time = stepper_position/1000 # 计算需要的时间
        rospy.sleep(stepper_time)

        jetmax.set_position((x, y, z), 0.5) # 到放置位置
        rospy.sleep(0.5)

        sucker.release(3)
        jetmax.set_position((x, y, z + 50), 0.5) 
        rospy.sleep(1)

    finally:
        jetmax.set_position((cur_x, cur_y, cur_z), 1) # 机械臂回到初始位置
        stepper.goto(-stepper_position) # 滑轨回到初始位置
        stepper_time = stepper_position/1000
        rospy.sleep(stepper_time)
        stepper.set_mode(stepper.EN) # 解除滑轨锁定
        state.moving_box = None
        state.runner = None
        print("FINISHED")

# 图像处理函数
def image_proc(image):
    if state.runner is not None:
        return image

    outputs = yolov5.detect(image) # 模型推理
    boxes, confs, classes = yolov5.post_process(image, outputs, 0.5)
    width = image.shape[1]
    height = image.shape[0]
    cards = []
    waste_name = 'None'
    for box, cls_conf, cls_id in zip(boxes, confs, classes):
        x1 = int(box[0] / TRT_INPUT_SIZE * width)
        y1 = int(box[1] / TRT_INPUT_SIZE * height)
        x2 = int(box[2] / TRT_INPUT_SIZE * width)
        y2 = int(box[3] / TRT_INPUT_SIZE * height)
        waste_name = TRT_CLASS_NAMES[cls_id]  # 得到检测到的分类名称
        
        waste_class_name = ''
        for k, v in WASTE_CLASSES.items():
            if waste_name in v:
                waste_class_name = k
                break
        cards.append((cls_conf, x1, y1, x2, y2, waste_class_name))
        cv2.putText(image, waste_class_name, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[waste_class_name], 2)
        cv2.putText(image, waste_name + "{:0.2f}".format(float(cls_conf)), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[waste_class_name], 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[waste_class_name], 3)

    if len(cards) == 0:
        state.count = 0
        state.moving_box = None
    else:
        if state.moving_box is None:
            moving_box = max(cards, key=lambda card: card[0])
            conf, x1, y1, x2, y2, waste_class_name = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            state.moving_box = c_x, c_y, waste_class_name
        else:
            l_c_x, l_c_y, l_waste_class_name = state.moving_box
            moving_box = min(cards, key=lambda card: math.sqrt((l_c_x - card[1]) ** 2 + (l_c_y - card[2]) ** 2))

            conf, x1, y1, x2, y2, waste_class_name = moving_box
            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
            # 在画面中标记出来
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 6)
            image = cv2.circle(image, (int(c_x), int(c_y)), 1, (255, 255, 255), 10)

            if math.sqrt((l_c_x - c_x) ** 2 + (l_c_y - c_y) ** 2) > 30:
                state.count = 0
            else:
                c_x = l_c_x * 0.2 + c_x * 0.8
                c_y = l_c_y * 0.2 + c_y * 0.8
                state.count += 1
            state.moving_box = c_x, c_y, waste_class_name
            if state.count > 50: #多次检测
                state.count = 0
                print(waste_name,waste_class_name)
                state.runner = threading.Thread(target=moving, daemon=True) #开启移动子线程
                state.runner.start()

    return image


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def image_proc_b():
    ros_image = image_queue.get(block=True)
    image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
    result_img = image_proc(image)
    # fps cal
    fps_t1 = time.time()
    fps_cur = (1.0 / (fps_t1 - state.fps_t0))
    state.fps = fps_cur if state.fps == 0.0 else (state.fps * 0.8 + fps_cur * 0.2)
    state.fps_t0 = fps_t1
    show_fps(result_img, state.fps)
    image_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imshow(ROS_NODE_NAME, image_bgr)
    cv2.waitKey(1)


def image_callback(ros_image):
    try:
        image_queue.put_nowait(ros_image)
    except queue.Full:
        pass

if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    image_queue = queue.Queue(maxsize=2)
    state = WasteClassification()
    state.load_camera_params()
    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()
    stepper = hiwonder.Stepper(1)
    jetmax.go_home(2,2)
    stepper.go_home(True)
    stepper.set_mode(stepper.EN)
    stepper.set_div(stepper.DIV_1_4)
    hiwonder.pwm_servo1.set_position(90, 0.5)
    rospy.sleep(1)
    yolov5 = Yolov5TensorRT(TRT_ENGINE_PATH, TRT_INPUT_SIZE, TRT_NUM_CLASSES)
    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback) # 订阅摄像头画面

    while True:
        try:
            image_proc_b()
            if rospy.is_shutdown():
                sys.exit(0)
        except KeyboardInterrupt:
            break

```

---

### `docs\robot-reference\jetmax_control\scripts\jetmax_control_main.py`

*Dung lượng: 8.67 KB | Số dòng: 182*

```python
#!/usr/bin/env python3
import actionlib
import rospy
from jetmax_control.msg import Mecanum
from jetmax_control.msg import JetMax as JetMaxState
from jetmax_control.msg import SetServo, SetJoint, SetJetMax
from std_msgs.msg import UInt8, UInt16, Float32, Bool
from std_srvs.srv import Empty
import hiwonder
import threading
import action_set


# import jetmax_control_joystick as joystick


class JetMaxControl:
    def __init__(self):
        rospy.init_node('jetmax_control', anonymous=True)
        self.pwm_servos = [hiwonder.pwm_servo1, hiwonder.pwm_servo2]

        """
        麦轮相关的topic
        """
        try:
            self.chassis = hiwonder.MecanumChassis()
            self.mecanum_pub = rospy.Publisher('/jetmax/mecanum/status', Mecanum, queue_size=1)
            self.mecanum_sub = rospy.Subscriber('/jetmax/mecanum/command', Mecanum,
                                                queue_size=1,
                                                callback=lambda msg: self.chassis.set_velocity(msg.velocity,
                                                                                               msg.direction,
                                                                                               msg.angular_rate))
        except Exception:
            self.chassis = None
            self.mecanum_sub = None
            self.mecanum_pub = None

        """
        机械臂相关的topic
        """
        self.jetmax = hiwonder.JetMax()
        # 机械臂末端位置控制topic
        self.jetmax_pub = rospy.Publisher('/jetmax/status', JetMaxState, queue_size=1)
        self.jetmax_sub = rospy.Subscriber('/jetmax/command', SetJetMax,
                                           queue_size=1,
                                           callback=lambda msg: self.jetmax.set_position((msg.x, msg.y, msg.z),
                                                                                         msg.duration))
        # 机械臂末端带速度的位置控制 topic
        self.jetmax_speed_sub = rospy.Subscriber('/jetmax/speed_command', SetJetMax,
                                                 queue_size=1,
                                                 callback=lambda msg: self.jetmax.set_position_with_speed(
                                                     (msg.x, msg.y, msg.z), max(msg.duration, 0.001)))
        # 机械臂末端相对位置控制topic
        self.jetmax_relative_sub = rospy.Subscriber('/jetmax/relative_command', SetJetMax,
                                                    queue_size=1,
                                                    callback=lambda msg: self.jetmax.set_position_relatively(
                                                        (msg.x, msg.y, msg.z), msg.duration))

        self.jetmax_joint_pubs = []
        self.jetmax_joint_subs = []
        self.jetmax_servo_pubs = []
        self.jetmax_servo_subs = []
        self.jetmax_servo_speed_subs = []
        # 机械臂各个关节的topic
        for i in range(1, 4):
            joint_pub = rospy.Publisher('/jetmax/joint{}/status'.format(i), Float32, queue_size=1)
            joint_cmd = rospy.Subscriber('/jetmax/joint{}/command'.format(i), SetJoint,
                                         queue_size=1,
                                         callback_args=i,
                                         callback=lambda msg, joint_id: self.jetmax.set_joint(joint_id,
                                                                                              msg.data,
                                                                                              msg.duration))
            servo_pub = rospy.Publisher('/jetmax/servo{}/status'.format(i), UInt16, queue_size=1)
            servo_cmd = rospy.Subscriber('/jetmax/servo{}/command'.format(i), SetServo,
                                         queue_size=1,
                                         callback_args=i,
                                         callback=lambda msg, servo_id: self.jetmax.set_servo(servo_id,
                                                                                              msg.data,
                                                                                              msg.duration))
            servo_speed_cmd = rospy.Subscriber('/jetmax/servo{}/speed_command'.format(i), SetServo,
                                               queue_size=1,
                                               callback_args=i,
                                               callback=lambda msg, servo_id: self.jetmax.set_servo_with_speed(
                                                   servo_id,
                                                   msg.data,
                                                   msg.duration))
            self.jetmax_joint_pubs.append(joint_pub)
            self.jetmax_joint_subs.append(joint_cmd)
            self.jetmax_servo_pubs.append(servo_pub)
            self.jetmax_servo_subs.append(servo_cmd)
            self.jetmax_servo_speed_subs.append(servo_speed_cmd)

        """
        末端执行器的topic
        """
        self.end_effector_servo_pubs = []
        self.end_effector_servo_subs = []
        # 末端的几个小的topic
        for i in range(1, 3):
            pub = rospy.Publisher('/jetmax/end_effector/servo{}/status'.format(i), UInt8, queue_size=1)
            cmd = rospy.Subscriber('/jetmax/end_effector/servo{}/command'.format(i), SetServo,
                                   queue_size=1,
                                   callback_args=i - 1,
                                   callback=lambda msg, servo_id: self.pwm_servos[servo_id].set_position(msg.data,
                                                                                                         msg.duration))
            self.end_effector_servo_pubs.append(pub)
            self.end_effector_servo_subs.append(cmd)

        self.sucker = hiwonder.Sucker()
        # 末端吸嘴的topic
        self.sucker_pub = rospy.Publisher('/jetmax/end_effector/sucker/status', Bool, queue_size=1)
        self.sucker_sub = rospy.Subscriber('/jetmax/end_effector/sucker/command', Bool,
                                           queue_size=1,
                                           callback=lambda msg: self.sucker.set_state(msg.data))

        # 回初始位置的topic
        self.go_home_srv = rospy.Service('/jetmax/go_home', Empty, self.go_home)
        self.action_set = action_set.ActionSetRunner(self.jetmax, self.sucker)

        self.rate = rospy.Rate(10)  # 10hz
        self.go_home() # 让机械臂回初始位置

    def go_home(self, req=None):
        self.jetmax.go_home(2)
        return []

    def update(self):
        while not rospy.is_shutdown():
            """
            更新麦轮底盘的状态
            """
            if self.chassis:
                mecanum_status = Mecanum(
                    velocity=self.chassis.velocity,
                    direction=self.chassis.direction,
                    angular_rate=self.chassis.angular_rate
                )
                self.mecanum_pub.publish(mecanum_status)

            """
            更新末端执行器的状态
            """
            pwm_servo_states = []
            for i in range(2):
                servo_state = UInt8(data=int(self.pwm_servos[i].get_position()))
                self.end_effector_servo_pubs[i].publish(servo_state)
                pwm_servo_states.append(servo_state)
            self.sucker_pub.publish(self.sucker.get_state())

            """
            更新机械臂的状态
            """
            servo_states = []
            joint_states = []
            for i in range(3):
                servo_state = int(self.jetmax.servos[i])
                joint_state = self.jetmax.joints[i]
                self.jetmax_servo_pubs[i].publish(UInt16(data=servo_state))
                self.jetmax_joint_pubs[i].publish(Float32(data=joint_state))
                servo_states.append(servo_state)
                joint_states.append(joint_state)
            x, y, z = self.jetmax.position
            self.jetmax_pub.publish(JetMaxState(
                x=x, y=y, z=z,
                joint1=joint_states[0], joint2=joint_states[1], joint3=joint_states[2],
                servo1=servo_states[0], servo2=servo_states[1], servo3=servo_states[2],
                pwm1=hiwonder.pwm_servo1.get_position(), pwm2=hiwonder.pwm_servo2.get_position(),
                sucker=self.sucker.get_state()
            ))

            self.rate.sleep()


if __name__ == '__main__':
    jetmax = JetMaxControl()
    # joystick.jetmax = jetmax.jetmax
    # joystick.sucker = jetmax.sucker
    try:
        #    threading.Thread(target=joystick.joystick_loop, daemon=True).start()
        jetmax.update()
    except rospy.ROSInterruptException:
        pass
```

---

## Nhóm 4: Chỉ gửi nếu tôi hỏi thêm

### `tests\test_game.py`

*Dung lượng: 6.51 KB | Số dòng: 153*

```python
#!/usr/bin/env python3
# Kiểm thử nhận diện nước đi, engine, ánh xạ ảnh -> ô cờ và cả ván mô phỏng. Không cần robot/ROS.
#   python3 tests/test_game.py

import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

from xiangqi_rules import Position, move_uci
from engine import SimpleEngine
from game import detect_move, BoardStabilizer, Game, make_info
from vision_board import ImageBoardMapper
from play_sim import build
from game import play


def test_detect_every_move():
    rng = random.Random(0)
    pos = Position.start()
    for _ in range(40):                           # đi ngẫu nhiên 40 nước, kiểm tra mọi nước hợp lệ ở mỗi thế cờ
        for m in pos.legal_moves():
            r = detect_move(pos, pos.make(m).to_dict())
            assert r.status == 'move' and r.move == m, (move_uci(m), r)
        assert detect_move(pos, pos.to_dict()).status == 'nochange'
        moves = pos.legal_moves()
        if not moves:
            break
        pos = pos.make(rng.choice(moves))


def test_detect_noise_never_wrong():
    rng = random.Random(1)
    pos = Position.start()
    wrong = ok = 0
    for _ in range(30):
        moves = pos.legal_moves()
        if not moves:
            break
        for m in moves:
            obs = pos.make(m).to_dict()
            del obs[rng.choice(sorted(obs))]      # YOLO bỏ sót một quân bất kỳ
            r = detect_move(pos, obs, tolerance=1)
            if r.status == 'move':
                if r.move == m:
                    ok += 1
                else:
                    wrong += 1
        pos = pos.make(rng.choice(moves))
    assert wrong == 0, 'nhận nhầm nước đi %d lần' % wrong
    assert ok > 0
    print('   (bỏ sót 1 quân: nhận đúng %d, không bao giờ nhận sai)' % ok)


def test_no_move_with_noise():
    pos = Position.start()
    obs = pos.to_dict()
    del obs[(4, 9)]                               # mất tướng đen trong ảnh, quân đen không thể là nước của người đi đỏ
    assert detect_move(pos, obs).status == 'nochange'
    obs = pos.to_dict()
    del obs[(4, 0)]                               # mất tướng đỏ: có thể là nhiễu, cũng có thể tướng vừa đi mà ảnh không
    assert detect_move(pos, obs).status == 'ambiguous'   # thấy ở ô mới -> không đoán, chờ khung hình sau


def test_invalid_board():
    pos = Position.start()
    obs = pos.to_dict()
    del obs[(0, 3)], obs[(2, 3)], obs[(4, 3)]     # mất 3 tốt: không nước nào khớp
    assert detect_move(pos, obs).status == 'invalid'


def test_stabilizer():
    s = BoardStabilizer(3)
    a = {(0, 0): 'R', (1, 0): 'N'}
    assert s.update(a) is None and s.update(a) is None
    assert s.update(a) == a
    flicker = {(0, 0): 'R'}                       # một khung hình bỏ sót quân vẫn qua được nhờ bỏ phiếu
    assert s.update(flicker) == a


def test_human_confirm():
    g = Game('red', SimpleEngine(seed=0), confirm=2)
    after = g.pos.make(g.pos.legal_moves()[0]).to_dict()
    r, info = g.human_observed(after)
    assert r.status == 'pending' and info is None
    r, info = g.human_observed(after)
    assert r.status == 'move' and info is not None and info.by == 'human'


def test_engine_legal_and_captures():
    # robot (đỏ) có xe ăn được pháo đen không được bảo vệ -> với độ sâu 2 phải ăn khi không "đi lệch"
    pos = Position.from_fen('5k3/9/9/9/4c4/9/9/9/4R4/3K5 w')
    eng = SimpleEngine('hard', seed=0)
    assert move_uci(eng.choose(pos)) == 'e1e5'


def test_uci_engine_with_fake():
    from engine import UciEngine
    fake = os.path.join(HERE, 'fake_uci_engine.py')
    eng = UciEngine(sys.executable, level='hard', args=[fake], seed=0)    # hard: luôn chọn nước tốt nhất
    try:
        pos = Position.start()
        assert eng.options >= {'Threads', 'Hash', 'MultiPV', 'EvalFile'}
        # phải lấy kết quả ở độ sâu LỚN NHẤT (depth 2), không phải depth 1: nước đầu tiên trong danh sách hợp lệ
        assert eng.choose(pos) == pos.legal_moves()[0]
        for _ in range(3):                         # nhiều lần liên tiếp trên nhiều thế cờ khác nhau
            pos = pos.make(eng.choose(pos))
            assert pos.legal_moves()
        easy = UciEngine(sys.executable, level='easy', args=[fake], seed=1)
        try:
            legal = set(pos.legal_moves())
            for _ in range(20):
                assert easy.choose(pos) in legal
        finally:
            easy.close()
    finally:
        eng.close()


def test_image_mapper():
    # ảnh giả: bàn cờ phóng 50 px/ô, góc a0 ở (100, 600), trục y ảnh hướng LÊN khi hàng tăng (xoay lật tuỳ ý)
    corners = {'a0': (100, 600), 'i0': (500, 600), 'a9': (100, 150), 'i9': (500, 150)}
    m = ImageBoardMapper(corners)
    names = ['%s_%s' % (c, p) for c in ('black', 'red')
             for p in ('advisor', 'bishop', 'cannon', 'king', 'knight', 'pawn', 'rook')]
    cx, cy = 100 + 4 * 50, 600 - 0 * 50           # ô e0
    dets = [(cx - 20, cy - 20, cx + 20, cy + 20, 0.9, names.index('red_king')),
            (100 + 7 * 50 - 20, 600 - 2 * 50 - 20, 100 + 7 * 50 + 20, 600 - 2 * 50 + 20, 0.8, names.index('red_cannon')),
            (100 + 3.5 * 50 - 20, 150 - 20, 100 + 3.5 * 50 + 20, 150 + 20, 0.7, names.index('black_rook'))]  # lệch giữa 2 ô
    board, warns = m.board_from_detections(dets, names)
    assert board == {(4, 0): 'K', (7, 2): 'C'}, board
    assert len(warns) == 1


def test_full_game_with_noise():
    for human in ('red', 'black'):
        game, sim = build(human, level='easy', noise=0.03, auto=True, seed=7, max_plies=60, log=lambda *a: None)
        play(game, sim.observe, log=lambda *a: None, poll=0)
        # mọi nước trong lịch sử phải hợp lệ khi phát lại từ đầu, và thế cờ cuối khớp thế cờ vật lý
        pos = Position.start()
        for info in game.history:
            assert pos.is_legal(info.move), info.uci
            pos = pos.make(info.move)
        assert pos.fen() == game.pos.fen()
        robot_first = game.history[0].by
        assert robot_first == ('robot' if human == 'black' else 'human')
        print('   (%s: %d nước, kết quả %s)' % (human, len(game.history), game.result()))


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        fn()
        print('OK  ', name)
    print('TẤT CẢ ĐỀU ĐẠT')
```

---

### `tests\test_rules.py`

*Dung lượng: 2.92 KB | Số dòng: 79*

```python
#!/usr/bin/env python3
# Kiểm thử luật cờ tướng. Không cần robot/ROS.
#   python3 tests/test_rules.py            # perft độ sâu 1-3 + các tình huống riêng (vài giây)
#   python3 tests/test_rules.py --deep     # thêm perft độ sâu 4 (3.290.240 nút, chậm trên Jetson)
# Số perft từ thế cờ ban đầu của cờ tướng: 44, 1920, 79666, 3290240 (số chuẩn dùng để kiểm tra bộ sinh nước đi).

import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'robot')]

from xiangqi_rules import Position, perft, move_uci, parse_uci, START_FEN

PERFT = {1: 44, 2: 1920, 3: 79666, 4: 3290240}


def uci_set(fen, only_from=None):
    pos = Position.from_fen(fen)
    ms = [move_uci(m) for m in pos.legal_moves()]
    if only_from:
        ms = [m for m in ms if m.startswith(only_from)]
    return set(ms)


def test_fen_roundtrip():
    assert Position.start().fen() == START_FEN
    p = Position.from_fen('R3k4/R8/9/9/9/9/9/9/9/3K5 b')
    assert Position.from_fen(p.fen()).fen() == p.fen()


def test_flying_general():
    # tướng đỏ không được sang cột d vì sẽ đối mặt tướng đen ở d9
    assert uci_set('3k5/9/9/9/9/9/9/9/9/4K4 w') == {'e0f0', 'e0e1'}


def test_knight_leg():
    free = uci_set('5k3/9/9/9/9/9/9/9/4N4/3K5 w', 'e1')
    assert len(free) == 6, free
    blocked = uci_set('5k3/9/9/9/9/9/9/4p4/4N4/3K5 w', 'e1')    # tốt đen ở e2 cản chân mã
    assert len(blocked) == 4 and 'e1f3' not in blocked and 'e1d3' not in blocked, blocked


def test_cannon_screen():
    ms = uci_set('5k3/9/9/9/4r4/9/4P4/9/4C4/3K5 w', 'e1')
    assert 'e1e2' in ms                # đi không ăn
    assert 'e1e5' in ms                # ăn xe đen qua ngòi là tốt e3
    assert 'e1e3' not in ms and 'e1e4' not in ms


def test_bishop_river():
    ms = uci_set('5k3/9/9/9/9/2B6/9/9/9/3K5 w', 'c4')
    assert ms == {'c4a2', 'c4e2'}, ms


def test_pawn():
    assert uci_set('5k3/9/9/9/9/9/P8/9/9/4K4 w', 'a3') == {'a3a4'}             # chưa qua sông: chỉ đi thẳng
    assert uci_set('5k3/9/9/9/4P4/9/9/9/9/3K5 w', 'e5') == {'e5e6', 'e5d5', 'e5f5'}   # qua sông: thêm đi ngang


def test_checkmate():
    assert Position.from_fen('R3k4/R8/9/9/9/9/9/9/9/3K5 b').result() == 'checkmate'
    assert Position.start().result() == 'ongoing'


def test_perft(deep=False):
    pos = Position.start()
    for d in (1, 2, 3, 4) if deep else (1, 2, 3):
        t = time.time()
        n = perft(pos, d)
        print('  perft(%d) = %d  (chuẩn %d)  %.1fs' % (d, n, PERFT[d], time.time() - t))
        assert n == PERFT[d], 'perft(%d) sai: %d != %d' % (d, n, PERFT[d])


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_') and k != 'test_perft'):
        fn()
        print('OK  ', name)
    test_perft('--deep' in sys.argv)
    print('OK   test_perft')
    print('TẤT CẢ ĐỀU ĐẠT')
```

---

### `README.md`

*Dung lượng: 153 B | Số dòng: 2*

```markdown
# xiangqi-sorting-jetmax
Real-time Xiangqi piece detection (YOLO11 + OpenCV) and sorting on a moving conveyor belt with a Hiwonder JetMax robotic arm.
```

---

