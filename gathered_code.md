# GATHERED SOURCE CODE - XIANGQI SORTING JETMAX

- **Thư mục gốc:** `D:\2026_2027\Chuyen-de\xiangqi-sorting-jetmax`
- **Tổng số file đã gôm:** 13

## Mục lục

### Nhóm 1: Core Robot (Board, Detect View, Motion)
- [robot\board.py](#robotboardpy)
- [robot\detect_view.py](#robotdetect_viewpy)
- [robot\motion.py](#robotmotionpy)

### Nhóm 2: Cấu hình hệ thống (Configs)
- [configs\board.yaml](#configsboardyaml)

### Nhóm 3: Test Scripts & Kịch bản thử nghiệm
- [tests\move_piece.py](#testsmove_piecepy)
- [tests\pick_place.py](#testspick_placepy)
- [tests\scan_detect.py](#testsscan_detectpy)

### Nhóm 4: Huấn luyện, Cấu trúc Class & Dataset
- [training\README.md](#trainingreadmemd)
- [training\build_dataset.ipynb](#trainingbuild_datasetipynb)
- [training\train_yolo11.ipynb](#trainingtrain_yolo11ipynb)

### Nhóm 5: Cấu hình danh sách Class & Dataset Config
  *(Không tìm thấy file nào)*

### Nhóm 6: Reference Scripts & API mẫu
- [docs\robot-reference\waste_classification_stepper.py](#docsrobot-referencewaste_classification_stepperpy)
- [docs\robot-reference\jetmax_control\scripts\jetmax_control_main.py](#docsrobot-referencejetmax_controlscriptsjetmax_control_mainpy)

### Nhóm 7: Tổng quan dự án
- [README.md](#readmemd)

---

## Nhóm 1: Core Robot (Board, Detect View, Motion)

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

## Nhóm 2: Cấu hình hệ thống (Configs)

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

## Nhóm 3: Test Scripts & Kịch bản thử nghiệm

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

## Nhóm 4: Huấn luyện, Cấu trúc Class & Dataset

### `training\README.md`

*Dung lượng: 0 B | Số dòng: 0*

```markdown

```

---

### `training\build_dataset.ipynb`

*File Jupyter Notebook | Dung lượng: 3.49 MB*

<!-- Cell 1 (Markdown) -->
<a href="https://colab.research.google.com/github/JulianNguyen05/xiangqi-sorting-jetmax/blob/main/training/build_dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- Cell 2 (Markdown) -->
# Mount Drive và tạo thư mục

```python
# Cell 3 (Code)
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/datn'
RAW_DRIVE = f'{BASE}/raw'          # bản nén .zip của từng bộ (lưu lâu dài)
RAW_LOCAL = '/content/raw'         # bản giải nén trên máy Colab (đọc nhanh)
for d in ['raw', 'merged', 'runs', 'own_images']:
    os.makedirs(f'{BASE}/{d}', exist_ok=True)
os.makedirs(RAW_LOCAL, exist_ok=True)
print('OK:', BASE)
```

<!-- Cell 4 (Markdown) -->
# Cài thư viện

```python
# Cell 5 (Code)
!pip install -q roboflow imagehash pyyaml
```

<!-- Cell 6 (Markdown) -->
# Lấy API key


```python
# Cell 7 (Code)
from google.colab import userdata
from roboflow import Roboflow

API_KEY = userdata.get('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=API_KEY)
print('Đã kết nối Roboflow')
```

<!-- Cell 8 (Markdown) -->
## Khai báo 4 bộ dataset
Danh sách 4 dataset trên Roboflow Universe.

| Tên ngắn | Link |
|---|---|
| super | https://universe.roboflow.com/312s-workspace/xiangqi-super-dataset |
| scara | https://universe.roboflow.com/312s-workspace/xiangqi-scara-yolo26 |
| new_chinese_chess | https://universe.roboflow.com/quoc-rnmzr/new-chinese-chess-ayph4/dataset/1 |
| duytranle | https://universe.roboflow.com/duytranle/xiangqi-ubjij/dataset/18 |

```python
# Cell 9 (Code)
DATASETS = {
    'super':             {'workspace': '312s-workspace', 'project': 'xiangqi-super-dataset',   'version': None},
    'scara':             {'workspace': '312s-workspace', 'project': 'xiangqi-scara-yolo26',    'version': None},
    'new_chinese_chess': {'workspace': 'quoc-rnmzr',     'project': 'new-chinese-chess-ayph4', 'version': 1},
    'duytranle':         {'workspace': 'duytranle',      'project': 'xiangqi-ubjij',           'version': 18,
                          'format': 'yolov8'},   # dự án keypoint không hỗ trợ yolov11
}
```

<!-- Cell 10 (Markdown) -->
### Tải về, giải nén và sao lưu lên Drive
Với mỗi bộ: (1) đã có trên Colab thì bỏ qua; (2) đã có file .zip trên Drive thì giải nén; (3) chưa có thì tải từ Roboflow (định dạng yolov11) rồi nén lưu lên Drive. Phiên bản đã tải được ghi vào versions_used.json để đưa vào báo cáo.

```python
# Cell 11 (Code)
import shutil, json

def latest_version(project):
    vs = project.versions()
    return max(int(str(v.version).split('/')[-1]) for v in vs)

VERSIONS_FILE = f'{BASE}/raw/versions_used.json'
versions_used = json.load(open(VERSIONS_FILE)) if os.path.exists(VERSIONS_FILE) else {}
paths = {}

for name, cfg in DATASETS.items():
    out_dir  = f'{RAW_LOCAL}/{name}'
    zip_base = f'{RAW_DRIVE}/{name}'

    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f'[{name}] đã có trên Colab, bỏ qua')
    elif os.path.exists(zip_base + '.zip'):
        print(f'[{name}] giải nén từ Drive...')
        shutil.unpack_archive(zip_base + '.zip', out_dir)
    else:
        print(f'[{name}] tải từ Roboflow...')
        project = rf.workspace(cfg['workspace']).project(cfg['project'])
        ver = cfg['version'] or latest_version(project)
        fmt = cfg.get('format', 'yolov11')
        project.version(ver).download(fmt, location=out_dir)
        shutil.make_archive(zip_base, 'zip', out_dir)
        versions_used[name] = {**cfg, 'version': ver, 'format': fmt}
        with open(VERSIONS_FILE, 'w') as f:
            json.dump(versions_used, f, indent=2, ensure_ascii=False)
        print(f'[{name}] đã tải version {ver} (định dạng {fmt}) và lưu lên Drive')

    paths[name] = out_dir

print('Xong:', list(paths))
```

<!-- Cell 12 (Markdown) -->
### Kiểm tra tên lớp và số ảnh của từng bộ
In danh sách lớp (names trong data.yaml) và số ảnh mỗi tập. Với mỗi bộ: in tên lớp, số ảnh (và số ảnh gốc ước lượng) mỗi tập, bảng số box theo lớp ở từng tập, và các lỗi nhãn.

```python
# Cell 13 (Code)
import yaml, glob, os
import pandas as pd
from collections import Counter

SPLITS = ['train', 'valid', 'test']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
MIN_BOX_WARN = 300          # lớp có ít box hơn mức này sẽ bị cảnh báo
pd.set_option('display.width', 220)

def load_names(yaml_path):
    with open(yaml_path, encoding='utf-8') as f:
        names = yaml.safe_load(f)['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names

def source_id(filename):
    # Roboflow: <tên gốc>.rf.<hash>.jpg -> bản augment cùng một ảnh gốc
    return filename.split('.rf.')[0]

reports = {}
for name, p in paths.items():
    print('=' * 80)
    print(name.upper())
    yaml_files = glob.glob(f'{p}/**/data.yaml', recursive=True)
    if not yaml_files:
        print('  Không thấy data.yaml')
        continue
    root = os.path.dirname(yaml_files[0])
    names = load_names(yaml_files[0])
    print(f'  Số lớp: {len(names)}')
    print(f'  Tên lớp: {names}')

    box = {s: Counter() for s in SPLITS}      # số box theo lớp
    img_with = Counter()                      # số ảnh có chứa lớp (cả 3 tập)
    problems = Counter()

    for s in SPLITS:
        img_files = [f for f in glob.glob(f'{root}/{s}/images/*')
                     if f.lower().endswith(IMG_EXT)]
        n_src = len({source_id(os.path.basename(f)) for f in img_files})
        print(f'  {s:6s}: {len(img_files):5d} ảnh  (~{n_src} ảnh gốc ước lượng)')

        lbl_files = glob.glob(f'{root}/{s}/labels/*.txt')
        img_stems = {os.path.splitext(os.path.basename(f))[0] for f in img_files}
        lbl_stems = {os.path.splitext(os.path.basename(f))[0] for f in lbl_files}
        problems['ảnh không có file nhãn'] += len(img_stems - lbl_stems)
        problems['file nhãn không có ảnh'] += len(lbl_stems - img_stems)

        for lf in lbl_files:
            with open(lf) as f:
                lines = [l.split() for l in f if l.strip()]
            if not lines:
                problems['file nhãn rỗng'] += 1
            seen = set()
            for parts in lines:
                if len(parts) != 5:
                    problems['dòng nhãn không phải bbox (5 giá trị)'] += 1
                    continue
                try:
                    cid = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    problems['dòng nhãn sai định dạng'] += 1
                    continue
                if not (0 <= cid < len(names)):
                    problems['id lớp ngoài phạm vi'] += 1
                    continue
                if w <= 0 or h <= 0:
                    problems['box có w hoặc h <= 0'] += 1
                if x - w/2 < -0.01 or y - h/2 < -0.01 or x + w/2 > 1.01 or y + h/2 > 1.01:
                    problems['box lệch ra ngoài khung ảnh'] += 1
                box[s][names[cid]] += 1
                seen.add(names[cid])
            for c in seen:
                img_with[c] += 1

    df = pd.DataFrame({f'{s}_box': pd.Series(box[s]) for s in SPLITS}).reindex(names).fillna(0).astype(int)
    df['tong_box'] = df.sum(axis=1)
    df['tong_anh_co_lop'] = pd.Series(img_with).reindex(names).fillna(0).astype(int)
    reports[name] = df

    print('\n  Số box theo lớp:')
    print(df.to_string())
    print(f"\n  Tổng box: {df['tong_box'].sum()}")
    weak = df[df['tong_box'] < MIN_BOX_WARN].index.tolist()
    if weak:
        print(f'  ⚠ Lớp ít hơn {MIN_BOX_WARN} box: {weak}')
    if problems:
        print('  ⚠ Vấn đề phát hiện:')
        for k, v in problems.items():
            print(f'     - {k}: {v}')
    else:
        print('  Không phát hiện lỗi nhãn cơ bản.')
```

<!-- Cell 14 (Markdown) -->
### Chẩn đoán dòng nhãn không phải bbox
Với mỗi bộ: đếm dòng 5 giá trị (bbox chuẩn) và các dòng khác, gom theo (lớp, số giá trị mỗi dòng), in kèm một dòng mẫu. Dùng để biết dòng lạ là polygon (segmentation) hay pose (keypoint) và thuộc lớp nào.

```python
# Cell 15 (Code)
from collections import Counter

for name, p in paths.items():
    yf = glob.glob(f'{p}/**/data.yaml', recursive=True)[0]
    root = os.path.dirname(yf)
    names = load_names(yf)
    odd, sample, ok = Counter(), {}, 0
    for lf in glob.glob(f'{root}/*/labels/*.txt'):
        for line in open(lf):
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 5:
                ok += 1
                continue
            key = (names[int(parts[0])], len(parts))
            odd[key] += 1
            sample.setdefault(key, ' '.join(parts[:12]))
    print('=' * 70)
    print(f'{name} | dòng 5 giá trị: {ok} | dòng khác: {sum(odd.values())}')
    for (cls, n), c in odd.most_common(6):
        print(f'  lớp={cls:15s} số giá trị/dòng={n:3d}  số dòng={c}')
        print('     mẫu:', sample[(cls, n)])
```

<!-- Cell 16 (Markdown) -->
## Làm sạch và ánh xạ về 14 lớp chuẩn
Duyệt từng bộ: đổi tên lớp về 14 lớp chuẩn, bỏ dòng `board`, bỏ ảnh có lớp `None`, chuyển polygon (super) và pose (duytranle) về bbox, kẹp box vào khung ảnh. Ảnh và nhãn sạch được gom vào /content/pool kèm manifest.csv (bộ, tập gốc, mã ảnh gốc). In bảng số box theo lớp và thống kê các thao tác đã làm.

```python
# Cell 17 (Code)
import shutil, glob, os, yaml
import pandas as pd
from collections import Counter

SPLITS = ['train', 'valid', 'test']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
POOL = '/content/pool'

CLASSES = [f'{c}_{p}' for c in ['black', 'red']
           for p in ['advisor', 'bishop', 'cannon', 'king', 'knight', 'pawn', 'rook']]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
PIECE_ALIAS = {'elephant': 'bishop', 'chariot': 'rook', 'general': 'king',
               'horse': 'knight', 'guard': 'advisor', 'soldier': 'pawn'}
DROP_LINES = {'board'}      # bỏ dòng nhãn, giữ ảnh
DROP_IMAGES = {'none'}      # bỏ cả ảnh có lớp này
POSE_DATASETS = {'duytranle'}   # nhãn dạng: lớp x y w h + keypoint

def load_names(yaml_path):
    with open(yaml_path, encoding='utf-8') as f:
        names = yaml.safe_load(f)['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names

def source_id(filename):
    return filename.split('.rf.')[0]

def canon(raw):
    n = raw.strip().lower().replace('-', '_').replace(' ', '_')
    if n in DROP_LINES:
        return 'DROP_LINE'
    if n in DROP_IMAGES:
        return 'DROP_IMAGE'
    color, piece = n.split('_', 1)
    name = f'{color}_{PIECE_ALIAS.get(piece, piece)}'
    if name not in CLASS_ID:
        raise ValueError(f'Lớp lạ chưa ánh xạ: {raw}')
    return name

if os.path.exists(POOL):
    shutil.rmtree(POOL)
os.makedirs(f'{POOL}/images'); os.makedirs(f'{POOL}/labels')

rows, cls_count, stats = [], {}, {}
for name, p in paths.items():
    yf = glob.glob(f'{p}/**/data.yaml', recursive=True)[0]
    root = os.path.dirname(yf)
    names = load_names(yf)
    pose = name in POSE_DATASETS
    st, cnt = Counter(), Counter()

    for split in SPLITS:
        img_dir, lbl_dir = f'{root}/{split}/images', f'{root}/{split}/labels'
        if not os.path.isdir(lbl_dir):
            continue
        imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)}
        for lf in os.listdir(lbl_dir):
            stem = os.path.splitext(lf)[0]
            if stem not in imgs:
                continue
            out, kept, skip = [], [], False
            for line in open(f'{lbl_dir}/{lf}'):
                parts = line.split()
                if not parts:
                    continue
                c = canon(names[int(parts[0])])
                if c == 'DROP_LINE':
                    st['dòng board bỏ'] += 1
                    continue
                if c == 'DROP_IMAGE':
                    skip = True
                    break
                vals = list(map(float, parts[1:]))
                if len(vals) == 4:
                    x, y, w, h = vals
                elif pose and len(vals) >= 4:
                    x, y, w, h = vals[:4]; st['pose→bbox'] += 1
                elif len(vals) >= 6 and len(vals) % 2 == 0:
                    xs, ys = vals[0::2], vals[1::2]
                    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                    x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
                    st['polygon→bbox'] += 1
                else:
                    st['dòng lỗi bỏ'] += 1
                    continue
                x0, y0 = max(0.0, x - w / 2), max(0.0, y - h / 2)
                x1, y1 = min(1.0, x + w / 2), min(1.0, y + h / 2)
                w, h = x1 - x0, y1 - y0
                if w <= 0.005 or h <= 0.005:
                    st['box quá nhỏ/lỗi bỏ'] += 1
                    continue
                out.append(f'{CLASS_ID[c]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {w:.6f} {h:.6f}')
                kept.append(c)
            if skip:
                st['ảnh bỏ (lớp None)'] += 1
                continue
            if not out:
                st['ảnh bỏ (không còn box)'] += 1
                continue
            new = f'{name}__{imgs[stem]}'
            shutil.copy(f'{img_dir}/{imgs[stem]}', f'{POOL}/images/{new}')
            with open(f'{POOL}/labels/{os.path.splitext(new)[0]}.txt', 'w') as f:
                f.write('\n'.join(out) + '\n')
            cnt.update(kept)
            rows.append({'file': new, 'dataset': name, 'orig_split': split,
                         'source_id': f'{name}__{source_id(imgs[stem])}', 'n_box': len(out)})
    cls_count[name], stats[name] = cnt, st

manifest = pd.DataFrame(rows)
manifest.to_csv(f'{POOL}/manifest.csv', index=False)

print('Số ảnh sạch theo bộ và tập gốc:')
print(manifest.groupby(['dataset', 'orig_split']).size().unstack(fill_value=0))
table = pd.DataFrame({n: [c.get(k, 0) for k in CLASSES] for n, c in cls_count.items()}, index=CLASSES)
table['tong'] = table.sum(axis=1)
print('\nSố box theo lớp chuẩn:')
print(table)
print('\nThao tác đã thực hiện:')
for n, s in stats.items():
    print(f'  {n}: {dict(s)}')
```

<!-- Cell 18 (Markdown) -->
## Gom nhóm ảnh trùng (cùng ảnh gốc hoặc gần giống nhau)
Tính perceptual hash cho từng ảnh trong /content/pool rồi gom nhóm bằng union-find: ảnh cùng source_id là một nhóm; ảnh cùng bộ chỉ nối khi hash trùng khít (SAME_DS_T); ảnh khác bộ nối khi khoảng cách hash <= CROSS_DS_T. In kích thước nhóm, thành phần của nhóm lớn nhất, số nhóm nối nhiều bộ, và vẽ vài cặp ảnh trùng khác bộ để kiểm tra bằng mắt. Chưa xóa gì.

```python
# Cell 19 (Code)
import numpy as np, imagehash, matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

SAME_DS_T = 0    # ngưỡng phash nối ảnh CÙNG bộ (0 = hash trùng khít; đặt -1 để chỉ gom theo source_id)
CROSS_DS_T = 4   # ngưỡng phash nối ảnh KHÁC bộ
manifest = pd.read_csv(f'{POOL}/manifest.csv')
n = len(manifest)

# 1) perceptual hash
hashes = []
for f in manifest['file']:
    try:
        with Image.open(f'{POOL}/images/{f}') as im:
            hashes.append(int(str(imagehash.phash(im)), 16))
    except Exception as e:
        print('Lỗi ảnh:', f, e); hashes.append(0)
H = np.array(hashes, dtype=np.uint64)
ds = manifest['dataset'].values

# 2) union-find
parent = list(range(n))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

first = {}
for i, s in enumerate(manifest['source_id']):
    if s in first: union(first[s], i)
    else: first[s] = i

def popcount64(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)

for i in range(n - 1):
    d = popcount64(H[i + 1:] ^ H[i])
    thr = np.where(ds[i + 1:] == ds[i], SAME_DS_T, CROSS_DS_T)
    for j in (np.nonzero(d <= thr)[0] + i + 1):
        union(i, int(j))

manifest['group'] = [find(i) for i in range(n)]

# 3) thống kê
g = manifest.groupby('group')
sizes = g.size().sort_values(ascending=False)
multi_ds = g['dataset'].nunique()
multi_sp = g['orig_split'].nunique()
print(f'Ảnh: {n} | nhóm: {len(sizes)} | nhóm lớn nhất: {sizes.head(5).tolist()}')
top = sizes.index[0]
print('Thành phần nhóm lớn nhất:', manifest[manifest.group == top].groupby(['dataset', 'orig_split']).size().to_dict())
print(f'Nhóm trải qua >1 bộ: {(multi_ds > 1).sum()} | nhóm nằm ở >1 tập gốc: {(multi_sp > 1).sum()}')
if sizes.iloc[0] > 0.03 * n:
    print('⚠ Nhóm lớn nhất quá lớn: đặt SAME_DS_T = -1 và chạy lại.')

print('\nCặp bộ có ảnh trùng nhau:')
pair = defaultdict(int)
for gid in multi_ds[multi_ds > 1].index:
    sets = tuple(sorted(manifest.loc[manifest.group == gid, 'dataset'].unique()))
    pair[sets] += 1
for k, v in sorted(pair.items(), key=lambda kv: -kv[1]):
    print(' ', k, v)

# 4) vẽ vài cặp trùng khác bộ để kiểm tra bằng mắt
cross = multi_ds[multi_ds > 1].index[:3]
if len(cross):
    fig, axes = plt.subplots(len(cross), 2, figsize=(8, 3.5 * len(cross)), squeeze=False)
    for r, gid in enumerate(cross):
        sub = manifest[manifest.group == gid].drop_duplicates('dataset').head(2)
        for c, (_, row) in enumerate(sub.iterrows()):
            axes[r][c].imshow(Image.open(f'{POOL}/images/{row.file}'))
            axes[r][c].set_title(row.dataset); axes[r][c].axis('off')
    plt.tight_layout(); plt.show()
```

<!-- Cell 20 (Markdown) -->
## Chia train/val/test theo nhóm và xuất dataset hợp nhất
Chia ~80/10/10 theo ảnh, giữ nguyên mỗi nhóm trong cùng một tập và giữ tỷ lệ các bộ trong từng tập (StratifiedGroupKFold). Kiểm tra tỷ lệ val/test của từng bộ trước; nếu lệch thì dừng, không ghi file. Nếu ổn thì copy ảnh/nhãn vào /content/merged, ghi data.yaml với 14 lớp chuẩn, in số box theo lớp ở mỗi tập, rồi nén lưu bản sao lên Drive (ghi đè merged_v1.zip cũ).

```python
# Cell 21 (Code)
from sklearn.model_selection import StratifiedGroupKFold
import yaml

MERGED = '/content/merged'
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
fold = np.zeros(len(manifest), dtype=int)
for k, (_, te) in enumerate(sgkf.split(manifest, manifest['dataset'], manifest['group'])):
    fold[te] = k
manifest['split'] = np.where(fold == 0, 'test', np.where(fold == 1, 'val', 'train'))

# kiểm tra tỷ lệ trước khi ghi file
share = manifest.groupby(['dataset', 'split']).size().unstack(fill_value=0)
for c in ['train', 'val', 'test']:
    if c not in share: share[c] = 0
pct = share[['val', 'test']].div(share[['train', 'val', 'test']].sum(axis=1), axis=0) * 100
print('Số ảnh theo bộ và tập:'); print(share[['train', 'val', 'test']])
print('\nTỷ lệ val/test mỗi bộ (%):'); print(pct.round(1))
bad = pct[(share[['train', 'val', 'test']].sum(axis=1) >= 100) & ((pct < 5).any(axis=1) | (pct > 20).any(axis=1))]
if len(bad):
    raise ValueError(f'Chia tập lệch ở bộ: {list(bad.index)}. Xem lại bước gom nhóm (nhóm quá lớn?), chưa ghi file.')

if os.path.exists(MERGED):
    shutil.rmtree(MERGED)
for s in ['train', 'val', 'test']:
    os.makedirs(f'{MERGED}/{s}/images'); os.makedirs(f'{MERGED}/{s}/labels')

for _, r in manifest.iterrows():
    stem = os.path.splitext(r.file)[0]
    shutil.copy(f'{POOL}/images/{r.file}', f'{MERGED}/{r.split}/images/{r.file}')
    shutil.copy(f'{POOL}/labels/{stem}.txt', f'{MERGED}/{r.split}/labels/{stem}.txt')

with open(f'{MERGED}/data.yaml', 'w') as f:
    yaml.safe_dump({'path': MERGED, 'train': 'train/images', 'val': 'val/images',
                    'test': 'test/images', 'nc': len(CLASSES), 'names': CLASSES},
                   f, allow_unicode=True, sort_keys=False)
manifest.to_csv(f'{MERGED}/manifest.csv', index=False)

box = {s: Counter() for s in ['train', 'val', 'test']}
for s in box:
    for lf in glob.glob(f'{MERGED}/{s}/labels/*.txt'):
        for line in open(lf):
            if line.strip():
                box[s][CLASSES[int(line.split()[0])]] += 1
print('\nSố box theo lớp và tập:')
print(pd.DataFrame(box, index=CLASSES))

shutil.make_archive(f'{BASE}/merged/merged_v1', 'zip', MERGED)
print('\nĐã lưu', f'{BASE}/merged/merged_v1.zip')
```

<!-- Cell 22 (Markdown) -->
## Kiểm tra nhãn bằng mắt
Vẽ box và tên lớp lên 2 ảnh ngẫu nhiên của mỗi bộ trong tập train của /content/merged, để chắc việc chuyển đổi nhãn (polygon/pose → bbox) và ánh xạ lớp đúng trước khi train.

```python
# Cell 23 (Code)
import cv2, matplotlib.pyplot as plt

picks = []
for dsn in ['super', 'duytranle', 'new_chinese_chess', 'scara']:
    sub = manifest[(manifest.dataset == dsn) & (manifest.split == 'train')]
    picks += sub.sample(2, random_state=7)[['file', 'dataset']].values.tolist()

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

for ax, (f, dsn) in zip(axes.flatten(), picks):
    img = cv2.cvtColor(cv2.imread(f'{MERGED}/train/images/{f}'), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    thick, fs = max(2, W // 300), max(0.5, W / 1100)
    stem = os.path.splitext(f)[0]
    for line in open(f'{MERGED}/train/labels/{stem}.txt'):
        c, x, y, w, h = line.split()
        c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)
        p1 = (int((x - w / 2) * W), int((y - h / 2) * H))
        p2 = (int((x + w / 2) * W), int((y + h / 2) * H))
        color = (255, 60, 60) if CLASSES[c].startswith('red') else (60, 120, 255)
        cv2.rectangle(img, p1, p2, color, thick)
        cv2.putText(img, CLASSES[c].split('_')[1][:3], (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 0), max(1, thick - 1))
    ax.imshow(img); ax.set_title(dsn); ax.axis('off')
plt.tight_layout(); plt.show()
```

---

### `training\train_yolo11.ipynb`

*File Jupyter Notebook | Dung lượng: 13.46 KB*

<!-- Cell 1 (Markdown) -->
<a href="https://colab.research.google.com/github/JulianNguyen05/xiangqi-sorting-jetmax/blob/main/training/train_yolo11.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- Cell 2 (Markdown) -->
## Chuẩn bị môi trường
Kiểm tra GPU, mount Drive và cài ultralytics.

```python
# Cell 3 (Code)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ GPU - đổi runtime sang T4')

from google.colab import drive
drive.mount('/content/drive')

!pip install -q ultralytics
```

```python
# Cell 4 (Code)
import os
BASE = '/content/drive/MyDrive/datn'
for p in [
    f'{BASE}/merged/merged_v1.zip',
    f'{BASE}/runs/yolo11n_640/weights/last.pt',
    f'{BASE}/runs/yolo11n_640/args.yaml',
    f'{BASE}/runs/yolo11n_640/results.csv',
]:
    print(os.path.exists(p), p)
```

<!-- Cell 5 (Markdown) -->
## Giải nén dataset hợp nhất
Copy merged_v1.zip từ Drive về máy Colab (đọc nhanh hơn đọc trực tiếp từ Drive) và kiểm tra số ảnh/nhãn mỗi tập.

```python
# Cell 6 (Code)
import os, shutil

BASE = '/content/drive/MyDrive/datn'
MERGED = '/content/merged'

if not os.path.isdir(f'{MERGED}/train'):
    shutil.unpack_archive(f'{BASE}/merged/merged_v1.zip', MERGED)

for s in ['train', 'val', 'test']:
    print(s, len(os.listdir(f'{MERGED}/{s}/images')), 'ảnh |', len(os.listdir(f'{MERGED}/{s}/labels')), 'nhãn')

print(open(f'{MERGED}/data.yaml').read())
```

<!-- Cell 7 (Markdown) -->
## Kiểm tra nhãn trước khi huấn luyện
Đảm bảo mỗi ảnh đều có file nhãn tương ứng và mọi class id nằm trong khoảng 0-13. Nhãn sai class id sẽ làm quá trình train báo lỗi hoặc cho kết quả sai.

```python
# Cell 8 (Code)
import glob

bad = 0
for s in ['train', 'val', 'test']:
    imgs = {os.path.splitext(f)[0] for f in os.listdir(f'{MERGED}/{s}/images')}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(f'{MERGED}/{s}/labels')}
    print(s, '| ảnh thiếu nhãn:', len(imgs - lbls), '| nhãn thiếu ảnh:', len(lbls - imgs))
    for p in glob.glob(f'{MERGED}/{s}/labels/*.txt'):
        for line in open(p):
            parts = line.split()
            if parts and not (0 <= int(parts[0]) <= 13):
                bad += 1
print('Dòng nhãn có class id sai:', bad)
```

<!-- Cell 9 (Markdown) -->
## Khai báo tham số huấn luyện
Gom các siêu tham số vào một chỗ để dễ chỉnh. Kết quả được lưu thẳng vào Drive (`/datn/runs`) nên không mất khi Colab ngắt kết nối. Tên run `yolo11n_640` khớp với thư mục `runs/` trong dự án.

```python
# Cell 10 (Code)
from ultralytics import YOLO

DATA    = f'{MERGED}/data.yaml'
PROJECT = f'{BASE}/runs'
NAME    = 'yolo11n_640'

EPOCHS  = 100
IMGSZ   = 640
BATCH   = 32
```

<!-- Cell 11 (Markdown) -->
## Nạp mô hình YOLO11n pretrained
Dùng trọng số `yolo11n.pt` đã huấn luyện sẵn trên COCO (transfer learning). Bản `n` (nano) nhỏ và nhanh, phù hợp triển khai lên Jetson Nano. Trọng số sẽ tự tải về lần đầu.

```python
# Cell 12 (Code)
model = YOLO('yolo11n.pt')
```

<!-- Cell 13 (Markdown) -->
## Huấn luyện mô hình
Các tham số đáng chú ý:
- `patience=30`: dừng sớm nếu 30 epoch liền không cải thiện.
- `fliplr=0.0`: **tắt lật ngang** vì lật sẽ làm chữ Hán trên quân cờ bị đảo ngược, gây nhầm lớp.
- `degrees=10`: xoay nhẹ, vì quân cờ trên băng chuyền có thể nằm lệch góc.
- `save_period=10`: lưu checkpoint mỗi 10 epoch để dự phòng khi Colab ngắt.
- `close_mosaic=10`: tắt mosaic ở 10 epoch cuối để mô hình ổn định hơn.

Thời gian ước tính khoảng 1-2 giờ trên T4.

```python
# Cell 14 (Code)
results = model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=30,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    save_period=10,
    workers=2,
    fliplr=0.0,
    degrees=10,
    close_mosaic=10,
    plots=True,
)
```

<!-- Cell 15 (Markdown) -->
## Tiếp tục huấn luyện nếu bị ngắt giữa chừng
Chỉ chạy ô này khi Colab ngắt kết nối lúc đang train. Chạy lại phần chuẩn bị môi trường và giải nén dataset, bỏ qua bước 4, rồi chạy ô dưới để train tiếp từ checkpoint `last.pt`.

```python
# Cell 16 (Code)
# model = YOLO(f'{PROJECT}/{NAME}/weights/last.pt')
# model.train(resume=True)
```

<!-- Cell 17 (Markdown) -->
## Xem kết quả huấn luyện
Đọc `results.csv` và vẽ đường loss cùng mAP theo từng epoch. Loss giảm đều và mAP tăng rồi đi ngang là dấu hiệu huấn luyện tốt. Nếu loss val tăng trong khi loss train giảm thì mô hình đang overfit.

```python
# Cell 18 (Code)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(f'{PROJECT}/{NAME}/results.csv')
df.columns = df.columns.str.strip()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for c in ['train/box_loss', 'val/box_loss', 'train/cls_loss', 'val/cls_loss']:
    ax[0].plot(df['epoch'], df[c], label=c)
ax[0].set_title('Loss'); ax[0].set_xlabel('epoch'); ax[0].legend()

for c in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
    ax[1].plot(df['epoch'], df[c], label=c)
ax[1].set_title('mAP (val)'); ax[1].set_xlabel('epoch'); ax[1].legend()
plt.show()

print(df[['epoch', 'metrics/precision(B)', 'metrics/recall(B)',
          'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].tail(3))
```

<!-- Cell 19 (Markdown) -->
## Đánh giá trên tập test
Tập test chưa từng được dùng khi huấn luyện hay chọn mô hình, nên đây là con số khách quan để báo cáo. Mục tiêu trong đề cương là **mAP@0.5 ≥ 95%**. Bảng bên dưới cho biết từng lớp để phát hiện lớp nào còn yếu.

```python
# Cell 20 (Code)
best = YOLO(f'{PROJECT}/{NAME}/weights/best.pt')
metrics = best.val(data=DATA, split='test', imgsz=IMGSZ, project=PROJECT, name=f'{NAME}_test', exist_ok=True)

print(f'mAP@0.5     : {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Precision   : {metrics.box.mp:.4f} | Recall: {metrics.box.mr:.4f}')

print('\nmAP@0.5 theo lớp:')
for i, name in best.names.items():
    print(f'  {name:15s} {metrics.box.ap50[i]:.4f}')
```

<!-- Cell 21 (Markdown) -->
## Dự đoán thử trên ảnh test
Chọn ngẫu nhiên vài ảnh từ tập test để xem mô hình khoanh vùng và gán nhãn có hợp lý không. Bước này hữu ích để đưa vào báo cáo.

```python
# Cell 22 (Code)
import random
from IPython.display import Image, display

samples = random.sample(glob.glob(f'{MERGED}/test/images/*'), 4)
preds = best.predict(samples, imgsz=IMGSZ, conf=0.5, save=True,
                     project=PROJECT, name=f'{NAME}_pred', exist_ok=True)

for r in preds:
    display(Image(filename=os.path.join(r.save_dir, os.path.basename(r.path))))
```

<!-- Cell 23 (Markdown) -->
## Xuất mô hình sang ONNX
`best.pt` dùng được ngay với Ultralytics, còn ONNX là định dạng trung gian để chuyển sang TensorRT. **Lưu ý:** file TensorRT (`.engine`) phải được tạo trực tiếp trên Jetson, không tạo trên Colab được, vì nó phụ thuộc vào GPU và phiên bản TensorRT của thiết bị.

```python
# Cell 24 (Code)
onnx_path = best.export(format='onnx', imgsz=IMGSZ, simplify=True)
print('Đã xuất:', onnx_path)
print('Các file trong thư mục weights:', os.listdir(f'{PROJECT}/{NAME}/weights'))
```

<!-- Cell 25 (Markdown) -->
## Tải kết quả về máy
Nén thư mục kết quả và tải về, giải nén vào `runs/yolo11n_640/` trong dự án. Bản gốc vẫn còn trên Drive.

```python
# Cell 26 (Code)
from google.colab import files

shutil.make_archive('/content/yolo11n_640', 'zip', f'{PROJECT}/{NAME}')
files.download('/content/yolo11n_640.zip')
```

---

## Nhóm 5: Cấu hình danh sách Class & Dataset Config

> ⚠️ Không tìm thấy file nào thuộc nhóm này trong dự án.

## Nhóm 6: Reference Scripts & API mẫu

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

## Nhóm 7: Tổng quan dự án

### `README.md`

*Dung lượng: 153 B | Số dòng: 2*

```markdown
# xiangqi-sorting-jetmax
Real-time Xiangqi piece detection (YOLO11 + OpenCV) and sorting on a moving conveyor belt with a Hiwonder JetMax robotic arm.
```

---

