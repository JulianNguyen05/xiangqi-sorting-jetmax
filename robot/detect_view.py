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
