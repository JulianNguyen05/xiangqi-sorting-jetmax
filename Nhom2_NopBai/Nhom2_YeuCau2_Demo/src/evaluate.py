# -*- coding: utf-8 -*-
"""Đánh giá mô hình trên tập test và xuất ONNX.

    python src/evaluate.py                      # mAP, Precision, Recall trên tập test
    python src/evaluate.py --split val
    python src/evaluate.py --export             # xuất thêm models/best.onnx (opset 12, cho TensorRT 8 trên Jetson Nano)

Kết quả tham chiếu trong báo cáo (713 ảnh test, 11 950 quân): mAP@0.5 = 0.964, mAP@0.5:0.95 = 0.815, P = 0.990, R = 0.959.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent / "Nhom2_YeuCau1" / "Nhom2_YeuCau1" / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(HERE / "models" / "best.pt"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--device", default=None, help="0 = GPU, cpu = CPU; mặc định tự chọn")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    m = model.val(data=a.data, split=a.split, imgsz=640, batch=32, device=a.device, plots=True,
                  project=str(HERE / "runs" / "detect"), name=f"eval_{a.split}", exist_ok=True)
    print(f"\n===== KẾT QUẢ TRÊN TẬP {a.split.upper()} =====")
    print(f"mAP@0.5      = {m.box.map50:.3f}")
    print(f"mAP@0.5:0.95 = {m.box.map:.3f}")
    print(f"Precision    = {m.box.mp:.3f}")
    print(f"Recall       = {m.box.mr:.3f}")
    print("\nmAP@0.5 theo lớp:")
    for i, c in enumerate(m.box.ap_class_index):
        print(f"  {model.names[int(c)]:<15}{m.box.ap50[i]:.3f}")
    if a.export:
        out = model.export(format="onnx", opset=12, simplify=True, imgsz=640)
        print("Đã xuất:", out)


if __name__ == "__main__":
    main()
