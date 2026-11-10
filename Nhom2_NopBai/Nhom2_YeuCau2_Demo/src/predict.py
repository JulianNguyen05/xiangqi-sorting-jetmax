# -*- coding: utf-8 -*-
"""Nhận dạng quân cờ trên một ảnh hoặc một thư mục ảnh và lưu ảnh đã vẽ khung.

    python src/predict.py xiangqi/samples/start_position.jpg
    python src/predict.py thu_muc_anh --conf 0.5
    python src/predict.py anh.jpg --weights models/best.onnx     # chạy bằng ONNX Runtime, không cần GPU

Ảnh kết quả nằm trong runs/detect/predict/.
"""
import argparse
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="ảnh hoặc thư mục ảnh")
    ap.add_argument("--weights", default=str(HERE / "models" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights, task="detect")
    results = model.predict(a.source, imgsz=640, conf=a.conf, save=True, project=str(HERE / "runs" / "detect"),
                            name="predict", exist_ok=True, verbose=False)
    for r in results:
        names = Counter(r.names[int(c)] for c in r.boxes.cls.tolist())
        print(f"{Path(r.path).name}: {len(r.boxes)} quân")
        for n, k in sorted(names.items()):
            print(f"    {n:<15}{k}")
    print("\nẢnh đã vẽ khung:", HERE / "runs" / "detect" / "predict")


if __name__ == "__main__":
    main()
