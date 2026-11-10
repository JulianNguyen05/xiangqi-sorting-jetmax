# -*- coding: utf-8 -*-
"""Huấn luyện YOLO11n nhận dạng 14 loại quân cờ tướng theo hai giai đoạn (giống train_yolo11.ipynb).

    python src/train_yolo11.py                 # chạy cả hai giai đoạn
    python src/train_yolo11.py --stage 2       # chỉ tinh chỉnh từ kết quả giai đoạn 1
    python src/train_yolo11.py --epochs 3      # chạy thử nhanh để kiểm tra môi trường

Giai đoạn 1: từ trọng số COCO (yolo11n.pt), 100 epoch, xoay nhẹ +-10 độ.
Giai đoạn 2: từ best.pt của giai đoạn 1, 60 epoch, xoay +-180 độ, AdamW lr0 = 0.001.
Không lật ảnh ở cả hai giai đoạn vì lật làm chữ Hán thành ảnh gương.
Cần GPU; trên Tesla T4 giai đoạn 2 mất khoảng 1,7 giờ.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE.parent / "Nhom2_YeuCau1" / "Nhom2_YeuCau1" / "data.yaml"
RUNS = HERE / "runs" / "detect"
COMMON = dict(imgsz=640, batch=32, patience=20, close_mosaic=10, save_period=10, fliplr=0.0, flipud=0.0,
              hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, mosaic=1.0, seed=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--stage", type=int, choices=[1, 2], help="mặc định chạy cả hai")
    ap.add_argument("--epochs", type=int, help="ghi đè số epoch (để chạy thử)")
    ap.add_argument("--device", default="0", help="0 = GPU đầu tiên, cpu = chạy bằng CPU (rất chậm)")
    a = ap.parse_args()

    from ultralytics import YOLO
    stage1_best = RUNS / "yolo11n_640" / "weights" / "best.pt"
    if a.stage in (None, 1):
        print("===== GIAI ĐOẠN 1: yolo11n.pt (COCO) -> quân cờ, xoay +-10 độ =====")
        YOLO("yolo11n.pt").train(data=a.data, epochs=a.epochs or 100, degrees=10, device=a.device,
                                 project=str(RUNS), name="yolo11n_640", exist_ok=True, **COMMON)
    if a.stage in (None, 2):
        if not stage1_best.exists():
            raise SystemExit(f"Chưa có {stage1_best}. Chạy giai đoạn 1 trước, hoặc chép best.pt của giai đoạn 1 vào đó.")
        print("===== GIAI ĐOẠN 2: tinh chỉnh, xoay +-180 độ =====")
        YOLO(str(stage1_best)).train(data=a.data, epochs=a.epochs or 60, degrees=180, optimizer="AdamW", lr0=0.001,
                                     device=a.device, project=str(RUNS), name="yolo11n_640_rot", exist_ok=True, **COMMON)
        print("Mô hình tốt nhất:", RUNS / "yolo11n_640_rot" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
