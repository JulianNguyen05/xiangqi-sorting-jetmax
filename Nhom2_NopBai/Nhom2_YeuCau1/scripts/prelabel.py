# -*- coding: utf-8 -*-
"""Gán nhãn sơ bộ cho ảnh mới bằng mô hình đã huấn luyện, để sau đó rà soát và sửa bằng LabelImg / Roboflow.

    pip install ultralytics
    python scripts/prelabel.py thu_muc_anh_moi --weights ../Nhom2_YeuCau2_Demo/models/best.pt

Kết quả: thu_muc_anh_moi/labels/<tên ảnh>.txt (định dạng YOLO) và classes.txt (đúng thứ tự 14 lớp, LabelImg cần tệp này).
Nhãn do mô hình sinh ra CHƯA phải nhãn đúng: phải mở từng ảnh bằng công cụ gán nhãn để sửa box sai, thêm quân bị sót.
"""
import argparse
from pathlib import Path

NAMES = ["black_advisor", "black_bishop", "black_cannon", "black_king", "black_knight", "black_pawn", "black_rook",
         "red_advisor", "red_bishop", "red_cannon", "red_king", "red_knight", "red_pawn", "red_rook"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="thư mục chứa ảnh cần gán nhãn")
    ap.add_argument("--weights", default="../Nhom2_YeuCau2_Demo/models/best.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng tin cậy; hạ thấp để ít sót quân hơn")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    from ultralytics import YOLO
    img_dir = Path(a.images)
    out = img_dir / "labels"; out.mkdir(exist_ok=True)
    model = YOLO(a.weights)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"Không thấy ảnh trong {img_dir}")
    n_box = 0
    for im in imgs:
        r = model.predict(str(im), imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        (out / (im.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_box += len(lines)
        print(f"{im.name}: {len(lines)} quân")
    for d in (out, img_dir):
        (d / "classes.txt").write_text("\n".join(NAMES) + "\n", encoding="utf-8")
    print(f"\nĐã gán nhãn sơ bộ {len(imgs)} ảnh, {n_box} quân. Nhãn nằm ở {out}")
    print("Bước tiếp theo: mở LabelImg, chọn định dạng YOLO, mở thư mục ảnh và thư mục labels này để rà soát.")


if __name__ == "__main__":
    main()
