# -*- coding: utf-8 -*-
"""Kiểm tra cơ sở dữ liệu định dạng YOLO: đếm ảnh, nhãn, số quân theo lớp và phát hiện nhãn lỗi.

    python scripts/check_dataset.py                 # kiểm tra thư mục Nhom2_YeuCau1 bên cạnh
    python scripts/check_dataset.py duong/dan/dataset

Chỉ dùng thư viện chuẩn của Python, không cần cài thêm gì.
"""
import sys
from collections import Counter
from pathlib import Path

NAMES = ["black_advisor", "black_bishop", "black_cannon", "black_king", "black_knight", "black_pawn", "black_rook",
         "red_advisor", "red_bishop", "red_cannon", "red_king", "red_knight", "red_pawn", "red_rook"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check(root: Path):
    total = Counter(); errors = []; ok = True
    print(f"Dataset: {root}\n")
    print(f"{'tập':<7}{'ảnh':>8}{'tệp nhãn':>10}{'quân':>9}{'ảnh thiếu nhãn':>16}")
    for split in ("train", "valid", "test"):
        img_dir, lab_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir():
            print(f"{split:<7}  (không có thư mục {img_dir})"); ok = False; continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        labs = {p.stem: p for p in lab_dir.glob("*.txt")} if lab_dir.is_dir() else {}
        boxes = 0; no_label = 0
        for im in imgs:
            lab = labs.get(im.stem)
            if lab is None:
                no_label += 1; continue
            for ln, line in enumerate(lab.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts: continue
                try:
                    c = int(parts[0]); x, y, w, h = map(float, parts[1:5])
                except (ValueError, IndexError):
                    errors.append(f"{lab.name}:{ln} sai định dạng"); continue
                if len(parts) != 5: errors.append(f"{lab.name}:{ln} có {len(parts)} cột (cần 5)")
                if not 0 <= c < len(NAMES): errors.append(f"{lab.name}:{ln} lớp {c} ngoài 0-13")
                if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0: errors.append(f"{lab.name}:{ln} toạ độ ngoài [0, 1]")
                total[c] += 1; boxes += 1
        print(f"{split:<7}{len(imgs):>8}{len(labs):>10}{boxes:>9}{no_label:>16}")
    print("\nSố quân theo lớp:")
    for i, n in enumerate(NAMES):
        print(f"  {i:>2} {n:<15}{total[i]:>8}")
    print(f"  {'tổng':<18}{sum(total.values()):>8}")
    if errors:
        print(f"\n{len(errors)} dòng nhãn lỗi (in 20 dòng đầu):")
        for e in errors[:20]: print("  -", e)
    else:
        print("\nKhông có dòng nhãn lỗi.")
    return ok and not errors


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "Nhom2_YeuCau1"
    sys.exit(0 if check(root) else 1)
