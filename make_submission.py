# -*- coding: utf-8 -*-
"""Tạo thư mục nộp bài Nhom2_NopBai/ từ repo xiangqi-sorting-jetmax.

Đặt tệp này (và thư mục Nhom2_NopBai/ đi kèm) ở THƯ MỤC GỐC của repo rồi chạy:

    python make_submission.py
    python make_submission.py --dataset D:\\tai-ve\\merged_v1.zip --runs D:\\tai-ve\\runs --zip

Tham số (đều không bắt buộc):
    --dataset  merged_v1.zip hoặc thư mục merged đã giải nén (tải từ Drive MyDrive/datn/)
    --runs     thư mục runs/ tải từ Drive (chứa yolo11n_640, yolo11n_640_rot)
    --weights  đường dẫn best.pt nếu không nằm trong --runs hay models/
    --zip      nén Nhom2_NopBai thành Nhom2_NopBai.zip sau khi chép

Tập lệnh chỉ CHÉP, không xoá hay sửa gì trong repo. Chạy lại nhiều lần được.
"""
import argparse, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Nhom2_NopBai"
YC1 = OUT / "Nhom2_YeuCau1"
YC2 = OUT / "Nhom2_YeuCau2_Demo"
NAMES = ["black_advisor", "black_bishop", "black_cannon", "black_king", "black_knight", "black_pawn", "black_rook",
         "red_advisor", "red_bishop", "red_cannon", "red_king", "red_knight", "red_pawn", "red_rook"]
# Thư viện + giao diện demo: giữ nguyên bố cục gốc để run_demo.bat và các lệnh trong DEMO_WINDOWS.md chạy được
LIB_DIRS = ["board", "brain", "config", "vision", "planner", "robot", "tests", "tools", "samples", "models"]
LIB_FILES = ["main.py", "run_demo.bat", "install.sh", "API.md", "DEMO_WINDOWS.md", "RUNBOOK.md", "README.md", "requirements.txt", "requirements-demo.txt"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".ipynb_checkpoints", "*.engine", ".deps_ok")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
done, missing = [], []


def copy(src: Path, dst: Path, note=""):
    if not src.exists():
        missing.append(f"{src.relative_to(ROOT) if src.is_relative_to(ROOT) else src}  {note}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    done.append(str(dst.relative_to(OUT)))
    return True


def fix_bat(bat: Path):
    """Thêm một dòng vào run_demo.bat (bản trong thư mục nộp) để bỏ qua biến SSL_CERT_FILE trỏ tới tệp không tồn tại."""
    if not bat.exists():
        return
    data = bat.read_bytes()
    if b"SSL_CERT_FILE" in data:
        return
    line = b'if defined SSL_CERT_FILE if not exist "%SSL_CERT_FILE%" set "SSL_CERT_FILE="\r\n'
    first, sep, rest = data.partition(b"\n")
    if first.strip().lower().startswith(b"@echo off"):
        data = first + sep + line + rest
    else:
        data = line + data
    bat.write_bytes(data)
    done.append("Nhom2_YeuCau2_Demo/xiangqi/run_demo.bat  (đã thêm dòng bỏ qua SSL_CERT_FILE hỏng)")


def count_images(d: Path):
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMG_EXT) if d.exists() else 0


def put_dataset(src):
    """Chép dataset về Nhom2_YeuCau1/Nhom2_YeuCau1/{train,valid,test}/{images,labels} và viết lại data.yaml."""
    dst = YC1 / "Nhom2_YeuCau1"
    if src is None:
        missing.append("dataset merged_v1  (chạy lại với --dataset <merged_v1.zip hoặc thư mục merged>)")
        return
    src = Path(src)
    if not src.exists():
        missing.append(f"{src}  (đường dẫn --dataset không tồn tại)")
        return
    tmp = None
    if src.is_file():
        tmp = OUT / "_tmp_dataset"
        shutil.rmtree(tmp, ignore_errors=True)
        print("  giải nén", src.name, "...")
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    # tìm thư mục chứa train/
    base = next((p.parent for p in [src / "train"] + sorted(src.rglob("train")) if p.is_dir() and (p / "images").is_dir()), None)
    if base is None:
        missing.append(f"{src}  (không thấy train/images bên trong)")
        return
    for split_src, split_dst in [("train", "train"), ("valid", "valid"), ("val", "valid"), ("test", "test")]:
        s = base / split_src
        if s.is_dir():
            print(f"  chép {split_src} -> {split_dst} ...")
            shutil.copytree(s, dst / split_dst, dirs_exist_ok=True)
    lines = ["# Nhom2 - Co so du lieu quan co tuong (merged_v1), dinh dang YOLO",
             "train: train/images", "val: valid/images", "test: test/images", "", f"nc: {len(NAMES)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (dst / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    n = {s: count_images(dst / s / "images") for s in ("train", "valid", "test")}
    done.append(f"Nhom2_YeuCau1/Nhom2_YeuCau1  (train {n['train']} / valid {n['valid']} / test {n['test']} ảnh)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset"); ap.add_argument("--runs"); ap.add_argument("--weights"); ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    if not (ROOT / "vision").is_dir() or not (ROOT / "training").is_dir():
        sys.exit("Hãy đặt make_submission.py ở thư mục gốc của repo (nơi có vision/, brain/, training/).")
    OUT.mkdir(exist_ok=True)

    print("[1/3] Yêu cầu 1 - cơ sở dữ liệu")
    copy(ROOT / "training" / "build_dataset.ipynb", YC1 / "build_dataset.ipynb")
    put_dataset(a.dataset)

    print("[2/3] Yêu cầu 2 - mô hình và demo")
    lib = YC2 / "xiangqi"
    for d in LIB_DIRS:
        copy(ROOT / d, lib / d)
    for f in LIB_FILES:
        if (ROOT / f).exists():
            copy(ROOT / f, lib / f)
    # engine Pikafish bản Windows (do package.ipynb đóng gói vào dist/)
    eng = next((d for d in sorted((ROOT / "dist").glob("xiangqi_demo_windows*/xiangqi/engine")) if (d / "pikafish.exe").exists()), None) if (ROOT / "dist").is_dir() else None
    if eng: copy(eng, lib / "engine")
    elif not (lib / "engine" / "pikafish.exe").exists(): missing.append("engine/pikafish.exe  (demo sẽ dùng engine Python yếu hơn; lấy từ xiangqi_demo_windows.zip)")
    fix_bat(lib / "run_demo.bat")
    for nb in sorted((ROOT / "training").glob("*.ipynb")):
        if nb.name != "build_dataset.ipynb":
            copy(nb, YC2 / "notebooks" / nb.name)
    copy(ROOT / "training" / "README.md", YC2 / "notebooks" / "README.md")
    copy(ROOT / "training" / "images", YC2 / "notebooks" / "images")
    # trọng số: ưu tiên --weights, rồi --runs, rồi models/ của repo
    runs = Path(a.runs) if a.runs else None
    if runs and runs.exists():
        for run in ("yolo11n_640", "yolo11n_640_rot"):
            r = next((p for p in [runs / run] + sorted(runs.rglob(run)) if p.is_dir()), None)
            if r is None:
                missing.append(f"runs/{run}  (không thấy trong {runs})"); continue
            for item in ("args.yaml", "results.csv", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "weights"):
                if (r / item).exists():
                    copy(r / item, YC2 / "runs" / "detect" / run / item)
    else:
        missing.append("runs/ (log huấn luyện, results.csv)  (chạy lại với --runs <thư mục runs tải từ Drive>)")
    cands = [Path(a.weights)] if a.weights else []
    cands += [YC2 / "runs" / "detect" / "yolo11n_640_rot" / "weights" / "best.pt", ROOT / "models" / "best.pt"]
    pt = next((c for c in cands if c.exists()), None)
    if pt: copy(pt, YC2 / "models" / "best.pt")
    else: missing.append("best.pt  (chạy lại với --weights <best.pt> hoặc --runs)")
    onnx = next((c for c in [ROOT / "models" / "best.onnx", (pt.parent / "best.onnx") if pt else ROOT / "_"] if c.exists()), None)
    if onnx: copy(onnx, YC2 / "models" / "best.onnx")
    else: missing.append("models/best.onnx  (giao diện demo cần tệp này; tải từ Drive MyDrive/datn/)")
    if not any((lib / "models").glob("*.onnx")) and onnx:
        copy(onnx, lib / "models" / "best.onnx")

    print("[3/3] Báo cáo")
    if not (OUT / "Nhom2_BaoCaoCuoiKi.docx").exists():
        missing.append("Nhom2_NopBai/Nhom2_BaoCaoCuoiKi.docx  (chép báo cáo vào đây)")

    print("\nĐã chép %d mục vào %s" % (len(done), OUT))
    for d in done: print("  +", d)
    if missing:
        print("\nCÒN THIẾU %d mục:" % len(missing))
        for m in missing: print("  -", m)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print("\nDung lượng thư mục nộp: %.0f MB" % size)
    if a.zip:
        print("Đang nén ...")
        shutil.make_archive(str(ROOT / "Nhom2_NopBai"), "zip", ROOT, "Nhom2_NopBai")
        print("Đã tạo", ROOT / "Nhom2_NopBai.zip")


if __name__ == "__main__":
    main()
