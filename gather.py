import json
import os
from pathlib import Path

# Thư mục gốc của dự án (thư mục chứa file gather.py)
PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_FILE = PROJECT_ROOT / "gathered_code.md"

# Định nghĩa danh sách các nhóm file cần thu thập
GROUPS = [
    {
        "name": "Nhóm 1: Core Robot (Board, Detect View, Motion)",
        "patterns": [
            "robot/board.py",
            "robot/detect_view.py",
            "robot/motion.py",
        ],
    },
    {
        "name": "Nhóm 2: Cấu hình hệ thống (Configs)",
        "patterns": [
            "configs/board.yaml",
        ],
    },
    {
        "name": "Nhóm 3: Test Scripts & Kịch bản thử nghiệm",
        "patterns": [
            "tests/move_piece.py",
            "tests/pick_place.py",
            "tests/scan_detect.py",
        ],
    },
    {
        "name": "Nhóm 4: Huấn luyện, Cấu trúc Class & Dataset",
        "patterns": [
            "training/README.md",
            "training/build_dataset.ipynb",
            "training/train_yolo11.ipynb",
        ],
    },
    {
        "name": "Nhóm 5: Cấu hình danh sách Class & Dataset Config",
        "patterns": [
            "**/data.yaml",
            "data.yaml",
        ],
    },
    {
        "name": "Nhóm 6: Reference Scripts & API mẫu",
        "patterns": [
            "docs/robot-reference/waste_classification_stepper.py",
            "docs/robot-reference/jetmax_control/scripts/jetmax_control_main.py",
        ],
    },
    {
        "name": "Nhóm 7: Tổng quan dự án",
        "patterns": [
            "README.md",
        ],
    },
]

# Các định dạng file trọng số mô hình (chỉ lấy thông tin tên và kích thước)
WEIGHTS_EXTENSIONS = {".pt", ".onnx", ".engine", ".pth", ".weights", ".tflite", ".bin", ".pkl"}


def format_size(size_bytes: int) -> str:
    """Định dạng dung lượng file sang B, KB, MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def resolve_files(pattern: str) -> list[Path]:
    """Tìm danh sách file phù hợp dựa trên đường dẫn tuyệt đối, tương đối hoặc wildcard pattern."""
    found = []
    if "*" in pattern or "?" in pattern:
        for p in PROJECT_ROOT.glob(pattern):
            if p.is_file() and p.name != "gathered_code.md":
                found.append(p)
    else:
        direct_path = PROJECT_ROOT / pattern
        if direct_path.exists() and direct_path.is_file():
            found.append(direct_path)
        else:
            filename = Path(pattern).name
            matches = [
                p for p in PROJECT_ROOT.rglob(filename) 
                if p.is_file() and p.name != "gathered_code.md"
            ]
            if matches:
                found.append(matches[0])
    return found


def get_language_tag(filepath: Path) -> str:
    """Xác định ngôn ngữ để highlight code trong Markdown."""
    ext = filepath.suffix.lower()
    mapping = {
        ".py": "python",
        ".msg": "text",
        ".srv": "text",
        ".action": "text",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".csv": "csv",
        ".ipynb": "python",
    }
    return mapping.get(ext, "")


def render_notebook(filepath: Path) -> str:
    """Trích xuất cell code và markdown từ Jupyter Notebook để hiển thị rõ ràng."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = json.load(f)
        output = []
        for i, cell in enumerate(nb.get("cells", []), 1):
            cell_type = cell.get("cell_type", "code")
            source = "".join(cell.get("source", []))
            if cell_type == "markdown":
                output.append(f"<!-- Cell {i} (Markdown) -->\n{source}")
            elif cell_type == "code":
                output.append(f"```python\n# Cell {i} (Code)\n{source}\n```")
        return "\n\n".join(output)
    except Exception:
        return filepath.read_text(encoding="utf-8", errors="ignore")


def main():
    seen_paths = set()
    grouped_results = []
    total_found = 0

    for group in GROUPS:
        group_files = []
        for pattern in group["patterns"]:
            matches = resolve_files(pattern)
            for file_path in matches:
                if file_path not in seen_paths:
                    seen_paths.add(file_path)
                    group_files.append(file_path)
        grouped_results.append((group["name"], group_files))
        total_found += len(group_files)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# GATHERED SOURCE CODE - XIANGQI SORTING JETMAX\n\n")
        out.write(f"- **Thư mục gốc:** `{PROJECT_ROOT}`\n")
        out.write(f"- **Tổng số file đã gôm:** {total_found}\n\n")

        out.write("## Mục lục\n\n")
        for group_name, files in grouped_results:
            out.write(f"### {group_name}\n")
            if not files:
                out.write("  *(Không tìm thấy file nào)*\n")
            for file_path in files:
                rel_path = file_path.relative_to(PROJECT_ROOT)
                anchor = str(rel_path).replace("\\", "/").replace("/", "").replace(".", "").lower()
                out.write(f"- [{rel_path}](#{anchor})\n")
            out.write("\n")

        out.write("---\n\n")

        # Ghi nội dung từng nhóm file
        for group_name, files in grouped_results:
            out.write(f"## {group_name}\n\n")
            if not files:
                out.write("> ⚠️ Không tìm thấy file nào thuộc nhóm này trong dự án.\n\n")
                continue

            for file_path in files:
                rel_path = file_path.relative_to(PROJECT_ROOT)
                ext = file_path.suffix.lower()
                size_str = format_size(file_path.stat().st_size)

                out.write(f"### `{rel_path}`\n\n")

                if ext in WEIGHTS_EXTENSIONS:
                    out.write(f"- **Loại file:** Model Weights ({ext})\n")
                    out.write(f"- **Kích thước file:** {size_str}\n")
                    out.write("> ℹ️ *Nội dung file nhị phân được bỏ qua.* \n\n")
                elif ext == ".ipynb":
                    out.write(f"*File Jupyter Notebook | Dung lượng: {size_str}*\n\n")
                    nb_content = render_notebook(file_path)
                    out.write(nb_content)
                    out.write("\n\n")
                else:
                    lang = get_language_tag(file_path)
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        line_count = len(content.splitlines())
                        out.write(f"*Dung lượng: {size_str} | Số dòng: {line_count}*\n\n")
                        out.write(f"```{lang}\n")
                        out.write(content)
                        if not content.endswith("\n"):
                            out.write("\n")
                        out.write("```\n\n")
                    except Exception as e:
                        out.write(f"> ❌ Không thể đọc file: {e}\n\n")

                out.write("---\n\n")

    print(f"✅ Đã cập nhật xong `gathered_code.md` với tổng cộng {total_found} file.")


if __name__ == "__main__":
    main()