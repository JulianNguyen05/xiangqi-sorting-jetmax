import json
import os
from pathlib import Path

# Thư mục gốc của dự án (thư mục chứa file gather.py)
PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_FILE = PROJECT_ROOT / "gathered_code.md"

# Cấu hình 3 nhóm file cần thu thập
GROUPS = [
    {
        "name": "Nhóm 1: API Robot (Quan trọng nhất)",
        "patterns": [
            "docs/robot-reference/jetmax_control/scripts/jetmax_control_main.py",
            "docs/robot-reference/jetmax_control/msg/JetMax.msg",
            "docs/robot-reference/jetmax_control/msg/SetJetMax.msg",
            "docs/robot-reference/jetmax_control/msg/SetJoint.msg",
            "docs/robot-reference/jetmax_control/msg/Sucker.msg",
            "docs/robot-reference/jetmax_control/msg/SuckerState.msg",
            "docs/robot-reference/waste_classification_stepper.py",
        ],
    },
    {
        "name": "Nhóm 2: Code cũ có thể tái sử dụng",
        "patterns": [
            "robot/motion.py",
            "robot/aruco_locator.py",
            "robot/rig.py",
            "tests/pick_place.py",
            "tests/joint_teleop.py",
        ],
    },
    {
        "name": "Nhóm 3: Thông tin Model & Training",
        "patterns": [
            "training/README.md",
        ],
    },
]

# Đuôi file binary/model cần bỏ qua nội dung
WEIGHTS_EXTENSIONS = {
    ".pt",
    ".onnx",
    ".engine",
    ".pth",
    ".weights",
    ".tflite",
    ".bin",
    ".pkl",
    ".nnue",
    ".pdf",
    ".docx",
}


def format_size(size_bytes: int) -> str:
    """Định dạng dung lượng file sang B, KB, MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def resolve_file(rel_path_str: str) -> Path | None:
    """Xử lý đường dẫn linh hoạt trên cả Windows và Linux/macOS."""
    file_path = PROJECT_ROOT / rel_path_str
    if file_path.exists() and file_path.is_file():
        return file_path

    alt_path = PROJECT_ROOT / rel_path_str.replace("/", os.sep)
    if alt_path.exists() and alt_path.is_file():
        return alt_path

    return None


def get_language_tag(filepath: Path) -> str:
    """Gán ngôn ngữ syntax highlighting cho khối Markdown code block."""
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
    """Trích xuất cell code và markdown nếu có file Jupyter Notebook."""
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
    except Exception as e:
        return f"> ❌ Lỗi đọc notebook: {e}\n"


def main():
    seen_paths = set()
    grouped_results = []
    total_found = 0

    for group in GROUPS:
        group_files = []
        for rel_path_str in group["patterns"]:
            file_path = resolve_file(rel_path_str)
            if file_path and file_path not in seen_paths:
                seen_paths.add(file_path)
                group_files.append(file_path)
            elif not file_path:
                print(f"⚠️  Không tìm thấy file: {rel_path_str}")

        grouped_results.append((group["name"], group_files))
        total_found += len(group_files)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# GATHERED SOURCE CODE - XIANGQI SORTING JETMAX\n\n")
        out.write(f"- **Thư mục gốc:** `{PROJECT_ROOT}`\n")
        out.write(f"- **Tổng số file đã gôm:** {total_found}\n\n")

        # Tạo mục lục
        out.write("## Mục lục\n\n")
        for group_name, files in grouped_results:
            out.write(f"### {group_name}\n")
            if not files:
                out.write("  *(Không tìm thấy file nào)*\n")
            for file_path in files:
                rel_path = file_path.relative_to(PROJECT_ROOT)
                anchor = (
                    str(rel_path)
                    .replace("\\", "/")
                    .replace("/", "")
                    .replace(".", "")
                    .replace("_", "")
                    .lower()
                )
                out.write(f"- [{rel_path}](#{anchor})\n")
            out.write("\n")

        out.write("---\n\n")

        # Ghi nội dung chi tiết từng file theo nhóm
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
                    out.write(f"- **Loại file:** Binary / Model / Unsupported ({ext})\n")
                    out.write(f"- **Kích thước file:** {size_str}\n")
                    out.write("> ℹ️ *Nội dung nhị phân/tài liệu được bỏ qua.* \n\n")
                elif ext == ".ipynb":
                    out.write(f"*File Jupyter Notebook | Dung lượng: {size_str}*\n\n")
                    nb_content = render_notebook(file_path)
                    out.write(nb_content)
                    out.write("\n\n")
                else:
                    lang = get_language_tag(file_path)
                    try:
                        content = file_path.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                        line_count = len(content.splitlines())
                        out.write(
                            f"*Dung lượng: {size_str} | Số dòng: {line_count}*\n\n"
                        )
                        out.write(f"```{lang}\n")
                        out.write(content)
                        if not content.endswith("\n"):
                            out.write("\n")
                        out.write("```\n\n")
                    except Exception as e:
                        out.write(f"> ❌ Không thể đọc file: {e}\n\n")

                out.write("---\n\n")

    print(f"✅ Đã gôm xong {total_found} file vào `gathered_code.md`.")


if __name__ == "__main__":
    main()