import os
from pathlib import Path

# Thư mục gốc của dự án (thư mục chứa file gather.py này)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Danh sách các đường dẫn/tên file cần gôm
TARGET_PATTERNS = [
    # 1. Các file lõi điều khiển
    "jetmax.py",
    "stepper.py",
    "pwm_servo.py",
    # 2. File tham khảo chính hãng
    "docs/robot-reference/waste_classification_stepper.py",
    # 3. File ROS control
    "docs/robot-reference/jetmax_control/scripts/jetmax_control_main.py",
    "SetJetMax.msg",
    "JetMax.msg",
    "SetServo.msg",
    # 4. Các file test hiện có
    "tests/test_move.py",
    "tests/capture_images.py",
    "tests/stepper_teleop.py",
]

OUTPUT_FILE = PROJECT_ROOT / "gathered_code.md"


def find_file(pattern: str) -> Path | None:
    """Tìm file dựa trên đường dẫn tương đối hoặc tìm kiếm toàn dự án theo tên file."""
    # 1. Kiểm tra đường dẫn trực tiếp
    direct_path = PROJECT_ROOT / pattern
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    # 2. Nếu không thấy trực tiếp, tìm kiếm theo tên file trong toàn dự án
    filename = Path(pattern).name
    matches = list(PROJECT_ROOT.rglob(filename))
    if matches:
        return matches[0]

    return None


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
    }
    return mapping.get(ext, "")


def main():
    collected_files = []
    missing_files = []

    for pattern in TARGET_PATTERNS:
        found_path = find_file(pattern)
        if found_path:
            # Lưu đường dẫn tương đối tính từ gốc dự án
            rel_path = found_path.relative_to(PROJECT_ROOT)
            collected_files.append((rel_path, found_path))
        else:
            missing_files.append(pattern)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# GATHERED SOURCE CODE\n\n")
        out.write(f"- **Thư mục gốc:** `{PROJECT_ROOT}`\n")
        out.write(f"- **Tổng số file gôm được:** {len(collected_files)}/{len(TARGET_PATTERNS)}\n\n")

        if missing_files:
            out.write("### ⚠️ Các file chưa tìm thấy:\n")
            for m in missing_files:
                out.write(f"- `{m}`\n")
            out.write("\n")

        out.write("## Mục lục\n\n")
        for rel_path, _ in collected_files:
            anchor = str(rel_path).replace("\\", "/").replace("/", "").replace(".", "").lower()
            out.write(f"- [{rel_path}](#{anchor})\n")
        out.write("\n---\n\n")

        # Đọc và ghi nội dung từng file
        for rel_path, full_path in collected_files:
            lang = get_language_tag(full_path)
            out.write(f"### `{rel_path}`\n\n")

            try:
                content = full_path.read_text(encoding="utf-8")
                line_count = len(content.splitlines())
                out.write(f"*Số dòng: {line_count}*\n\n")
                out.write(f"```{lang}\n")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
                out.write("```\n\n")
            except Exception as e:
                out.write(f"> ❌ Không thể đọc file: {e}\n\n")

            out.write("---\n\n")

    print(f"✅ Đã gôm thành công {len(collected_files)} file vào `{OUTPUT_FILE.name}`")
    if missing_files:
        print(f"⚠️ Không tìm thấy {len(missing_files)} file:")
        for m in missing_files:
            print(f"   - {m}")


if __name__ == "__main__":
    main()