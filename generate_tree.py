import fnmatch
import os
import sys
from datetime import datetime
from pathlib import Path

# Cấu hình Emoji cho từng loại file & thư mục
FOLDER_EMOJI = "📁"
DEFAULT_FILE_EMOJI = "📄"

EMOJI_MAP = {
    ".py": "🐍",
    ".ipynb": "📄",
    ".md": "📝",
    ".docx": "📘",
    ".doc": "📘",
    ".pdf": "📕",
    ".txt": "📝",
    ".rar": "📦",
    ".zip": "📦",
    ".7z": "📦",
    ".yaml": "⚙️",
    ".yml": "⚙️",
    ".json": "⚙️",
    ".toml": "⚙️",
    ".ini": "⚙️",
    ".xml": "⚙️",
    ".exe": "⚙️",
    ".sh": "⚙️",
    ".bat": "⚙️",
    ".gitkeep": "⚙️",
    ".gitignore": "⚙️",
}


def load_ignore_patterns(root_path: Path) -> tuple[list[str], list[str]]:
    """Đọc quy tắc loại trừ từ file .gitignore và mặc định."""
    ignores = [
        ".git",
        ".git/*",
        ".vscode",
        ".vscode/*",
        ".idea",
        ".idea/*",
        "__pycache__",
        "__pycache__/*",
        ".pytest_cache",
        ".pytest_cache/*",
        ".venv",
        ".venv/*",
        "venv",
        "venv/*",
        ".DS_Store",
        "Thumbs.db",
        "*.pyc",
        "*.swp",
        "*.tmp",
        "*.png",
        "*.jpg",
        "*.jpeg",
    ]
    unignores = []

    gitignore_file = root_path / ".gitignore"
    if gitignore_file.exists():
        with open(gitignore_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("!"):
                    unignores.append(line[1:].strip())
                else:
                    ignores.append(line)

    return ignores, unignores


def is_path_ignored(
    rel_path: Path, ignores: list[str], unignores: list[str]
) -> bool:
    """Kiểm tra xem file/thư mục có bị ignore theo quy tắc .gitignore hay không."""
    path_str = rel_path.as_posix()
    name = rel_path.name
    parts = rel_path.parts

    # 1. Kiểm tra danh sách ngoại lệ (!file)
    for unpat in unignores:
        clean_unpat = unpat.rstrip("/")
        if fnmatch.fnmatch(name, clean_unpat) or fnmatch.fnmatch(
            path_str, clean_unpat
        ):
            return False

    # 2. Kiểm tra quy tắc ignore
    for pat in ignores:
        clean_pat = pat.rstrip("/")

        # Trùng tên file/thư mục
        if fnmatch.fnmatch(name, clean_pat):
            return True

        # Trùng đường dẫn
        if fnmatch.fnmatch(path_str, clean_pat):
            return True

        # Kiểm tra từng thư mục cha
        for part in parts:
            if fnmatch.fnmatch(part, clean_pat):
                return True

        # Xử lý quy tắc dạng folder/*
        if clean_pat.endswith("/*"):
            parent_folder = clean_pat[:-2]
            if path_str == parent_folder or path_str.startswith(
                parent_folder + "/"
            ):
                return True

    return False


def get_emoji(path: Path) -> str:
    """Xác định emoji dựa trên loại file/thư mục."""
    if path.is_dir():
        return FOLDER_EMOJI
    if path.name in EMOJI_MAP:
        return EMOJI_MAP[path.name]
    return EMOJI_MAP.get(path.suffix.lower(), DEFAULT_FILE_EMOJI)


def build_tree_lines(
    current_dir: Path,
    root_path: Path,
    prefix: str = "",
    ignores: list[str] = None,
    unignores: list[str] = None,
) -> list[str]:
    """Hàm đệ quy dựng các dòng cây thư mục."""
    lines = []

    try:
        entries = []
        for p in current_dir.iterdir():
            rel_p = p.relative_to(root_path)
            if not is_path_ignored(rel_p, ignores, unignores):
                entries.append(p)
    except PermissionError:
        return lines

    # Sắp xếp: Thư mục lên trước, sau đó đến file
    entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

    count = len(entries)
    for index, path in enumerate(entries):
        is_last = index == count - 1
        connector = "└── " if is_last else "├── "
        emoji = get_emoji(path)

        if path.is_dir():
            extension = "    " if is_last else "│   "
            sub_lines = build_tree_lines(
                path, root_path, prefix + extension, ignores, unignores
            )
            # In tên thư mục
            lines.append(f"{prefix}{connector}{emoji} {path.name}")
            # Nếu bên trong có chứa file con không bị ignore thì in ra
            lines.extend(sub_lines)
        else:
            lines.append(f"{prefix}{connector}{emoji} {path.name}")

    return lines


def generate_file_tree(
    target_dir: str = ".", output_file: str = "file_tree.md"
):
    """Tạo file markdown chứa file tree chuẩn định dạng."""
    root_path = Path(target_dir).resolve()

    if not root_path.exists() or not root_path.is_dir():
        print(
            f"❌ Đường dẫn không tồn tại hoặc không phải là thư mục: {root_path}"
        )
        return

    ignores, unignores = load_ignore_patterns(root_path)

    now_str = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
    project_name = root_path.name
    abs_path = str(root_path)

    tree_body = "\n".join(
        build_tree_lines(
            root_path,
            root_path,
            prefix="",
            ignores=ignores,
            unignores=unignores,
        )
    )
    ticks = "```"

    markdown_content = f"""# File Tree: {project_name}

**Generated:** {now_str}
**Root Path:** `{abs_path}`

{ticks}
{tree_body}
{ticks}

---
*Generated by FileTree Generator Script*
"""

    out_p = Path(output_file)
    out_p.write_text(markdown_content, encoding="utf-8")

    print(markdown_content)
    print(f"\n✅ Đã xuất file thành công tại: {out_p.resolve()}")


if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    generate_file_tree(target_path)