"""路径处理工具。"""

from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path):
    """确保目录 path 存在，并以 Path 对象返回。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(folder):
    """列出 folder 下常见图片格式文件，并按文件名排序。"""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    files = [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name.lower())


def paired_by_name(source_dir, target_dir):
    """按相同文件名配对两个目录中的图片。

    输入:
        source_dir: 源目录。
        target_dir: 目标目录。
    输出:
        [(source_path, target_path), ...]。
    """
    source_files = list_image_files(source_dir)
    target_files = {path.name: path for path in list_image_files(target_dir)}
    pairs = []
    missing = []
    for source_path in source_files:
        target_path = target_files.get(source_path.name)
        if target_path is None:
            missing.append(source_path.name)
        else:
            pairs.append((source_path, target_path))
    if missing:
        raise FileNotFoundError(f"Missing paired files in {target_dir}: {missing[:5]}")
    return pairs
