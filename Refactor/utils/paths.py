"""Path helpers used by training and inference scripts."""

from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path):
    """Create directory `path` when needed and return it as a `Path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(folder):
    """Return sorted image file paths from `folder` using common image extensions."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    files = [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda path: path.name.lower())


def paired_by_name(source_dir, target_dir):
    """Pair files by identical names and return `(source, target)` path tuples."""
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
