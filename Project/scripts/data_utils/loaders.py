"""
loaders.py
----------
Load NVD annotations from YOLO-format files on disk.

Public entry point:
  load_yolo_split_from_txt — read a NVD-style split .txt listing image paths

Returns a DatasetSplit with bboxes in absolute pixel coordinates.
"""

from pathlib import Path

from PIL import Image as PILImage

from .models import BBox, DatasetSplit, ImageAnnotation

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height) by reading only the image header via PIL."""
    try:
        with PILImage.open(img_path) as pil_img:
            return pil_img.size
    except Exception:
        return 0, 0


def _parse_label_file(
    label_path: Path,
    width: int,
    height: int,
    class_names: list[str],
) -> list[BBox]:
    """
    Parse a single YOLO .txt label file and return a list of BBox objects.

    Converts normalised (cx, cy, bw, bh) coordinates to absolute pixels.
    Extends *class_names* in-place when an unseen class_id is encountered.
    """
    bboxes: list[BBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])

        while len(class_names) <= cls_id:
            class_names.append(f"class_{len(class_names)}")

        bboxes.append(BBox(
            class_id=cls_id,
            class_name=class_names[cls_id],
            x_min=(cx - bw / 2) * width,
            y_min=(cy - bh / 2) * height,
            x_max=(cx + bw / 2) * width,
            y_max=(cy + bh / 2) * height,
        ))
    return bboxes


def _make_image_annotation(
    img_path: Path,
    images_dir: Path,
    label_path: Path,
    class_names: list[str],
) -> ImageAnnotation:
    """Build one ImageAnnotation from an image path and its label file."""
    width, height = _read_image_size(img_path)
    img_ann = ImageAnnotation(
        file_name=img_path.name,
        images_dir=str(images_dir.resolve()),
        width=width,
        height=height,
    )
    if label_path.exists() and width > 0 and height > 0:
        img_ann.bboxes = _parse_label_file(label_path, width, height, class_names)
    return img_ann


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_yolo_split_from_txt(
    txt_path: str,
    root: str,
    split_name: str = "train",
    class_names: list[str] | None = None,
) -> DatasetSplit:
    """
    Load one dataset split from a .txt file listing image paths (one per line).

    Matches the NVD flat layout: splits are defined by train.txt / val.txt /
    test.txt files where each line is a relative path such as
    ``./images/<filename>.png``.

    Label files are resolved by replacing the ``images`` folder with
    ``labels`` and swapping the extension to ``.txt``.

    Parameters
    ----------
    txt_path : str
        Path to the .txt split file (e.g. ``data/raw/train.txt``).
    root : str
        Root directory that the paths inside the .txt are relative to.
    split_name : str
        Label for this split ('train', 'val', or 'test').
    class_names : list[str] | None
        Canonical class list. Extended in-place when unknown ids are found.
    """
    txt = Path(txt_path).resolve()
    root_dir = Path(root).resolve()
    names: list[str] = list(class_names) if class_names is not None else []

    images: list[ImageAnnotation] = []
    for raw_line in txt.read_text(encoding="utf-8").splitlines():
        rel = raw_line.strip()
        if not rel:
            continue
        img_path = (root_dir / rel).resolve()
        if not img_path.exists():
            continue
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        images.append(
            _make_image_annotation(img_path, img_path.parent, label_path, names)
        )

    return DatasetSplit(name=split_name, class_names=names, images=images)
