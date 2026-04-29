"""
models.py
---------
Domain model dataclasses for the NVD annotation pipeline.

BBox, ImageAnnotation, and DatasetSplit are the sole shared data
structures used by every other module in this package.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BBox:
    """One bounding-box annotation. Coordinates are absolute pixels."""

    class_id: int        # index into DatasetSplit.class_names
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    is_crowd: bool = False

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0


@dataclass
class ImageAnnotation:
    """One image and all its bounding-box annotations."""

    file_name: str      # filename only (no directory component)
    images_dir: str     # folder that contains the actual image file
    width: int
    height: int
    bboxes: list[BBox] = field(default_factory=list)

    @property
    def image_path(self) -> Path:
        return Path(self.images_dir) / self.file_name

    @property
    def num_boxes(self) -> int:
        return len(self.bboxes)


@dataclass
class DatasetSplit:
    """All annotated images for one split (train / val / test)."""

    name: str                 # 'train', 'val', or 'test'
    class_names: list[str]    # ordered list; BBox.class_id is the index into this
    images: list[ImageAnnotation] = field(default_factory=list)

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_boxes(self) -> int:
        return sum(img.num_boxes for img in self.images)
