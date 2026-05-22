"""
data_utils — shared data layer for the NVD car-detection project.

Public API
----------
BBox               – single bounding-box annotation (abs pixel coords)
ImageAnnotation    – one image's metadata + its bounding boxes
DatasetSplit       – all annotated images for one split (train/val/test)

load_yolo_split_from_txt     – load one split from a .txt file listing image paths (NVD flat layout)
compute_stats                – compute dataset statistics for one split
"""

from .models import BBox, ImageAnnotation, DatasetSplit
from .loaders import load_yolo_split_from_txt
from .config_loader import load_splits_from_data_yaml
from .dataset_stats import compute_stats
from .reporting import print_stats

__all__ = [
    "BBox",
    "ImageAnnotation",
    "DatasetSplit",
    "load_yolo_split_from_txt",
    "load_splits_from_data_yaml",
    "compute_stats",
    "print_stats",
]
