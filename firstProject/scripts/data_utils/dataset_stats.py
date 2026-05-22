"""
dataset_stats.py
----------------
Compute descriptive statistics for a DatasetSplit.

All heavy computation lives here; EDA visualisation calls these functions
and receives plain Python dicts — no matplotlib/cv2 in this module.
"""

from collections import Counter

from .models import DatasetSplit


def compute_stats(split: DatasetSplit) -> dict:
    """
    Return a statistics dictionary for *split*.

    Keys
    ----
    split               : str   — split name
    num_images          : int
    num_boxes           : int
    class_names         : list[str]
    class_counts        : dict[str, int]  — per-class box count
    images_without_boxes: int
    boxes_per_image     : list[int]       — length == num_images
    box_widths          : list[float]     — absolute pixels
    box_heights         : list[float]
    box_areas           : list[float]     — width × height
    box_aspect_ratios   : list[float]     — width / height
    bbox_centers_x      : list[float]     — normalised [0, 1]
    bbox_centers_y      : list[float]     — normalised [0, 1]

    Quality flags (useful for annotation QC)
    -----------------------------------------
    crowd_boxes         : int  — boxes with iscrowd=1 (not individually annotated)
    tiny_boxes          : int  — boxes with area < 32×32 px (often truncated or noise)
    missing_dims        : int  — images where width or height is 0 (parser couldn't read size)
    """
    all_bboxes = [bbox for img in split.images 
                            for bbox in img.bboxes]

    widths = [b.width for b in all_bboxes]
    heights = [b.height for b in all_bboxes]
    areas = [b.area for b in all_bboxes]
    aspects = [b.aspect_ratio for b in all_bboxes]

    # Normalised centre positions (x, y both in [0, 1]) for the heatmap
    cx_list: list[float] = []
    cy_list: list[float] = []
    for img in split.images:
        if img.width <= 0 or img.height <= 0:
            continue
        for b in img.bboxes:
            cx_list.append((b.x_min + b.x_max) / 2.0 / img.width)
            cy_list.append((b.y_min + b.y_max) / 2.0 / img.height)

    return {
        "split": split.name,
        "num_images": split.num_images,
        "num_boxes": split.num_boxes,
        "class_names": split.class_names,
        "class_counts": dict(Counter(b.class_name for b in all_bboxes)),
        "images_without_boxes": sum(1 for img in split.images if img.num_boxes == 0),
        "boxes_per_image": [img.num_boxes for img in split.images],
        "box_widths": widths,
        "box_heights": heights,
        "box_areas": areas,
        "box_aspect_ratios": aspects,
        "bbox_centers_x": cx_list,
        "bbox_centers_y": cy_list,
        # Annotation quality flags
        "crowd_boxes": sum(1 for b in all_bboxes if b.is_crowd),
        "tiny_boxes": sum(1 for b in all_bboxes if b.area < 32 * 32),
        "missing_dims": sum(1 for img in split.images if img.width <= 0 or img.height <= 0),
    }


