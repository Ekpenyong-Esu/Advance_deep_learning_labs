"""
coco_formatter.py
-----------------
Re-serialise a DatasetSplit as a COCO JSON file.

This is needed for DETR (and other HuggingFace) training pipelines that
expect the standard COCO annotation schema.

The output JSON follows the official COCO schema:
  {
    "info":        {...},
    "licenses":    [],
    "categories":  [{"id": 1, "name": "car", ...}, ...],
    "images":      [{"id": 1, "file_name": "...", "width": ..., "height": ...}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                      "bbox": [x, y, w, h], "area": ..., "iscrowd": 0}]
  }

Note: COCO category ids are 1-indexed; BBox.class_id is 0-indexed.
"""

import json
from pathlib import Path
from typing import List

from .models import DatasetSplit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_coco_dict(split: DatasetSplit) -> dict:
    categories = [
        {"id": i + 1, "name": name, "supercategory": "vehicle"}
        for i, name in enumerate(split.class_names)
    ]

    images: List[dict] = []
    annotations: List[dict] = []
    ann_id = 1

    for img_id, img_ann in enumerate(split.images, start=1):
        images.append(
            {
                "id": img_id,
                "file_name": img_ann.file_name,
                "width": img_ann.width,
                "height": img_ann.height,
            }
        )
        for bbox in img_ann.bboxes:
            coco_w = bbox.x_max - bbox.x_min
            coco_h = bbox.y_max - bbox.y_min
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": bbox.class_id + 1,  # COCO is 1-indexed
                    "bbox": [bbox.x_min, bbox.y_min, coco_w, coco_h],
                    "area": bbox.area,
                    "iscrowd": int(bbox.is_crowd),
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": "NVD — Nordic Vehicle Dataset (COCO format)",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_coco_split(split: DatasetSplit, output_path: str) -> None:
    """
    Write *split* as a COCO JSON file to *output_path*.

    Parameters
    ----------
    split : DatasetSplit
    output_path : str
        Full path to the output .json file
        (e.g. 'data/processed/annotations/instances_train.json').
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    coco_dict = _build_coco_dict(split)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(coco_dict, fh, indent=2, ensure_ascii=False)
