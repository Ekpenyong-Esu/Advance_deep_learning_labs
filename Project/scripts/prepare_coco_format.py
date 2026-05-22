#!/usr/bin/env python
"""
prepare_coco_format.py
----------------------
Re-serialise NVD YOLO annotations as COCO JSON files for DETR / HuggingFace training.

Output layout
-------------
<output>/
    annotations/
        instances_train.json
        instances_val.json
        instances_test.json

Usage
-----
python scripts/prepare_coco_format.py \\
    --data-yaml configs/data.yaml \\
    --output    data/processed/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import load_splits_from_data_yaml
from data_utils.coco_formatter import save_coco_split


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare COCO-format annotations for DETR training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-yaml",
        metavar="YAML",
        required=True,
        help="Path to a YOLO data.yaml (e.g. the one shipped with the NVD download).",
    )
    p.add_argument(
        "--output",
        default="data/processed",
        metavar="DIR",
        help="Root output directory (default: data/processed).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    splits = load_splits_from_data_yaml(args.data_yaml)

    out_root = Path(args.output).resolve()
    ann_dir = out_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"[prepare_coco] Processing split '{split.name}': "
              f"{split.num_images} images, {split.num_boxes} boxes")
        json_path = ann_dir / f"instances_{split.name}.json"
        save_coco_split(split, str(json_path))
        print(f"[prepare_coco] Wrote {json_path}")

def convert_yolo_to_coco(
    images_dir,
    labels_dir,
    split_file,
    output_json,
    categories=None,
):
    """
    Convert a YOLO-format tiled dataset to COCO JSON.

    Parameters
    ----------
    images_dir : str or Path
        Directory containing tile images.
    labels_dir : str or Path
        Directory containing YOLO .txt label files.
    split_file : str or Path
        Text file listing relative image paths (one per line).
    output_json : str or Path
        Output COCO JSON path.
    categories : list of dict, optional
        List of {"id": int, "name": str}. Defaults to [{"id": 0, "name": "car"}].
    """
    import json
    from PIL import Image as PILImage

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    split_file = Path(split_file)
    output_json = Path(output_json)

    if categories is None:
        categories = [{"id": 0, "name": "car"}]

    # COCO categories are 1-indexed
    coco_categories = [
        {"id": cat["id"] + 1, "name": cat["name"], "supercategory": "vehicle"}
        for cat in categories
    ]

    lines = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

    images_list = []
    annotations_list = []
    ann_id = 1

    for img_id, rel_path in enumerate(lines, start=1):
        # rel_path is like ./images/name.png
        fname = Path(rel_path).name
        img_path = images_dir / fname

        if not img_path.exists():
            continue

        im = PILImage.open(img_path)
        w, h = im.size

        images_list.append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h,
        })

        # Load corresponding label
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert normalized YOLO to COCO absolute [x, y, w, h]
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            annotations_list.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,  # COCO is 1-indexed
                "bbox": [x, y, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_dict = {
        "info": {"description": "NVD tiled dataset (COCO format)", "version": "1.0"},
        "licenses": [],
        "categories": coco_categories,
        "images": images_list,
        "annotations": annotations_list,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(coco_dict, fh, indent=2, ensure_ascii=False)

    print(f"[convert_yolo_to_coco] {len(images_list)} images, "
          f"{ann_id - 1} annotations → {output_json}")
    
if __name__ == "__main__":
    main()
