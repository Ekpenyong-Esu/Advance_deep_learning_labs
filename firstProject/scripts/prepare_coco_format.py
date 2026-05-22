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


if __name__ == "__main__":
    main()
