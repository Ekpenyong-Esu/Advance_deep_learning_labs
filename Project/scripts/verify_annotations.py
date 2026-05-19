#!/usr/bin/env python
"""
verify_annotations.py
---------------------
Visually verify YOLO label files by drawing bounding boxes from the .txt
labels onto a random sample of images and saving the results to disk.

Delegates all parsing to data_utils and all drawing to eda.visualizer —
this script is purely CLI orchestration.

Usage
-----
# Verify train split using the project data.yaml
python scripts/verify_annotations.py \\
    --data-yaml configs/data.yaml \\
    --split     train \\
    --n         10 \\
    --output    results/figures/verify_train/
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import load_splits_from_data_yaml
from eda.visualizer import draw_annotations


def _save_verified_images(
    split,
    n: int,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Sample *n* images from *split*, draw boxes via eda.visualizer, and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    available = [img for img in split.images if Path(img.images_dir, img.file_name).exists()]
    rng = random.Random(seed)
    sample = rng.sample(available, min(n, len(available)))

    for img_ann in sample:
        img_path = Path(img_ann.images_dir) / img_ann.file_name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [warn] Cannot read {img_path}")
            continue
        annotated = draw_annotations(bgr, img_ann)
        cv2.imwrite(str(output_dir / img_ann.file_name), annotated)

    print(f"Saved {len(sample)} verified images to: {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visually verify YOLO label files by drawing boxes on sampled images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-yaml",
        required=True,
        metavar="YAML",
        help="Path to a YOLO data.yaml (e.g. configs/data.yaml).",
    )
    p.add_argument(
        "--split",
        default="train",
        metavar="NAME",
    )
    p.add_argument(
        "--n",
        type=int,
        default=10,
        metavar="INT",
        help="Number of images to sample (default: 10).",
    )
    p.add_argument(
        "--output",
        default="results/figures/verify",
        metavar="DIR",
        help="Directory to save verification images (default: results/figures/verify).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    splits = load_splits_from_data_yaml(args.data_yaml)
    matched = [s for s in splits if s.name == args.split]
    if not matched:
        print(f"ERROR: split '{args.split}' not found in {args.data_yaml}", file=sys.stderr)
        sys.exit(1)

    _save_verified_images(matched[0], args.n, Path(args.output))


if __name__ == "__main__":
    main()

