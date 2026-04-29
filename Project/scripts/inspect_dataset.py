#!/usr/bin/env python
"""
inspect_dataset.py
------------------
Print per-split statistics for the NVD dataset.

Usage
-----
python scripts/inspect_dataset.py --data-yaml configs/data.yaml
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import compute_stats, load_splits_from_data_yaml
from data_utils.reporting import print_stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Print dataset statistics for the NVD dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-yaml",
        metavar="YAML",
        required=True,
        help="Path to a YOLO data.yaml (e.g. the one shipped with the NVD download).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    splits = load_splits_from_data_yaml(args.data_yaml)

    for split in splits:
        stats = compute_stats(split)
        print_stats(stats)


if __name__ == "__main__":
    main()
