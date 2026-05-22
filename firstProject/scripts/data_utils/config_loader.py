"""
config_loader.py
----------------
Load NVD dataset splits from a YOLO data.yaml configuration file.

Public entry point:
  load_splits_from_data_yaml  — loads from a YOLO data.yaml (NVD flat layout)
"""

from pathlib import Path

import yaml

from .loaders import load_yolo_split_from_txt
from .models import DatasetSplit


def load_splits_from_data_yaml(data_yaml_path: str) -> list[DatasetSplit]:
    """
    Load all dataset splits defined in a YOLO data.yaml file.

    Recommended entry point for the NVD flat layout where splits are listed
    by ``train.txt`` / ``val.txt`` / ``test.txt`` files.

    Parameters
    ----------
    data_yaml_path : str
        Path to a YOLO data.yaml (e.g. ``configs/data.yaml``).

    Returns
    -------
    list[DatasetSplit]
        One entry per split found in the YAML. Splits missing from disk are
        skipped with a warning printed to stdout.
    """
    p = Path(data_yaml_path).resolve()
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    root = Path(cfg.get("path", str(p.parent)))
    if not root.is_absolute():
        root = p.parent / root

    names_raw = cfg.get("names", {})
    class_names: list[str] = (
        [names_raw[i] for i in range(len(names_raw))]
        if isinstance(names_raw, dict)
        else list(names_raw)
    )

    splits: list[DatasetSplit] = []
    for split_name in ("train", "val", "test"):
        rel = cfg.get(split_name)
        if rel is None:
            continue
        if str(rel).endswith(".txt"):
            txt_path = root / rel
            if not txt_path.exists():
                print(f"[data_utils] Skipping '{split_name}': {txt_path} not found")
                continue
            splits.append(
                load_yolo_split_from_txt(str(txt_path), str(root), split_name, list(class_names))
            )
    return splits
