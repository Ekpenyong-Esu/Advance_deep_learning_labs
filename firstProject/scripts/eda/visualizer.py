"""
visualizer.py
-------------
Draw bounding boxes on images and produce spatial density heatmaps.

All functions return matplotlib Figure objects so callers can decide
whether to display inline (notebook) or save to disk.

Dependencies: opencv-python, matplotlib, numpy
"""

import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from data_utils.models import DatasetSplit, ImageAnnotation


# ---------------------------------------------------------------------------
# Colour palette — one BGR colour per class (OpenCV convention)
# ---------------------------------------------------------------------------

_PALETTE = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
]


def _class_colour(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Core draw helper
# ---------------------------------------------------------------------------

def draw_annotations(
    image: np.ndarray,
    img_ann: ImageAnnotation,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw bounding boxes and class labels on *image* (BGR ndarray).

    Returns a new array (does not mutate *image*).
    """
    out = image.copy()
    for bbox in img_ann.bboxes:
        colour = _class_colour(bbox.class_id)
        x1, y1, x2, y2 = (
            int(bbox.x_min),
            int(bbox.y_min),
            int(bbox.x_max),
            int(bbox.y_max),
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        label = bbox.class_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), colour, -1)
        cv2.putText(
            out, label, (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Multi-image sample grids
# ---------------------------------------------------------------------------

def plot_annotated_samples(
    split: DatasetSplit,
    n: int = 25,
    cols: int = 5,
    seed: int = 42,
    figsize_per_cell: tuple[float, float] = (3.5, 2.5),
) -> plt.Figure:
    """
    Return a matplotlib Figure showing *n* random images with drawn bboxes.

    Only images that actually exist on disk are selected.
    """
    rng = random.Random(seed)
    available = [img for img in split.images if img.image_path.exists()]
    if not available:
        raise FileNotFoundError(
            f"No images found on disk for split '{split.name}'. "
            "Run the download / extraction step first."
        )
    sample = rng.sample(available, min(n, len(available)))

    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_cell[0] * cols, figsize_per_cell[1] * rows),
    )
    # plt.subplots returns a 2D array when rows>1, 1D when rows==1.
    # Flattening to 1D lets us iterate the same way in both cases.
    axes = np.array(axes).flatten()

    for ax, img_ann in zip(axes, sample):
        bgr = cv2.imread(str(img_ann.image_path))
        if bgr is None:
            ax.axis("off")
            continue
        annotated = draw_annotations(bgr, img_ann)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(
            f"{Path(img_ann.file_name).name}\n{img_ann.num_boxes} boxes",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"{split.name} split — annotated samples", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spatial heatmap of bounding-box centres
# ---------------------------------------------------------------------------

def plot_bbox_heatmap(
    stats: dict,
    bins: int = 50,
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
) -> plt.Figure:
    """
    Plot a 2-D density heatmap of normalised bounding-box centre positions.

    Parameters
    ----------
    stats : dict
        Dictionary returned by dataset_stats.compute_stats().
    bins : int
        Number of bins in each axis of the 2-D histogram.
    """
    cx = np.array(stats["bbox_centers_x"])
    cy = np.array(stats["bbox_centers_y"])

    if len(cx) == 0:
        raise ValueError("No bounding boxes found — cannot plot heatmap.")

    heatmap, xedges, yedges = np.histogram2d(cx, cy, bins=bins, range=[[0, 1], [0, 1]])

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="auto",
        cmap="hot",
        interpolation="bilinear",
    )
    fig.colorbar(img, ax=ax, label="Box count")
    ax.set_xlabel("Normalised x (→ right)")
    ax.set_ylabel("Normalised y (↑ up in image coords)")
    ax.set_title(title or f"{stats['split']} — bounding-box centre heatmap")
    fig.tight_layout()
    return fig
