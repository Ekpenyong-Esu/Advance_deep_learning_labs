"""
visualiser.py
-------------
Responsibility: render failure-case images and save grid figures to disk.

This module knows about OpenCV, matplotlib, and the case-dict shapes produced
by ``categoriser.py``.  It has no knowledge of model frameworks or metrics.

Public API
----------
  load_image_rgb(path)                            → np.ndarray
  crop_with_pad(img, box, pad)                    → (crop, rel_box)
  draw_box(img, box, color, label, thickness)     → np.ndarray
  render_fn_case(case, pad)                       → np.ndarray
  render_fp_case(case, pad)                       → np.ndarray
  render_poor_case(case, pad)                     → np.ndarray
  save_failure_grid(cases, title, render_fn, ...) → None
  save_all_grids(all_errors, figures_dir, ...)    → None
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PAD   = 40     # pixels of context added around each error box
_DEFAULT_COLS  = 3
_DEFAULT_ROWS  = 3
_DEFAULT_N     = _DEFAULT_COLS * _DEFAULT_ROWS   # panels per grid

# RGB colours for box annotations
_COLOR_MISSED = (220,  50,  50)   # red      — missed GT (False Negative)
_COLOR_GHOST  = ( 50, 100, 220)   # blue     — ghost prediction (False Positive)
_COLOR_GT     = ( 50, 200,  50)   # green    — ground-truth box
_COLOR_PRED   = (220, 140,  30)   # orange   — predicted box (Poor Localisation)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_rgb(path: Path) -> np.ndarray | None:
    """Load an image from *path* and return an RGB ndarray.

    Returns ``None`` if the file cannot be read.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Low-level drawing primitives
# ---------------------------------------------------------------------------

def crop_with_pad(
    img: np.ndarray,
    box: list[float],
    pad: int = _DEFAULT_PAD,
) -> tuple[np.ndarray, list[int]]:
    """Crop a region around *box* with *pad* pixels of context.

    Parameters
    ----------
    img : RGB ndarray (H, W, 3).
    box : [x1, y1, x2, y2] in absolute pixel coordinates.
    pad : Padding in pixels added to each side of the box.

    Returns
    -------
    crop    : Cropped RGB ndarray.
    rel_box : [x1, y1, x2, y2] coordinates of *box* relative to the crop origin.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
    x2c, y2c = min(w, x2 + pad), min(h, y2 + pad)
    crop    = img[y1c:y2c, x1c:x2c].copy()
    rel_box = [x1 - x1c, y1 - y1c, x2 - x1c, y2 - y1c]
    return crop, rel_box


def draw_box(
    img: np.ndarray,
    box: list[float | int],
    color: tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
) -> np.ndarray:
    """Draw one bounding box on *img* in-place.

    Parameters
    ----------
    img       : RGB ndarray (mutated).
    box       : [x1, y1, x2, y2] in pixel coordinates relative to *img*.
    color     : RGB tuple.
    label     : Optional text drawn above the box.
    thickness : Rectangle line width in pixels.

    Returns
    -------
    The same *img* array (mutated).
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    # OpenCV uses BGR, so swap R and B channels for the colour
    bgr = (color[2], color[1], color[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)
    if label:
        cv2.putText(
            img, label, (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA,
        )
    return img


# ---------------------------------------------------------------------------
# Case renderers — one per error type (single responsibility)
# ---------------------------------------------------------------------------

def render_fn_case(case: dict[str, Any], pad: int = _DEFAULT_PAD) -> np.ndarray:
    """Render a False Negative case: missed GT box shown in red.

    The image is loaded here so the categoriser stays memory-efficient.
    """
    img = load_image_rgb(case["path"])
    if img is None:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    crop, rel = crop_with_pad(img, case["missed_gt"], pad)
    draw_box(crop, rel, _COLOR_MISSED, "MISSED", thickness=2)
    return crop


def render_fp_case(case: dict[str, Any], pad: int = _DEFAULT_PAD) -> np.ndarray:
    """Render a False Positive case: ghost prediction shown in blue."""
    img = load_image_rgb(case["path"])
    if img is None:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    crop, rel = crop_with_pad(img, case["ghost_box"], pad)
    draw_box(crop, rel, _COLOR_GHOST, f"FP {case['score']:.2f}", thickness=2)
    return crop


def render_poor_case(case: dict[str, Any], pad: int = _DEFAULT_PAD) -> np.ndarray:
    """Render a Poor Localisation case: GT in green, prediction in orange."""
    img = load_image_rgb(case["path"])
    if img is None:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    crop, gt_rel = crop_with_pad(img, case["gt_box"], pad)
    off_x = max(0, int(case["gt_box"][0]) - pad)
    off_y = max(0, int(case["gt_box"][1]) - pad)
    pb        = case["pred_box"]
    pred_rel  = [pb[0] - off_x, pb[1] - off_y, pb[2] - off_x, pb[3] - off_y]
    draw_box(crop, gt_rel,  _COLOR_GT,   "GT",  thickness=2)
    draw_box(crop, pred_rel, _COLOR_PRED, f"IoU={case['iou']:.2f}", thickness=2)
    return crop


# ---------------------------------------------------------------------------
# Grid renderer
# ---------------------------------------------------------------------------

def save_failure_grid(
    cases: list[dict[str, Any]],
    title: str,
    render_fn: Callable[[dict[str, Any]], np.ndarray],
    save_path: Path,
    n: int = _DEFAULT_N,
    cols: int = _DEFAULT_COLS,
    show: bool = False,
    seed: int = 42,
) -> None:
    """Sample up to *n* cases and save a matplotlib image grid to *save_path*.

    Parameters
    ----------
    cases      : List of case dicts (FN, FP, or Poor).
    title      : Figure super-title.
    render_fn  : One of ``render_fn_case``, ``render_fp_case``, ``render_poor_case``.
    save_path  : Absolute path for the output PNG.
    n          : Maximum number of panels.
    cols       : Number of columns in the grid.
    show       : If ``True``, call ``plt.show()``; otherwise close the figure.
    seed       : Random seed for reproducible sampling.
    """
    if not cases:
        return

    rng    = random.Random(seed)
    sample = rng.sample(cases, min(n, len(cases)))
    cols_  = min(cols, len(sample))
    rows   = (len(sample) + cols_ - 1) // cols_

    fig, axes = plt.subplots(rows, cols_, figsize=(cols_ * 4, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for ax, case in zip(axes, sample):
        ax.imshow(render_fn(case))
        ax.axis("off")
        ax.set_title(Path(case["path"]).name[:22], fontsize=7)

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"{title}  (n={len(cases)} total)", fontsize=11, y=1.01)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Batch grid saver
# ---------------------------------------------------------------------------

def save_all_grids(
    all_errors: dict[str, dict[str, list]],
    figures_dir: Path,
    best_model: str | None = None,
    n: int = _DEFAULT_N,
    seed: int = 42,
) -> None:
    """Save three grids (FN / FP / Poor) for every model in *all_errors*.

    Parameters
    ----------
    all_errors  : Output of :func:`~categoriser.categorise_all_models`.
    figures_dir : Directory where PNG files are written.
    best_model  : If set, grids for this model are also shown inline.
    n           : Maximum panels per grid.
    seed        : Random seed for reproducible sampling.
    """
    for name, errors in all_errors.items():
        slug = name.lower().replace(" · ", "_").replace(" ", "_").replace("-", "")
        show = (name == best_model)

        for error_key, render_fn, label in [
            ("fn",   render_fn_case,   "False Negatives"),
            ("fp",   render_fp_case,   "False Positives"),
            ("poor", render_poor_case, "Poor Localisation"),
        ]:
            save_failure_grid(
                cases     = errors[error_key],
                title     = f"{name} — {label}",
                render_fn = render_fn,
                save_path = figures_dir / f"error_{slug}_{error_key}.png",
                n         = n,
                show      = show,
                seed      = seed,
            )

        status = "(shown above)" if show else "saved"
        print(
            f"  {name}: "
            f"FN={len(errors['fn'])}  "
            f"FP={len(errors['fp'])}  "
            f"Poor={len(errors['poor'])}  [{status}]"
        )

    print(f"\nAll grids saved to {figures_dir}")
