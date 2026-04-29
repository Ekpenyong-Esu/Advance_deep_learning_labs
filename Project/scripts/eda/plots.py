"""
plots.py
--------
Matplotlib / seaborn distribution plots for EDA.

All functions accept the statistics dict returned by
dataset_stats.compute_stats() and return a matplotlib Figure.

No I/O side effects — callers save figures as needed.
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    stats: dict,
    figsize: tuple[float, float] = (7, 4),
    title: str | None = None,
) -> plt.Figure:
    """Bar chart of bounding-box count per class."""
    sns.set_theme(style="whitegrid", palette="muted")

    counts = stats["class_counts"]
    
    if not counts:
        raise ValueError("No class counts found in stats dict.")

    names = sorted(counts, key=lambda k: -counts[k])
    values = [counts[n] for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names[::-1], values[::-1], color=sns.color_palette("muted", len(names)))
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Number of bounding boxes")
    ax.set_title(title or f"{stats['split']} — class distribution")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bounding-box size distribution
# ---------------------------------------------------------------------------

def plot_bbox_size_distribution(
    stats: dict,
    bins: int = 50,
    figsize: tuple[float, float] = (10, 4),
    title: str | None = None,
) -> plt.Figure:
    """
    Side-by-side histograms of bounding-box width and height (absolute px).
    """
    sns.set_theme(style="whitegrid", palette="muted")
    widths = stats["box_widths"]
    heights = stats["box_heights"]

    fig, (ax_w, ax_h) = plt.subplots(1, 2, figsize=figsize)

    ax_w.hist(widths, bins=bins, color=sns.color_palette("muted")[0], edgecolor="white")
    ax_w.set_xlabel("Box width (px)")
    ax_w.set_ylabel("Count")
    ax_w.set_title("Width distribution")

    ax_h.hist(heights, bins=bins, color=sns.color_palette("muted")[1], edgecolor="white")
    ax_h.set_xlabel("Box height (px)")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Height distribution")

    fig.suptitle(title or f"{stats['split']} — bounding-box size distribution")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Aspect ratio distribution
# ---------------------------------------------------------------------------

def plot_bbox_aspect_ratio(
    stats: dict,
    bins: int = 50,
    figsize: tuple[float, float] = (7, 4),
    title: str | None = None,
) -> plt.Figure:
    """Histogram of bounding-box aspect ratios (width / height)."""
    sns.set_theme(style="whitegrid", palette="muted")
    aspects = [a for a in stats["box_aspect_ratios"] if a > 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(aspects, bins=bins, color=sns.color_palette("muted")[2], edgecolor="white")
    ax.axvline(np.median(aspects), color="red", linestyle="--", linewidth=1.2,
               label=f"Median {np.median(aspects):.2f}")
    ax.set_xlabel("Aspect ratio  (width / height)")
    ax.set_ylabel("Count")
    ax.set_title(title or f"{stats['split']} — bounding-box aspect ratio")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Objects per frame
# ---------------------------------------------------------------------------

def plot_objects_per_frame(
    stats: dict,
    figsize: tuple[float, float] = (7, 4),
    title: str | None = None,
) -> plt.Figure:
    """
    Bar chart of the distribution of the number of annotated objects per frame.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    bpi = stats["boxes_per_image"]
    if not bpi:
        raise ValueError("No boxes_per_image data found in stats dict.")

    freq = Counter(bpi)  # {num_objects: num_frames}
    x = sorted(freq)
    counts = [freq[i] for i in x]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, counts, color=sns.color_palette("muted")[3])
    ax.set_xlabel("Annotated objects per frame")
    ax.set_ylabel("Number of frames")
    ax.set_title(title or f"{stats['split']} — objects per frame")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined EDA dashboard (all four plots in one figure)
# ---------------------------------------------------------------------------

def plot_eda_dashboard(
    stats: dict,
    figsize: tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    All four EDA plots in one figure — handy for a quick notebook summary.

    The subplot code intentionally duplicates the individual plot functions
    above so this one call produces a self-contained figure without needing
    to stitch separate figures together.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    counts = stats["class_counts"]
    names = sorted(counts, key=lambda k: -counts[k])
    values = [counts[n] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"EDA Dashboard — {stats['split']} split", fontsize=14, y=1.01)

    # -- Class distribution
    ax = axes[0, 0]
    bars = ax.barh(names[::-1], values[::-1], color=sns.color_palette("muted", len(names)))
    ax.bar_label(bars, padding=4, fontsize=8)
    ax.set_xlabel("Bounding boxes")
    ax.set_title("Class distribution")

    # -- Box size
    ax = axes[0, 1]
    ax.hist(stats["box_widths"], bins=50, alpha=0.7, label="width",
            color=sns.color_palette("muted")[0])
    ax.hist(stats["box_heights"], bins=50, alpha=0.7, label="height",
            color=sns.color_palette("muted")[1])
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Count")
    ax.set_title("BBox width vs height")
    ax.legend()

    # -- Aspect ratio
    ax = axes[1, 0]
    aspects = [a for a in stats["box_aspect_ratios"] if a > 0]
    ax.hist(aspects, bins=50, color=sns.color_palette("muted")[2], edgecolor="white")
    if aspects:
        ax.axvline(np.median(aspects), color="red", linestyle="--",
                   label=f"Median {np.median(aspects):.2f}")
    ax.set_xlabel("Width / Height")
    ax.set_title("Aspect ratio distribution")
    ax.legend()

    # -- Objects per frame
    ax = axes[1, 1]
    bpi = stats["boxes_per_image"]
    freq = Counter(bpi)
    x = sorted(freq)
    ax.bar(x, [freq[i] for i in x], color=sns.color_palette("muted")[3])
    ax.set_xlabel("Objects per frame")
    ax.set_ylabel("Frames")
    ax.set_title("Objects-per-frame distribution")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    return fig
