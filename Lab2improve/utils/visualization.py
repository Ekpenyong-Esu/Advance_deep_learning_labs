"""
utils/visualization.py — Image Saving and Grid Plotting
========================================================
Shared helpers for saving generated and adversarial images to disk.
All functions write PNG files and close the figure to avoid memory leaks.

Functions
---------
  save_image_grid           — 4×4 grid of greyscale flat images
  save_adversarial_comparison — side-by-side original vs adversarial pairs
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # non-interactive backend; safe for file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Grid of generated images
# ─────────────────────────────────────────────────────────────────────────────

def save_image_grid(images: np.ndarray, path: str,
                    nrows: int = 4, ncols: int = 4,
                    img_size: int = 28) -> None:
    """
    Save a grid of greyscale images to a PNG file.

    Parameters
    ----------
    images   : float array of shape (N, img_size*img_size) or (N, img_size, img_size)
               Expected value range [0, 1].
    path     : destination file path; parent directories are created if absent.
    nrows    : number of rows in the grid
    ncols    : number of columns in the grid
    img_size : side length of each image in pixels (default 28 for MNIST)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    images = images[: nrows * ncols]

    fig = plt.figure(figsize=(ncols, nrows))
    gs  = gridspec.GridSpec(nrows, ncols, wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.axis("off")
        ax.set_aspect("equal")
        plt.imshow(img.reshape(img_size, img_size), cmap="Greys_r", vmin=0, vmax=1)

    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def save_adversarial_comparison(originals: torch.Tensor,
                                adversarials: torch.Tensor,
                                orig_preds: torch.Tensor,
                                adv_preds: torch.Tensor,
                                path: str,
                                num: int = 5) -> None:
    """
    Save a two-row figure: originals on top, adversarial versions below.

    Parameters
    ----------
    originals    : (N, 1, 28, 28) tensor — original images in [-1, 1]
    adversarials : (N, 1, 28, 28) tensor — adversarial images in [-1, 1]
    orig_preds   : (N,) LongTensor — model predictions on originals
    adv_preds    : (N,) LongTensor — model predictions on adversarial images
    path         : save path (parent directories created if absent)
    num          : number of image pairs to plot
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = min(num, originals.size(0))

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))

    for i in range(n):
        # Rescale from [-1, 1] to [0, 1] for display
        orig_img = (originals[i, 0].cpu().numpy() + 1.0) / 2.0
        adv_img  = (adversarials[i, 0].cpu().numpy() + 1.0) / 2.0

        axes[0, i].imshow(orig_img, cmap="Greys_r", vmin=0, vmax=1)
        axes[0, i].set_title(f"Pred: {orig_preds[i].item()}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(adv_img, cmap="Greys_r", vmin=0, vmax=1)
        axes[1, i].set_title(f"Pred: {adv_preds[i].item()}", fontsize=8)
        axes[1, i].axis("off")

    plt.suptitle("Row 1: Original    Row 2: Adversarial (FGSM)", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
