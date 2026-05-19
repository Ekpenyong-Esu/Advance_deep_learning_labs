"""
utils/visualization.py — Caption and Training Visualisation Helpers
====================================================================
Functions
---------
  plot_loss_curves       — plot train and validation loss over epochs
  show_sample_captions   — display a grid of images with ground-truth and
                           generated captions side-by-side
  save_sample_captions   — same as above but saves to a PNG file

All functions are designed to work inside Jupyter notebooks (inline display)
as well as saving files to disk.  The Agg backend is used for file saves to
avoid GUI dependencies on headless servers.

Usage
-----
    from utils.visualization import plot_loss_curves, show_sample_captions

    plot_loss_curves(train_losses, val_losses)
    show_sample_captions(encoder, decoder, test_ref_loader, vocab, device, n=6)
"""

import os
import random
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import torch

import config
from data.vocabulary import Vocabulary
from utils.evaluation import generate_caption


# ─────────────────────────────────────────────────────────────────────────────
# Loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: List[float],
    val_losses:   List[float],
    save_path:    str = None,
) -> None:
    """
    Plot training and validation loss over epochs.

    Parameters
    ----------
    train_losses : list of per-epoch average training loss values
    val_losses   : list of per-epoch average validation loss values
    save_path    : if given, the figure is saved to this path (PNG);
                   if None, the figure is displayed inline (notebook mode)
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train loss", linewidth=2)
    ax.plot(epochs, val_losses,   marker="s", label="Val loss",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss (per token)")
    ax.set_title("Image Captioning — Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Loss curve saved → {save_path}")
    else:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Sample captions grid
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_display(image: torch.Tensor) -> "np.ndarray":
    """
    Convert a normalised image tensor (3, H, W) back to a uint8 numpy array
    suitable for ``imshow``.
    """
    import numpy as np
    mean = torch.tensor(config.IMAGE_MEAN).view(3, 1, 1)
    std  = torch.tensor(config.IMAGE_STD).view(3, 1, 1)
    img  = image.cpu() * std + mean            # un-normalise
    img  = img.clamp(0, 1).permute(1, 2, 0)   # (H, W, 3)
    return (img.numpy() * 255).astype("uint8")


def show_sample_captions(
    encoder,
    decoder,
    test_ref_loader: torch.utils.data.DataLoader,
    vocabulary:      Vocabulary,
    device:          torch.device,
    n:               int  = None,
    save_path:       str  = None,
    seed:            int  = 0,
) -> None:
    """
    Display ``n`` test images with their ground-truth and generated captions.

    Parameters
    ----------
    encoder          : ImageEncoder (in eval mode)
    decoder          : CaptionDecoder (in eval mode)
    test_ref_loader  : DataLoader over TestReferenceDataset (batch_size=1)
    vocabulary       : Vocabulary instance
    device           : torch.device
    n                : number of images to show (defaults to config.NUM_SAMPLE_IMAGES)
    save_path        : if given, save the figure here instead of showing inline
    seed             : random seed for sample selection
    """
    n = n or config.NUM_SAMPLE_IMAGES

    encoder.eval()
    decoder.eval()

    # Collect all items from the loader so we can sample randomly
    all_items = []
    for images, captions_list in test_ref_loader:
        all_items.append((images[0], captions_list))  # (3,H,W) and list of strings
        if len(all_items) >= 50:          # cap to avoid loading entire test set
            break

    random.seed(seed)
    samples = random.sample(all_items, min(n, len(all_items)))

    ncols = 2
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))

    if nrows == 1:
        axes = [axes]

    for row_idx, (image, captions_list) in enumerate(samples):
        # Ground truth — pick the first reference caption
        ref_strings = [cap[0] if isinstance(cap, (list, tuple)) else cap
                       for cap in captions_list]
        ground_truth = ref_strings[0]

        # Generated caption
        generated = generate_caption(encoder, decoder, image, vocabulary, device)

        img_np = _tensor_to_display(image)

        # Left column — image
        ax_img  = axes[row_idx][0]
        ax_txt  = axes[row_idx][1]

        ax_img.imshow(img_np)
        ax_img.axis("off")
        ax_img.set_title(f"Sample {row_idx + 1}", fontsize=10)

        # Right column — captions as text
        caption_text = (
            f"Ground truth:\n{ground_truth}\n\n"
            f"Generated:\n{generated}"
        )
        ax_txt.text(
            0.05, 0.5, caption_text,
            transform=ax_txt.transAxes,
            fontsize=10,
            verticalalignment="center",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        )
        ax_txt.axis("off")

    plt.suptitle("Image Captioning — Sample Predictions", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Sample captions saved → {save_path}")
    else:
        plt.show()
