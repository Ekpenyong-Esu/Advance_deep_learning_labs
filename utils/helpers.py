"""
utils/helpers.py — Shared Utility Functions
============================================
Provides model checkpoint save / load and a parameter counter.
Used by the transfer-learning experiments to persist a model trained
on one dataset and reload it for another.
"""

import os
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: str) -> None:
    """
    Save model weights to a file.

    Parameters
    ----------
    model : nn.Module
    path  : str   — file path ending in .pth  (e.g. "checkpoints/mnist_cnn.pth")
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(model: nn.Module, path: str,
                    device: torch.device = None) -> nn.Module:
    """
    Load model weights from a checkpoint file.

    Parameters
    ----------
    model  : nn.Module   — must have the same architecture as when saved
    path   : str         — path to the .pth file
    device : torch.device (optional)

    Returns
    -------
    model : nn.Module with loaded weights
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else torch.device("cpu")
    state_dict   = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    print(f"Checkpoint loaded ← {path}")
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Returns
    -------
    dict with keys 'total' and 'trainable'
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
