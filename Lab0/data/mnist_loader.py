"""
data/mnist_loader.py — MNIST Data Loader
=========================================
MNIST is the classic handwritten-digit dataset.

  Training set : 60,000 grayscale images (28×28 px)
  This loader further splits the training set into:
    Train      : 48,000 images (80%)
    Validation : 12,000 images (20%)
  Test set     :  10,000 images
  Classes      : digits 0 – 9

Usage
-----
    from data.mnist_loader import get_mnist_loaders
    train_loader, val_loader, test_loader = get_mnist_loaders()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import config


def get_mnist_loaders(batch_size: int = config.BATCH_SIZE):
    """
    Download (if needed) and return DataLoaders for MNIST.

    Parameters
    ----------
    batch_size : int

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    """

    # Normalise to mean=0.5, std=0.5  (maps [0,1] → [-1,1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # 1. Load raw training data without transform
    full_train_raw = datasets.MNIST(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=None,          # No transform yet
    )

    # Load test set with transform
    test_dataset = datasets.MNIST(
        root=config.DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    # 2. Split the raw dataset (indices only)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_train_raw))
    val_size = len(full_train_raw) - train_size

    train_raw, val_raw = random_split(full_train_raw, [train_size, val_size], generator=generator)

    # 3. Apply transform to train and validation separately
    train_full = datasets.MNIST(
        root=config.DATA_ROOT,
        train=True,
        download=False,
        transform=transform
    )

    val_full = datasets.MNIST(
        root=config.DATA_ROOT,
        train=True,
        download=False,
        transform=transform
    )

    train_dataset = Subset(train_full, train_raw.indices)
    val_dataset   = Subset(val_full,   val_raw.indices)

    # ── Create DataLoaders ─────────────────────────────────────────────────── #
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"MNIST ready | "
          f"train: {len(train_dataset):,} | "
          f"val: {len(val_dataset):,} | "
          f"test: {len(test_dataset):,} | "
          f"image size: 28×28 | channels: 1 (grayscale)")

    return train_loader, val_loader, test_loader