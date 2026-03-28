"""
data/svhn_loader.py — Street View House Numbers (SVHN) Data Loader
===================================================================
SVHN is a real-world digit recognition dataset collected from Google Street View.

  Training set : ~73,257 colour images (32×32 px)
  Extra split  : ~531,131 additional training images  ← Grade-5 "larger dataset"
  Test set     : ~26,032 images
  Classes      : digits 0 – 9

Transfer-learning mode (for MNIST → SVHN)
------------------------------------------
SVHN images are RGB (3 channels), but MNIST is grayscale (1 channel).
We provide a grayscale loader that converts SVHN to grayscale AND resizes
images to 28×28 so the same network trained on MNIST can be used directly.

Usage
-----
    from data.svhn_loader import get_svhn_loaders, get_svhn_loaders_grayscale

    # Standard colour loader (for independent SVHN experiments)
    train_loader, test_loader = get_svhn_loaders()

    # Grayscale 28×28 loader (for transfer from MNIST)
    train_loader, test_loader = get_svhn_loaders_grayscale(use_extra=True)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import config


# ─────────────────────────────────────────────────────────────────────────────
# Standard colour loader
# ─────────────────────────────────────────────────────────────────────────────

def get_svhn_loaders(batch_size: int = config.BATCH_SIZE):
    """
    Standard SVHN loader — RGB 32×32 images.

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.SVHN(
        root=config.DATA_ROOT, 
        split="train", download=True, 
        transform=transform
    )

    test_dataset = datasets.SVHN(
        root=config.DATA_ROOT, 
        split="test",  
        download=True, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
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

    print(f"SVHN (colour) ready  |  train: {len(train_dataset):,}  |  "
          f"test: {len(test_dataset):,}  |  32×32 RGB")

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Grayscale loader for MNIST → SVHN transfer learning
# ─────────────────────────────────────────────────────────────────────────────

def get_svhn_loaders_grayscale(batch_size: int = config.BATCH_SIZE,
                               use_extra: bool = False):
    """
    SVHN loader for transfer learning from MNIST.

    Images are converted to grayscale and resized to 28×28 so that
    the exact same CNN trained on MNIST can be evaluated and fine-tuned
    on SVHN without any architecture changes.

    Parameters
    ----------
    batch_size : int
    use_extra : bool
        If True, the 'extra' SVHN split (~531 K images, ~1 GB) is concatenated
        with the standard training split.  This satisfies the Grade-5
        requirement of using a larger public dataset (≈1 GB).

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    """

    # Convert RGB → grayscale, resize to match MNIST dimensions, then normalise
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # 3 channels → 1 channel
        transforms.Resize((28, 28)),                    # 32×32 → 28×28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.SVHN(
        root=config.DATA_ROOT, 
        split="train", 
        download=True, 
        transform=transform
    )

    if use_extra:
        # Download the large 'extra' split for Grade-5 (≈1 GB download)
        print("Downloading SVHN 'extra' split (~1 GB) for Grade-5 larger-dataset requirement …")
        extra_dataset = datasets.SVHN(
            root=config.DATA_ROOT, 
            split="extra", 
            download=True, 
            transform=transform
        )
        # Merge standard training + extra into one big dataset
        train_dataset = ConcatDataset([train_dataset, extra_dataset])
        
        print(f"Combined train set size: {len(train_dataset):,} images")

    test_dataset = datasets.SVHN(
        root=config.DATA_ROOT, 
        split="test", 
        download=True, 
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
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

    total_train = len(train_dataset)
    print(f"SVHN (grayscale 28×28) ready  |  train: {total_train:,}  |  "
          f"test: {len(test_dataset):,}  |  extra split: {use_extra}")

    return train_loader, test_loader
