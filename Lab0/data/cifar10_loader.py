"""
data/cifar10_loader.py — CIFAR-10 Data Loader
==============================================
CIFAR-10 is a standard benchmark dataset of 60,000 colour images (32×32 px)
spread across 10 classes.

   Training set : 50,000 images
   This loader further splits the training set into:
    Train      : 40,000 images (80%)
    Validation : 10,000 images (20%)

    Test set     : 10,000 images
    Classes      : airplane, automobile, bird, cat, deer,
                 dog, frog, horse, ship, truck

Usage
-----
    from data.cifar10_loader import get_cifar10_loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders()          # 32×32 for CNNs
    train_loader, val_loader, test_loader = get_cifar10_loaders(224)       # 224×224 for AlexNet/ViT
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import config


def get_cifar10_loaders(image_size: int = 32, batch_size: int = config.BATCH_SIZE):
    """
    Download (if needed) and return DataLoaders for CIFAR-10.

    Parameters
    ----------
    image_size : int
        32  → standard CNN experiments (keeps original resolution)
        224 → AlexNet / ViT / Swin (models that require ImageNet resolution)
    batch_size : int
        Number of images per batch.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    """

    # ── Normalisation values ───────────────────────────────────────────────── #
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2023, 0.1994, 0.2010)

    # ── Build transforms ───────────────────────────────────────────────────── #
    if image_size == 224:
        # For pretrained models (AlexNet, ViT, etc.)
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        # For training from scratch at native 32×32
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

    # 1. Load raw training data (no transform)
    full_train_raw = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=None,
    )

    # Load test set with test_transform
    test_dataset = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform,
    )

    # 2. Split into train / validation (indices only)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_train_raw))
    val_size = len(full_train_raw) - train_size

    train_raw, val_raw = random_split(full_train_raw, [train_size, val_size], generator=generator)

    # 3. Create datasets with proper transforms + apply split indices
    train_full = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=True,
        download=False,
        transform=train_transform
    )

    val_full = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=True,
        download=False,
        transform=test_transform
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

    print(f"CIFAR-10 ready | "
          f"train: {len(train_dataset):,} | "
          f"val: {len(val_dataset):,} | "
          f"test: {len(test_dataset):,} | "
          f"image size: {image_size}×{image_size}")

    return train_loader, val_loader, test_loader