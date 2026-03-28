"""
data/cifar10_loader.py — CIFAR-10 Data Loader
==============================================
CIFAR-10 is a standard benchmark dataset of 60,000 colour images (32×32 px)
spread across 10 classes.

  Training set : 50,000 images
  Test set     : 10,000 images
  Classes      : airplane, automobile, bird, cat, deer,
                 dog, frog, horse, ship, truck

Usage
-----
    from data.cifar10_loader import get_cifar10_loaders
    train_loader, test_loader = get_cifar10_loaders()          # 32×32 for CNNs
    train_loader, test_loader = get_cifar10_loaders(224)       # 224×224 for AlexNet/ViT
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
    test_loader  : DataLoader
    """

    # ── Normalisation values ───────────────────────────────────────────────── #
    # When using pretrained ImageNet models we must match their normalisation.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    # When training from scratch, use CIFAR-10 specific statistics.
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2023, 0.1994, 0.2010)

    # ── Build transforms ───────────────────────────────────────────────────── #
    if image_size == 224: # imagenet was trained on 224×224, so we must match that for pretrained models
        # For pretrained models (AlexNet, ViT, Swin …)
        # Data augmentation: random crop + horizontal flip
        train_transform = transforms.Compose([
            transforms.Resize(256), # standard practice for imagenet is to resize to 256 then random crop to 224
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        # No augmentation on the test set — just resize and normalise
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        # For training from scratch at original 32×32 resolution
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # pad by 4 then random crop
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

    # ── Download and wrap datasets ─────────────────────────────────────────── #
    train_dataset = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform,
    )

    # ── Create DataLoaders ─────────────────────────────────────────────────── #
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                   # shuffle so every epoch sees a different order
        num_workers=config.NUM_WORKERS,
        pin_memory=True,                # faster host → GPU transfer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,                  # no shuffling needed for evaluation
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"CIFAR-10 ready  |  train: {len(train_dataset):,}  |  "
          f"test: {len(test_dataset):,}  |  image size: {image_size}×{image_size}")

    return train_loader, test_loader
