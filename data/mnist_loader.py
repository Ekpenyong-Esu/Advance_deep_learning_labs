"""
data/mnist_loader.py — MNIST Data Loader
=========================================
MNIST is the classic handwritten-digit dataset.

  Training set : 60,000 grayscale images (28×28 px)
  Test set     :  10,000 images
  Classes      : digits 0 – 9

Usage
-----
    from data.mnist_loader import get_mnist_loaders
    train_loader, test_loader = get_mnist_loaders()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
    test_loader  : DataLoader
    """

    # Normalise to mean=0.5, std=0.5  (maps pixel values from [0,1] → [-1,1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.MNIST(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=config.DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
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

    print(f"MNIST ready  |  train: {len(train_dataset):,}  |  test: {len(test_dataset):,}  "
          f"|  image size: 28×28  |  channels: 1 (grayscale)")

    return train_loader, test_loader
