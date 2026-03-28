"""data/cifar10_loader.py — CIFAR-10 Data Loader"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_cifar10_loaders(image_size: int = 32, batch_size: int = config.BATCH_SIZE):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)
    cifar10_mean  = (0.4914, 0.4822, 0.4465)
    cifar10_std   = (0.2023, 0.1994, 0.2010)

    if image_size == 224:
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

    train_dataset = datasets.CIFAR10(root=config.DATA_ROOT, train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root=config.DATA_ROOT, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True)

    print(f"CIFAR-10 ready  |  train: {len(train_dataset):,}  |  test: {len(test_dataset):,}  |  {image_size}x{image_size}")
    return train_loader, test_loader
