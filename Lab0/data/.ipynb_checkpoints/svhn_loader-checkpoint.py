"""data/svhn_loader.py — SVHN Data Loader"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import config


def get_svhn_loaders(batch_size: int = config.BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.SVHN(root=config.DATA_ROOT, split="train", download=True, transform=transform)
    test_dataset  = datasets.SVHN(root=config.DATA_ROOT, split="test",  download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                               num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"SVHN (colour) ready  |  train: {len(train_dataset):,}  |  test: {len(test_dataset):,}  |  32x32 RGB")
    return train_loader, test_loader


def get_svhn_loaders_grayscale(batch_size: int = config.BATCH_SIZE, use_extra: bool = False):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.SVHN(root=config.DATA_ROOT, split="train", download=True, transform=transform)
    if use_extra:
        print("Downloading SVHN 'extra' split (~1 GB) ...")
        extra_dataset = datasets.SVHN(root=config.DATA_ROOT, split="extra", download=True, transform=transform)
        train_dataset = ConcatDataset([train_dataset, extra_dataset])
        print(f"Combined train set size: {len(train_dataset):,} images")
    test_dataset = datasets.SVHN(root=config.DATA_ROOT, split="test", download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"SVHN (grayscale 28x28) ready  |  train: {len(train_dataset):,}  |  test: {len(test_dataset):,}  |  extra: {use_extra}")
    return train_loader, test_loader
