"""data/mnist_loader.py — MNIST Data Loader"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_mnist_loaders(batch_size: int = config.BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST(root=config.DATA_ROOT, train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=config.DATA_ROOT, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"MNIST ready  |  train: {len(train_dataset):,}  |  test: {len(test_dataset):,}  |  28x28 grayscale")
    return train_loader, test_loader
