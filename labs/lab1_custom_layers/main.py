"""
Lab 1: Custom Neural Network Layers
====================================
Demonstrates building advanced reusable building blocks in PyTorch:
  - Depthwise Separable Convolution
  - Squeeze-and-Excitation (SE) Block
  - Residual Block with SE
  - A small ResNet-style model trained on CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Custom Layer Definitions
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise conv + pointwise conv.

    Reduces parameters compared to a standard convolution while preserving
    most of the representational power.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block.

    Adaptively recalibrates channel-wise feature responses by modelling
    interdependencies between channels.

    Args:
        channels: Number of input (and output) channels.
        reduction: Reduction ratio for the bottleneck FC layers.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """Pre-activation residual block with an optional SE sub-layer.

    Uses two 3×3 depthwise separable convolutions and a skip connection.
    A 1×1 projection is added automatically when ``in_channels != out_channels``
    or ``stride != 1``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution (default 1).
        use_se: Whether to include a Squeeze-and-Excitation block.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels,
                                            stride=stride, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels,
                                            stride=1, padding=1)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SmallResNet(nn.Module):
    """A lightweight ResNet-style network for CIFAR-10 (32×32 images).

    Architecture summary::

        stem  → stage1 (64ch, s1) → stage2 (128ch, s2)
              → stage3 (256ch, s2) → global-avg-pool → fc(10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(32, 64, stride=1, blocks=2)
        self.stage2 = self._make_stage(64, 128, stride=2, blocks=2)
        self.stage3 = self._make_stage(128, 256, stride=2, blocks=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, stride: int, blocks: int) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return train/test DataLoaders."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(root="./data", train=True,
                                 download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> float:
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
    return correct / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    model = SmallResNet(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        print(f"Epoch [{epoch:02d}/{epochs}]  "
              f"Loss: {train_loss:.4f}  "
              f"Test Acc: {test_acc * 100:.2f}%")

    print("Training complete.")


if __name__ == "__main__":
    main()
