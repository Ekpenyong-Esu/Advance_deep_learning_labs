"""
Lab 5: Transfer Learning & Fine-tuning
========================================
Demonstrates using a pretrained ResNet-18 on CIFAR-10:

Phase 1 – Feature Extraction
  Freeze the entire backbone; train only the newly added classifier head.
  Runs for ``FEATURE_EPOCHS`` epochs.

Phase 2 – Fine-tuning
  Unfreeze all parameters; train end-to-end with a smaller learning rate.
  Runs for ``FINETUNE_EPOCHS`` epochs.

Accuracy (top-1 and top-5) is reported after each epoch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_CLASSES = 10
BATCH_SIZE = 128
FEATURE_LR = 1e-3       # Phase 1 learning rate
FINETUNE_LR = 1e-4      # Phase 2 learning rate
FEATURE_EPOCHS = 5
FINETUNE_EPOCHS = 10
WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_loaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    """Return CIFAR-10 train and test DataLoaders with ImageNet normalisation.

    ResNet-18 was pretrained on ImageNet so we use ImageNet statistics here.
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=28),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_set = datasets.CIFAR10("./data", train=True, download=True,
                                 transform=train_transform)
    test_set = datasets.CIFAR10("./data", train=False, download=True,
                                transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Load pretrained ResNet-18 and replace the final FC layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the final FC layer."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def topk_accuracy(output: torch.Tensor, target: torch.Tensor,
                  topk: tuple[int, ...] = (1, 5)) -> list[float]:
    """Compute top-k accuracy for each k in ``topk``."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.unsqueeze(0).expand_as(pred))
    return [correct[:k].any(dim=0).float().sum().item() / batch_size
            for k in topk]


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device) -> float:
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
             device: torch.device) -> tuple[float, float]:
    model.eval()
    top1_correct, top5_correct = 0.0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        acc1, acc5 = topk_accuracy(out, labels, topk=(1, 5))
        top1_correct += acc1 * images.size(0)
        top5_correct += acc5 * images.size(0)
    n = len(loader.dataset)
    return top1_correct / n, top5_correct / n


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def run_phase(model: nn.Module, loader_train: DataLoader,
              loader_test: DataLoader, optimizer: optim.Optimizer,
              scheduler: optim.lr_scheduler.LRScheduler,
              criterion: nn.Module, device: torch.device,
              epochs: int, phase_name: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"{phase_name}  |  Trainable params: {count_trainable(model):,}")
    print(f"{'=' * 50}")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader_train, optimizer, criterion, device)
        top1, top5 = evaluate(model, loader_test, device)
        scheduler.step()
        print(f"  Epoch [{epoch:02d}/{epochs}]  "
              f"Loss: {loss:.4f}  "
              f"Top-1: {top1 * 100:.2f}%  "
              f"Top-5: {top5 * 100:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    model = build_model(NUM_CLASSES).to(device)

    # ---- Phase 1: Feature Extraction ----
    freeze_backbone(model)
    opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=FEATURE_LR, weight_decay=WEIGHT_DECAY)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=FEATURE_EPOCHS)
    run_phase(model, train_loader, test_loader, opt1, sched1, criterion,
              device, FEATURE_EPOCHS, "Phase 1: Feature Extraction")

    # ---- Phase 2: Fine-tuning ----
    unfreeze_all(model)
    opt2 = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    sched2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=FINETUNE_EPOCHS)
    run_phase(model, train_loader, test_loader, opt2, sched2, criterion,
              device, FINETUNE_EPOCHS, "Phase 2: Fine-tuning")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
