"""
models/alexnet_model.py — AlexNet for Transfer Learning on CIFAR-10
====================================================================
AlexNet was the breakthrough CNN that won ImageNet 2012.
It was trained on 1,000 classes from the ImageNet dataset (224×224 images).
We adapt it for CIFAR-10 (10 classes) in two different ways:

  1. Fine-tuning (Task 0.2.1 experiment A)
     ─────────────────────────────────────
     ALL layers are unlocked and updated during training.
     The whole network learns to specialise for CIFAR-10.
     Slower but typically achieves higher accuracy.

  2. Feature Extraction (Task 0.2.1 experiment B)
     ───────────────────────────────────────────────
     The convolutional backbone is FROZEN — its weights never change.
     Only the final fully-connected output layer is trained.
     Much faster, but the features were optimised for ImageNet, not CIFAR-10.

Why is there a performance difference?
  Fine-tuning lets every layer adapt to CIFAR-10's visual patterns.
  Feature extraction forces the model to use ImageNet features "as-is",
  which are general enough to still work, but less specialised.

Note: AlexNet requires 224×224 input images.  CIFAR-10 images (32×32) must
      be upscaled using the cifar10_loader.py (image_size=224 option).

Usage
-----
    from models.alexnet_model import get_alexnet_finetune, get_alexnet_feature_extractor

    model = get_alexnet_finetune(num_classes=10)
    model = get_alexnet_feature_extractor(num_classes=10)
"""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import AlexNet_Weights


def get_alexnet_finetune(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    AlexNet prepared for FULL fine-tuning.

    All parameters are trainable, so every layer updates during training.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        If True, load ImageNet pretrained weights (recommended).

    Returns
    -------
    model : nn.Module
    """
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)

    # AlexNet's original final layer: Linear(4096 → 1000)  (for 1000 ImageNet classes)
    # We replace it with:            Linear(4096 → num_classes)
    in_features = model.classifier[6].in_features   # 4096
    model.classifier[6] = nn.Linear(in_features, num_classes)

    # All parameters are trainable (fine-tuning mode)
    for param in model.parameters():
        param.requires_grad = True

    _print_param_summary(model, "AlexNet (Fine-Tuning)")
    return model


def get_alexnet_feature_extractor(num_classes: int = 10) -> nn.Module:
    """
    AlexNet used as a FROZEN feature extractor.

    Only the newly added output layer is trained; the rest is frozen.

    Parameters
    ----------
    num_classes : int

    Returns
    -------
    model : nn.Module
    """
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Replace — and unfreeze — only the final classifier layer
    in_features = model.classifier[6].in_features   # 4096
    model.classifier[6] = nn.Linear(in_features, num_classes)
    # nn.Linear is created with requires_grad=True by default

    _print_param_summary(model, "AlexNet (Feature Extraction)")
    return model


def _print_param_summary(model: nn.Module, name: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}:")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  Frozen parameters    : {total - trainable:,}")
