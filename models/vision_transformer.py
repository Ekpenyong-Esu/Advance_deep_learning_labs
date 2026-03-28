"""
models/vision_transformer.py — ViT and Swin Transformer for Grade 5
====================================================================
Just trying to play with transformers for vision tasks. These are large models pretrained on ImageNet,

What is a Vision Transformer (ViT)?
------------------------------------
Originally, Transformers were designed for text (NLP).
ViT splits an image into fixed-size patches (e.g. 16×16 pixels) and treats
each patch like a word token. A standard Transformer then processes the
sequence of patches using self-attention.

ViT-B/16 (used here)
  • Splits 224×224 images into 14×14 = 196 patches of size 16×16
  • Processes patches through 12 Transformer encoder layers
  • Hidden dimension: 768

What is Swin Transformer?
--------------------------
Swin Transformer introduces a hierarchical (multi-scale) design with
"shifted windows" of attention, making it more efficient and better suited
for dense-prediction tasks than the original ViT.

Swin-T (Tiny, used here)
  • Hierarchical: 4 stages of feature extraction (like a CNN)
  • Local attention windows instead of global attention → much less memory
  • Hidden dimension: 768 (at the deepest stage)

Both models are PRETRAINED on ImageNet and fine-tuned for CIFAR-10.
CIFAR-10 images (32×32) are upscaled to 224×224 for these models.

Usage
-----
    from models.vision_transformer import get_vit_model, get_swin_transformer

    vit_model  = get_vit_model(num_classes=10)
    swin_model = get_swin_transformer(num_classes=10)
"""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights, Swin_T_Weights


def get_vit_model(num_classes: int = 10) -> nn.Module:
    """
    Vision Transformer ViT-B/16 fine-tuned for CIFAR-10.

    Parameters
    ----------
    num_classes : int

    Returns
    -------
    model : nn.Module
        All parameters are trainable (full fine-tuning).
    """
    # Load the pretrained ViT-B/16 checkpoint (ImageNet weights)
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # Replace the classification head.
    # Original: Linear(768 → 1000)
    # New     : Linear(768 → num_classes)
    in_features = model.heads.head.in_features   # 768
    model.heads.head = nn.Linear(in_features, num_classes)

    _print_summary(model, "ViT-B/16")
    return model


def get_swin_transformer(num_classes: int = 10) -> nn.Module:
    """
    Swin Transformer Tiny (Swin-T) fine-tuned for CIFAR-10.

    Parameters
    ----------
    num_classes : int

    Returns
    -------
    model : nn.Module
        All parameters are trainable (full fine-tuning).
    """
    # Load the pretrained Swin-T checkpoint (ImageNet weights)
    model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

    # Replace the final classification head.
    # Original: Linear(768 → 1000)
    # New     : Linear(768 → num_classes)
    in_features = model.head.in_features   # 768
    model.head = nn.Linear(in_features, num_classes)

    _print_summary(model, "Swin-T")
    return model


def _print_summary(model: nn.Module, name: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}:")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")
