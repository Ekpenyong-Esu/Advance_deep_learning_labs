"""models/vision_transformer.py — ViT and Swin Transformer for Grade 5"""
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights, Swin_T_Weights


def get_vit_model(num_classes: int = 10) -> nn.Module:
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    _print_summary(model, "ViT-B/16")
    return model


def get_swin_transformer(num_classes: int = 10) -> nn.Module:
    model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    _print_summary(model, "Swin-T")
    return model


def _print_summary(model: nn.Module, name: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}:  Total={total:,}  Trainable={trainable:,}")
