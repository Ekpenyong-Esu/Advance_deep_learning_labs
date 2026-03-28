"""models/alexnet_model.py — AlexNet for Transfer Learning"""
import torch.nn as nn
import torchvision.models as models
from torchvision.models import AlexNet_Weights


def get_alexnet_finetune(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = True
    _print_param_summary(model, "AlexNet (Fine-Tuning)")
    return model


def get_alexnet_feature_extractor(num_classes: int = 10) -> nn.Module:
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    _print_param_summary(model, "AlexNet (Feature Extraction)")
    return model


def _print_param_summary(model: nn.Module, name: str) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}:  Total={total:,}  Trainable={trainable:,}  Frozen={total-trainable:,}")
