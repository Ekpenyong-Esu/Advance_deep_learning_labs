"""
utils/helpers.py — Shared Utility Functions
"""
import os
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(model: nn.Module, path: str,
                    device: torch.device = None) -> nn.Module:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    map_location = device if device is not None else torch.device("cpu")
    state_dict   = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    print(f"Checkpoint loaded <- {path}")
    return model


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
