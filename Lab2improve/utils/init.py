"""
utils/init.py — Shared Weight Initialisation
=============================================
Xavier normal initialisation for Linear layers.
Used by vanilla_gan.py and cgan.py.
"""

import torch.nn as nn


def xavier_init(module: nn.Module) -> None:
    """Apply Xavier normal initialisation to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
