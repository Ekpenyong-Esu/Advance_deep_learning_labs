"""
training/optimizer.py — Optimizer Factory
==========================================
Builds the correct PyTorch optimizer from a config dict.

Supported optimizers
--------------------
  "SGD"   — stochastic gradient descent with optional momentum + weight decay
  "ADAM"  — adaptive moment estimation
  "ADAMW" — Adam with decoupled weight decay (recommended for transformers)

Usage
-----
    from training.optimizer import build_optimizer
    optimizer = build_optimizer(model, config)
"""

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create the optimiser requested in config.

    Only trainable parameters (requires_grad=True) are passed so that
    frozen layers (e.g. BERT base layers during feature extraction) are
    correctly excluded from weight updates.

    Parameters
    ----------
    model  : nn.Module
    config : dict — must contain 'optimizer' and 'learning_rate';
                    optional keys: 'momentum', 'weight_decay', 'betas', 'eps'

    Returns
    -------
    torch.optim.Optimizer
    """
    params = filter(lambda p: p.requires_grad, model.parameters())
    name   = config.get("optimizer", "Adam").upper()
    lr     = config.get("learning_rate", 0.001)

    if name == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),
        )
    elif name == "ADAMW":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.01),
        )
    elif name == "ADAM":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Use 'SGD', 'Adam', or 'AdamW'.")
