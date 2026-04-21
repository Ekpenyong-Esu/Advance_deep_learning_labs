"""
models/vanilla_gan.py — Vanilla GAN Generator and Discriminator (Tasks 1 & 2)
==============================================================================
Architecture from Goodfellow et al. (2014) adapted for flat MNIST images.

Generator
---------
  z  (z_dim)  →  Linear → ReLU  →  Linear → Sigmoid  →  x̂  (x_dim)

  Output is in [0, 1] to match the ToTensor-normalised MNIST pixel range.

Discriminator
-------------
  BCE variant  (use_sigmoid=True)  — Task 1
    x  (x_dim)  →  Linear → ReLU  →  Linear → Sigmoid  →  probability  (1,)

  Logistic variant  (use_sigmoid=False)  — Task 2
    x  (x_dim)  →  Linear → ReLU  →  Linear  →  raw logit  (1,)
    BCEWithLogitsLoss applies Sigmoid internally for numerical stability.

Xavier initialisation is applied to all Linear layers in both models.
"""

import torch.nn as nn
from utils.init import xavier_init


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Maps a latent noise vector z to a flat image x̂ in [0, 1]."""

    def __init__(self, z_dim: int, h_dim: int, x_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid(),           # pixel values in [0, 1]
        )
        self.apply(xavier_init)

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Classifies a flat image as real (1) or fake (0).

    Parameters
    ----------
    use_sigmoid : bool
        True  → append Sigmoid; output is a probability in [0, 1].
                 Pair with bce_loss      (Task 1).
        False → output is a raw logit.
                 Pair with logistic_loss (Task 2).
    """

    def __init__(self, x_dim: int, h_dim: int, use_sigmoid: bool = True):
        super().__init__()
        layers = [
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.apply(xavier_init)

    def forward(self, x):
        return self.net(x)
