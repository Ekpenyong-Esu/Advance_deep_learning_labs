"""
models/cgan.py — Conditional GAN Generator and Discriminator (Task 3)
======================================================================
Extends the vanilla GAN so that both Generator and Discriminator accept
a class label as a second input.  The label is projected through a learned
embedding and concatenated with the noise / image vector before the first
Linear layer.

ConditionalGenerator
--------------------
  [z (z_dim) ‖ Embed(y) (embed_dim)]  →  Linear → ReLU  →  Linear → Sigmoid  →  x̂

ConditionalDiscriminator
------------------------
  [x (x_dim) ‖ Embed(y) (embed_dim)]  →  Linear → ReLU  →  Linear → Sigmoid  →  p

By conditioning on the label at both ends, the Generator learns to produce
images of a specific class, and the Discriminator learns whether the image
is a real example of that class or a fake one.
"""

import torch
import torch.nn as nn
from utils.init import xavier_init


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Generator
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalGenerator(nn.Module):
    """Generates a flat image conditioned on class label y."""

    def __init__(self, z_dim: int, h_dim: int, x_dim: int,
                 num_classes: int, embed_dim: int):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.net = nn.Sequential(
            nn.Linear(z_dim + embed_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid(),
        )

        self.apply(xavier_init)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)                   # (B, embed_dim)
        inp = torch.cat([z, emb], dim=1)               # (B, z_dim + embed_dim)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalDiscriminator(nn.Module):
    """Classifies a flat image as real or fake, conditioned on class label y."""

    def __init__(self, x_dim: int, h_dim: int,
                 num_classes: int, embed_dim: int):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        
        self.net = nn.Sequential(
            nn.Linear(x_dim + embed_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )
        self.apply(xavier_init)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)                   # (B, embed_dim)
        inp = torch.cat([x, emb], dim=1)               # (B, x_dim + embed_dim)
        return self.net(inp)
