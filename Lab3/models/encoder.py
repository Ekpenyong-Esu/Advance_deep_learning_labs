"""
models/encoder.py — CNN Image Encoder
======================================
Wraps a pretrained ResNet-50 to extract a fixed-size feature vector for
each image.  The final classification head of ResNet-50 is replaced by a
projection layer that maps 2048-dim pool features to ``embed_dim``.

Architecture
------------
  Image (3, 224, 224)
    → ResNet-50 backbone  (pretrained on ImageNet, last FC removed)
    → AdaptiveAvgPool2d   → (2048,)
    → Linear(2048, embed_dim)
    → BatchNorm1d(embed_dim)
    → ReLU
  Output: (batch, embed_dim)

Fine-tuning
-----------
When ``fine_tune=True`` the parameters of the last two ResNet layer groups
(``layer3`` and ``layer4``) are unfrozen and trained along with the
projection head.  All earlier layers remain frozen to preserve low-level
ImageNet features and reduce training time.

When ``fine_tune=False`` (default) only the projection head is trained.

Usage
-----
    from models.encoder import ImageEncoder

    encoder = ImageEncoder(embed_dim=256, fine_tune=False)
    features = encoder(images)   # (batch, 256)
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """
    ResNet-50 backbone with a learnable projection head.

    Parameters
    ----------
    embed_dim : int
        Dimension of the output feature vector (matches the decoder's
        word-embedding dimension for alignment).
    fine_tune : bool
        If True, unfreeze layer3 and layer4 of ResNet-50 for end-to-end
        fine-tuning.  Defaults to False.
    """

    def __init__(self, embed_dim: int, fine_tune: bool = False):
        super().__init__()

        # ── Load pretrained ResNet-50 ────────────────────────────────────── #
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the original average-pool and fc classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Global average pool: (batch, 2048, 7, 7) → (batch, 2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head: 2048 → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # ── Freeze / unfreeze backbone layers ───────────────────────────── #
        self._freeze_backbone()
        if fine_tune:
            self.fine_tune(True)

    # ── forward pass ─────────────────────────────────────────────────────── #

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : FloatTensor  (batch, 3, 224, 224)

        Returns
        -------
        FloatTensor  (batch, embed_dim)
        """
        features = self.backbone(images)        # (batch, 2048, 7, 7)
        features = self.pool(features)          # (batch, 2048, 1, 1)
        features = features.flatten(start_dim=1)  # (batch, 2048)
        features = self.projection(features)    # (batch, embed_dim)
        return features

    # ── fine-tuning helpers ──────────────────────────────────────────────── #

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def fine_tune(self, enable: bool = True) -> None:
        """
        Enable or disable gradient computation for layer3 and layer4
        of the ResNet-50 backbone.

        Parameters
        ----------
        enable : bool
            True  → unfreeze layer3 + layer4 (fine-tuning on).
            False → re-freeze them.
        """
        # ResNet children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        # Indices in self.backbone Sequential: 0-7
        FINETUNE_FROM = 6   # layer3 starts at index 6
        for i, child in enumerate(self.backbone.children()):
            if i >= FINETUNE_FROM:
                for param in child.parameters():
                    param.requires_grad = enable
