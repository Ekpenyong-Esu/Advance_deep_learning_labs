"""
models/simple_cnn.py — Simple CNN for CIFAR-10
===============================================
Architecture overview
---------------------
Input  : (batch, 3, 32, 32)  — CIFAR-10 colour image

  Conv Block 1 : Conv2d(3→32)  + BatchNorm + Activation + MaxPool  → (batch, 32, 16, 16)
  Conv Block 2 : Conv2d(32→64) + BatchNorm + Activation + MaxPool  → (batch, 64,  8,  8)
  Conv Block 3 : Conv2d(64→128)+ BatchNorm + Activation + MaxPool  → (batch,128,  4,  4)
  Flatten      :                                                      (batch, 2048)
  FC 1         : Linear(2048→512) + Dropout + Activation
  FC 2         : Linear(512→256)  + Dropout + Activation
  FC 3 (output): Linear(256→10)

The activation function is configurable: pass 'leakyrelu' or 'tanh'.

Usage
-----
    from models.simple_cnn import SimpleCNN
    model = SimpleCNN(num_classes=10, activation='leakyrelu')
"""

import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build an activation function by name
# ─────────────────────────────────────────────────────────────────────────────

def _make_activation(name: str) -> nn.Module:
    """
    Return a fresh activation-function module by name.

    We call this every time we need one so each layer gets its own module
    instance — required by PyTorch when the same layer type appears in
    multiple Sequential blocks.
    """
    name = name.lower()
    if name == "leakyrelu":
        # LeakyReLU lets a small gradient flow for negative inputs,
        # which helps avoid the "dying ReLU" problem.
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "tanh":
        # Tanh squashes values to (−1, 1); historically popular, smooth gradient.
        return nn.Tanh()
    else:
        raise ValueError(
            f"Unknown activation '{name}'. Choose 'leakyrelu' or 'tanh'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Configurable CNN for image classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (10 for CIFAR-10).
    activation : str
        Which activation function to use: 'leakyrelu' or 'tanh'.
    """

    def __init__(self, num_classes: int = 10, activation: str = "leakyrelu"):
        super(SimpleCNN, self).__init__()

        act = activation   # shorthand used in _make_activation calls below

        # ── Convolutional feature extractor ─────────────────────────────── #
        # Each block: Conv → BatchNorm → Activation → MaxPool
        #
        # BatchNorm stabilises training by normalising layer outputs;
        # MaxPool halves the spatial dimensions (width and height).

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            _make_activation(act),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32×32 → 16×16
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            _make_activation(act),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16×16 → 8×8
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            _make_activation(act),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 8×8 → 4×4
        )

        # ── Fully connected classifier ───────────────────────────────────── #
        # After 3 MaxPool(2×2) operations on a 32×32 input:
        #   32 → 16 → 8 → 4   (spatial dimension)
        # Flattened size = 128 channels × 4 × 4 = 2048

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),              # randomly zero 50% of activations to prevent overfitting
            nn.Linear(128 * 4 * 4, 512),
            _make_activation(act),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            _make_activation(act),
            nn.Linear(256, num_classes),    # raw scores (logits) for each class
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor  shape (batch, 3, 32, 32)

        Returns
        -------
        logits : Tensor  shape (batch, num_classes)
        """
        x = self.conv_block1(x)          # (B,  32, 16, 16)
        x = self.conv_block2(x)          # (B,  64,  8,  8)
        x = self.conv_block3(x)          # (B, 128,  4,  4)
        x = x.view(x.size(0), -1)        # (B, 2048)  — flatten
        x = self.classifier(x)           # (B, num_classes)
        return x
