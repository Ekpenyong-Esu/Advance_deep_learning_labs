"""
models/mnist_cnn.py — CNN for MNIST and Transfer to SVHN
=========================================================
This CNN is first trained on MNIST (handwritten digits), then the learned
weights are transferred to SVHN (Street View house numbers).

Transfer learning strategy
--------------------------
MNIST  : 28×28 grayscale images (1 channel)
SVHN   : 32×32 colour images    (3 channels)

To reuse the same network architecture on both datasets we:
  • Convert SVHN images to grayscale         (3 ch → 1 ch, done in svhn_loader.py)
  • Resize SVHN images to 28×28              (done in svhn_loader.py)

This means the CNN architecture is identical for both tasks, so the weights
trained on MNIST can be loaded and directly used / fine-tuned on SVHN.

The network has two clear sections:
  features   — convolutional layers (learn visual features like edges, shapes)
  classifier — fully-connected layers (make the final digit prediction)

When doing transfer learning we optionally FREEZE features so that
only the classifier adapts to the new data (SVHN).

Usage
-----
    from models.mnist_cnn import MnistCNN

    # Create model for MNIST (default: input_size=28)
    model = MnistCNN(num_classes=10)

    # After saving and loading checkpoint:
    model.freeze_features()   # lock the convolutional extractor
    # … fine-tune only the classifier on SVHN …
"""

import torch.nn as nn


class MnistCNN(nn.Module):
    """
    CNN for single-channel (grayscale) digit images.

    Parameters
    ----------
    num_classes : int
        10 for digit recognition (both MNIST and SVHN).
    input_size : int
        28 for MNIST, 28 for SVHN after resize (see svhn_loader.py).
    """

    def __init__(self, num_classes: int = 10, input_size: int = 28):
        super(MnistCNN, self).__init__()

        # ── Convolutional feature extractor ─────────────────────────────── #
        # These layers learn low-level features (edges, curves, strokes) that
        # are shared between MNIST and SVHN digits.
        self.features = nn.Sequential(
            # Block 1: 1 channel → 32 feature maps
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),     # 28×28 → 14×14

            # Block 2: 32 → 64 feature maps
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),     # 14×14 → 7×7
        )

        # Flattened size after two MaxPool(2×2) on a 28×28 image:
        # 28 → 14 → 7  →  64 channels × 7 × 7 = 3136
        flat_size = 64 * (input_size // 4) * (input_size // 4)

        # ── Fully connected classifier ───────────────────────────────────── #
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor  shape (batch, 1, 28, 28)

        Returns
        -------
        logits : Tensor  shape (batch, num_classes)
        """
        x = self.features(x)          # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)     # (B, 3136)  — flatten
        x = self.classifier(x)        # (B, num_classes)
        return x

    # ── Transfer-learning helpers ──────────────────────────────────────── #

    def freeze_features(self):
        """
        Freeze the convolutional feature extractor.

        Call this before fine-tuning on SVHN so only the classifier adapts.
        The pretrained MNIST features stay fixed.
        """
        for param in self.features.parameters():
            param.requires_grad = False
        print("Feature extractor FROZEN — only the classifier will be updated.")

    def unfreeze_all(self):
        """Unfreeze every layer so the entire network can be fine-tuned."""
        for param in self.parameters():
            param.requires_grad = True
        print("All layers UNFROZEN — entire network will be trained.")
