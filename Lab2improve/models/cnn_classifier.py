"""
models/cnn_classifier.py — CNN MNIST Classifier (Task 4)
=========================================================
A lightweight two-block convolutional network for 28×28 greyscale images.

Architecture
------------
  Input : (B, 1, 28, 28)  — pixel values in [-1, 1]

  Block 1 : Conv2d(1  → 32, 3×3, padding=1) → ReLU → MaxPool(2×2)   → (B, 32, 14, 14)
  Block 2 : Conv2d(32 → 64, 3×3, padding=1) → ReLU → MaxPool(2×2)   → (B, 64,  7,  7)

  Classifier head:
    Flatten → Linear(64×7×7 → 128) → ReLU → Dropout(0.25) → Linear(128 → 10)

  Output  : raw logits of shape (B, 10)  — use CrossEntropyLoss for training.

Usage
-----
    from models.cnn_classifier import MNISTClassifier

    model  = MNISTClassifier()
    logits = model(images)          # images: (B, 1, 28, 28)
    preds  = logits.argmax(dim=1)   # predicted class indices
"""

import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """Two-block CNN for MNIST digit classification (outputs raw logits)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop  = nn.Dropout(p=0.25)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))    # (B, 64,  7,  7)
        x = x.view(x.size(0), -1)              # (B, 3136)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)                      # (B, 10) — raw logits
