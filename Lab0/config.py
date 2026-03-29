"""
config.py — Central Configuration for All Experiments
======================================================
All hyperparameters and settings live here.
Change values in this file to quickly modify experiment behaviour.
"""

import torch
from pathlib import Path

# Project root (directory containing this config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Device — automatically uses GPU if one is available
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Data settings
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = str(PROJECT_ROOT / "data_files")   # Where datasets are downloaded to
BATCH_SIZE  = 64               # Number of images per training batch
NUM_WORKERS = 2                # Parallel data-loading workers (set 0 on Windows if you get errors)

# Human-readable class labels for CIFAR-10 (for reference / debugging)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ─────────────────────────────────────────────────────────────────────────────
# Weights & Biases project name
# All experiment runs are grouped under this project at https://wandb.ai
# ─────────────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "advanced-ai-lab"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint directory — saved models are stored here
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")

# ─────────────────────────────────────────────────────────────────────────────
# Task 0.1 — Simple CNN experiments on CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

# Experiment 1: SGD optimiser + LeakyReLU
CNN_SGD_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "SGD",
    "activation":    "leakyrelu",
    "epochs":        20,
    "momentum":      0.9,   # default SGD momentum
    "weight_decay":  0.0,   # default SGD weight decay
}

# Experiment 2: Adam optimiser + LeakyReLU
CNN_ADAM_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "activation":    "leakyrelu",
    "epochs":        20,
    "betas":         (0.9, 0.999),  # default Adam betas
    "eps":           1e-8,          # default Adam epsilon
    "weight_decay":  0.0,
}

# Experiment 3 (separate file): Adam optimiser + Tanh
CNN_TANH_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "activation":    "tanh",
    "epochs":        20,
    "betas":         (0.9, 0.999),
    "eps":           1e-8,
    "weight_decay":  0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 0.2.1 — AlexNet transfer learning on CIFAR-10
# AlexNet expects 224×224 images, so CIFAR-10 images are upscaled.
# ─────────────────────────────────────────────────────────────────────────────
ALEXNET_FINETUNE_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "epochs":        10,
    "batch_size":    32,    # smaller batch because images are 224×224
    "num_classes":   10,
}

ALEXNET_FEATURE_CONFIG = {
    "learning_rate": 0.001,   # Higher LR: only the tiny FC head is trained
    "optimizer":     "Adam",
    "epochs":        5,
    "batch_size":    32,
    "num_classes":   10,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 0.2.2 — Transfer learning: MNIST → SVHN
# ─────────────────────────────────────────────────────────────────────────────
MNIST_CONFIG = {
    "learning_rate": 0.001,
    "optimizer":     "Adam",
    "epochs":        10,
    "num_classes":   10,
}

SVHN_TRANSFER_CONFIG = {
    "learning_rate": 0.0001,  # Lower LR for fine-tuning
    "optimizer":     "Adam",
    "epochs":        10,
    "num_classes":   10,
    # Set True to also download SVHN 'extra' split (~530 K extra images, ~1 GB).
    # This satisfies the Grade-5 "larger public dataset" requirement.
    "use_extra_data": True,
}

# ─────────────────────────────────────────────────────────────────────────────
# Grade 5 — Vision Transformer experiments on CIFAR-10
# Images are upscaled to 224×224 for both ViT and Swin.
# ─────────────────────────────────────────────────────────────────────────────
VIT_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "epochs":        5,
    "batch_size":    16,   # Large model — needs smaller batch
    "num_classes":   10,
    "image_size":    224,
}

SWIN_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "epochs":        5,
    "batch_size":    16,
    "num_classes":   10,
    "image_size":    224,
}
