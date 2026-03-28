"""
config.py — Central Configuration for All Experiments
======================================================
All hyperparameters and settings live here.
Change values in this file to quickly modify experiment behaviour.
"""

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Device — automatically uses GPU if one is available
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Data settings
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = "./data_files"
BATCH_SIZE  = 64
NUM_WORKERS = 2

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard log directory
# ─────────────────────────────────────────────────────────────────────────────
TENSORBOARD_LOG_DIR = "./runs"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint directory
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "./checkpoints"

# ─────────────────────────────────────────────────────────────────────────────
# Task 0.1 — Simple CNN experiments on CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────
CNN_SGD_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "SGD",
    "activation":    "leakyrelu",
    "epochs":        20,
    "momentum":      0.9,
    "weight_decay":  0.0,
}

CNN_ADAM_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "activation":    "leakyrelu",
    "epochs":        20,
    "betas":         (0.9, 0.999),
    "eps":           1e-8,
    "weight_decay":  0.0,
}

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
# ─────────────────────────────────────────────────────────────────────────────
ALEXNET_FINETUNE_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "epochs":        10,
    "batch_size":    32,
    "num_classes":   10,
}

ALEXNET_FEATURE_CONFIG = {
    "learning_rate": 0.001,
    "optimizer":     "Adam",
    "epochs":        5,
    "batch_size":    32,
    "num_classes":   10,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 0.2.2 — Transfer learning: MNIST -> SVHN
# ─────────────────────────────────────────────────────────────────────────────
MNIST_CONFIG = {
    "learning_rate": 0.001,
    "optimizer":     "Adam",
    "epochs":        10,
    "num_classes":   10,
}

SVHN_TRANSFER_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "epochs":        10,
    "num_classes":   10,
    "use_extra_data": True,
}

# ─────────────────────────────────────────────────────────────────────────────
# Grade 5 — Vision Transformer experiments on CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────
VIT_CONFIG = {
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "epochs":        5,
    "batch_size":    16,
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
