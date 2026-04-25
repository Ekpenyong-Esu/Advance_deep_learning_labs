"""
config.py — Central Configuration for All Lab 2 Experiments
============================================================
All hyperparameters and task-specific settings live here.
Edit this file to adjust any experiment without touching model or training code.
"""

import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent
MNIST_ROOT     = str(PROJECT_ROOT.parent / "MNIST")   # shared MNIST cache
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")
OUTPUT_DIR     = str(PROJECT_ROOT / "outputs")

# ─────────────────────────────────────────────────────────────────────────────
# Device — automatically selects GPU when available
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Weights & Biases
# ─────────────────────────────────────────────────────────────────────────────

WANDB_PROJECT = "advanced-ai-lab-2"

# ─────────────────────────────────────────────────────────────────────────────
# Shared GAN settings
# ─────────────────────────────────────────────────────────────────────────────

GAN_BATCH_SIZE = 64
GAN_Z_DIM      = 100    # latent dimension
GAN_H_DIM      = 256    # hidden layer width
GAN_X_DIM      = 784    # 28 × 28 flattened MNIST image
GAN_LR         = 2e-4

# Epochs at which comparison samples are saved (Task 1 vs Task 2)
COMPARISON_EPOCHS = {5, 10, 50}

# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Vanilla GAN with standard BCE loss
# ─────────────────────────────────────────────────────────────────────────────

VANILLA_GAN_CONFIG = {
    "z_dim":      GAN_Z_DIM,
    "h_dim":      GAN_H_DIM,
    "x_dim":      GAN_X_DIM,
    "lr":         GAN_LR,
    "batch_size": GAN_BATCH_SIZE,
    "epochs":     50,
    "loss":       "bce",            # Discriminator uses Sigmoid output
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Vanilla GAN with Logistic (BCEWithLogits) loss
#
# Key difference: the Discriminator returns raw logits (no Sigmoid).
# BCEWithLogitsLoss applies the Sigmoid internally for numerical stability,
# matching the logistic non-saturating loss described on Brandon Amos's blog.
# ─────────────────────────────────────────────────────────────────────────────

LOGISTIC_GAN_CONFIG = {
    "z_dim":      GAN_Z_DIM,
    "h_dim":      GAN_H_DIM,
    "x_dim":      GAN_X_DIM,
    "lr":         GAN_LR,
    "batch_size": GAN_BATCH_SIZE,
    "epochs":     50,
    "loss":       "logistic",       # Discriminator returns logits; no Sigmoid
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Conditional GAN (cGAN)
# ─────────────────────────────────────────────────────────────────────────────

CGAN_NUM_CLASSES  = 10
CGAN_EMBED_DIM    = 50      # class-embedding dimension
CGAN_TARGET_DIGIT = 3       # digit shown in saved sample grids

CGAN_CONFIG = {
    "z_dim":       GAN_Z_DIM,
    "h_dim":       GAN_H_DIM,
    "x_dim":       GAN_X_DIM,
    "num_classes": CGAN_NUM_CLASSES,
    "embed_dim":   CGAN_EMBED_DIM,
    "lr":          GAN_LR,
    "batch_size":  GAN_BATCH_SIZE,
    "epochs":      50,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — CNN Classifier + FGSM Adversarial Attack
# ─────────────────────────────────────────────────────────────────────────────

CNN_CONFIG = {
    "lr":                     1e-3,
    "batch_size":             64,
    "epochs":                 20,
    "early_stopping_patience": 3,   # stop if loss does not improve for N epochs
}

ADVERSARIAL_CONFIG = {
    "source_class": 4,      # digit to perturb
    "target_class": 9,      # desired mis-classification label
    "epsilon":      0.3,    # FGSM perturbation magnitude
    "num_samples":  10,     # how many adversarial examples to produce
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Conditional Diffusion Model (DDPM)
# ─────────────────────────────────────────────────────────────────────────────

DIFFUSION_CONFIG = {
    "timesteps":    1000,
    "beta_start":   1e-4,
    "beta_end":     0.02,
    "lr":           2e-4,
    "batch_size":   128,   # ← increased from 64
    "epochs":       500,
    "num_classes":  10,
    "t_emb_dim":    256,   # ← doubled from 128
    "h_dim":        1024,  # ← doubled from 512
    "target_digit": 3,
}