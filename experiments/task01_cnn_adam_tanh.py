"""
experiments/task01_cnn_adam_tanh.py
=====================================
Task 0.1 — Experiment 3  (required separate file)
  Dataset   : CIFAR-10
  Model     : Simple CNN
  Activation: Tanh          ← CHANGED from LeakyReLU
  Optimiser : Adam  lr=0.0001

Tanh squashes activations to the range (−1, 1), giving smooth, symmetric
gradients.  Historically it was the default choice before ReLU variants
became dominant.  It can sometimes cause "vanishing gradients" in very
deep networks because the gradient near saturation (±1) approaches zero.

This experiment lets you compare Tanh vs LeakyReLU directly on TensorBoard.

Run from the project root:
    python experiments/task01_cnn_adam_tanh.py

View all three Task 0.1 experiments side-by-side in TensorBoard:
    tensorboard --logdir=runs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.cifar10_loader import get_cifar10_loaders
from models.simple_cnn   import SimpleCNN
from training.trainer    import train_model

# ─────────────────────────────────────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Task01_CNN_Adam_Tanh"

experiment_config = {
    **config.CNN_TANH_CONFIG,
    "device": config.DEVICE,
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print("\nLoading CIFAR-10 …")
    train_loader, test_loader = get_cifar10_loaders(
        image_size=32,
        batch_size=config.BATCH_SIZE,
    )

    # NOTE: The only change from task01_cnn_adam_leakyrelu.py is activation="tanh"
    print("\nBuilding SimpleCNN (Tanh) …")
    model = SimpleCNN(num_classes=10, activation="tanh")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    best_acc = train_model(
        model           = model,
        train_loader    = train_loader,
        test_loader     = test_loader,
        config          = experiment_config,
        experiment_name = EXPERIMENT_NAME,
        log_dir         = config.TENSORBOARD_LOG_DIR,
    )

    print(f"[Result] Experiment : {EXPERIMENT_NAME}")
    print(f"[Result] Best Test Accuracy : {best_acc:.2f}%")
    print("[Result] Optimiser: Adam | LR: 0.0001 | Activation: Tanh\n")


if __name__ == "__main__":
    main()
