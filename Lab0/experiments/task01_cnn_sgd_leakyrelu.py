"""
experiments/task01_cnn_sgd_leakyrelu.py
========================================
Task 0.1 — Experiment 1
  Dataset   : CIFAR-10
  Model     : Simple CNN
  Activation: LeakyReLU
  Optimiser : SGD   lr=0.0001 (all other parameters kept at defaults)

Run from the project root:
    python experiments/task01_cnn_sgd_leakyrelu.py

View in TensorBoard:
    tensorboard --logdir=runs
"""

import sys
import os
# Allow imports from the project root regardless of where this script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.cifar10_loader   import get_cifar10_loaders
from models.simple_cnn     import SimpleCNN
from training.trainer      import train_model

# ─────────────────────────────────────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Task01_CNN_SGD_LeakyReLU"

experiment_config = {
    **config.CNN_SGD_CONFIG,        # pulls learning_rate, epochs, momentum etc.
    "device": config.DEVICE,        # automatically selects GPU if available
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Step 1: Load data ─────────────────────────────────────────────────── #
    print("\nLoading CIFAR-10 …")
    train_loader, test_loader = get_cifar10_loaders(
        image_size=32,
        batch_size=config.BATCH_SIZE,
    )

    # ── Step 2: Create model ──────────────────────────────────────────────── #
    print("\nBuilding SimpleCNN (LeakyReLU) …")

    model = SimpleCNN(num_classes=10, activation="leakyrelu")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Step 3: Train ─────────────────────────────────────────────────────── #
    best_acc = train_model(
        model           = model,
        train_loader    = train_loader,
        test_loader     = test_loader,
        config          = experiment_config,
        experiment_name = EXPERIMENT_NAME,
        project         = config.WANDB_PROJECT,
    )

    print(f"[Result] Experiment : {EXPERIMENT_NAME}")
    print(f"[Result] Best Test Accuracy : {best_acc:.2f}%")
    print("[Result] Optimiser: SGD | LR: 0.0001 | Activation: LeakyReLU\n")


if __name__ == "__main__":
    main()
