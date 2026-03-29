"""
experiments/task01_cnn_adam_leakyrelu.py
=========================================
Task 0.1 — Experiment 2
  Dataset   : CIFAR-10
  Model     : Simple CNN            (same architecture as Experiment 1)
  Activation: LeakyReLU             (same as Experiment 1)
  Optimiser : Adam  lr=0.0001       ← CHANGED from SGD

Adam adapts the learning rate per parameter using first- and second-moment
estimates of the gradients.  It usually converges faster than vanilla SGD
and is less sensitive to the initial learning-rate choice.

Run from the project root:
    python experiments/task01_cnn_adam_leakyrelu.py

View in TensorBoard alongside Experiment 1:
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
EXPERIMENT_NAME = "Task01_CNN_Adam_LeakyReLU"

experiment_config = {
    **config.CNN_ADAM_CONFIG,
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

    print("\nBuilding SimpleCNN (LeakyReLU) …")

    model = SimpleCNN(num_classes=10, activation="leakyrelu")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

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
    print("[Result] Optimiser: Adam | LR: 0.0001 | Activation: LeakyReLU\n")


if __name__ == "__main__":
    main()
