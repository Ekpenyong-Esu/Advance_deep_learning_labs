"""
experiments/task02_mnist_to_svhn.py
=====================================
Task 0.2.2 — Transfer Learning: MNIST → SVHN

This experiment has two stages:

  Stage 1 — Train on MNIST
  ─────────────────────────
  Train a CNN from scratch on MNIST (handwritten greyscale digits, 28×28).
  Save the trained weights to disk.

  Stage 2 — Transfer to SVHN
  ────────────────────────────
  Load the MNIST weights.
  Freeze the convolutional feature extractor.
  Fine-tune only the classifier on SVHN (street-sign colour digits).

  SVHN images are converted to greyscale and resized to 28×28 in
  svhn_loader.py so the same network architecture fits without changes.

  Setting use_extra=True in SVHN_TRANSFER_CONFIG downloads the 'extra'
  SVHN split (~531 K additional images, ~1 GB) — this satisfies the
  Grade-5 "larger public dataset" requirement.

Why does performance differ on SVHN vs MNIST?
  MNIST digits are clean, centred, white-on-black.  SVHN digits appear
  inside cluttered street-view photos with variable lighting, perspective
  distortion, and multiple overlapping digits.  The CNN's MNIST features
  are still useful (same digit shapes), but the mismatch causes a drop in
  accuracy.  You can reduce this gap by unfreezing all layers (full fine-tuning)
  or by training for more epochs.

Run from the project root:
    python experiments/task02_mnist_to_svhn.py

View in TensorBoard:
    tensorboard --logdir=runs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.mnist_loader  import get_mnist_loaders
from data.svhn_loader   import get_svhn_loaders_grayscale
from models.mnist_cnn   import MnistCNN
from training.trainer   import train_model
from utils.helpers      import save_checkpoint, load_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint path — where the MNIST-trained weights are saved
# ─────────────────────────────────────────────────────────────────────────────
MNIST_CHECKPOINT = os.path.join(config.CHECKPOINT_DIR, "mnist_cnn.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Train on MNIST
# ─────────────────────────────────────────────────────────────────────────────

def stage1_train_mnist():
    print("\n" + "═" * 62)
    print("  STAGE 1 — Train CNN on MNIST")
    print("═" * 62)

    train_loader, test_loader = get_mnist_loaders(batch_size=config.BATCH_SIZE)

    model = MnistCNN(num_classes=10, input_size=28)
    total = sum(p.numel() for p in model.parameters())
    print(f"MnistCNN parameters: {total:,}")

    mnist_config = {
        **config.MNIST_CONFIG,
        "device": config.DEVICE,
    }

    best_acc = train_model(
        model           = model,
        train_loader    = train_loader,
        test_loader     = test_loader,
        config          = mnist_config,
        experiment_name = "Task02_MNIST_Training",
        project         = config.WANDB_PROJECT,
    )

    # Save weights for Stage 2
    save_checkpoint(model, MNIST_CHECKPOINT)

    print(f"[Stage 1 Result] MNIST Test Accuracy : {best_acc:.2f}%\n")
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Transfer to SVHN
# ─────────────────────────────────────────────────────────────────────────────

def stage2_transfer_svhn():
    print("\n" + "═" * 62)
    print("  STAGE 2 — Transfer CNN from MNIST to SVHN")
    print("═" * 62)

    use_extra = config.SVHN_TRANSFER_CONFIG.get("use_extra_data", False)
    train_loader, test_loader = get_svhn_loaders_grayscale(
        batch_size=config.BATCH_SIZE,
        use_extra=use_extra,
    )

    # Re-create the architecture and load MNIST weights
    model = MnistCNN(num_classes=10, input_size=28)
    model = load_checkpoint(model, MNIST_CHECKPOINT, device=config.DEVICE)

    # Freeze the feature extractor — only the classifier adapts to SVHN
    model.freeze_features()

    svhn_config = {
        **config.SVHN_TRANSFER_CONFIG,
        "device": config.DEVICE,
    }
    # Remove the non-standard key before passing to trainer
    svhn_config.pop("use_extra_data", None)

    best_acc = train_model(
        model           = model,
        train_loader    = train_loader,
        test_loader     = test_loader,
        config          = svhn_config,
        experiment_name = "Task02_SVHN_Transfer",
        project         = config.WANDB_PROJECT,
    )

    print(f"[Stage 2 Result] SVHN Test Accuracy (after transfer) : {best_acc:.2f}%\n")
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    mnist_acc = stage1_train_mnist()
    svhn_acc  = stage2_transfer_svhn()

    print("═" * 62)
    print("  Task 0.2.2 Summary")
    print("─" * 62)
    print(f"  MNIST accuracy (source task)           : {mnist_acc:.2f}%")
    print(f"  SVHN accuracy  (after transfer learning): {svhn_acc:.2f}%")
    print("═" * 62)
    print(
        "\nExpected behaviour:\n"
        "  MNIST accuracy is typically 99%+.\n"
        "  SVHN accuracy after transfer is lower (often 60–80%) because\n"
        "  SVHN images are noisier and harder than MNIST.\n"
        "  To improve SVHN accuracy, try un-freezing all layers (full fine-tuning).\n"
    )


if __name__ == "__main__":
    main()
