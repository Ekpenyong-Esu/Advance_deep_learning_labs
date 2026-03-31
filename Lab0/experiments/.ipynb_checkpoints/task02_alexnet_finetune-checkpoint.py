"""
experiments/task02_alexnet_finetune.py
=======================================
Task 0.2.1 — AlexNet Fine-Tuning on CIFAR-10
  Pretrained on : ImageNet (1,000 classes, 224×224)
  Fine-tuned on : CIFAR-10 (10 classes)
  Mode          : ALL layers are trainable

Fine-tuning means we start from ImageNet weights and update every layer
during training on CIFAR-10.  Because the pretrained features are already
very good at recognising visual patterns, the model converges faster and
to a higher accuracy than training from scratch.

Key step: The final classifier layer (originally 4096→1000) is replaced
          with a new layer (4096→10) before training begins.

Note: AlexNet requires 224×224 input.  CIFAR-10 images are upscaled in
      the data loader.

Run from the project root:
    python experiments/task02_alexnet_finetune.py

View in TensorBoard:
    tensorboard --logdir=runs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.cifar10_loader    import get_cifar10_loaders
from models.alexnet_model   import get_alexnet_finetune
from training.trainer       import train_model

# ─────────────────────────────────────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Task02_AlexNet_FineTuning"

experiment_config = {
    **config.ALEXNET_FINETUNE_CONFIG,
    "device": config.DEVICE,
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # AlexNet requires 224×224 — use the large-image CIFAR-10 loader
    print("\nLoading CIFAR-10 (upscaled to 224×224 for AlexNet) …")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        image_size=224,
        batch_size=config.ALEXNET_FINETUNE_CONFIG["batch_size"],
    )

    print("\nBuilding AlexNet (pretrained=True, all layers trainable) …")

    model = get_alexnet_finetune(num_classes=10, pretrained=True)

    best_acc = train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        test_loader     = test_loader,
        config          = experiment_config,
        experiment_name = EXPERIMENT_NAME,
        project         = config.WANDB_PROJECT,
    )

    print(f"[Result] Experiment : {EXPERIMENT_NAME}")
    print(f"[Result] Best Test Accuracy : {best_acc:.2f}%")
    print("[Result] Mode: Fine-Tuning (all layers updated)\n")


if __name__ == "__main__":
    main()
