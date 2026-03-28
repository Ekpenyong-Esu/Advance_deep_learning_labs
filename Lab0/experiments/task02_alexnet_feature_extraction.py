"""
experiments/task02_alexnet_feature_extraction.py
=================================================
Task 0.2.1 — AlexNet as Feature Extractor on CIFAR-10
  Pretrained on : ImageNet
  Tested on     : CIFAR-10 (10 classes)
  Mode          : Backbone FROZEN — only the final FC layer is trained

Feature extraction means the convolutional backbone acts as a fixed
feature-extractor.  We only train the new output layer (4096 → 10).

Why is this slower to improve vs fine-tuning?
  The backbone learned ImageNet features.  Those features are good but not
  optimised for CIFAR-10's low-resolution 10-class images.  Because we
  cannot update the backbone, accuracy is limited by how well ImageNet
  features transfer.  Fine-tuning overcomes this by adapting every layer.

Run from the project root:
    python experiments/task02_alexnet_feature_extraction.py

Compare with fine-tuning on TensorBoard:
    tensorboard --logdir=runs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.cifar10_loader  import get_cifar10_loaders
from models.alexnet_model import get_alexnet_feature_extractor
from training.trainer     import train_model

# ─────────────────────────────────────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Task02_AlexNet_FeatureExtraction"

experiment_config = {
    **config.ALEXNET_FEATURE_CONFIG,
    "device": config.DEVICE,
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print("\nLoading CIFAR-10 (upscaled to 224×224 for AlexNet) …")
    
    train_loader, test_loader = get_cifar10_loaders(
        image_size=224,
        batch_size=config.ALEXNET_FEATURE_CONFIG["batch_size"],
    )

    print("\nBuilding AlexNet (backbone frozen, only FC head trains) …")

    model = get_alexnet_feature_extractor(num_classes=10)

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
    print("[Result] Mode: Feature Extraction (backbone frozen)\n")
    print(
        "Why is feature extraction usually lower than fine-tuning?\n"
        "  AlexNet's backbone learned ImageNet patterns (1000 classes, high-res).\n"
        "  CIFAR-10 images are only 32×32 with 10 classes — a different distribution.\n"
        "  Because we cannot adjust the backbone, some feature mismatch remains.\n"
        "  Fine-tuning resolves this by updating every layer on CIFAR-10 data."
    )


if __name__ == "__main__":
    main()
