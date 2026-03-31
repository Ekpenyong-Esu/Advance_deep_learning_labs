"""
experiments/grade5_transformers_cifar10.py
===========================================
Grade 5 — Multiple Transformer Models on CIFAR-10
  Models : ViT-B/16 (Vision Transformer) + Swin-T (Swin Transformer)
  Dataset: CIFAR-10  (images upscaled to 224×224)
  GPU    : Required / strongly recommended (these are large models)
  Logs   : Weights & Biases — compare both models side-by-side

Why two Transformer models?
  ViT (Vision Transformer) processes image patches as a flat sequence with
  GLOBAL self-attention across all patches — excellent at capturing
  long-range dependencies but memory-intensive.

  Swin Transformer uses LOCAL attention within shifted windows arranged in a
  HIERARCHICAL (multi-scale) structure.  It is more efficient than ViT and
  generally achieves higher accuracy on dense-prediction tasks.

  Running both lets wandb show a direct performance comparison.

Grade-5 requirements covered in this project:
  ✓ Multiple transformer models  — ViT-B/16 and Swin-T (this file)
  ✓ Larger dataset (~1 GB)       — SVHN extra split in task02_mnist_to_svhn.py
  ✓ GPU support                  — automatic via config.DEVICE
  ✓ Weights & Biases visualisation  — all experiments write to wandb

Run from the project root:
    python experiments/grade5_transformers_cifar10.py

Compare all experiments at https://wandb.ai
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.cifar10_loader       import get_cifar10_loaders
from models.vision_transformer import get_vit_model, get_swin_transformer
from training.trainer          import train_model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one transformer experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_transformer_experiment(model_name: str, model, exp_config: dict,
                                train_loader, val_loader, test_loader) -> float:
    print(f"\n{'▶' * 3}  Starting transformer experiment: {model_name}")
    best_acc = train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader
        test_loader     = test_loader,
        config          = exp_config,
        experiment_name = model_name,
        project         = config.WANDB_PROJECT,
    )
    print(f"[Result] {model_name} — Best Test Accuracy: {best_acc:.2f}%")
    return best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    
    print(f"\nDevice: {config.DEVICE}")

    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    else:
        print(
            "\n⚠  WARNING: No GPU detected.  ViT and Swin are large models;\n"
            "   training on CPU will be very slow.  Consider using a machine\n"
            "   with a CUDA-capable GPU, or reduce 'epochs' in config.py.\n"
        )

    # Both ViT and Swin require 224×224 images
    vit_batch  = config.VIT_CONFIG["batch_size"]
    swin_batch = config.SWIN_CONFIG["batch_size"]

    # ── Load CIFAR-10 at 224×224 ────────────────────────────────────────── #
    # Both models use the same resolution, so we can share one pair of loaders
    # if their batch sizes are equal; otherwise load separately.
    if vit_batch == swin_batch:
        print(f"\nLoading CIFAR-10 (224×224, batch={vit_batch}) …")
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            image_size=224, batch_size=vit_batch
        )
        vit_train, vit_val, vit_test   = train_loader, val_loader, test_loader
        
        swin_train, swin_loader, swin_test = train_loader, val_loader, test_loader
    else:
        print(f"\nLoading CIFAR-10 for ViT (224×224, batch={vit_batch}) …")
        vit_train, vit_val, vit_test = get_cifar10_loaders(
            image_size=224, batch_size=vit_batch
        )
        print(f"\nLoading CIFAR-10 for Swin (224×224, batch={swin_batch}) …")
        swin_train, swine_val, swin_test = get_cifar10_loaders(
            image_size=224, batch_size=swin_batch
        )

    results = {}

    # ── Experiment A: Vision Transformer (ViT-B/16) ─────────────────────── #
    print("\nBuilding ViT-B/16 (pretrained on ImageNet) …")

    vit_model  = get_vit_model(num_classes=10)
    vit_config = {**config.VIT_CONFIG, "device": config.DEVICE}
    vit_config.pop("image_size", None)   # trainer does not use image_size

    results["ViT-B/16"] = run_transformer_experiment(
        model_name   = "Grade5_ViT_B16_CIFAR10",
        model        = vit_model,
        exp_config   = vit_config,
        train_loader = vit_train,
        val_loader   = vit_val,
        test_loader  = vit_test,
    )

    # ── Experiment B: Swin Transformer (Swin-T) ──────────────────────────── #
    print("\nBuilding Swin-T (pretrained on ImageNet) …")
    
    swin_model  = get_swin_transformer(num_classes=10)
    swin_config = {**config.SWIN_CONFIG, "device": config.DEVICE}
    swin_config.pop("image_size", None)

    results["Swin-T"] = run_transformer_experiment(
        model_name   = "Grade5_SwinT_CIFAR10",
        model        = swin_model,
        exp_config   = swin_config,
        train_loader = swin_train,
        val_loader   = swin_val,
        test_loader  = swin_test,
    )

    # ── Summary ──────────────────────────────────────────────────────────── #
    print("\n" + "═" * 62)
    print("  Grade-5 Transformer Comparison Summary")
    print("─" * 62)
    for name, acc in results.items():
        print(f"  {name:<30} Best Test Acc: {acc:.2f}%")
    print("─" * 62)
    print("  View plots: https://wandb.ai")
    print("═" * 62)

    print(
        "\nGrade-5 checklist:\n"
        "  ✓ Multiple transformer models  → ViT-B/16 + Swin-T (this file)\n"
        "  ✓ Larger public dataset ~1 GB  → SVHN extra split"
        " (task02_mnist_to_svhn.py, use_extra_data=True)\n"
        "  ✓ GPU support                  → automatic via config.DEVICE\n"
        "  ✓ Weights & Biases visualisation → https://wandb.ai, all experiments\n"
    )


if __name__ == "__main__":
    main()
