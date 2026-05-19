"""
experiments/task02_bert.py
===========================
Task 1.2 — BERT Fine-Tuning for Sentiment Analysis

  Model     : bert-base-uncased  (110 M parameters, 12 transformer layers)
  Pre-trained on: BookCorpus + English Wikipedia (masked LM + next-sentence)
  Fine-tuned on : Amazon product reviews  (25 K large dataset by default)
  Optimiser : Adam   lr=2e-5  weight_decay=0.01

BERT (Bidirectional Encoder Representations from Transformers) uses deep
bidirectional self-attention to build contextual word representations.
Fine-tuning replaces the pre-trained classification head with a fresh
two-class linear layer and updates all weights jointly.

Compare with task01_ann.py and task01_bilstm.py to see how a pre-trained
transformer compares to models trained from scratch.

Run from the project root:
    python experiments/task02_bert.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.transformer_loader import get_transformer_loaders
from models.bert_model       import BertSentiment
from training.trainer       import train_model
from utils.helpers          import count_parameters, save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Task02_BERT_Large"

experiment_config = {
    **config.BERT_LARGE_CONFIG,
    "device": config.DEVICE,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(dataset: str | None = None) -> dict:
    """
    Fine-tune BERT for sentiment classification.

    Parameters
    ----------
    dataset : "small" | "large" | "public" | None
        Which dataset config to use: small → BERT_SMALL_CONFIG, large → BERT_LARGE_CONFIG, public → BERT_PUBLIC_CONFIG.

    Returns
    -------
    dict with keys: best_val_accuracy, test_accuracy, test_f1
    """
    dataset_configs = {
        "small":  config.BERT_SMALL_CONFIG,
        "large":  config.BERT_LARGE_CONFIG,
        "public": config.BERT_PUBLIC_CONFIG,
    }
    chosen_dataset = dataset if dataset is not None else config.BERT_LARGE_CONFIG["dataset"]
    exp_cfg  = {**dataset_configs[chosen_dataset], "device": config.DEVICE}
    exp_name       = f"Task02_BERT_{chosen_dataset.capitalize()}"
    model_name     = exp_cfg["model_name"]

    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    else:
        print(
            "\n⚠  WARNING: No GPU detected.  BERT has 110 M parameters.\n"
            "   Training on CPU will be very slow — consider reducing 'epochs'\n"
            "   in config.py (e.g. to 1) or switching to a GPU machine.\n"
        )

    # ── Step 1: Load data ──────────────────────────────────────────────── #
    train_loader, val_loader, test_loader = get_transformer_loaders(
        model_name=model_name,
        dataset=chosen_dataset,
        batch_size=exp_cfg["batch_size"],
        max_len=exp_cfg.get("max_len", config.TRANSFORMER_MAX_LEN),
    )

    # ── Step 2: Build model ────────────────────────────────────────────── #
    print(f"\nBuilding {model_name} (pretrained on BookCorpus + Wikipedia) …")
    model = BertSentiment(model_name=model_name)

    params = count_parameters(model)
    print(f"  Total parameters    : {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")

    # ── Step 3: Train ──────────────────────────────────────────────────── #
    results = train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        test_loader     = test_loader,
        config          = exp_cfg,
        experiment_name = exp_name,
        project         = config.WANDB_PROJECT,
    )

    # ── Step 4: Save checkpoint ────────────────────────────────────────── #
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{exp_name}.pth")
    save_checkpoint(model, ckpt_path)

    print(f"[Result] Experiment       : {exp_name}")
    print(f"[Result] Best Val Accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"[Result] Test Accuracy    : {results['test_accuracy']:.2f}%")
    print(f"[Result] Test F1          : {results['test_f1']:.4f}\n")

    return results


if __name__ == "__main__":
    main()
