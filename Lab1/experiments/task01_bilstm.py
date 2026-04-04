"""
experiments/task01_bilstm.py
=============================
Task 1.1 — Bidirectional LSTM for Sentiment Analysis

  Model     : BiLSTM with learned word embeddings
              (Embedding → 2-layer BiLSTM → Dropout → Linear classifier)
  Features  : Word-index sequences (padded to 256 tokens)
  Datasets  : "small"  — 1 K Amazon product reviews
              "large"  — 25 K Amazon product reviews
  Optimiser : Adam   lr=0.001

The BiLSTM reads the full sentence in both directions, giving each
position access to past AND future context.  This makes it significantly
better than the ANN at capturing word order and long-range dependencies
that carry sentiment (e.g. negations, emphasis).

Compare results with task01_ann.py to see the benefit of sequence models
over bag-of-words approaches.

Run from the project root:
    python experiments/task01_bilstm.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.lstm_loader   import get_lstm_loaders
from models.lstm_model  import BiLSTMSentiment
from training.trainer      import train_model
from utils.helpers         import count_parameters, save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _run(dataset: str, exp_config: dict, experiment_name: str) -> dict:
    """Load data, build model, train, and return results."""
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Step 1: Load data ──────────────────────────────────────────────── #
    train_loader, val_loader, test_loader, vocab_size = get_lstm_loaders(
        dataset=dataset,
        batch_size=exp_config["batch_size"],
        max_len=config.LSTM_MAX_LEN,
        max_vocab=config.LSTM_MAX_VOCAB,
    )

    # ── Step 2: Build model ────────────────────────────────────────────── #
    print(f"\nBuilding BiLSTMSentiment  (vocab_size={vocab_size:,}) …")
    model = BiLSTMSentiment(
        vocab_size=vocab_size,
        embed_dim=exp_config.get("embed_dim", 128),
        hidden_dim=exp_config.get("hidden_dim", 256),
        num_layers=exp_config.get("num_layers", 2),
        dropout=exp_config.get("dropout", 0.3),
    )

    params = count_parameters(model)
    print(f"  Total parameters    : {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")

    # ── Step 3: Train ──────────────────────────────────────────────────── #
    results = train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        test_loader     = test_loader,
        config          = exp_config,
        experiment_name = experiment_name,
        project         = config.WANDB_PROJECT,
    )

    # ── Step 4: Save checkpoint ────────────────────────────────────────── #
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}.pth")
    save_checkpoint(model, ckpt_path)

    print(f"[Result] Experiment       : {experiment_name}")
    print(f"[Result] Best Val Accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"[Result] Test Accuracy    : {results['test_accuracy']:.2f}%")
    print(f"[Result] Test F1          : {results['test_f1']:.4f}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def main(dataset: str = "small") -> dict:
    """
    Run the BiLSTM sentiment experiment on the chosen dataset.

    Parameters
    ----------
    dataset : "small" (1 K reviews) | "large" (25 K reviews) | "public"

    Returns
    -------
    dict with keys: best_val_accuracy, test_accuracy, test_f1
    """
    dataset_configs = {
        "small":  config.BILSTM_SMALL_CONFIG,
        "large":  config.BILSTM_LARGE_CONFIG,
        "public": config.BILSTM_PUBLIC_CONFIG,
    }
    if dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'small', 'large', or 'public'.")

    exp_cfg  = {**dataset_configs[dataset], "device": config.DEVICE}
    exp_name = f"Task01_BiLSTM_{dataset.capitalize()}"

    return _run(dataset=dataset, exp_config=exp_cfg, experiment_name=exp_name)


if __name__ == "__main__":
    print("=" * 62)
    print("  Task 1.1 — BiLSTM on SMALL dataset (1 K reviews)")
    print("=" * 62)
    small_results = main("small")

    print("=" * 62)
    print("  Task 1.1 — BiLSTM on LARGE dataset (25 K reviews)")
    print("=" * 62)
    large_results = main("large")

    print("\n" + "═" * 62)
    print("  BiLSTM Dataset Comparison")
    print("─" * 62)
    for name, res in [("Small (1 K)", small_results), ("Large (25 K)", large_results)]:
        print(f"  {name:<15}  Test Acc: {res['test_accuracy']:.2f}%  "
              f"F1: {res['test_f1']:.4f}")
    print("═" * 62)
