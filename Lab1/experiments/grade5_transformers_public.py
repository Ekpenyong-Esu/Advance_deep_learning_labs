"""
experiments/grade5_transformers_public.py
==========================================
Grade 5 — Multiple Transformer Models on a Large Public Dataset (~1 GB)

  Datasets  : amazon_polarity from Hugging Face  (~3.6 M reviews, ~1 GB)
              Capped to PUBLIC_MAX_SAMPLES rows (set in config.py) for
              practical training speed.
  Models    : BERT-base-uncased      (110 M parameters)
              DistilBERT-base-uncased (~66 M parameters)
  Optimiser : Adam   lr=2e-5  weight_decay=0.01
  GPU       : Strongly recommended — both models are large

What does this experiment show?
---------------------------------
• Training on the full public dataset sizes tests whether the improvements
  seen on 25 K data continue to scale with 100 K+ examples.
• Running BERT and DistilBERT back-to-back on the SAME data produces a
  direct complexity / accuracy / speed comparison:
    - BERT    → highest expected accuracy (larger model, more layers)
    - DistilBERT → ~97% of BERT accuracy at 60% of the inference time
• All metrics stream live to Weights & Biases for side-by-side curve plots.

Grade-5 checklist
------------------
  ✓  Larger public dataset (~1 GB)  — amazon_polarity
  ✓  Multiple transformer models    — BERT-base + DistilBERT-base
  ✓  GPU support                    — automatic via config.DEVICE
  ✓  W&B / wandb visualisation      — all runs logged to https://wandb.ai

Run from the project root:
    python experiments/grade5_transformers_public.py

View all experiment plots at https://wandb.ai  (project: advanced-ai-lab-1)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from data.transformer_loader import get_transformer_loaders
from models.bert_model       import BertSentiment, DistilBertSentiment
from training.trainer       import train_model
from utils.helpers          import count_parameters, save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_transformer(
    model_name:      str,
    model_cls,
    exp_config:      dict,
    train_loader,
    val_loader,
    test_loader,
    experiment_name: str,
) -> dict:
    """Instantiate, train and evaluate one transformer model."""
    print(f"\n{'▶' * 3}  Starting: {experiment_name}")

    model  = model_cls(model_name=model_name)
    
    params = count_parameters(model)
    
    print(f"  Parameters — total: {params['total']:,}  "
          f"trainable: {params['trainable']:,}")

    results = train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        test_loader     = test_loader,
        config          = exp_config,
        experiment_name = experiment_name,
        project         = config.WANDB_PROJECT,
    )

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}.pth")
    
    save_checkpoint(model, ckpt_path)

    print(f"[Result] {experiment_name}")
    print(f"         Test Accuracy: {results['test_accuracy']:.2f}%  "
          f"Test F1: {results['test_f1']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> dict:
    """
    Run BERT and DistilBERT on the large public dataset and compare.

    Returns
    -------
    dict mapping model name → results dict
    """
    print(f"\nDevice: {config.DEVICE}")
    
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    else:
        print(
            "\n⚠  WARNING: No GPU detected.  Both BERT and DistilBERT are large.\n"
            "   Reduce 'epochs' in BERT_PUBLIC_CONFIG / DISTILBERT_PUBLIC_CONFIG\n"
            "   in config.py (e.g. to 1) before running on CPU.\n"
        )

    results = {}

    # ── Experiment A: BERT on public dataset ─────────────────────────── #
    bert_cfg  = {**config.BERT_PUBLIC_CONFIG, "device": config.DEVICE}
    
    bert_name = bert_cfg["model_name"]

    print(f"\nLoading public dataset for BERT ({bert_name}) …")
    
    bert_train, bert_val, bert_test = get_transformer_loaders(
        model_name=bert_name,
        dataset="public",
        batch_size=bert_cfg["batch_size"],
        max_len=bert_cfg.get("max_len", config.TRANSFORMER_MAX_LEN),
    )

    results["BERT"] = _run_transformer(
        model_name      = bert_name,
        model_cls       = BertSentiment,
        exp_config      = bert_cfg,
        train_loader    = bert_train,
        val_loader      = bert_val,
        test_loader     = bert_test,
        experiment_name = "Grade5_BERT_Public",
    )

    # ── Experiment B: DistilBERT on public dataset ───────────────────── #
    dbert_cfg  = {**config.DISTILBERT_PUBLIC_CONFIG, "device": config.DEVICE}
    
    dbert_name = dbert_cfg["model_name"]

    print(f"\nLoading public dataset for DistilBERT ({dbert_name}) …")
    
    dbert_train, dbert_val, dbert_test = get_transformer_loaders(
        model_name=dbert_name,
        dataset="public",
        batch_size=dbert_cfg["batch_size"],
        max_len=dbert_cfg.get("max_len", config.TRANSFORMER_MAX_LEN),
    )

    results["DistilBERT"] = _run_transformer(
        model_name      = dbert_name,
        model_cls       = DistilBertSentiment,
        exp_config      = dbert_cfg,
        train_loader    = dbert_train,
        val_loader      = dbert_val,
        test_loader     = dbert_test,
        experiment_name = "Grade5_DistilBERT_Public",
    )

    # ── Summary ──────────────────────────────────────────────────────── #
    print("\n" + "═" * 62)
    print("  Grade-5 Transformer Comparison on Public Dataset")
    print("─" * 62)
    for name, res in results.items():
        print(
            f"  {name:<20}  Test Acc: {res['test_accuracy']:.2f}%  "
            f"Test F1: {res['test_f1']:.4f}"
        )
    print("─" * 62)
    print("  View all charts: https://wandb.ai  (project: advanced-ai-lab-1)")
    print("═" * 62)

    print(
        "\nGrade-5 checklist:\n"
        f"  ✓  Larger public dataset (~1 GB)  — {config.PUBLIC_DATASET_NAME}"
        f" (up to {config.PUBLIC_MAX_SAMPLES:,} samples used for training)\n"
        "  ✓  Multiple transformer models   — BERT-base-uncased + DistilBERT-base-uncased\n"
        "  ✓  GPU support                   — automatic via config.DEVICE\n"
        "  ✓  W&B visualisation             — https://wandb.ai  (advanced-ai-lab-1)\n"
    )

    return results


if __name__ == "__main__":
    main()
