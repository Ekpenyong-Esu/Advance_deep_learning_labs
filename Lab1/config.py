"""
config.py — Central Configuration for All Lab 1 Experiments
============================================================
All hyperparameters and settings live here.
Change values in this file to quickly modify experiment behaviour.
"""

import torch
from pathlib import Path

# Project root (directory containing this config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Device — automatically uses GPU if one is available
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset paths
# ─────────────────────────────────────────────────────────────────────────────
SMALL_DATASET_PATH  = str(PROJECT_ROOT / "amazon_cells_labelled.txt")          # 1 K rows
LARGE_DATASET_PATH  = str(PROJECT_ROOT / "amazon_cells_labelled_LARGE_25K.txt") # 25 K rows
PUBLIC_DATASET_NAME = "amazon_polarity"  # Hugging Face dataset identifier (~3.6 M rows, ~1 GB)
# Cap training samples from the public dataset for practical training speed.
# The full dataset is still downloaded (~1 GB), satisfying the Grade-5 requirement.
# Set to None to use the entire dataset.
PUBLIC_MAX_SAMPLES  = 100_000

# ─────────────────────────────────────────────────────────────────────────────
# Weights & Biases project name
# All experiment runs are grouped under this project at https://wandb.ai
# ─────────────────────────────────────────────────────────────────────────────
WANDB_PROJECT = "advanced-ai-lab-1"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint directory — saved models are stored here
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")

# ─────────────────────────────────────────────────────────────────────────────
# Common data settings
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
NUM_WORKERS  = 0          # 0 on Windows to avoid multiprocessing issues
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO is implicitly 1 − TRAIN_RATIO − VAL_RATIO = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing settings
# ─────────────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES  = 8_000   # TF-IDF vocabulary cap (used by the Simple ANN)
LSTM_MAX_VOCAB      = 30_000   # Word embedding vocabulary cap (used by the BiLSTM)
LSTM_MAX_LEN        = 256      # Max tokens per sentence for the BiLSTM
TRANSFORMER_MAX_LEN = 128      # Max tokens per sentence for BERT / DistilBERT

# Metrics logged for every model during evaluation
METRICS = ["accuracy", "f1", "precision", "recall"]

# ─────────────────────────────────────────────────────────────────────────────
# Task 1.1 — Simple ANN (TF-IDF features → feed-forward layers)
# ─────────────────────────────────────────────────────────────────────────────
ANN_SMALL_CONFIG = {
    "dataset":       "small",
    "learning_rate": 0.0005,   # was 0.001 — reduce overshoot
    "optimizer":     "Adam",
    "epochs":        50,
    "batch_size":    16,
    "dropout":       0.5,      # was 0.1 — crank dropout back up hard
    "weight_decay":  1e-2,     # was 1e-5 — much stronger L2
    "early_stopping_patience": 5,
    "use_scheduler": True,
    "warmup_ratio":  0.1,
}

ANN_LARGE_CONFIG = {
    "dataset":       "large",
    "learning_rate": 0.00005,
    "optimizer":     "Adam",
    "epochs":        20,
    "batch_size":    64,
    "dropout":       0.5,
    "weight_decay":  5e-3,
    "early_stopping_patience": 2,
    "warmup_ratio":  0.05,
}

ANN_PUBLIC_CONFIG = {
    "dataset":       "public",
    "learning_rate": 0.0001,
    "optimizer":     "Adam",
    "epochs":        10,
    "batch_size":    128,
    "dropout":       0.5,
    "weight_decay":  5e-3,
    "early_stopping_patience": 3,
    "warmup_ratio":  0.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 1.1 — Bidirectional LSTM (word embeddings → BiLSTM → classifier)
# ─────────────────────────────────────────────────────────────────────────────
BILSTM_SMALL_CONFIG = {
    "dataset":       "small",
    "learning_rate": 0.001,
    "optimizer":     "Adam",
    "epochs":        20,
    "batch_size":    64,
    "embed_dim":     128,
    "hidden_dim":    256,
    "num_layers":    2,
    "dropout":       0.5,
    "weight_decay":  1e-4,
    "grad_clip":     1.0,   # prevent exploding gradients in LSTM
    "early_stopping_patience": 5,
}

BILSTM_LARGE_CONFIG = {
    "dataset":       "large",
    "learning_rate": 0.001,
    "optimizer":     "Adam",
    "epochs":        20,
    "batch_size":    64,
    "embed_dim":     128,
    "hidden_dim":    256,
    "num_layers":    2,
    "dropout":       0.5,
    "weight_decay":  1e-4,
    "grad_clip":     1.0,   # prevent exploding gradients in LSTM
    "early_stopping_patience": 5,
}

BILSTM_PUBLIC_CONFIG = {
    "dataset":       "public",
    "learning_rate": 0.001,
    "optimizer":     "Adam",
    "epochs":        10,
    "batch_size":    128,
    "embed_dim":     128,
    "hidden_dim":    256,
    "num_layers":    2,
    "dropout":       0.5,
    "weight_decay":  1e-4,
    "grad_clip":     1.0,   # prevent exploding gradients in LSTM
    "early_stopping_patience": 3,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 1.2 — BERT (bert-base-uncased fine-tuned for sentiment)
# ─────────────────────────────────────────────────────────────────────────────
BERT_CONFIG = {
    "model_name":    "bert-base-uncased",
    "dataset":       "large",           # train on the 25 K Amazon reviews
    "learning_rate": 2e-5,
    "optimizer":     "AdamW",           # decoupled weight decay (critical for transformers)
    "epochs":        10,
    "batch_size":    16,                # BERT is large — keep batch small
    "max_len":       128,
    "weight_decay":  0.01,
    "use_scheduler": True,             # linear warmup + decay
    "warmup_ratio":  0.1,              # 10% of steps used for warmup
    "early_stopping_patience": 3,      # stop if val_loss doesn't improve for 3 epochs
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 1.2 — DistilBERT (distilbert-base-uncased, 40% smaller than BERT)
# Provides the second transformer model required for Grade 5.
# ─────────────────────────────────────────────────────────────────────────────
DISTILBERT_CONFIG = {
    "model_name":    "distilbert-base-uncased",
    "dataset":       "large",
    "learning_rate": 2e-5,
    "optimizer":     "AdamW",           # decoupled weight decay (critical for transformers)
    "epochs":        10,
    "batch_size":    32,
    "max_len":       128,
    "weight_decay":  0.01,
    "use_scheduler": True,             # linear warmup + decay
    "warmup_ratio":  0.1,
    "early_stopping_patience": 3,      # stop if val_loss doesn't improve for 3 epochs
}

# ─────────────────────────────────────────────────────────────────────────────
# Grade 5 — Transformers fine-tuned on the PUBLIC large dataset
# Both models train on the same data for a direct head-to-head comparison.
# ─────────────────────────────────────────────────────────────────────────────
BERT_PUBLIC_CONFIG = {
    "model_name":    "bert-base-uncased",
    "dataset":       "public",
    "learning_rate": 2e-5,
    "optimizer":     "AdamW",
    "epochs":        5,
    "batch_size":    16,
    "max_len":       128,
    "weight_decay":  0.01,
    "use_scheduler": True,
    "warmup_ratio":  0.1,
    "early_stopping_patience": 2,
}

DISTILBERT_PUBLIC_CONFIG = {
    "model_name":    "distilbert-base-uncased",
    "dataset":       "public",
    "learning_rate": 2e-5,
    "optimizer":     "AdamW",
    "epochs":        5,
    "batch_size":    32,
    "max_len":       128,
    "weight_decay":  0.01,
    "use_scheduler": True,
    "warmup_ratio":  0.1,
    "early_stopping_patience": 2,
}
