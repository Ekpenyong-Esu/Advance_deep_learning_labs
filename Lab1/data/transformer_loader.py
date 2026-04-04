"""
data/transformer_loader.py — BERT / DistilBERT Data Loader
===========================================================
Provides DataLoaders for BERT and DistilBERT using Hugging Face tokenisation.

Usage
-----
    from data.transformer_loader import get_transformer_loaders
    train_loader, val_loader, test_loader = get_transformer_loaders(
        model_name="bert-base-uncased", dataset="small"
    )

Supported datasets
------------------
  "small"   — 1 K Amazon product reviews  (amazon_cells_labelled.txt)
  "large"   — 25 K Amazon product reviews (amazon_cells_labelled_LARGE_25K.txt)
  "public"  — amazon_polarity from Hugging Face (~3.6 M reviews, capped)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import config
from utils.text_preprocessing import batch_preprocess
from data.base_loader import _load_dataset, _split


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _TransformerDataset(Dataset):
    """
    Stores pre-tokenised Hugging Face inputs.

    Each item is a 3-tuple: (input_ids, attention_mask, label).
    This format is handled transparently by the shared trainer.
    """

    def __init__(self, encodings: dict, labels: list):
        self.input_ids      = encodings["input_ids"]       # (N, max_len) LongTensor
        self.attention_mask = encodings["attention_mask"]  # (N, max_len) LongTensor
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_transformer_loaders(
    model_name: str,
    dataset:    str = "small",
    batch_size: int = 16,
    max_len:    int = config.TRANSFORMER_MAX_LEN,
):
    """
    Build DataLoaders for BERT / DistilBERT using Hugging Face tokenisation.

    Pipeline
    --------
    1. Load raw texts and integer labels.
    2. Stratified 70 / 15 / 15 split.
    3. Minimal text preprocessing (tokeniser handles normalisation).
    4. Tokenise all splits with AutoTokenizer (padding + truncation).
    5. Wrap in TransformerDatasets returning (input_ids, attention_mask, label).

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    print(f"\nLoading transformer data  [{dataset} dataset, {model_name}] …")
    texts, labels = _load_dataset(dataset)
    tr_t, va_t, te_t, tr_l, va_l, te_l = _split(texts, labels)

    print("  Preprocessing text (minimal, transformer mode) …")
    tr_t = batch_preprocess(tr_t, mode="transformer")
    va_t = batch_preprocess(va_t, mode="transformer")
    te_t = batch_preprocess(te_t, mode="transformer")

    print(f"  Tokenising with '{model_name}' (max_len={max_len}) …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(text_list: list) -> dict:
        return tokenizer(
            text_list,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    tr_enc = _tokenize(tr_t)
    va_enc = _tokenize(va_t)
    te_enc = _tokenize(te_t)

    print(f"  Split: {len(tr_l):,} train / {len(va_l):,} val / {len(te_l):,} test")

    train_loader = DataLoader(
        _TransformerDataset(tr_enc, tr_l),
        batch_size=batch_size, shuffle=True,  num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        _TransformerDataset(va_enc, va_l),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        _TransformerDataset(te_enc, te_l),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader
