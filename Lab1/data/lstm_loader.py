"""
data/lstm_loader.py — BiLSTM Data Loader (Padded Word Sequences)
=================================================================
Provides DataLoaders for the BiLSTM model using padded integer sequences.

Usage
-----
    from data.lstm_loader import get_lstm_loaders
    train_loader, val_loader, test_loader, vocab_size = get_lstm_loaders(dataset="small")

Supported datasets
------------------
  "small"   — 1 K Amazon product reviews  (amazon_cells_labelled.txt)
  "large"   — 25 K Amazon product reviews (amazon_cells_labelled_LARGE_25K.txt)
  "public"  — amazon_polarity from Hugging Face (~3.6 M reviews, capped)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils.text_preprocessing import batch_preprocess
from data.base_loader import _load_dataset, _split


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class _Vocabulary:
    """
    Build a word → integer index mapping from training texts.

    Special reserved indices
    ------------------------
    0 → <PAD>   padding token (also used for unseen words in val / test)
    1 → <UNK>   unknown / rare words (words not in the training vocabulary)
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_size: int = config.LSTM_MAX_VOCAB):
        self.max_size  = max_size
        self.word2idx: dict = {"<PAD>": 0, "<UNK>": 1}

    def build(self, train_texts: list) -> None:
        """Fit on training texts ONLY to prevent information leakage."""
        
        counter = Counter()
        
        for text in train_texts:
            counter.update(text.split())

        # Retain the most frequent words up to (max_size − 2) to reserve
        # indices 0 and 1 for the special tokens.
        for word, _ in counter.most_common(self.max_size - 2):
            self.word2idx[word] = len(self.word2idx)

    def encode(self, text: str, max_len: int) -> list:
        """Convert a cleaned text string into a padded list of integer indices."""
        
        tokens = text.split()[:max_len]
        
        ids    = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        # Right-pad to max_len with the PAD index
        return ids + [self.PAD_IDX] * (max_len - len(ids))

    @property
    def size(self) -> int:
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _SequenceDataset(Dataset):
    """Stores padded integer sequences for the BiLSTM."""

    def __init__(self, sequences: list, labels: list):
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_lstm_loaders(
    dataset:   str = "small",
    batch_size: int = config.BATCH_SIZE,
    max_len:   int = config.LSTM_MAX_LEN,
    max_vocab: int = config.LSTM_MAX_VOCAB,
):
    """
    Build DataLoaders for the BiLSTM using padded word-index sequences.

    Pipeline
    --------
    1. Load raw texts and integer labels.
    2. Stratified 70 / 15 / 15 split.
    3. Full text preprocessing.
    4. Build word vocabulary from TRAINING split only.
    5. Encode and right-pad all splits to max_len.
    6. Wrap in Datasets and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, vocab_size : int
        vocab_size is required to create the Embedding layer in BiLSTMSentiment.
    """
    print(f"\nLoading BiLSTM data  [{dataset} dataset] …")
    
    texts, labels = _load_dataset(dataset)
    
    tr_t, va_t, te_t, tr_l, va_l, te_l = _split(texts, labels)

    print("  Preprocessing text (classical) …")
    
    tr_t = batch_preprocess(tr_t, mode="classical")
    va_t = batch_preprocess(va_t, mode="classical")
    te_t = batch_preprocess(te_t, mode="classical")

    print("  Building vocabulary from training split only …")
    
    vocab = _Vocabulary(max_size=max_vocab)
    
    vocab.build(tr_t)
    
    vocab_size = vocab.size
    
    print(f"  Vocabulary size: {vocab_size:,} tokens (max {max_vocab:,})")
    print(f"  Split: {len(tr_l):,} train / {len(va_l):,} val / {len(te_l):,} test")

    tr_seq = [vocab.encode(t, max_len) for t in tr_t]
    va_seq = [vocab.encode(t, max_len) for t in va_t]
    te_seq = [vocab.encode(t, max_len) for t in te_t]

    train_loader = DataLoader(
        _SequenceDataset(tr_seq, tr_l),
        batch_size=batch_size, shuffle=True,  num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        _SequenceDataset(va_seq, va_l),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        _SequenceDataset(te_seq, te_l),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader, vocab_size
