"""
data/ann_loader.py — Simple ANN Data Loader (TF-IDF)
=====================================================
Provides DataLoaders for the Simple ANN model using TF-IDF feature vectors.

Usage
-----
    from data.ann_loader import get_ann_loaders
    train_loader, val_loader, test_loader, vocab_size = get_ann_loaders(dataset="small")

Supported datasets
------------------
  "small"   — 1 K Amazon product reviews  (amazon_cells_labelled.txt)
  "large"   — 25 K Amazon product reviews (amazon_cells_labelled_LARGE_25K.txt)
  "public"  — amazon_polarity from Hugging Face (~3.6 M reviews, capped)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from utils.text_preprocessing import batch_preprocess
from data.base_loader import _load_dataset, _split


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _SparseTFIDFDataset(Dataset):
    """
    Wraps a scipy sparse TF-IDF matrix for memory-efficient loading.

    Each sample is converted to a dense float32 tensor on access, so only
    one batch worth of data is ever resident in memory at once.
    """

    def __init__(self, sparse_X: sp.spmatrix, labels: list):
        self.X = sparse_X
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(
            np.asarray(self.X[idx].todense()).squeeze(0), dtype=torch.float32
        )
        return x, self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_ann_loaders(dataset: str = "small", batch_size: int = config.BATCH_SIZE):
    """
    Build DataLoaders for the Simple ANN using TF-IDF feature vectors.

    Pipeline
    --------
    1. Load raw texts and integer labels.
    2. Stratified 70 / 15 / 15 split.
    3. Full text preprocessing (lowercase, stopword removal, etc.).
    4. Fit TF-IDF vectoriser on TRAINING split; transform all three splits.
    5. Wrap in sparse-to-dense Datasets and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, vocab_size : int
    """
    print(f"\nLoading ANN data  [{dataset} dataset] …")
    texts, labels = _load_dataset(dataset)
    tr_t, va_t, te_t, tr_l, va_l, te_l = _split(texts, labels)

    print("  Preprocessing text (classical) …")
    tr_t = batch_preprocess(tr_t, mode="classical")
    va_t = batch_preprocess(va_t, mode="classical")
    te_t = batch_preprocess(te_t, mode="classical")

    print("  Fitting TF-IDF on training split (will NOT see val / test) …")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=config.TFIDF_MAX_FEATURES,
        max_df=0.5,
        use_idf=True,
        norm="l2",
    )
    tr_X = vectorizer.fit_transform(tr_t)   # fit + transform training only
    va_X = vectorizer.transform(va_t)        # transform only
    te_X = vectorizer.transform(te_t)        # transform only

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  TF-IDF vocab: {vocab_size:,} features")
    print(f"  Split: {len(tr_l):,} train / {len(va_l):,} val / {len(te_l):,} test")

    train_loader = DataLoader(
        _SparseTFIDFDataset(tr_X, tr_l),
        batch_size=batch_size, shuffle=True,  num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        _SparseTFIDFDataset(va_X, va_l),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        _SparseTFIDFDataset(te_X, te_l),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader, vocab_size
