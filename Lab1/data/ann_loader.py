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
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = _split(texts, labels)

    print("  Preprocessing text (classical) …")

    train_texts = batch_preprocess(train_texts, mode="classical")
    val_texts   = batch_preprocess(val_texts,   mode="classical")
    test_texts  = batch_preprocess(test_texts,  mode="classical")

    print("  Fitting TF-IDF on training split (will NOT see val / test) …")

    min_df = 1 if dataset == "small" else 3

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),                       # unigrams + bigrams
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=min_df,                                 # drop terms seen in fewer than 3 docs (noise filter)
        max_df=0.5,                               # drop terms in more than 50% of docs (too common)
        use_idf=True,
        norm="l2",
    )
    train_tfidf = vectorizer.fit_transform(train_texts)  # fit + transform training only

    val_tfidf   = vectorizer.transform(val_texts)        # transform only
    test_tfidf  = vectorizer.transform(test_texts)       # transform only

    vocab_size = len(vectorizer.vocabulary_)
    
    print(f"  TF-IDF vocab: {vocab_size:,} features")
    print(f"  Split: {len(train_labels):,} train / {len(val_labels):,} val / {len(test_labels):,} test")

    train_loader = DataLoader(
        _SparseTFIDFDataset(train_tfidf, train_labels),
        batch_size=batch_size, shuffle=True,  num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        _SparseTFIDFDataset(val_tfidf, val_labels),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        _SparseTFIDFDataset(test_tfidf, test_labels),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader, vocab_size
