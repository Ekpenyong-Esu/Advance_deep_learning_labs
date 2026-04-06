"""
models/ann_model.py — Simple Feed-Forward ANN for Sentiment Analysis
=====================================================================
Two architectures are provided, selected automatically by build_ann():

  SmallANN  — for small datasets (< ~5 K training samples)
  ─────────
  Input  : (batch, vocab_size)
    Block 1 : Linear(vocab_size → 128) + BatchNorm1d + ReLU + Dropout
    Block 2 : Linear(128 → 64)                        + ReLU + Dropout
    Output  : Linear(64 → num_classes)

  Rationale: fewer parameters prevent memorisation on limited data.

  LargeANN  — for large / public datasets (≥ ~5 K training samples)
  ─────────
  Input  : (batch, vocab_size)
    Block 1 : Linear(vocab_size → 512) + BatchNorm1d + ReLU + Dropout
    Block 2 : Linear(512 → 256)        + BatchNorm1d + ReLU + Dropout
    Block 3 : Linear(256 → 64)                        + ReLU + Dropout
    Output  : Linear(64 → num_classes)

  num_classes = 2  →  [negative, positive]  (CrossEntropyLoss)

Usage
-----
    from models.ann_model import build_ann

    model = build_ann(dataset="small", vocab_size=15000)
    model = build_ann(dataset="large", vocab_size=15000)
    model = build_ann(dataset="public", vocab_size=15000)
"""

import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Small dataset model  (~1 K samples)
# ─────────────────────────────────────────────────────────────────────────────

class SmallANN(nn.Module):
    """Shallow ANN for small datasets — reduced capacity to avoid overfitting."""

    def __init__(self, vocab_size: int, num_classes: int = 2, dropout: float = 0.5):
        super(SmallANN, self).__init__()

        self.network = nn.Sequential(
            # ── Block 1 ──────────────────────────────────────────────────── #
            nn.Linear(vocab_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Block 2 ──────────────────────────────────────────────────── #
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Output ───────────────────────────────────────────────────── #
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# Large / public dataset model  (25 K+ samples)
# ─────────────────────────────────────────────────────────────────────────────

class LargeANN(nn.Module):
    """Deeper ANN for large datasets — higher capacity to exploit more data."""

    def __init__(self, vocab_size: int, num_classes: int = 2, dropout: float = 0.5):
        super(LargeANN, self).__init__()

        self.network = nn.Sequential(
            # ── Block 1 ──────────────────────────────────────────────────── #
            nn.Linear(vocab_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Block 2 ──────────────────────────────────────────────────── #
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Block 3 ──────────────────────────────────────────────────── #
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # ── Output ───────────────────────────────────────────────────── #
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory — picks the right model for the dataset
# ─────────────────────────────────────────────────────────────────────────────


def build_ann(dataset: str, vocab_size: int, num_classes: int = 2, dropout: float = 0.5):
    """
    Return the appropriate ANN architecture for the given dataset.

    "small"          →  SmallANN  (2 hidden blocks, narrow)
    "large"/"public" →  LargeANN  (3 hidden blocks, wide)

    Parameters
    ----------
    dataset     : str   — "small", "large", or "public"
    vocab_size  : int   — TF-IDF input dimension (from the data loader)
    num_classes : int   — number of output classes (default 2)
    dropout     : float — dropout probability

    Returns
    -------
    nn.Module
    """
    if dataset == "small":
        print(f"  [build_ann] dataset='{dataset}' → SmallANN (2 blocks, 128→64)")
        return SmallANN(vocab_size=vocab_size, num_classes=num_classes, dropout=dropout)
    else:
        print(f"  [build_ann] dataset='{dataset}' → LargeANN (3 blocks, 512→256→64)")
        return LargeANN(vocab_size=vocab_size, num_classes=num_classes, dropout=dropout)