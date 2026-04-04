"""
models/ann_model.py — Simple Feed-Forward ANN for Sentiment Analysis
=====================================================================
Architecture
------------
Input  : (batch, vocab_size)  — dense TF-IDF feature vector

  Block 1  : Linear(vocab_size → 512) + BatchNorm1d + ReLU + Dropout
  Block 2  : Linear(512 → 256)        + BatchNorm1d + ReLU + Dropout
  Block 3  : Linear(256 → 64)         +              ReLU + Dropout
  Output   : Linear(64 → num_classes)

  num_classes = 2  →  [negative, positive]   (used with CrossEntropyLoss)

BatchNorm1d after each linear layer stabilises training and allows higher
learning rates.  Dropout prevents co-adaptation of features.

Usage
-----
    from models.ann_model import SimpleANN

    model = SimpleANN(vocab_size=50000)
    logits = model(tfidf_tensor)   # (batch, 2)
"""

import torch.nn as nn


class SimpleANN(nn.Module):
    """
    Feed-forward ANN for binary sentiment classification.

    Parameters
    ----------
    vocab_size  : int   — TF-IDF input dimension
    num_classes : int   — number of output classes (2 for binary sentiment)
    dropout     : float — dropout probability applied after each hidden block
    """

    def __init__(
        self,
        vocab_size:  int,
        num_classes: int   = 2,
        dropout:     float = 0.3,
    ):
        super(SimpleANN, self).__init__()

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
        """
        Parameters
        ----------
        x : FloatTensor  shape (batch, vocab_size)

        Returns
        -------
        logits : FloatTensor  shape (batch, num_classes)
        """
        logits = self.network(x)
        return logits
