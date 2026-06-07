"""
models/lstm_model.py — Bidirectional LSTM for Sentiment Analysis
================================================================
Architecture
------------
Input  : (batch, seq_len)  — padded integer word indices

  Embedding : nn.Embedding(vocab_size, embed_dim, padding_idx=0)
              Maps each token index to a trainable dense vector.
              Words at padding positions contribute zero gradient.

  BiLSTM    : 2 stacked bidirectional LSTM layers, hidden_dim units per direction.
              Processes the sequence left-to-right (forward) AND right-to-left
              (backward) simultaneously, capturing both past and future context.

  Concat    : The last hidden state from the forward direction and the first
              hidden state from the backward direction are concatenated →
              shape (batch, 2 * hidden_dim).  This represents the full
              sequence context in a single fixed-size vector.

  Dropout   : Applied after embedding and after the final hidden state.

  Classifier: Linear(2 * hidden_dim → num_classes)

Why Bidirectional?
------------------
A regular LSTM only reads left-to-right, so the hidden state at position t
has no information about future words.  A BiLSTM reads in both directions
and merges the two representations, giving each position access to the full
sentence context — important for sentiment where the negation of a word at
the end of a sentence changes the meaning of earlier words.

Usage
-----
    from models.lstm_model import BiLSTMSentiment

    model = BiLSTMSentiment(vocab_size=30002)
    logits = model(sequences_tensor)   # (batch, 2)
"""

import torch
import torch.nn as nn


class BiLSTMSentiment(nn.Module):
    """
    Bidirectional LSTM for binary sentiment classification.

    Parameters
    ----------
    vocab_size  : int   — number of words in the vocabulary (from the data loader)
    embed_dim   : int   — dimension of each learned word embedding
    hidden_dim  : int   — number of LSTM units per direction
    num_layers  : int   — number of stacked BiLSTM layers
    num_classes : int   — number of output classes (2 for binary sentiment)
    dropout     : float — dropout probability (applied between layers and at output)
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 128,
        hidden_dim:  int   = 256,
        num_layers:  int   = 2,
        num_classes: int   = 2,
        dropout:     float = 0.3,
    ):
        super(BiLSTMSentiment, self).__init__()

        # Word embedding — padding_idx=0 means the <PAD> token always maps to
        # an all-zero vector and its gradient is set to zero during training.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Bidirectional LSTM
        # dropout between layers: only applied when num_layers > 1
        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Sentinel classifier: 2 * hidden_dim because forward + backward
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : LongTensor  shape (batch, seq_len)

        Returns
        -------
        logits : FloatTensor  shape (batch, num_classes)
        """
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # lstm returns:
        #   output : (batch, seq_len, 2*hidden_dim)  — all hidden states
        #   hidden : (num_layers*2, batch, hidden_dim)
        _, (hidden, _) = self.lstm(embedded)

        # hidden[-2] : last layer, forward direction   (batch, hidden_dim)
        # hidden[-1] : last layer, backward direction  (batch, hidden_dim)
        forward_h  = hidden[-2]
        backward_h = hidden[-1]

        # Concatenate both directions → (batch, 2 * hidden_dim)
        combined = torch.cat((forward_h, backward_h), dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)
        return logits
