"""
models/decoder.py — LSTM Caption Decoder with Concatenation Embedding Fusion
=============================================================================
The decoder generates a word sequence conditioned on an image feature vector
produced by the CNN encoder.

Embedding combination: CONCATENATION
--------------------------------------
At every time step the image feature vector and the current word embedding
are concatenated before being fed to the LSTM cell.  This is the
**PAR-inject** (parallel injection) strategy from Tanti & Gatt (2018):

    x_t = [image_feature ; word_embedding(w_t)]   (batch, 2 * embed_dim)
    h_t, c_t = LSTM(x_t, (h_{t-1}, c_{t-1}))
    logits_t  = fc(dropout(h_t))                  (batch, vocab_size)

Why concatenation?
------------------
Concatenation preserves the full information of both streams and lets the
LSTM learn how to mix them, at the cost of doubling the input dimensionality.
See Task 3.1.1 in the notebook for a detailed pros / cons discussion.

Training (teacher forcing)
--------------------------
  input  captions : [<SOS>, w_1,  w_2,  ..., w_{T-1}]   (batch, T)
  target captions : [w_1,   w_2,  ..., w_T, <EOS>  ]    (batch, T)

The whole sequence is processed in a single LSTM call for efficiency.

Inference (greedy decoding)
---------------------------
Use ``generate_caption`` from ``utils/evaluation.py`` which calls
``decoder.step()`` one token at a time.

Usage
-----
    from models.decoder import CaptionDecoder

    decoder = CaptionDecoder(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.5,
    )
    logits = decoder(image_features, captions)   # (batch, T, vocab_size)
"""

import torch
import torch.nn as nn

from data.vocabulary import PAD_IDX


class CaptionDecoder(nn.Module):
    """
    LSTM-based caption decoder that injects image features at every step
    via concatenation with the word embedding.

    Parameters
    ----------
    vocab_size  : int   — total vocabulary size (output classes)
    embed_dim   : int   — word embedding dimension (must match encoder output)
    hidden_dim  : int   — LSTM hidden state dimension
    num_layers  : int   — number of stacked LSTM layers
    dropout     : float — dropout probability (applied after LSTM output)
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        hidden_dim:  int,
        num_layers:  int   = 1,
        dropout:     float = 0.5,
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Word embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX,
        )

        # LSTM — input is (image_feature ∥ word_embedding) = 2 * embed_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,   # concatenation doubles the dimension
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projection from hidden state to vocabulary logits
        self.fc      = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────── #

    def _init_weights(self) -> None:
        """Uniform initialisation for embedding; Xavier for fc."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    # ── training forward pass (teacher forcing) ───────────────────────────── #

    def forward(
        self,
        image_features: torch.Tensor,
        captions:       torch.Tensor,
    ) -> torch.Tensor:
        """
        Process a full caption sequence using teacher forcing.

        Parameters
        ----------
        image_features : FloatTensor  (batch, embed_dim)
            Projected CNN features from the encoder.
        captions       : LongTensor   (batch, seq_len)
            Token indices.  The last token (typically <EOS>) is excluded
            from the input — the caller passes ``captions[:, :-1]``.

        Returns
        -------
        FloatTensor  (batch, seq_len, vocab_size)
            Raw logits for each time step.
        """
        batch, seq_len = captions.shape

        # Embed the caption tokens
        embeddings = self.embedding(captions)          # (batch, seq_len, embed_dim)

        # Broadcast image feature across all time steps
        img_feat = image_features.unsqueeze(1).expand(-1, seq_len, -1)
        # (batch, seq_len, embed_dim)

        # Concatenate along the feature dimension
        lstm_input = torch.cat([img_feat, embeddings], dim=2)
        # (batch, seq_len, 2 * embed_dim)

        # Run LSTM over the full sequence in one call
        lstm_out, _ = self.lstm(lstm_input)            # (batch, seq_len, hidden_dim)

        # Project to vocabulary logits
        logits = self.fc(self.dropout(lstm_out))       # (batch, seq_len, vocab_size)
        return logits

    # ── single-step inference helper ─────────────────────────────────────── #

    def step(
        self,
        image_feature: torch.Tensor,
        word_idx:      torch.Tensor,
        hidden:        tuple,
    ) -> tuple:
        """
        Perform one LSTM step for autoregressive caption generation.

        Parameters
        ----------
        image_feature : FloatTensor  (1, embed_dim)
        word_idx      : LongTensor   (1, 1)  — current input token index
        hidden        : (h, c) — previous LSTM hidden and cell states

        Returns
        -------
        logits : FloatTensor (1, vocab_size)
        hidden : updated (h, c) tuple
        """
        embedding  = self.embedding(word_idx)               # (1, 1, embed_dim)
        img_feat   = image_feature.unsqueeze(1)             # (1, 1, embed_dim)
        lstm_input = torch.cat([img_feat, embedding], dim=2)  # (1, 1, 2*embed_dim)

        lstm_out, hidden = self.lstm(lstm_input, hidden)    # (1, 1, hidden_dim)
        logits = self.fc(lstm_out.squeeze(1))               # (1, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """
        Return zero-initialised (h_0, c_0) for the LSTM.

        Parameters
        ----------
        batch_size : int
        device     : torch.device
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h, c
