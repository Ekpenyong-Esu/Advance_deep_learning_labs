"""
Lab 2: Attention Mechanisms & Transformers
==========================================
Implements the Transformer encoder from scratch:
  - Scaled dot-product attention
  - Multi-head attention
  - Sinusoidal positional encoding
  - Transformer encoder block (attention + FFN + layer norm)
  - Full encoder model for sequence classification

A small synthetic dataset is used so the lab runs quickly on CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Args:
        query: ``(B, heads, T_q, d_k)``
        key:   ``(B, heads, T_k, d_k)``
        value: ``(B, heads, T_k, d_v)``
        mask:  Optional boolean mask ``(B, 1, T_q, T_k)`` — True means *ignore*.

    Returns:
        output: ``(B, heads, T_q, d_v)``
        weights: ``(B, heads, T_q, T_k)``
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention module.

    Linearly projects queries, keys, and values ``num_heads`` times,
    applies scaled dot-product attention in parallel, and concatenates
    the results.

    Args:
        d_model: Total model dimension.
        num_heads: Number of attention heads. Must divide ``d_model``.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model)`` → ``(B, heads, T, d_k)``."""
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B = query.size(0)
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)
        # (B, heads, T, d_k) → (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.out_proj(self.dropout(attn_out))


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        d_model: Model dimension.
        max_len: Maximum supported sequence length.
        dropout: Dropout probability applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape ``(B, T, d_model)``."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear → GELU → Linear."""

    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder layer (Pre-LN variant).

    Pre-LN applies layer normalisation *before* each sub-layer, which
    generally leads to more stable training.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dim_ff: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int,
                 dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, mask))
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full Encoder Model for Sequence Classification
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """Transformer encoder stack for sequence classification.

    Uses a [CLS] token prepended to the sequence; its final hidden
    state is passed to a classification head.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Model dimension (embedding size).
        num_heads: Attention heads per layer.
        num_layers: Number of encoder blocks.
        dim_ff: Feed-forward hidden dimension.
        num_classes: Number of output classes.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4,
                 num_layers: int = 3, dim_ff: int = 256, num_classes: int = 2,
                 max_len: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len + 1, dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, token_ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B = token_ids.size(0)
        x = self.embedding(token_ids)                          # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                        # (B, T+1, d_model)
        x = self.pos_enc(x)
        for block in self.encoder:
            x = block(x, mask)
        x = self.norm(x[:, 0])                                # CLS hidden state
        return self.head(x)


# ---------------------------------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------------------------------

def make_synthetic_dataset(
    n_samples: int = 2000,
    seq_len: int = 32,
    vocab_size: int = 50,
    num_classes: int = 2,
    seed: int = 42,
) -> TensorDataset:
    """Create a simple synthetic sequence-classification dataset.

    Positive class: sequences whose sum of tokens > vocab_size * seq_len / 2.
    """
    torch.manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len))
    labels = (tokens.float().mean(dim=1) > vocab_size / 2).long()
    return TensorDataset(tokens, labels)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(tokens), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * tokens.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        correct += model(tokens).argmax(dim=1).eq(labels).sum().item()
    return correct / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_size = 50
    seq_len = 32
    dataset = make_synthetic_dataset(n_samples=3000, seq_len=seq_len,
                                     vocab_size=vocab_size)
    n_train = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train]
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    model = TransformerClassifier(
        vocab_size=vocab_size, d_model=64, num_heads=4,
        num_layers=2, dim_ff=128, num_classes=2, max_len=seq_len,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 10
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch:02d}/{epochs}]  Loss: {loss:.4f}  Test Acc: {acc * 100:.2f}%")

    print("Training complete.")


if __name__ == "__main__":
    main()
