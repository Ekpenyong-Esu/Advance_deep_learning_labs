"""
Unit tests for Lab 2 attention & transformer components.
Run with: python -m pytest test_transformer.py -v
"""

import math
import torch
import pytest
from main import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
    TransformerClassifier,
    make_synthetic_dataset,
)


class TestScaledDotProductAttention:
    def test_output_shape(self) -> None:
        B, H, T, d_k = 2, 4, 10, 16
        q = k = v = torch.randn(B, H, T, d_k)
        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, T, d_k)
        assert weights.shape == (B, H, T, T)

    def test_weights_sum_to_one(self) -> None:
        q = k = v = torch.randn(1, 1, 5, 8)
        _, weights = scaled_dot_product_attention(q, k, v)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, 5), atol=1e-5)

    def test_mask_sets_neg_inf(self) -> None:
        q = k = v = torch.randn(1, 1, 3, 4)
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        mask[0, 0, :, 2] = True  # mask last key position
        _, weights = scaled_dot_product_attention(q, k, v, mask)
        assert (weights[0, 0, :, 2] < 1e-6).all()


class TestMultiHeadAttention:
    def test_output_shape(self) -> None:
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        assert mha(x, x, x).shape == (2, 10, 64)

    def test_invalid_head_count(self) -> None:
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4)


class TestSinusoidalPositionalEncoding:
    def test_output_shape_preserved(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=50, dropout=0.0)
        x = torch.zeros(3, 20, 32)
        assert pe(x).shape == x.shape

    def test_encoding_is_deterministic(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=16, max_len=10, dropout=0.0)
        pe.eval()
        x = torch.zeros(1, 5, 16)
        assert torch.equal(pe(x), pe(x))


class TestTransformerEncoderBlock:
    def test_output_shape(self) -> None:
        block = TransformerEncoderBlock(d_model=32, num_heads=4, dim_ff=64)
        x = torch.randn(2, 8, 32)
        assert block(x).shape == x.shape


class TestTransformerClassifier:
    def test_forward_shape(self) -> None:
        model = TransformerClassifier(vocab_size=20, d_model=32, num_heads=4,
                                      num_layers=1, dim_ff=64, num_classes=3)
        model.eval()
        tokens = torch.randint(0, 20, (4, 16))
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (4, 3)


class TestSyntheticDataset:
    def test_dataset_length(self) -> None:
        ds = make_synthetic_dataset(n_samples=100, seq_len=10, vocab_size=20)
        assert len(ds) == 100

    def test_label_values(self) -> None:
        ds = make_synthetic_dataset(n_samples=200, seq_len=8, vocab_size=10)
        labels = torch.stack([ds[i][1] for i in range(len(ds))])
        assert set(labels.tolist()).issubset({0, 1})
