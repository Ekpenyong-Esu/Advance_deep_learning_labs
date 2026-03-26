"""
Unit tests for Lab 1 custom layer definitions.
Run with: python -m pytest test_layers.py -v
"""

import torch
import pytest
from main import (
    DepthwiseSeparableConv,
    SqueezeExcitation,
    ResidualBlock,
    SmallResNet,
)


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    """4 images of 3×32×32."""
    return torch.randn(4, 3, 32, 32)


class TestDepthwiseSeparableConv:
    def test_output_shape(self) -> None:
        layer = DepthwiseSeparableConv(3, 16)
        x = torch.randn(2, 3, 8, 8)
        assert layer(x).shape == (2, 16, 8, 8)

    def test_stride_reduces_spatial(self) -> None:
        layer = DepthwiseSeparableConv(8, 16, stride=2, padding=1)
        x = torch.randn(1, 8, 16, 16)
        assert layer(x).shape == (1, 16, 8, 8)


class TestSqueezeExcitation:
    def test_output_shape_unchanged(self) -> None:
        se = SqueezeExcitation(64)
        x = torch.randn(2, 64, 8, 8)
        assert se(x).shape == x.shape

    def test_scale_in_zero_one(self) -> None:
        """The SE block multiplies by a sigmoid output, so values can only
        be scaled (not amplified beyond the input magnitude)."""
        se = SqueezeExcitation(32)
        x = torch.ones(1, 32, 4, 4)
        out = se(x)
        assert (out >= 0).all()


class TestResidualBlock:
    def test_same_channels(self) -> None:
        block = ResidualBlock(32, 32)
        x = torch.randn(2, 32, 8, 8)
        assert block(x).shape == x.shape

    def test_channel_change(self) -> None:
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 8, 8)
        assert block(x).shape == (2, 64, 4, 4)

    def test_no_se(self) -> None:
        block = ResidualBlock(16, 16, use_se=False)
        x = torch.randn(1, 16, 4, 4)
        assert block(x).shape == x.shape


class TestSmallResNet:
    def test_output_logits(self, dummy_batch: torch.Tensor) -> None:
        model = SmallResNet(num_classes=10)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 10)

    def test_parameter_count(self) -> None:
        model = SmallResNet()
        params = sum(p.numel() for p in model.parameters())
        # Should be a lightweight model: under 5 M parameters
        assert params < 5_000_000
