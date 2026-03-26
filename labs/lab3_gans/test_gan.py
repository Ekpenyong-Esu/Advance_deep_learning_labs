"""
Unit tests for Lab 3 GAN components.
Run with: python -m pytest test_gan.py -v
"""

import torch
import pytest
from main import Generator, Discriminator, LATENT_DIM, IMAGE_CHANNELS


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


class TestGenerator:
    def test_output_shape(self, device: torch.device) -> None:
        G = Generator().to(device)
        z = torch.randn(4, LATENT_DIM, device=device)
        out = G(z)
        assert out.shape == (4, IMAGE_CHANNELS, 28, 28)

    def test_output_in_range(self, device: torch.device) -> None:
        G = Generator().to(device)
        G.eval()
        with torch.no_grad():
            z = torch.randn(8, LATENT_DIM, device=device)
            out = G(z)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5


class TestDiscriminator:
    def test_output_shape(self, device: torch.device) -> None:
        D = Discriminator().to(device)
        x = torch.randn(4, IMAGE_CHANNELS, 28, 28, device=device)
        out = D(x)
        assert out.shape == (4,)

    def test_output_is_scalar_per_sample(self, device: torch.device) -> None:
        D = Discriminator().to(device)
        x = torch.randn(3, IMAGE_CHANNELS, 28, 28, device=device)
        assert D(x).ndim == 1


class TestGANForwardPass:
    def test_generator_discriminator_chain(self, device: torch.device) -> None:
        G = Generator().to(device)
        D = Discriminator().to(device)
        z = torch.randn(2, LATENT_DIM, device=device)
        fake = G(z)
        logit = D(fake)
        assert logit.shape == (2,)
