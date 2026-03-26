"""
Unit tests for Lab 4 VAE components.
Run with: python -m pytest test_vae.py -v
"""

import torch
import pytest
from main import Encoder, Decoder, VAE, elbo_loss, LATENT_DIM, IMAGE_CHANNELS


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def sample_images(device: torch.device) -> torch.Tensor:
    return torch.rand(4, IMAGE_CHANNELS, 28, 28, device=device)


class TestEncoder:
    def test_output_shapes(self, sample_images: torch.Tensor) -> None:
        encoder = Encoder(latent_dim=LATENT_DIM)
        mu, logvar = encoder(sample_images)
        assert mu.shape == (4, LATENT_DIM)
        assert logvar.shape == (4, LATENT_DIM)


class TestDecoder:
    def test_output_shape(self, device: torch.device) -> None:
        decoder = Decoder(latent_dim=LATENT_DIM)
        z = torch.randn(4, LATENT_DIM, device=device)
        out = decoder(z)
        assert out.shape == (4, IMAGE_CHANNELS, 28, 28)

    def test_output_in_zero_one(self, device: torch.device) -> None:
        decoder = Decoder(latent_dim=LATENT_DIM)
        z = torch.randn(4, LATENT_DIM, device=device)
        out = decoder(z)
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5


class TestVAE:
    def test_forward_shapes(self, sample_images: torch.Tensor) -> None:
        model = VAE(latent_dim=LATENT_DIM)
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(sample_images)
        assert recon.shape == sample_images.shape
        assert mu.shape == (4, LATENT_DIM)
        assert logvar.shape == (4, LATENT_DIM)

    def test_reparameterise_training_mode(self) -> None:
        model = VAE(latent_dim=8)
        model.train()
        mu = torch.zeros(5, 8)
        logvar = torch.zeros(5, 8)
        z1 = model.reparameterise(mu, logvar)
        z2 = model.reparameterise(mu, logvar)
        # In training mode, z should be stochastic
        assert not torch.equal(z1, z2)

    def test_reparameterise_eval_mode(self) -> None:
        model = VAE(latent_dim=8)
        model.eval()
        mu = torch.randn(5, 8)
        logvar = torch.zeros(5, 8)
        z = model.reparameterise(mu, logvar)
        assert torch.equal(z, mu)

    def test_sample(self, device: torch.device) -> None:
        model = VAE(latent_dim=LATENT_DIM).to(device)
        samples = model.sample(8, device)
        assert samples.shape == (8, IMAGE_CHANNELS, 28, 28)


class TestELBOLoss:
    def test_loss_is_positive(self) -> None:
        recon = torch.sigmoid(torch.randn(4, 1, 28, 28))
        original = torch.rand(4, 1, 28, 28)
        mu = torch.randn(4, LATENT_DIM)
        logvar = torch.zeros(4, LATENT_DIM)
        total, recon_l, kl_l = elbo_loss(recon, original, mu, logvar)
        assert total.item() > 0
        assert recon_l.item() > 0
        assert kl_l.item() >= 0

    def test_zero_kl_at_prior(self) -> None:
        """When µ=0 and σ=1 (logvar=0), KL divergence should be ~0."""
        recon = torch.rand(4, 1, 28, 28)
        original = recon.clone()
        mu = torch.zeros(4, LATENT_DIM)
        logvar = torch.zeros(4, LATENT_DIM)
        _, _, kl_l = elbo_loss(recon, original, mu, logvar)
        assert abs(kl_l.item()) < 1e-4
