"""
Lab 4: Variational Autoencoders (VAE)
======================================
Implements a convolutional VAE trained on MNIST:
  - Encoder:    image → (µ, log σ²) in latent space
  - Reparameterisation trick: z = µ + σ · ε,  ε ~ N(0,I)
  - Decoder:    z → reconstructed image
  - ELBO loss:  reconstruction BCE + KL divergence
  - Latent space interpolation between two images

Outputs are saved to ./output/ every epoch.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

LATENT_DIM = 16
IMAGE_CHANNELS = 1
IMAGE_SIZE = 28
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20
KL_WEIGHT = 1.0   # β in β-VAE (1.0 = standard VAE)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Convolutional encoder that maps an image to (µ, log σ²).

    Args:
        latent_dim: Dimensionality of the latent space.
        image_channels: Number of input image channels.
    """

    def __init__(self, latent_dim: int = LATENT_DIM,
                 image_channels: int = IMAGE_CHANNELS) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),  # 14×14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),              # 7×7
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Convolutional decoder that maps a latent vector to a reconstructed image.

    Args:
        latent_dim: Dimensionality of the latent space.
        image_channels: Number of output image channels.
    """

    def __init__(self, latent_dim: int = LATENT_DIM,
                 image_channels: int = IMAGE_CHANNELS) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 14×14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),  # 28×28
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), 64, 7, 7)
        return self.deconv(x)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """Convolutional Variational Autoencoder.

    Args:
        latent_dim: Dimensionality of the latent space.
        image_channels: Number of image channels.
    """

    def __init__(self, latent_dim: int = LATENT_DIM,
                 image_channels: int = IMAGE_CHANNELS) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim, image_channels)
        self.decoder = Decoder(latent_dim, image_channels)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(µ, σ²) using the reparameterisation trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Draw ``n`` samples from the prior N(0, I)."""
        z = torch.randn(n, self.encoder.fc_mu.out_features, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    steps: int = 8) -> torch.Tensor:
        """Linearly interpolate between two images in latent space."""
        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        zs = torch.stack([mu1 * (1 - a) + mu2 * a for a in alphas])
        return self.decoder(zs.squeeze(1))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def elbo_loss(recon: torch.Tensor, original: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              kl_weight: float = KL_WEIGHT) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the ELBO loss: -E[log p(x|z)] + β · KL[q(z|x) || p(z)].

    Args:
        recon: Reconstructed images, values in [0, 1].
        original: Original images, values in [0, 1].
        mu: Latent mean, shape ``(B, latent_dim)``.
        logvar: Latent log-variance, shape ``(B, latent_dim)``.
        kl_weight: β weight on the KL term.

    Returns:
        total: Combined ELBO loss (scalar).
        recon_loss: Per-batch reconstruction loss.
        kl_loss: Per-batch KL divergence.
    """
    recon_loss = F.binary_cross_entropy(recon, original, reduction="sum") / original.size(0)
    # Analytical KL divergence for Gaussian: -0.5 * sum(1 + logvar - mu² - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / original.size(0)
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_mnist_loaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),    # values in [0, 1] – matches Sigmoid decoder
    ])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model: VAE, loader: DataLoader,
                optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float, float]:
    model.train()
    total, recon_total, kl_total = 0.0, 0.0, 0.0
    n = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs)
        loss, recon_loss, kl_loss = elbo_loss(recon, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        b = imgs.size(0)
        total += loss.item() * b
        recon_total += recon_loss.item() * b
        kl_total += kl_loss.item() * b
        n += b
    return total / n, recon_total / n, kl_total / n


@torch.no_grad()
def eval_epoch(model: VAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, mu, logvar = model(imgs)
        loss, _, _ = elbo_loss(recon, imgs, mu, logvar)
        total += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("output", exist_ok=True)

    train_loader, test_loader = get_mnist_loaders()

    model = VAE(latent_dim=LATENT_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_recon, tr_kl = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, test_loader, device)
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train ELBO: {tr_loss:.2f}  "
              f"Recon: {tr_recon:.2f}  KL: {tr_kl:.2f}  "
              f"Val ELBO: {val_loss:.2f}")

        # Save reconstructions
        test_imgs, _ = next(iter(test_loader))
        test_imgs = test_imgs[:8].to(device)
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(test_imgs)
        comparison = torch.cat([test_imgs, recon])
        save_image(comparison, f"output/recon_epoch_{epoch:02d}.png", nrow=8)

        # Save samples from prior
        samples = model.sample(64, device)
        save_image(samples, f"output/sample_epoch_{epoch:02d}.png", nrow=8)
        model.train()

    print("Training complete. Outputs saved to ./output/")


if __name__ == "__main__":
    main()
