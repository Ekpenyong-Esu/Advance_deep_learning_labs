"""
Lab 3: Generative Adversarial Networks (DCGAN)
================================================
Implements DCGAN (Radford et al., 2015) trained on MNIST:
  - Generator:      latent vector → 28×28 grayscale image
  - Discriminator:  image → real/fake probability
  - Adversarial training loop with label smoothing

Sample images are saved to ./output/ after every epoch.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

LATENT_DIM = 100
IMAGE_CHANNELS = 1      # MNIST is grayscale
FEATURE_DIM = 64        # base feature map size
IMAGE_SIZE = 28
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
BETA1 = 0.5             # Adam β₁ recommended for GANs
BETA2 = 0.999
REAL_LABEL_SMOOTH = 0.9  # one-sided label smoothing for discriminator


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """DCGAN generator: latent vector → image via transposed convolutions.

    Architecture (for 28×28 output)::

        Linear(latent) → Reshape(7×7) →
        ConvTranspose2d ×2 → tanh activation
    """

    def __init__(self, latent_dim: int = LATENT_DIM,
                 feature_dim: int = FEATURE_DIM,
                 image_channels: int = IMAGE_CHANNELS) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_dim, feature_dim * 4 * 7 * 7, bias=False),
            nn.BatchNorm1d(feature_dim * 4 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.conv_blocks = nn.Sequential(
            # 7×7 → 14×14
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
            # 14×14 → 28×28
            nn.ConvTranspose2d(feature_dim * 2, image_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self._feature_dim = feature_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(x.size(0), self._feature_dim * 4, 7, 7)
        return self.conv_blocks(x)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """DCGAN discriminator: image → real/fake logit.

    Uses strided convolutions instead of pooling, and LeakyReLU activations.
    Spectral norm is applied for improved training stability.
    """

    def __init__(self, image_channels: int = IMAGE_CHANNELS,
                 feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()

        def _conv_block(in_ch: int, out_ch: int,
                        stride: int = 2) -> list[nn.Module]:
            return [
                nn.utils.spectral_norm(
                    nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.net = nn.Sequential(
            *_conv_block(image_channels, feature_dim),           # 28→14
            *_conv_block(feature_dim, feature_dim * 2),          # 14→7
            nn.Conv2d(feature_dim * 2, 1, kernel_size=7, bias=False),  # 7→1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0))


# ---------------------------------------------------------------------------
# Weight Initialization
# ---------------------------------------------------------------------------

def weights_init(module: nn.Module) -> None:
    """Apply DCGAN weight initialisation (mean=0, std=0.02)."""
    name = type(module).__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)  # type: ignore[arg-type]
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)  # type: ignore[arg-type]
        nn.init.constant_(module.bias.data, 0)           # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_mnist_loader(batch_size: int = BATCH_SIZE) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True,
                             download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=2, pin_memory=True,
                      drop_last=True)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(epochs: int = 20, device: torch.device | None = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("output", exist_ok=True)

    loader = get_mnist_loader()

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # Fixed noise for visualisation
    fixed_noise = torch.randn(64, LATENT_DIM, device=device)

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []

        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            B = real_imgs.size(0)

            real_labels = torch.full((B,), REAL_LABEL_SMOOTH, device=device)
            fake_labels = torch.zeros(B, device=device)

            # ---- Train Discriminator ----
            opt_D.zero_grad()
            d_real = criterion(D(real_imgs), real_labels)
            z = torch.randn(B, LATENT_DIM, device=device)
            d_fake = criterion(D(G(z).detach()), fake_labels)
            d_loss = d_real + d_fake
            d_loss.backward()
            opt_D.step()

            # ---- Train Generator ----
            opt_G.zero_grad()
            z = torch.randn(B, LATENT_DIM, device=device)
            gen_labels = torch.ones(B, device=device)   # generator wants D(G(z))=1
            g_loss = criterion(D(G(z)), gen_labels)
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        avg_d = sum(d_losses) / len(d_losses)
        avg_g = sum(g_losses) / len(g_losses)
        print(f"Epoch [{epoch:02d}/{epochs}]  D Loss: {avg_d:.4f}  G Loss: {avg_g:.4f}")

        with torch.no_grad():
            samples = G(fixed_noise) * 0.5 + 0.5   # rescale to [0, 1]
        save_image(samples, f"output/epoch_{epoch:02d}.png", nrow=8)

    print("Training complete. Images saved to ./output/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    train(epochs=20)


if __name__ == "__main__":
    main()
