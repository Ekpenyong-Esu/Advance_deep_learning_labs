"""
training/diffusion_trainer.py — Diffusion Model Training Loop (Task 5)
=======================================================================
Trains a GaussianDiffusion model using the L_simple objective:

    L = E_{x_0, t, ε} [ ‖ε − ε_θ(x_t, t, y)‖² ]

The caller is responsible for constructing the model and DataLoader.
This module owns only the training loop.
"""

import torch
import torch.optim as optim
import tqdm
import wandb


def train_diffusion(diffusion, loader,
                    device: torch.device,
                    lr: float = 2e-4,
                    epochs: int = 20) -> None:
    """
    Train the GaussianDiffusion model for `epochs` epochs.

    Parameters
    ----------
    diffusion : GaussianDiffusion instance (wraps the denoiser)
    loader    : DataLoader of (flat_image, label) pairs; images (B, 784) in [0, 1]
    device    : torch.device
    lr        : Adam learning rate
    epochs    : number of full passes over the training data

    Returns
    -------
    None — the model is updated in-place.
    """
    diffusion.to(device)
    optimizer = optim.Adam(diffusion.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in tqdm.tqdm(loader, leave=False,
                                        desc=f"Diffusion {epoch + 1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            loss = diffusion.p_losses(images, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"  Diffusion epoch {epoch + 1}/{epochs} — MSE loss: {avg:.6f}")
        wandb.log({"diffusion/mse_loss": avg}, step=epoch + 1)
