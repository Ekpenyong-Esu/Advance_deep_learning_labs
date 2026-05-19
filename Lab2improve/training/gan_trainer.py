"""
training/gan_trainer.py — Training Loops for Vanilla GAN and cGAN
==================================================================
Each function handles exactly one epoch and returns a metrics dict
so the caller (main.py) can log, print, and decide when to save.

  train_vanilla_gan_epoch  — Tasks 1 & 2 (BCE or Logistic loss)
  train_cgan_epoch         — Task 3       (conditional, BCE loss)

Design notes
------------
- Loss function is injected by the caller, keeping the loop agnostic
  to whether BCE or Logistic loss is used.
- `.detach()` is used on generated samples for the Discriminator step
  so that Generator gradients are not computed in that pass.
- A fresh noise vector is sampled for the Generator step to avoid bias.
"""

import torch
import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla GAN — one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_vanilla_gan_epoch(G, D, G_opt, D_opt, loss_fn,
                            loader, z_dim: int,
                            device: torch.device) -> dict:
    """
    Train Generator and Discriminator for a single epoch.

    Parameters
    ----------
    G, D     : vanilla Generator / Discriminator nn.Module
    G_opt    : optimiser for G
    D_opt    : optimiser for D
    loss_fn  : callable(preds, targets) → scalar  — bce_loss or logistic_loss
    loader   : DataLoader of flat MNIST images (784,)
    z_dim    : latent vector dimension
    device   : torch.device

    Returns
    -------
    dict
        "D_loss" — average Discriminator loss over the epoch
        "G_loss" — average Generator loss over the epoch
    """
    G.train()
    D.train()
    d_total = g_total = 0.0

    for X_real, _ in tqdm.tqdm(loader, leave=False, desc="GAN"):
        B      = X_real.size(0)
        X_real = X_real.to(device)
        ones   = torch.ones(B, 1, device=device)
        zeros  = torch.zeros(B, 1, device=device)

        # ── Discriminator step ──────────────────────────────────────────── #
        z      = torch.randn(B, z_dim, device=device)
        fake   = G(z).detach()          # stop Generator gradients here
        D_loss = loss_fn(D(X_real), ones) + loss_fn(D(fake), zeros)

        D_opt.zero_grad()
        D_loss.backward()
        D_opt.step()
        d_total += D_loss.item()

        # ── Generator step ───────────────────────────────────────────────── #
        z      = torch.randn(B, z_dim, device=device)
        G_loss = loss_fn(D(G(z)), ones)

        G_opt.zero_grad()
        G_loss.backward()
        G_opt.step()
        g_total += G_loss.item()

    n = len(loader)
    return {"D_loss": d_total / n, "G_loss": g_total / n}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional GAN — one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_cgan_epoch(G, D, G_opt, D_opt, loss_fn,
                     loader, z_dim: int,
                     device: torch.device) -> dict:
    """
    Train Conditional Generator and Discriminator for a single epoch.

    Both G and D accept an extra `labels` tensor as their second argument.

    Parameters
    ----------
    G, D     : ConditionalGenerator / ConditionalDiscriminator
    G_opt    : optimiser for G
    D_opt    : optimiser for D
    loss_fn  : callable(preds, targets) → scalar
    loader   : DataLoader of flat MNIST images with labels
    z_dim    : latent vector dimension
    device   : torch.device

    Returns
    -------
    dict
        "D_loss" — average Discriminator loss over the epoch
        "G_loss" — average Generator loss over the epoch
    """
    G.train()
    D.train()
    d_total = g_total = 0.0
    num_classes = G.label_emb.num_embeddings

    for X_real, y_real in tqdm.tqdm(loader, leave=False, desc="cGAN"):
        B      = X_real.size(0)
        X_real = X_real.to(device)
        y_real = y_real.to(device)
        ones   = torch.ones(B, 1, device=device)
        zeros  = torch.zeros(B, 1, device=device)

        # ── Discriminator step ──────────────────────────────────────────── #
        z      = torch.randn(B, z_dim, device=device)
        y_fake = torch.randint(0, num_classes, (B,), device=device)
        fake   = G(z, y_fake).detach()

        D_real = D(X_real, y_real)
        D_fake = D(fake, y_fake)
        D_loss = loss_fn(D_real, ones) + loss_fn(D_fake, zeros)

        D_opt.zero_grad()
        D_loss.backward()
        D_opt.step()
        d_total += D_loss.item()

        # ── Generator step ───────────────────────────────────────────────── #
        z      = torch.randn(B, z_dim, device=device)
        y_fake = torch.randint(0, num_classes, (B,), device=device)
        fake   = G(z, y_fake)
        G_loss = loss_fn(D(fake, y_fake), ones)

        G_opt.zero_grad()
        G_loss.backward()
        G_opt.step()
        g_total += G_loss.item()

    n = len(loader)
    return {"D_loss": d_total / n, "G_loss": g_total / n}
