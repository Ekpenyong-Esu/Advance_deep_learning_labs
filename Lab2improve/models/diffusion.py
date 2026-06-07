"""
models/diffusion.py — Conditional Diffusion Model for MNIST (Task 5)
=====================================================================
Implements a simple Denoising Diffusion Probabilistic Model (DDPM)
(Ho et al., 2020) with a conditional MLP denoiser suited for 28×28
greyscale images represented as flat 784-dim vectors.

Components
----------
  sinusoidal_embedding  — fixed-frequency encoding of the diffusion timestep t
  ConditionalDenoiser   — MLP that predicts the noise given (x_t, t, y)
  GaussianDiffusion     — wraps the denoiser with the full forward / reverse process

Training objective (L_simple)
------------------------------
  Sample x_0, t, ε.
  x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
  Loss = ‖ε − ε_θ(x_t, t, y)‖²

Inference (ancestral sampling)
-------------------------------
  Start from x_T ~ N(0, I).
  For t = T, …, 1:
      μ_θ = (1/√α_t) · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t, y))
      x_{t−1} = μ_θ + √(β̃_t) · z    (z ~ N(0,I) if t > 0, else 0)
  Return x_0.

References
----------
  Ho et al. (2020) — "Denoising Diffusion Probabilistic Models"
  https://arxiv.org/abs/2006.11239
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Encode integer diffusion timesteps as fixed sinusoidal vectors.

    Parameters
    ----------
    timesteps : (B,) long tensor of timestep indices
    dim       : output embedding dimension (should be even)

    Returns
    -------
    emb : (B, dim) float tensor
    """
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10_000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half - 1, 1)
    )
    args = timesteps[:, None].float() * freqs[None]    # (B, half)
    emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Denoiser network
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalDenoiser(nn.Module):
    """
    MLP-based conditional denoiser ε_θ(x_t, t, y).

    Input  : noisy image x_t (B, x_dim)  +  timestep t  +  class label y
    Output : predicted noise ε           (B, x_dim)

    Parameters
    ----------
    x_dim       : flat image dimension (784 for MNIST)
    t_emb_dim   : sinusoidal timestep embedding size
    num_classes : number of class labels
    h_dim       : hidden layer width
    """

    def __init__(self, x_dim: int = 784, t_emb_dim: int = 128,
                 num_classes: int = 10, h_dim: int = 512):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.label_emb = nn.Embedding(num_classes, t_emb_dim)
        self.time_proj  = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(x_dim + 2 * t_emb_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, x_dim),
        )

    def forward(self, x: torch.Tensor,
                t: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)   # (B, t_emb_dim)
        t_emb = self.time_proj(t_emb)                      # (B, t_emb_dim)
        y_emb = self.label_emb(y)                          # (B, t_emb_dim)
        inp   = torch.cat([x, t_emb, y_emb], dim=1)       # (B, x_dim + 2*t_emb_dim)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian Diffusion — forward process, training loss, and sampling
# ─────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion(nn.Module):
    """
    DDPM forward and reverse diffusion process.

    Parameters
    ----------
    denoiser   : ConditionalDenoiser
    timesteps  : total number of noise steps T
    beta_start : starting variance β_1
    beta_end   : ending variance β_T
    """

    def __init__(self, denoiser: nn.Module, timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.denoiser  = denoiser
        self.timesteps = timesteps

        betas              = torch.linspace(beta_start, beta_end, timesteps)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas",              betas)
        self.register_buffer("alphas",             alphas)
        self.register_buffer("alphas_cumprod",     alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",
                             torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas",
                             torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance",
                             betas * (1.0 - alphas_cumprod_prev)
                             / (1.0 - alphas_cumprod))

    # ── helpers ──────────────────────────────────────────────────────────── #

    def _gather(self, a: torch.Tensor, t: torch.Tensor,
                ref_shape) -> torch.Tensor:
        """Gather schedule values at timestep indices t and broadcast."""
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(ref_shape) - 1)))

    # ── forward process  q(x_t | x_0) ───────────────────────────────────── #

    def forward_sample(self, x0: torch.Tensor,
                       t: torch.Tensor):
        """
        Add noise to x0 at arbitrary timestep t.

        Returns
        -------
        x_noisy : x_t
        noise   : the Gaussian noise that was added (prediction target)
        """
        noise   = torch.randn_like(x0)
        sqrt_a  = self._gather(self.sqrt_alphas_cumprod,           t, x0.shape)
        sqrt_1a = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_a * x0 + sqrt_1a * noise, noise

    # ── training loss  L_simple ───────────────────────────────────────────── #

    def p_losses(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute L_simple = MSE(true_noise, predicted_noise) for a random t.

        Parameters
        ----------
        x0 : (B, x_dim) clean images in [0, 1]
        y  : (B,)        class label indices
        """
        B       = x0.size(0)
        t       = torch.randint(0, self.timesteps, (B,), device=x0.device)
        x_noisy, noise = self.forward_sample(x0, t)
        pred_noise     = self.denoiser(x_noisy, t, y)
        return F.mse_loss(pred_noise, noise)

    # ── reverse process  p_θ(x_0)  (ancestral sampling) ─────────────────── #

    @torch.no_grad()
    def sample(self, y: torch.Tensor, x_dim: int = 784) -> torch.Tensor:
        """
        Generate images by iteratively denoising from pure Gaussian noise.

        Parameters
        ----------
        y     : (B,) class label tensor
        x_dim : flat image dimension

        Returns
        -------
        x : (B, x_dim) generated images (approximate [0, 1] range)
        """
        device = self.betas.device
        B = y.size(0)
        x = torch.randn(B, x_dim, device=device)

        for t_idx in reversed(range(self.timesteps)):
            t_batch   = torch.full((B,), t_idx, device=device, dtype=torch.long)
            betas_t   = self._gather(self.betas,                          t_batch, x.shape)
            sqrt_1a_t = self._gather(self.sqrt_one_minus_alphas_cumprod,  t_batch, x.shape)
            sqrt_ra_t = self._gather(self.sqrt_recip_alphas,              t_batch, x.shape)

            pred_noise = self.denoiser(x, t_batch, y)
            model_mean = sqrt_ra_t * (x - betas_t / sqrt_1a_t * pred_noise)

            if t_idx > 0:
                var_t = self._gather(self.posterior_variance, t_batch, x.shape)
                x     = model_mean + torch.sqrt(var_t) * torch.randn_like(x)
            else:
                x = model_mean

        return x
