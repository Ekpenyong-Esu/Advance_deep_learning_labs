"""
utils/losses.py — GAN Loss Functions (Task 2)
=============================================
Two loss functions are provided for the GAN experiments:

BCE loss  (Task 1 — standard GAN)
----------------------------------
  Pair with Discriminator that has Sigmoid as its final activation.

  D_loss = −E[log D(x)]   − E[log(1 − D(G(z)))]
         = BCE(D(x_real), 1) + BCE(D(G(z)), 0)

  G_loss = −E[log D(G(z))]
         = BCE(D(G(z)), 1)

Logistic loss  (Task 2 — Brandon Amos blog)
--------------------------------------------
  Pair with Discriminator that returns raw logits (no Sigmoid).
  BCEWithLogitsLoss fuses Sigmoid + BCE for numerical stability.

  D_loss = −E[log σ(D(x))]   − E[log(1 − σ(D(G(z))))]
         = BCEWithLogitsLoss(D(x_real), 1) + BCEWithLogitsLoss(D(G(z)), 0)

  G_loss = −E[log σ(D(G(z)))]
         = BCEWithLogitsLoss(D(G(z)), 1)

  The non-saturating Generator loss (−log σ) avoids vanishing gradients
  that occur with the original minimax formulation (log(1 − σ)).

Usage
-----
    from utils.losses import get_loss_fn

    loss_fn = get_loss_fn("bce")       # Task 1
    loss_fn = get_loss_fn("logistic")  # Task 2
"""

import torch
import torch.nn.functional as F


def bce_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Standard binary cross-entropy.
    Requires the Discriminator to output probabilities via Sigmoid.
    """
    return F.binary_cross_entropy(preds, targets)


def logistic_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable logistic GAN loss (BCEWithLogitsLoss).
    Requires the Discriminator to output raw logits (no Sigmoid activation).
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def get_loss_fn(name: str):
    """
    Return the requested loss function by name.

    Parameters
    ----------
    name : "bce" for Task 1, "logistic" for Task 2

    Returns
    -------
    Callable[[Tensor, Tensor], Tensor]
    """
    if name == "bce":
        return bce_loss
    if name == "logistic":
        return logistic_loss
    raise ValueError(f"Unknown loss '{name}'. Expected 'bce' or 'logistic'.")
