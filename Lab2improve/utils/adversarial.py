"""
utils/adversarial.py — Adversarial Attack Utilities (Task 4)
=============================================================
Implements targeted FGSM (Fast Gradient Sign Method) to craft adversarial
examples that fool a trained CNN into misclassifying a source digit as a
chosen target digit.

Targeted FGSM formula
---------------------
  x_adv = clip( x − ε · sign( ∇_x L(f(x), y_target) ) )

  Subtracting the signed gradient minimises the cross-entropy loss toward
  the target class, which causes the model to predict the target label.

  Contrast with untargeted FGSM, which adds the gradient to maximise the
  loss w.r.t. the true label.

References
----------
  Goodfellow et al. (2014) — "Explaining and Harnessing Adversarial Examples"
  Jason Carter GitHub      — MNIST adversarial attack examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_targeted_attack(model: nn.Module,
                         images: torch.Tensor,
                         target_class: int,
                         epsilon: float,
                         device: torch.device) -> torch.Tensor:
    """
    Perturb `images` so that `model` classifies them as `target_class`.

    Parameters
    ----------
    model        : trained MNISTClassifier (outputs logits)
    images       : (B, 1, 28, 28) float tensor with values in [-1, 1]
    target_class : int in [0, 9] — desired mis-classification target
    epsilon      : perturbation magnitude (0.3 is a common default)
    device       : torch.device

    Returns
    -------
    adv_images : (B, 1, 28, 28) adversarial images clamped to [-1, 1]
    """
    model.eval()
    images = images.clone().detach().to(device)
    images.requires_grad_(True)

    targets = torch.full((images.size(0),), target_class,
                         dtype=torch.long, device=device)

    logits = model(images)
    loss   = F.cross_entropy(logits, targets)

    model.zero_grad()
    loss.backward()

    # Subtract the signed gradient → move toward target class
    adv_images = torch.clamp(
        images.detach() - epsilon * images.grad.sign(),
        min=-1.0, max=1.0,
    )
    return adv_images


def classify_random_noise(model: nn.Module,
                          num_samples: int,
                          device: torch.device) -> torch.Tensor:
    """
    Pass random Gaussian noise images through `model` and return predictions.

    Parameters
    ----------
    model       : trained MNISTClassifier
    num_samples : how many noise images to classify
    device      : torch.device

    Returns
    -------
    predictions : (num_samples,) LongTensor of predicted class indices
    """
    model.eval()
    noise = torch.randn(num_samples, 1, 28, 28, device=device)
    with torch.no_grad():
        logits = model(noise)
    return logits.argmax(dim=1)
