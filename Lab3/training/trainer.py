"""
training/trainer.py — Training and Validation Loops for Image Captioning
=========================================================================
Each function processes a single epoch and returns a metrics dictionary
so the caller (main.ipynb) can log, print, and decide when to checkpoint.

  train_epoch    — forward pass, backprop, gradient clip, optimiser step
  validate_epoch — forward pass only, no gradients, returns val loss

Design notes
------------
- Teacher forcing is applied during both training and validation:
    input   caption : tokens[0 .. T-1]  (includes <SOS>, excludes last token)
    target  caption : tokens[1 .. T  ]  (excludes <SOS>, includes <EOS>)
- PAD tokens are masked out of the loss via ``ignore_index`` in CrossEntropyLoss
  so they do not influence the gradient or the reported loss value.
- Gradient clipping (``GRAD_CLIP`` from config) prevents exploding gradients
  that are common in LSTM training.
- The encoder and decoder are passed separately so the caller can apply
  different learning rates or freeze the encoder independently.
"""

import torch
import torch.nn as nn
import tqdm

import config
from data.vocabulary import PAD_IDX


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def get_criterion() -> nn.CrossEntropyLoss:
    """
    Cross-entropy loss that ignores padding positions.

    The loss is computed over all non-PAD positions, which means shorter
    captions in a batch do not dominate the gradient.
    """
    return nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    encoder,
    decoder,
    loader:       torch.utils.data.DataLoader,
    criterion:    nn.Module,
    enc_optimizer: torch.optim.Optimizer,
    dec_optimizer: torch.optim.Optimizer,
    device:       torch.device,
    grad_clip:    float = None,
) -> dict:
    """
    Run one full training epoch.

    Parameters
    ----------
    encoder       : ImageEncoder  (models/encoder.py)
    decoder       : CaptionDecoder (models/decoder.py)
    loader        : training DataLoader — yields (images, captions) batches
    criterion     : loss function (CrossEntropyLoss with ignore_index=PAD_IDX)
    enc_optimizer : optimiser for encoder parameters (can be None if frozen)
    dec_optimizer : optimiser for decoder parameters
    device        : torch.device
    grad_clip     : gradient clipping norm; defaults to config.GRAD_CLIP

    Returns
    -------
    dict
        "train_loss"    — average cross-entropy loss per non-PAD token
        "train_batches" — number of batches processed
    """
    grad_clip = grad_clip if grad_clip is not None else config.GRAD_CLIP

    encoder.train()
    decoder.train()

    total_loss   = 0.0
    total_tokens = 0

    for images, captions in tqdm.tqdm(loader, leave=False, desc="Train"):
        images   = images.to(device)
        captions = captions.to(device)

        # ── split into input / target ──────────────────────────────────── #
        # input  : all tokens except the last  → [<SOS>, w1, …, w_{T-1}]
        # target : all tokens except the first → [w1, …, w_T, <EOS>]
        cap_input  = captions[:, :-1]   # (batch, T)
        cap_target = captions[:, 1:]    # (batch, T)

        # ── forward pass ──────────────────────────────────────────────── #
        image_features = encoder(images)                         # (batch, embed_dim)
        logits         = decoder(image_features, cap_input)      # (batch, T, vocab_size)

        # ── compute loss ────────────────────────────────────────────────── #
        # CrossEntropyLoss expects (N, C, ...) so we reshape:
        # logits  → (batch * T, vocab_size)
        # targets → (batch * T,)
        batch_t, seq_t, vocab_size = logits.shape
        loss = criterion(
            logits.reshape(batch_t * seq_t, vocab_size),
            cap_target.reshape(batch_t * seq_t),
        )

        # ── backprop ────────────────────────────────────────────────────── #
        if enc_optimizer is not None:
            enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        loss.backward()

        # Clip gradients to prevent LSTM explosion
        nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        if enc_optimizer is not None:
            nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)

        if enc_optimizer is not None:
            enc_optimizer.step()
        dec_optimizer.step()

        # Track loss weighted by the number of non-PAD tokens in the batch
        non_pad = (cap_target != PAD_IDX).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    return {"train_loss": avg_loss, "train_batches": len(loader)}


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

def validate_epoch(
    encoder,
    decoder,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    """
    Evaluate the model on the validation set for one epoch.

    No gradients are computed.  The function mirrors ``train_epoch``
    but skips backpropagation and optimiser steps.

    Parameters
    ----------
    encoder   : ImageEncoder
    decoder   : CaptionDecoder
    loader    : validation DataLoader
    criterion : same loss function used during training
    device    : torch.device

    Returns
    -------
    dict
        "val_loss"    — average cross-entropy loss per non-PAD token
        "val_batches" — number of batches processed
    """
    encoder.eval()
    decoder.eval()

    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for images, captions in tqdm.tqdm(loader, leave=False, desc="Val  "):
            images   = images.to(device)
            captions = captions.to(device)

            cap_input  = captions[:, :-1]
            cap_target = captions[:, 1:]

            image_features = encoder(images)
            logits         = decoder(image_features, cap_input)

            batch_t, seq_t, vocab_size = logits.shape
            loss = criterion(
                logits.reshape(batch_t * seq_t, vocab_size),
                cap_target.reshape(batch_t * seq_t),
            )

            non_pad = (cap_target != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    return {"val_loss": avg_loss, "val_batches": len(loader)}
