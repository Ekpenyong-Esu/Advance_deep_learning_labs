"""
training/engine.py — Per-Epoch Training and Evaluation Loops
=============================================================
Contains the two core loop functions used by the trainer:

  train_one_epoch(model, loader, optimizer, criterion, device, epoch)
      Runs a full forward + backward pass over the training loader.
      Logs per-batch loss to Weights & Biases.
      Returns (avg_loss, accuracy_percent).

  evaluate(model, loader, criterion, device, epoch, tag)
      Runs inference over any loader (val or test) with no gradient updates.
      Returns (avg_loss, accuracy_percent, f1_score).

Usage
-----
    from training.engine import train_one_epoch, evaluate
"""

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import f1_score, precision_score, recall_score

from training.batch_utils import unpack_batch, forward_pass


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    grad_clip: Optional[float] = None,
    scheduler        = None,
):
    """
    Train the model for one complete pass over the training loader.

    Parameters
    ----------
    epoch     : int   — 0-based epoch index (used for progress bar label and W&B step)
    grad_clip : float — max gradient norm; None disables clipping
    scheduler         — optional LR scheduler stepped every batch (e.g. warmup)

    Returns
    -------
    avg_loss : float
    accuracy : float  — training accuracy in percent
    """
    model.train()

    running_loss = 0.0
    correct      = 0
    total        = 0

    bar = tqdm(loader, desc=f"  Epoch {epoch + 1:>3} [train]", leave=False)

    for batch_idx, batch in enumerate(bar):
        inputs, labels = unpack_batch(batch)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = forward_pass(model, inputs, device)
        loss    = criterion(outputs, labels)
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

        bar.set_postfix(
            loss=f"{running_loss / (batch_idx + 1):.4f}",
            acc=f"{100. * correct / total:.1f}%",
        )

        global_step = epoch * len(loader) + batch_idx
        wandb.log({"batch_loss": loss.item()}, step=global_step)

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop (val and test share this)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    tag:       str,
):
    """
    Evaluate the model on any loader without updating weights.

    Parameters
    ----------
    tag   : "val" | "test" — used in the progress bar label only
    epoch : int            — 0-based (used for progress bar label only)

    Returns
    -------
    avg_loss  : float
    accuracy  : float  — accuracy in percent
    f1        : float  — binary F1 score (positive class = 1)
    precision : float  — binary precision
    recall    : float  — binary recall
    """
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    bar = tqdm(loader, desc=f"  Epoch {epoch + 1:>3} [{tag}]  ", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(bar):
            inputs, labels = unpack_batch(batch)
            labels = labels.to(device)

            outputs = forward_pass(model, inputs, device)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            bar.set_postfix(
                loss=f"{running_loss / (batch_idx + 1):.4f}",
                acc=f"{100. * correct / total:.1f}%",
            )

    avg_loss  = running_loss / len(loader)
    accuracy  = 100. * correct / total
    f1        = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="binary", zero_division=0)

    return avg_loss, accuracy, f1, precision, recall
