"""
training/cnn_trainer.py — CNN Training and Evaluation Loops (Task 4)
=====================================================================
Provides two functions:

  train_cnn    — trains a MNISTClassifier for N epochs using Adam +
                 CrossEntropyLoss and prints per-epoch accuracy.

  evaluate_cnn — evaluates a trained model on a DataLoader and returns
                 accuracy as a percentage.

Both functions accept (B, 1, 28, 28) image tensors in [-1, 1].
"""

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn(model: nn.Module, loader,
              device: torch.device,
              lr: float = 1e-3,
              epochs: int = 5,
              early_stopping_patience: int = 5) -> nn.Module:
    """
    Train `model` with Adam + CrossEntropyLoss.

    Parameters
    ----------
    model                   : MNISTClassifier (outputs raw logits)
    loader                  : DataLoader of (image, label) pairs; images (B, 1, 28, 28)
    device                  : torch.device
    lr                      : Adam learning rate
    epochs                  : number of full passes over the training data
    early_stopping_patience : stop if loss does not improve for this many consecutive epochs

    Returns
    -------
    model : the same nn.Module, now trained (in-place modification)
    """
    model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss     = float("inf")
    no_improve    = 0

    for epoch in range(epochs):
        total_loss = correct = total = 0

        for images, labels in tqdm.tqdm(loader, leave=False,
                                        desc=f"CNN {epoch + 1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        acc      = 100.0 * correct / total
        avg_loss = total_loss / len(loader)
        print(f"  CNN epoch {epoch + 1}/{epochs} "
              f"— loss: {avg_loss:.4f}  acc: {acc:.2f}%")
        wandb.log({"cnn/train_loss": avg_loss, "cnn/train_acc": acc}, step=epoch + 1)

        # Early stopping: check whether loss improved
        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"  Early stopping triggered after {epoch + 1} epochs "
                      f"(no loss improvement for {early_stopping_patience} consecutive epochs).")
                break

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cnn(model: nn.Module, loader,
                 device: torch.device) -> float:
    """
    Compute classification accuracy on a DataLoader.

    Parameters
    ----------
    model   : trained MNISTClassifier
    loader  : DataLoader of (image, label) pairs
    device  : torch.device

    Returns
    -------
    accuracy : float in [0, 100]
    """
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds   = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return 100.0 * correct / total
