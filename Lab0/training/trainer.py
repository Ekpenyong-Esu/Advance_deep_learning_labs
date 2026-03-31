"""
training/trainer.py — Generic Training Loop with Weights & Biases
==================================================================
This single trainer works for ALL experiments in this project.

It handles:
  • Creating the right optimiser (SGD or Adam) from the config dict
  • Training one epoch  (forward pass → loss → backward → weight update)
  • Evaluating on a validation set after every epoch (for early stopping / best model)
  • Evaluating on the test set ONCE after training is complete
  • Logging loss + accuracy to Weights & Biases at every batch and epoch
  • Printing a clean progress bar to the terminal with tqdm

Data split
----------
  70% training  — used every epoch to update weights
  15% validation — used every epoch to track best model (no weight updates)
  15% test       — used ONCE at the end to report final generalisation

Weights & Biases (wandb)
------------------------
All metrics are automatically synced to https://wandb.ai.
Log in once before running experiments:

    wandb login

Usage
-----
    from training.trainer import train_model

    best_acc = train_model(
        model           = my_model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        test_loader     = test_loader,
        config          = {"device": ..., "epochs": 10, "learning_rate": 0.001, "optimizer": "Adam"},
        experiment_name = "my_experiment",
    )
"""

import copy

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create the optimiser requested in `config`.

    Only trainable parameters (requires_grad=True) are passed so that
    frozen layers (e.g. AlexNet backbone in feature-extraction mode) are
    correctly skipped.
    """
    params = filter(lambda p: p.requires_grad, model.parameters())

    name = config.get("optimizer", "Adam").upper()
    lr   = config.get("learning_rate", 0.0001)

    if name == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),  # reduces overfitting by discouraging large weights
        )
    elif name == "ADAM":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-8),    # stability; protects against division by zero
            weight_decay=config.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimiser '{name}'. Use 'SGD' or 'Adam'.")


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch: train
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch.

    Returns
    -------
    avg_loss : float   — average loss over all batches
    accuracy : float   — training accuracy in percent
    """
    model.train()   # enables Dropout and BatchNorm training behaviour

    running_loss = 0.0
    correct = 0
    total   = 0

    bar = tqdm(loader, desc=f"  Epoch {epoch+1:>3} [train]", leave=False)

    for batch_idx, (images, labels) in enumerate(bar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()                      # clear gradients from previous step
        outputs = model(images)                    # forward pass
        loss    = criterion(outputs, labels)       # compute loss
        loss.backward()                            # backpropagation
        optimizer.step()                           # update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

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
# Shared evaluation: validation and test
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model, loader, criterion, device, epoch, tag):
    """
    Evaluate the model on any loader (validation or test).

    Parameters
    ----------
    tag : str   — label shown in the progress bar, e.g. "val" or "test"

    Returns
    -------
    avg_loss : float
    accuracy : float   — accuracy in percent
    """
    model.eval()   # disables Dropout; BatchNorm uses running statistics

    running_loss = 0.0
    correct = 0
    total   = 0

    bar = tqdm(loader, desc=f"  Epoch {epoch+1:>3} [{tag}]  ", leave=False)

    with torch.no_grad():   # no gradients needed → saves memory
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            bar.set_postfix(
                loss=f"{running_loss / (total / labels.size(0)):.4f}",
                acc=f"{100. * correct / total:.1f}%",
            )

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, test_loader,
                config: dict, experiment_name: str,
                project: str = "advanced-ai-lab") -> float:
    """
    Full training loop: train for N epochs, evaluate on the validation set
    after each epoch, and evaluate on the test set exactly once at the end.

    Data split expected
    -------------------
      train_loader  — 70% of the dataset
      val_loader    — 15% of the dataset (guides best-model selection)
      test_loader   — 15% of the dataset (touched only once, after training)

    The best model is determined by the highest validation accuracy across
    all epochs. That best model is then restored before test evaluation,
    ensuring the test result reflects the best checkpoint — not the last one.

    Parameters
    ----------
    model           : nn.Module
    train_loader    : DataLoader  (70%)
    val_loader      : DataLoader  (15%)
    test_loader     : DataLoader  (15%)
    config          : dict — must contain: 'device', 'epochs',
                             'learning_rate', 'optimizer'
    experiment_name : str  — used as wandb run name
    project         : str  — wandb project name

    Returns
    -------
    best_val_accuracy : float   — highest validation accuracy (%) achieved
    """
    device = config.get("device", torch.device("cpu"))
    epochs = config.get("epochs", 10)

    model = model.to(device)

    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    wandb.init(project=project,
               name=experiment_name,
               config=config,
               reinit=True)

    # ── Print experiment header ──────────────────────────────────────────── #
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'═' * 62}")
    print(f"  Experiment  : {experiment_name}")
    print(f"  Device      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Optimiser   : {config.get('optimizer')}  lr={config.get('learning_rate')}")
    print(f"  Epochs      : {epochs}")
    print(f"  Trainable   : {trainable:,} parameters")
    print(f"  Split       : 70% train / 15% val / 15% test")
    print(f"  W&B project : {project}  (view at https://wandb.ai)")
    print(f"{'═' * 62}\n")

    best_val_accuracy  = 0.0
    best_model_state   = None   # deep-copied weights of the best checkpoint

    # ── Training + validation loop ───────────────────────────────────────── #
    for epoch in range(epochs):

        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
        )

        val_loss, val_acc = _evaluate(
            model, val_loader, criterion, device, epoch, tag="val",
        )

        # Log all epoch-level metrics at the same step so curves overlay
        # correctly with the per-batch batch_loss curve in W&B.
        epoch_step = (epoch + 1) * len(train_loader)
        wandb.log({
            "train_loss"    : train_loss,
            "train_accuracy": train_acc,
            "val_loss"      : val_loss,
            "val_accuracy"  : val_acc,
            "epoch"         : epoch + 1,
        }, step=epoch_step)

        print(f"  Epoch {epoch+1:>2}/{epochs}  |  "
              f"Train  loss={train_loss:.4f}  acc={train_acc:.1f}%  |  "
              f"Val    loss={val_loss:.4f}  acc={val_acc:.1f}%")

        # Track the best checkpoint by validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state  = copy.deepcopy(model.state_dict())

    # ── Restore best checkpoint ──────────────────────────────────────────── #
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  ✓ Restored best checkpoint  (val acc = {best_val_accuracy:.2f}%)")

    # ── Test evaluation — runs exactly once, on the best model ──────────── #
    test_loss, test_acc = _evaluate(
        model, test_loader, criterion, device, epoch=epochs - 1, tag="test",
    )

   final_step = epochs * len(train_loader)

    wandb.log({
        "test_loss"    : test_loss,
        "test_accuracy": test_acc,
    }, step=final_step)

    # ── Summary ──────────────────────────────────────────────────────────── #
    wandb.finish()

    print(f"\n{'─' * 62}")
    print(f"  ✓ {experiment_name}")
    print(f"    Best Val Accuracy  : {best_val_accuracy:.2f}%")
    print(f"    Final Test Accuracy: {test_acc:.2f}%")
    print(F"    W&B dashboard      : https://wandb.ai")
    print(f"{'─' * 62}\n")

    return best_val_accuracy