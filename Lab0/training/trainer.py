"""
training/trainer.py — Generic Training Loop with TensorBoard
=============================================================
This single trainer works for ALL experiments in this project.

It handles:
  • Creating the right optimiser (SGD or Adam) from the config dict
  • Training one epoch  (forward pass → loss → backward → weight update)
  • Evaluating on the test set after every epoch
  • Logging loss + accuracy to TensorBoard at every batch and epoch
  • Printing a clean progress bar to the terminal with tqdm

TensorBoard
-----------
After running any experiment, open a terminal and run:

    tensorboard --logdir=runs

Then open http://localhost:6006 in your browser to see the plots.

Usage
-----
    from training.trainer import train_model

    best_acc = train_model(
        model          = my_model,
        train_loader   = train_loader,
        test_loader    = test_loader,
        config         = {"device": ..., "epochs": 10, "learning_rate": 0.001, "optimizer": "Adam"},
        experiment_name= "my_experiment",
    )
"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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
    # Filter to trainable parameters only
    params = filter(lambda p: p.requires_grad, model.parameters())

    name = config.get("optimizer", "Adam").upper()
    
    lr   = config.get("learning_rate", 0.0001)

    if name == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),
        )
    elif name == "ADAM":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=config.get("betas", (0.9, 0.999)),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimiser '{name}'. Use 'SGD' or 'Adam'.")


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch: train
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optimizer, criterion, device,
                     epoch, writer, tag):
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

        optimizer.zero_grad()           # clear gradients from previous step

        outputs = model(images)         # forward pass
        loss    = criterion(outputs, labels)   # compute loss

        loss.backward()                 # backpropagation: compute gradients
        optimizer.step()                # update weights

        # ── Statistics ──────────────────────────────────────────────────── #
        running_loss += loss.item()
        _, predicted = outputs.max(1)   # index of the highest-score class
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        bar.set_postfix(
            loss=f"{running_loss / (batch_idx + 1):.4f}",
            acc=f"{100. * correct / total:.1f}%",
        )

        # Log per-batch loss to TensorBoard
        step = epoch * len(loader) + batch_idx
        writer.add_scalar(f"{tag}/batch_loss", loss.item(), step)

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total

    # Log per-epoch metrics
    writer.add_scalar(f"{tag}/train_loss",     avg_loss, epoch)
    writer.add_scalar(f"{tag}/train_accuracy", accuracy, epoch)

    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch: evaluate
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model, loader, criterion, device, epoch, writer, tag):
    """
    Evaluate the model on the test/validation set.

    Returns
    -------
    avg_loss : float
    accuracy : float   — test accuracy in percent
    """
    model.eval()   # disables Dropout; BatchNorm uses running statistics

    running_loss = 0.0
    correct = 0
    total   = 0

    bar = tqdm(loader, desc=f"  Epoch {epoch+1:>3} [test] ", leave=False)

    with torch.no_grad():   # no gradients needed for evaluation → saves memory
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

    writer.add_scalar(f"{tag}/test_loss",     avg_loss, epoch)
    writer.add_scalar(f"{tag}/test_accuracy", accuracy, epoch)

    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Public API: train for N epochs + evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, test_loader,
                config: dict, experiment_name: str,
                log_dir: str = "./runs") -> float:
    """
    Full training loop: train for N epochs, evaluate after each epoch,
    log everything to TensorBoard, and return the best test accuracy.

    Parameters
    ----------
    model           : nn.Module   — any PyTorch model
    train_loader    : DataLoader
    test_loader     : DataLoader
    config          : dict        — must contain at minimum:
                        'device', 'epochs', 'learning_rate', 'optimizer'
    experiment_name : str         — used as TensorBoard tag and printed label
    log_dir         : str         — root folder for TensorBoard logs

    Returns
    -------
    best_accuracy : float   — highest test accuracy (%) achieved
    """
    device = config.get("device", torch.device("cpu"))
    epochs = config.get("epochs", 10)

    model = model.to(device)

    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()   # standard for multi-class classification

    tb_path = os.path.join(log_dir, experiment_name)
    writer  = SummaryWriter(log_dir=tb_path)

    # ── Print experiment header ──────────────────────────────────────────── #
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'═' * 62}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Device     : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Optimiser  : {config.get('optimizer')}  lr={config.get('learning_rate')}")
    print(f"  Epochs     : {epochs}")
    print(f"  Trainable  : {trainable:,} parameters")
    print(f"  TensorBoard: tensorboard --logdir={log_dir}")
    print(f"{'═' * 62}\n")

    best_accuracy = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, writer, experiment_name,
        )
        test_loss, test_acc = _evaluate(
            model, test_loader, criterion,
            device, epoch, writer, experiment_name,
        )

        print(f"  Epoch {epoch+1:>2}/{epochs}  |  "
              f"Train  loss={train_loss:.4f}  acc={train_acc:.1f}%  |  "
              f"Test   loss={test_loss:.4f}  acc={test_acc:.1f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc

    writer.close()

    print(f"\n{'─' * 62}")
    print(f"  ✓ {experiment_name}")
    print(f"    Best Test Accuracy : {best_accuracy:.2f}%")
    print(f"    TensorBoard logs   : {tb_path}")
    print(f"{'─' * 62}\n")

    return best_accuracy
