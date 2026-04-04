"""
training/trainer.py — Orchestration: Full Training + Evaluation Loop
=====================================================================
Public entry point for all experiments. Wires together the three
sub-modules that each own one concern:

  optimizer.py   — builds the right PyTorch optimiser from a config dict
  batch_utils.py — unpacks 2-tuple (ANN/LSTM) or 3-tuple (BERT) batches
  engine.py      — per-epoch train loop and shared evaluation loop

Data split expected
-------------------
  train_loader  — 70%  weights updated every step
  val_loader    — 15%  evaluated after every epoch; drives best-checkpoint
  test_loader   — 15%  evaluated ONCE on the best checkpoint after training

Usage
-----
    from training.trainer import train_model

    results = train_model(
        model           = my_model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        test_loader     = test_loader,
        config          = {"device": ..., "epochs": 10,
                           "learning_rate": 0.001, "optimizer": "Adam"},
        experiment_name = "Task01_ANN_small",
    )
    # results = {"best_val_accuracy": ..., "test_accuracy": ..., "test_f1": ...}
"""

import copy

import torch
import torch.nn as nn
import wandb

from training.optimizer  import build_optimizer
from training.engine     import train_one_epoch, evaluate
from transformers import get_linear_schedule_with_warmup


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    config:          dict,
    experiment_name: str,
    project:         str = "advanced-ai-lab-1",
) -> dict:
    """
    Train for N epochs, pick the best checkpoint by validation accuracy,
    and report final metrics on the held-out test set.

    Parameters
    ----------
    model           : nn.Module
    train_loader    : DataLoader  (70%)
    val_loader      : DataLoader  (15%)
    test_loader     : DataLoader  (15%)
    config          : dict — must contain 'device', 'epochs',
                             'learning_rate', 'optimizer'
    experiment_name : str  — W&B run name
    project         : str  — W&B project name

    Returns
    -------
    dict
        "best_val_accuracy" — highest val accuracy (%) across all epochs
        "test_accuracy"     — accuracy (%) on the best-checkpoint weights
        "test_f1"           — binary F1 on the test set
    """
    device    = config.get("device", torch.device("cpu"))
    epochs    = config.get("epochs", 10)
    grad_clip = config.get("grad_clip", None)

    model     = model.to(device)
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    # ── Optional LR scheduler (linear warmup → decay, for Transformers) ── #
    scheduler = None
    
    if config.get("use_scheduler", False):
        
        total_steps  = len(train_loader) * epochs
        warmup_steps = int(config.get("warmup_ratio", 0.1) * total_steps)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    wandb.init(project=project, name=experiment_name, config=config, reinit=True)

    # ── Experiment header ────────────────────────────────────────────────── #
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'═' * 62}")
    print(f"  Experiment  : {experiment_name}")
    print(
        f"  Device      : {device}"
        + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )
    print(f"  Optimiser   : {config.get('optimizer')}  lr={config.get('learning_rate')}")
    print(f"  Epochs      : {epochs}")
    print(f"  Trainable   : {trainable:,} parameters")
    print(f"  Split       : 70% train / 15% val / 15% test")
    print(f"  W&B project : {project}  (view at https://wandb.ai)")
    print(f"{'═' * 62}\n")

    best_val_accuracy = 0.0
    best_model_state  = None

    # ── Train + validate ─────────────────────────────────────────────────── #
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            grad_clip=grad_clip, scheduler=scheduler,
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(
            model, val_loader, criterion, device, epoch, tag="val",
        )

        epoch_step = (epoch + 1) * len(train_loader)
        wandb.log({
            "train_loss"      : train_loss,
            "train_accuracy"  : train_acc,
            "val_loss"        : val_loss,
            "val_accuracy"    : val_acc,
            "val_f1"          : val_f1,
            "val_precision"   : val_prec,
            "val_recall"      : val_rec,
            "epoch"           : epoch + 1,
        }, step=epoch_step)

        print(
            f"  Epoch {epoch + 1:>2}/{epochs}  |  "
            f"train  loss={train_loss:.4f}  acc={train_acc:.1f}%  |  "
            f"val  loss={val_loss:.4f}  acc={val_acc:.1f}%  f1={val_f1:.4f}  "
            f"prec={val_prec:.4f}  rec={val_rec:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state  = copy.deepcopy(model.state_dict())

    # ── Final test — best checkpoint only ───────────────────────────────── #
    model.load_state_dict(best_model_state)

    test_loss, test_acc, test_f1, test_prec, test_rec = evaluate(
        model, test_loader, criterion, device, epochs - 1, tag="test",
    )

    # Use summary instead of log so test metrics appear as a single scalar
    # in the W&B run overview rather than a one-point line on a time-series plot.
    wandb.run.summary["test_loss"]      = test_loss
    wandb.run.summary["test_accuracy"]  = test_acc
    wandb.run.summary["test_f1"]        = test_f1
    wandb.run.summary["test_precision"] = test_prec
    wandb.run.summary["test_recall"]    = test_rec

    print(f"\n{'─' * 62}")
    print(
        f"  [FINAL]  Test Accuracy : {test_acc:.2f}%   "
        f"F1 : {test_f1:.4f}   Precision : {test_prec:.4f}   Recall : {test_rec:.4f}"
    )
    print(f"{'─' * 62}\n")

    wandb.finish()

    return {
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy"    : test_acc,
        "test_f1"          : test_f1,
        "test_precision"   : test_prec,
        "test_recall"      : test_rec,
    }
