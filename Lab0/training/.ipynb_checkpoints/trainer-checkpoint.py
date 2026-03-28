"""training/trainer.py — Generic Training Loop with TensorBoard"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def _build_optimizer(model, config):
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = config.get("optimizer", "Adam").upper()
    lr   = config.get("learning_rate", 0.0001)
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr,
                               momentum=config.get("momentum", 0.9),
                               weight_decay=config.get("weight_decay", 0.0))
    elif name == "ADAM":
        return torch.optim.Adam(params, lr=lr,
                                betas=config.get("betas", (0.9, 0.999)),
                                eps=config.get("eps", 1e-8),
                                weight_decay=config.get("weight_decay", 0.0))
    else:
        raise ValueError(f"Unknown optimiser '{name}'.")


def _train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer, tag):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=f"  Epoch {epoch+1:>3} [train]", leave=False)
    for batch_idx, (images, labels) in enumerate(bar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        bar.set_postfix(loss=f"{running_loss/(batch_idx+1):.4f}", acc=f"{100.*correct/total:.1f}%")
        writer.add_scalar(f"{tag}/batch_loss", loss.item(), epoch * len(loader) + batch_idx)
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    writer.add_scalar(f"{tag}/train_loss",     avg_loss, epoch)
    writer.add_scalar(f"{tag}/train_accuracy", accuracy, epoch)
    return avg_loss, accuracy


def _evaluate(model, loader, criterion, device, epoch, writer, tag):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=f"  Epoch {epoch+1:>3} [test] ", leave=False)
    with torch.no_grad():
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            bar.set_postfix(loss=f"{running_loss/(total/labels.size(0)):.4f}", acc=f"{100.*correct/total:.1f}%")
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    writer.add_scalar(f"{tag}/test_loss",     avg_loss, epoch)
    writer.add_scalar(f"{tag}/test_accuracy", accuracy, epoch)
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, config: dict,
                experiment_name: str, log_dir: str = "./runs") -> float:
    device = config.get("device", torch.device("cpu"))
    epochs = config.get("epochs", 10)
    model  = model.to(device)
    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    writer    = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*62}\n  Experiment : {experiment_name}")
    print(f"  Device     : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Optimiser  : {config.get('optimizer')}  lr={config.get('learning_rate')}")
    print(f"  Epochs     : {epochs}  |  Trainable: {trainable:,}\n{'='*62}\n")
    best_accuracy = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, criterion,
                                                 device, epoch, writer, experiment_name)
        test_loss,  test_acc  = _evaluate(model, test_loader, criterion,
                                          device, epoch, writer, experiment_name)
        print(f"  Epoch {epoch+1:>2}/{epochs}  |  "
              f"Train loss={train_loss:.4f} acc={train_acc:.1f}%  |  "
              f"Test  loss={test_loss:.4f} acc={test_acc:.1f}%")
        if test_acc > best_accuracy:
            best_accuracy = test_acc
    writer.close()
    print(f"\n  Best Test Accuracy: {best_accuracy:.2f}%  ({experiment_name})\n")
    return best_accuracy
