# Advanced AI Lab — CNN & Transfer Learning

A clean, beginner-friendly implementation of all lab tasks with **TensorBoard** visualisation, **GPU** support, and **Grade-5** transformer models.

---

## Project Structure

```
Advance_AI/
├── config.py                               ← All hyperparameters in one place
├── requirements.txt                        ← Python dependencies
│
├── data/
│   ├── cifar10_loader.py                   ← CIFAR-10 data loader (32×32 or 224×224)
│   ├── mnist_loader.py                     ← MNIST data loader
│   └── svhn_loader.py                      ← SVHN data loader (colour + greyscale)
│
├── models/
│   ├── simple_cnn.py                       ← CNN with swappable activation function
│   ├── alexnet_model.py                    ← AlexNet: fine-tuning & feature extraction
│   ├── mnist_cnn.py                        ← CNN for MNIST → SVHN transfer
│   └── vision_transformer.py               ← ViT-B/16 and Swin-T (Grade 5)
│
├── training/
│   └── trainer.py                          ← Generic training loop + TensorBoard logger
│
├── utils/
│   └── helpers.py                          ← save/load checkpoints, count parameters
│
└── experiments/
    ├── task01_cnn_sgd_leakyrelu.py         ← Task 0.1 exp 1: SGD + LeakyReLU
    ├── task01_cnn_adam_leakyrelu.py        ← Task 0.1 exp 2: Adam + LeakyReLU
    ├── task01_cnn_adam_tanh.py             ← Task 0.1 exp 3: Adam + Tanh  (separate file)
    ├── task02_alexnet_finetune.py          ← Task 0.2.1: AlexNet fine-tuning
    ├── task02_alexnet_feature_extraction.py← Task 0.2.1: AlexNet feature extraction
    ├── task02_mnist_to_svhn.py             ← Task 0.2.2: MNIST → SVHN transfer
    └── grade5_transformers_cifar10.py      ← Grade 5: ViT + Swin Transformer
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Verify GPU availability

```python
import torch
print(torch.cuda.is_available())   # True = GPU detected
```

> **Tip — Windows `num_workers` issue**  
> If you see a `BrokenPipeError` on Windows, open `config.py` and set `NUM_WORKERS = 0`.

---

## Running the Experiments

Run every script from the **project root** folder (`Advance_AI/`).

### Task 0.1 — Simple CNN on CIFAR-10

| Experiment | File | What changes |
|---|---|---|
| SGD + LeakyReLU | `task01_cnn_sgd_leakyrelu.py` | baseline |
| Adam + LeakyReLU | `task01_cnn_adam_leakyrelu.py` | optimiser |
| Adam + Tanh | `task01_cnn_adam_tanh.py` | activation function |

```bash
python experiments/task01_cnn_sgd_leakyrelu.py
python experiments/task01_cnn_adam_leakyrelu.py
python experiments/task01_cnn_adam_tanh.py
```

### Task 0.2.1 — AlexNet Transfer Learning (CIFAR-10)

```bash
python experiments/task02_alexnet_finetune.py
python experiments/task02_alexnet_feature_extraction.py
```

### Task 0.2.2 — Transfer Learning: MNIST → SVHN

```bash
python experiments/task02_mnist_to_svhn.py
```

This script runs two stages automatically:
1. Trains a CNN on MNIST and saves the weights.
2. Loads those weights and fine-tunes only the classifier on SVHN.

> **Grade-5 larger dataset**: Set `"use_extra_data": True` in `config.py → SVHN_TRANSFER_CONFIG`
> to also download the SVHN *extra* split (~531 K images, ~1 GB).

### Grade 5 — Transformer Models on CIFAR-10

```bash
python experiments/grade5_transformers_cifar10.py
```

Trains **ViT-B/16** and **Swin-T** (both pretrained on ImageNet) on CIFAR-10.  
GPU is **strongly recommended** — these are large models.

---

## Viewing Results in TensorBoard

After running any experiment, start TensorBoard from the project root:

```bash
tensorboard --logdir=runs
```

Then open **http://localhost:6006** in your browser.

All experiments write to separate sub-folders inside `runs/`, so you can
compare every run side-by-side on the same graph.

---

## Grade-5 Checklist

| Requirement | How it is satisfied |
|---|---|
| ✅ Multiple transformer models | ViT-B/16 + Swin-T in `grade5_transformers_cifar10.py` |
| ✅ Larger public dataset (~1 GB) | SVHN *extra* split in `task02_mnist_to_svhn.py` (`use_extra_data=True`) |
| ✅ GPU support | `config.DEVICE` auto-selects CUDA if available |
| ✅ TensorBoard visualisation | Every experiment logs to `runs/` |

---

## Key Concepts for Beginners

### What is Transfer Learning?
Instead of training a model from random weights, we start from weights
already trained on a large dataset (e.g. ImageNet).  The model has already
learned to recognise basic visual features (edges, textures, shapes), so it
needs far less data and training time to adapt to a new task.

### Fine-Tuning vs Feature Extraction
- **Fine-tuning** — unfreeze all layers; the entire network adapts. Higher accuracy, slower.  
- **Feature extraction** — freeze the backbone; only the output layer trains. Faster, slightly lower accuracy.

### Why Does SVHN Accuracy Drop vs MNIST?
MNIST has clean, centred, white digits on a black background.  SVHN comes
from real street photos: digits overlap, lighting varies, and backgrounds are
cluttered.  The CNN's MNIST-trained features transfer the general concept of
"digit shape" but struggle with SVHN's noise and distortion.

### Why Two Transformer Architectures?
- **ViT** uses global self-attention — every image patch "talks" to every other patch.  
- **Swin** uses local shifted-window attention in a hierarchical layout — more efficient, often better.  
  Comparing both on TensorBoard shows which architecture learns CIFAR-10 faster.

---

## Adjusting Hyperparameters

Everything is controlled from **`config.py`**.  Common tweaks:

| Setting | Where | Effect |
|---|---|---|
| `epochs` | each `*_CONFIG` dict | more epochs → higher accuracy |
| `learning_rate` | each `*_CONFIG` dict | smaller → slower but more stable |
| `BATCH_SIZE` | top of `config.py` | larger → faster but needs more RAM/VRAM |
| `use_extra_data` | `SVHN_TRANSFER_CONFIG` | `True` downloads the ~1 GB SVHN extra split |
