# Advanced Deep Learning Labs

A collection of hands-on deep learning laboratory exercises covering advanced topics in modern deep learning. Each lab provides a self-contained Python implementation using PyTorch.

## Labs Overview

| Lab | Topic | Key Concepts |
|-----|-------|-------------|
| [Lab 1](labs/lab1_custom_layers/) | Custom Neural Network Layers | CNN, Residual Blocks, Batch Normalization |
| [Lab 2](labs/lab2_attention_transformers/) | Attention & Transformers | Multi-Head Attention, Positional Encoding, Transformer |
| [Lab 3](labs/lab3_gans/) | Generative Adversarial Networks | DCGAN, Generator, Discriminator, Training Loop |
| [Lab 4](labs/lab4_vaes/) | Variational Autoencoders | Encoder, Decoder, Reparameterization, ELBO Loss |
| [Lab 5](labs/lab5_transfer_learning/) | Transfer Learning & Fine-tuning | Pretrained Models, Feature Extraction, Fine-tuning |

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Each lab is a standalone Python script. Navigate to the lab directory and run:

```bash
python main.py
```

## Structure

```
labs/
├── lab1_custom_layers/         # Custom CNN layers & residual networks
├── lab2_attention_transformers/ # Self-attention & Transformer architecture
├── lab3_gans/                  # Generative Adversarial Networks
├── lab4_vaes/                  # Variational Autoencoders
└── lab5_transfer_learning/     # Transfer learning & fine-tuning
```

## Topics Covered

- **Custom Layers**: Building reusable convolutional blocks, residual connections, squeeze-and-excitation blocks, and depthwise separable convolutions.
- **Attention & Transformers**: Scaled dot-product attention, multi-head attention, positional encoding, encoder/decoder stacks, and Vision Transformers (ViT).
- **GANs**: Deep Convolutional GAN (DCGAN) architecture, progressive training strategies, and evaluation with FID score concepts.
- **VAEs**: Probabilistic encoder-decoder frameworks, KL divergence regularization, and latent space interpolation.
- **Transfer Learning**: Loading pretrained models from `torchvision`, freezing layers for feature extraction, and fine-tuning on custom datasets.
