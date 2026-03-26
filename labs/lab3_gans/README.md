# Lab 3: Generative Adversarial Networks (GANs)

This lab implements a Deep Convolutional GAN (DCGAN) trained on the MNIST dataset.

## Topics

- Generator: maps a random latent vector to a realistic image
- Discriminator: classifies images as real or fake
- Adversarial training loop with separate generator and discriminator updates
- Label smoothing for more stable training

## Running the Lab

```bash
python main.py
```

Generated sample images are saved to `output/` after each epoch.
