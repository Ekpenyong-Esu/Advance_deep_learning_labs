# Lab 4: Variational Autoencoders (VAEs)

This lab implements a convolutional VAE trained on MNIST.

## Topics

- Probabilistic encoder: maps input → latent mean and log-variance
- Reparameterization trick: enables backpropagation through stochastic sampling
- Decoder: maps latent vector → reconstructed image
- Evidence Lower BOund (ELBO) loss: reconstruction + KL divergence
- Latent space interpolation

## Running the Lab

```bash
python main.py
```

Reconstructions and samples are saved to `output/` after each epoch.
