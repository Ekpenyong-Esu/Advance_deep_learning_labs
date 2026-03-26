# Lab 5: Transfer Learning & Fine-tuning

This lab demonstrates how to leverage pretrained models from `torchvision` for a custom classification task.

## Topics

- Loading a pretrained ResNet-18 from `torchvision.models`
- **Feature extraction**: freeze all layers except the final classification head
- **Fine-tuning**: unfreeze all or part of the backbone and train end-to-end with a lower learning rate
- Learning rate scheduling (Cosine Annealing with Warm Restarts)
- Evaluating top-1 and top-5 accuracy

## Running the Lab

```bash
python main.py
```

The script runs two phases (feature extraction then fine-tuning) on CIFAR-10 and reports per-epoch accuracy.
