import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
Z_DIM = 100
H_DIM = 128
LR = 1e-3
EPOCHS = 50

WANDB_PROJECT = "gan-mnist"
CHECKPOINT_DIR = "checkpoints"