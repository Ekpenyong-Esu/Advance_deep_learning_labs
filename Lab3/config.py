"""
config.py — Central Configuration for Lab 3: Image Captioning
==============================================================
All hyperparameters and paths live here.
Edit this file to adjust any experiment without touching model or training code.
"""

import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent
DATA_DIR       = PROJECT_ROOT / "data" / "Flickr8k"
IMAGES_DIR     = DATA_DIR / "Images"
CAPTIONS_FILE  = DATA_DIR / "captions.txt"
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")
OUTPUT_DIR     = str(PROJECT_ROOT / "outputs")

# ─────────────────────────────────────────────────────────────────────────────
# Device — automatically selects GPU when available
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset & Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# Minimum word frequency to be included in the vocabulary
VOCAB_MIN_FREQ = 5

# Train / Validation / Test split proportions (must sum to 1.0).
# Splits are made at the IMAGE level — no image appears in multiple splits.
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# Random seed for reproducible splits
SPLIT_SEED  = 42

# ─────────────────────────────────────────────────────────────────────────────
# Image pre-processing
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 224          # spatial size fed to ResNet50
IMAGE_MEAN = (0.485, 0.456, 0.406)   # ImageNet mean
IMAGE_STD  = (0.229, 0.224, 0.225)   # ImageNet std

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

EMBED_DIM   = 256    # dimension of word embeddings and projected image features
HIDDEN_DIM  = 512    # LSTM hidden state dimension
NUM_LAYERS  = 1      # number of LSTM layers
DROPOUT     = 0.5    # dropout probability in decoder

# Fine-tune the CNN encoder's last two ResNet blocks during training.
# Set to False for faster training; True for better performance.
FINE_TUNE_ENCODER = False

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE     = 32
NUM_EPOCHS     = 10
ENCODER_LR     = 1e-4    # learning rate for encoder (when fine-tuning)
DECODER_LR     = 4e-4    # learning rate for decoder
GRAD_CLIP      = 5.0     # gradient clipping norm (prevents exploding gradients)

# ─────────────────────────────────────────────────────────────────────────────
# Caption Generation
# ─────────────────────────────────────────────────────────────────────────────

MAX_CAPTION_LENGTH = 50   # maximum number of tokens to generate at inference
NUM_SAMPLE_IMAGES  = 6    # number of images to visualise in the samples cell

# ─────────────────────────────────────────────────────────────────────────────
# Embedding Combination Strategy
# ─────────────────────────────────────────────────────────────────────────────
# Options: "concatenation" (used here), "addition", "multiplication"
# Affects the decoder input size; see models/decoder.py.

EMBEDDING_COMBINATION = "concatenation"
