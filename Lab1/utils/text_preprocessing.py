"""
utils/text_preprocessing.py — Text Cleaning and Tokenisation
=============================================================
Two preprocessing variants:

  preprocess_classical(text)    — used for Simple ANN and BiLSTM
      Aggressive cleaning: lowercase, remove emails / IPs / special chars /
      numbers, tokenise with NLTK, remove English stopwords.

  preprocess_transformer(text)  — used for BERT / DistilBERT
      Minimal cleaning suitable for pre-trained subword tokenisers.
      BERT understands context (including stopwords and punctuation), so
      heavy cleaning degrades its performance.

Usage
-----
    from utils.text_preprocessing import batch_preprocess

    clean_texts = batch_preprocess(raw_texts, mode="classical")   # ANN / LSTM
    clean_texts = batch_preprocess(raw_texts, mode="transformer") # BERT
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────────────────────────────
# NLTK resource bootstrap — downloads only if missing
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = None  # cached after first load


def _get_stopwords() -> set:
    global _STOPWORDS
    if _STOPWORDS is not None:
        return _STOPWORDS

    for resource in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt")
                           else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


# ─────────────────────────────────────────────────────────────────────────────
# Core cleaning functions
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_classical(text: str) -> str:
    """
    Full preprocessing pipeline for ANN and BiLSTM.

    Steps
    -----
    1. Lowercase
    2. Remove email addresses
    3. Remove IP addresses
    4. Remove special characters (keep alphanumeric + spaces)
    5. Remove standalone numbers
    6. Tokenise with NLTK word_tokenize
    7. Remove English stopwords

    Returns
    -------
    Space-joined string of cleaned tokens.
    """
    text = text.lower()
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text)  # emails
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "", text)                      # IPs
    text = re.sub(r"[^\w\s]", "", text)                                           # special chars
    text = re.sub(r"\b\d+\b", "", text)                                           # standalone numbers

    sw     = _get_stopwords()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t and t not in sw]
    cleaned = " ".join(tokens)
    return cleaned


def preprocess_transformer(text: str) -> str:
    """
    Minimal preprocessing for transformer tokenisers.

    Only collapses excessive whitespace.  All further normalisation
    (lowercasing, subword splitting, special-token insertion) is handled
    by the pre-trained tokeniser, so aggressive cleaning is intentionally
    avoided here.
    """
    cleaned = " ".join(text.split())
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Batch helper
# ─────────────────────────────────────────────────────────────────────────────

def batch_preprocess(texts: list, mode: str = "classical") -> list:
    """
    Apply preprocessing to a list of strings.

    Parameters
    ----------
    texts : list[str]
    mode  : "classical"    → preprocess_classical (ANN / BiLSTM)
            "transformer"  → preprocess_transformer (BERT / DistilBERT)

    Returns
    -------
    list[str]
    """
    if mode == "classical":
        processed = [preprocess_classical(t) for t in texts]
    elif mode == "transformer":
        processed = [preprocess_transformer(t) for t in texts]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'classical' or 'transformer'.")
    return processed
