"""
data/base_loader.py — Shared Raw-Data Helpers
==============================================
Private utilities used by all three model-specific loaders:

  _load_file(path)       — read a tab-separated .txt sentiment file
  _load_public(n)        — stream amazon_polarity from Hugging Face
  _load_dataset(dataset) — dispatch to the right raw loader
  _split(texts, labels)  — stratified 70 / 15 / 15 split

Public helper:

  get_raw_splits(dataset) — returns the six raw split lists so
                            Task 1.3 can verify that all model families
                            are evaluated on the exact same test reviews.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

import config


# ─────────────────────────────────────────────────────────────────────────────
# Raw-file loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_file(path: str):
    """
    Load a tab-separated sentiment file with no header.

    Expected format (one line per sample):
        <sentence>  TAB  <label>

    Returns
    -------
    texts  : list[str]
    labels : list[int]   (0 = negative, 1 = positive)
    """
    texts, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            # rsplit to handle the rare case of tabs inside the sentence
            parts = line.rsplit("\t", 1)
            texts.append(parts[0].strip())
            labels.append(int(parts[1].strip()))
    return texts, labels


def _load_public(max_samples: int | None = None):
    """
    Load the amazon_polarity dataset from Hugging Face.

    The dataset contains ~3.6 M training examples.
    We concatenate the 'title' and 'content' fields into a single text string
    and optionally cap the number of returned examples.

    Returns
    -------
    texts  : list[str]
    labels : list[int]   (0 = negative, 1 = positive)
    """
    from datasets import load_dataset

    print(f"Downloading / loading '{config.PUBLIC_DATASET_NAME}' from Hugging Face …")
    print("  (This may take a while on first run — the dataset is ~1 GB)")

    ds = load_dataset(config.PUBLIC_DATASET_NAME, split="train")

    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    texts  = [f"{row['title']} {row['content']}" for row in ds]
    labels = [int(row["label"]) for row in ds]

    print(f"  Loaded {len(texts):,} examples from '{config.PUBLIC_DATASET_NAME}'")
    return texts, labels


def _load_dataset(dataset: str):
    """
    Dispatch to the right raw loader.

    Parameters
    ----------
    dataset : "small" | "large" | "public"

    Returns
    -------
    texts  : list[str]
    labels : list[int]
    """
    if dataset == "small":
        return _load_file(config.SMALL_DATASET_PATH)
    elif dataset == "large":
        return _load_file(config.LARGE_DATASET_PATH)
    elif dataset == "public":
        return _load_public(max_samples=config.PUBLIC_MAX_SAMPLES)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose 'small', 'large', or 'public'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared split
# ─────────────────────────────────────────────────────────────────────────────

def _split(texts: list, labels: list):
    """
    Stratified 70 / 15 / 15 train / val / test split.

    Stratification keeps the class ratio identical in all three splits,
    which matters especially for the small (1 K) dataset.

    Returns
    -------
    tr_texts, va_texts, te_texts, tr_labels, va_labels, te_labels
    """
    test_ratio = 1.0 - config.TRAIN_RATIO - config.VAL_RATIO
    tr_t, te_t, tr_l, te_l = train_test_split(
        texts, labels,
        test_size=test_ratio,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    val_ratio_adj = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    tr_t, va_t, tr_l, va_l = train_test_split(
        tr_t, tr_l,
        test_size=val_ratio_adj,
        random_state=config.RANDOM_SEED,
        stratify=tr_l,
    )

    return tr_t, va_t, te_t, tr_l, va_l, te_l


# ─────────────────────────────────────────────────────────────────────────────
# Public helper
# ─────────────────────────────────────────────────────────────────────────────

def get_raw_splits(dataset: str = "large"):
    """
    Return the raw (unprocessed) text splits and labels for a dataset.

    This function exposes the underlying 70/15/15 stratified split so that
    Task 1.3 can confirm all three model families evaluate on the exact same
    test reviews — only the numerical representation differs.

    Parameters
    ----------
    dataset : "small" | "large" | "public"

    Returns
    -------
    tr_texts, va_texts, te_texts : list[str]   — raw review strings
    tr_labels, va_labels, te_labels : list[int] — 0 or 1
    """
    texts, labels = _load_dataset(dataset)
    return _split(texts, labels)
