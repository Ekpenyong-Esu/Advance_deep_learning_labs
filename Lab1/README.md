# Lab 1 — Sentiment Analysis: ANN, BiLSTM & Transformers

Binary sentiment classification (positive / negative) on Amazon product reviews.  
Three model families are compared across multiple dataset sizes, with all metrics logged to Weights & Biases.

---

## Project Structure

```
Lab1/
├── config.py                        ← All hyperparameters and dataset paths
├── main.ipynb                       ← Single notebook entry point (run this)
├── requirements.txt
│
├── data/
│   ├── sentiment_loader.py          ← DataLoaders for ANN, BiLSTM, and transformers
│   ├── amazon_cells_labelled.txt    ← Small dataset   (1 K reviews)
│   └── amazon_cells_labelled_LARGE_25K.txt  ← Large dataset  (25 K reviews)
│
├── models/
│   ├── ann_model.py                 ← Simple feed-forward ANN (TF-IDF features)
│   ├── lstm_model.py                ← Bidirectional LSTM (word embeddings)
│   └── bert_model.py                ← BERT and DistilBERT wrappers
│
├── training/
│   └── trainer.py                   ← Generic training loop (all three model types)
│
├── experiments/
│   ├── task01_ann.py                ← ANN on small + large datasets
│   ├── task01_bilstm.py             ← BiLSTM on small + large datasets
│   ├── task02_bert.py               ← BERT fine-tuning
│   ├── task02_distilbert.py         ← DistilBERT fine-tuning
│   └── grade5_comparison.py        ← BERT + DistilBERT on the public ~1 GB dataset
│
└── utils/
    ├── helpers.py                   ← Checkpoint save / load, parameter counter
    └── text_preprocessing.py        ← Classical cleaning (ANN/LSTM) + minimal (BERT)
```

---

## Tasks

### Task 1.1 — Simple ANN and BiLSTM

| Model | Input features | Key property |
|---|---|---|
| SimpleANN | TF-IDF bag-of-words (up to 50 K features) | Fast, no word order |
| BiLSTMSentiment | Padded word-index sequences (learned embeddings) | Captures word order and long-range context |

Each model is trained on two dataset sizes to measure the effect of data volume:
- **Small** — 1 K Amazon product reviews
- **Large** — 25 K Amazon product reviews

### Task 1.2 — Transformer Fine-Tuning

| Model | Parameters | Notes |
|---|---|---|
| BERT-base-uncased | ~110 M | 12 transformer layers, full fine-tuning |
| DistilBERT-base-uncased | ~66 M | 40% smaller, 60% faster, ~97% of BERT accuracy |

Both transformers are fine-tuned on the 25 K Amazon dataset by default.  
For Grade 5, they are also trained on the large public dataset (see below).

---

## Grading Criteria

| Grade | Requirement | How it is met |
|---|---|---|
| 3 | Both provided datasets (1 K + 25 K) | `task01_ann.py` and `task01_bilstm.py` run on both |
| 3 | ANN model + transformer model | `ann_model.py` + `bert_model.py` / `distilbert_model.py` |
| 4 | Larger public dataset (~1 GB) OR multiple transformers | Both satisfied |
| 5 | Larger public dataset (~1 GB) AND multiple transformers | `grade5_comparison.py` — `amazon_polarity` from Hugging Face + BERT + DistilBERT |
| 5 | GPU support | Automatic via `config.DEVICE` |
| 5 | W&B / wandb visualisation | All experiments log to `advanced-ai-lab-1` project |

The **Grade-5 public dataset** is `amazon_polarity` from Hugging Face (~3.6 M reviews, ~1 GB download).  
Training is capped at `PUBLIC_MAX_SAMPLES` rows (default 100 K) for practical speed — the full dataset is still downloaded satisfying the size requirement.  
Change `PUBLIC_MAX_SAMPLES` in `config.py` to `None` to train on the entire dataset.

---

## Data Split

All datasets use a stratified **70 / 15 / 15** train / val / test split with a fixed random seed.  
The TF-IDF vectoriser and word vocabulary are **always fitted on the training split only** — no information from val or test is ever used during fitting.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK resources (first run only)

The data loader downloads required NLTK resources automatically on first use.

### 3. Log in to Weights & Biases (once)

```bash
wandb login
```

Or run Section 0.1 in `main.ipynb` which handles the login interactively.

### 4. Run experiments

Open `main.ipynb` and run the cells top-to-bottom, or run any experiment directly:

```bash
# Task 1.1
python experiments/task01_ann.py
python experiments/task01_bilstm.py

# Task 1.2
python experiments/task02_bert.py
python experiments/task02_distilbert.py

# Grade 5
python experiments/grade5_comparison.py
```

### 5. View results

All training curves and metrics are available at **https://wandb.ai** under project `advanced-ai-lab-1`.

---

## Configuration

All hyperparameters live in `config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `PUBLIC_MAX_SAMPLES` | `100_000` | Cap on public dataset samples (set `None` for full ~3.6 M) |
| `TFIDF_MAX_FEATURES` | `50_000` | TF-IDF vocabulary size |
| `LSTM_MAX_VOCAB` | `30_000` | BiLSTM word vocabulary size |
| `LSTM_MAX_LEN` | `256` | Max tokens per sentence for BiLSTM |
| `TRANSFORMER_MAX_LEN` | `128` | Max tokens per sentence for BERT / DistilBERT |
| `WANDB_PROJECT` | `advanced-ai-lab-1` | W&B project name |

---

## Experiment Naming

All W&B runs follow the pattern `TaskXX_ModelName_DatasetSize`:

| Experiment file | W&B run name |
|---|---|
| `task01_ann.py` (small) | `Task01_ANN_Small` |
| `task01_ann.py` (large) | `Task01_ANN_Large` |
| `task01_bilstm.py` (small) | `Task01_BiLSTM_Small` |
| `task01_bilstm.py` (large) | `Task01_BiLSTM_Large` |
| `task02_bert.py` | `Task02_BERT_Large` |
| `task02_distilbert.py` | `Task02_DistilBERT_Large` |
| `grade5_comparison.py` | `Grade5_BERT_Public`, `Grade5_DistilBERT_Public` |
