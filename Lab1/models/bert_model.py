"""
models/bert_model.py — Transformer Models for Sentiment Analysis
================================================================
Thin PyTorch wrappers around Hugging Face transformer encoders.

Both models expose the same interface as SimpleANN and BiLSTMSentiment:
    forward(input_ids, attention_mask) → logits  (batch, num_classes)

This uniform interface means the shared trainer handles all three model
types without any special-casing for the model itself.

Models
------
  BertSentiment       — bert-base-uncased  (110 M parameters)
      Full BERT encoder with 12 transformer layers and 768 hidden dimensions.
      All weights are updated during fine-tuning.

  DistilBertSentiment — distilbert-base-uncased  (~66 M parameters)
      Knowledge-distilled version of BERT: 40% fewer parameters, 60% faster
      at inference, while retaining ~97% of BERT's accuracy.
      Running both side-by-side provides a direct complexity / performance
      trade-off comparison — required for the Grade-5 criteria.

Usage
-----
    from models.bert_model import BertSentiment, DistilBertSentiment

    model = BertSentiment()
    logits = model(input_ids, attention_mask)   # (batch, 2)
"""

import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class BertSentiment(nn.Module):
    """
    BERT-base-uncased fine-tuned for binary sentiment classification.

    The pretrained encoder is fully trainable — all 12 transformer layers
    are updated together with the classification head during fine-tuning.

    Parameters
    ----------
    model_name  : str   — Hugging Face model identifier
    num_classes : int   — number of output sentiment classes (default 2)
    """

    def __init__(
        self,
        model_name:  str = "bert-base-uncased",
        num_classes: int = 2,
    ):
        super(BertSentiment, self).__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
        )

    def forward(self, input_ids, attention_mask):
        """
        Parameters
        ----------
        input_ids      : LongTensor  (batch, seq_len)
        attention_mask : LongTensor  (batch, seq_len)
            1 for real tokens, 0 for padding positions.

        Returns
        -------
        logits : FloatTensor  (batch, num_classes)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        return logits


class DistilBertSentiment(nn.Module):
    """
    DistilBERT-base-uncased fine-tuned for binary sentiment classification.

    DistilBERT removes the token-type embeddings and every other layer from
    BERT, then trains the smaller model to mimic BERT via knowledge
    distillation.  The result is a much faster model with minimal accuracy
    loss — ideal for comparing architectural trade-offs.

    Parameters
    ----------
    model_name  : str   — Hugging Face model identifier
    num_classes : int   — number of output sentiment classes (default 2)
    """

    def __init__(
        self,
        model_name:  str = "distilbert-base-uncased",
        num_classes: int = 2,
    ):
        super(DistilBertSentiment, self).__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
        )

    def forward(self, input_ids, attention_mask):
        """
        Parameters
        ----------
        input_ids      : LongTensor  (batch, seq_len)
        attention_mask : LongTensor  (batch, seq_len)

        Returns
        -------
        logits : FloatTensor  (batch, num_classes)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        return logits
