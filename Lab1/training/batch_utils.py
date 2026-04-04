"""
training/batch_utils.py — Batch Unpacking and Forward Pass
===========================================================
Thin helpers that abstract over the two batch formats produced by the
data loaders, so the training engine never needs model-specific branching.

Batch formats
-------------
  2-tuple  (x, labels)                         — SimpleANN / BiLSTM
  3-tuple  (input_ids, attention_mask, labels)  — BERT / DistilBERT

Usage
-----
    from training.batch_utils import unpack_batch, forward_pass
"""

import torch
import torch.nn as nn


def unpack_batch(batch):
    """
    Split a raw DataLoader batch into (inputs, labels).

    For transformer batches the inputs are returned as the tuple
    (input_ids, attention_mask) so that forward_pass can pass them
    as separate keyword arguments.

    Returns
    -------
    inputs : Tensor | tuple[Tensor, Tensor]
    labels : LongTensor
    """
    if len(batch) == 3:
        # Transformer: (input_ids, attention_mask, labels)
        return (batch[0], batch[1]), batch[2]
    # ANN / BiLSTM: (x, labels)
    return batch[0], batch[1]


def forward_pass(model: nn.Module, inputs, device: torch.device) -> torch.Tensor:
    """
    Run the model's forward pass for any input type.

    - inputs is a Tensor                  → model(inputs.to(device))
    - inputs is (input_ids, attn_mask)    → model(*inputs_on_device)

    Returns
    -------
    logits : Tensor of shape (batch_size, num_classes)
    """
    if isinstance(inputs, tuple):
        return model(inputs[0].to(device), inputs[1].to(device))
    return model(inputs.to(device))
