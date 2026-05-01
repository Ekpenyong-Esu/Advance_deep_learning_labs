"""
utils/evaluation.py — Caption Generation and BLEU-Score Evaluation
===================================================================
Provides two public functions:

  generate_caption(encoder, decoder, image, vocabulary, device)
      Greedy-decode a single image tensor into a caption string.

  calculate_bleu(encoder, decoder, test_ref_loader, vocabulary, device)
      Compute corpus-level BLEU-1 through BLEU-4 scores on the test set.

BLEU score
----------
BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between a
generated caption and a set of reference captions.  We report:

  BLEU-1 — unigram precision
  BLEU-2 — up to bigram precision
  BLEU-3 — up to trigram precision
  BLEU-4 — up to 4-gram precision  (the standard benchmark metric)

We use ``nltk.translate.bleu_score.corpus_bleu`` which applies
corpus-level smoothing and handles short sentences gracefully.

Usage
-----
    from utils.evaluation import generate_caption, calculate_bleu

    caption = generate_caption(encoder, decoder, image_tensor, vocab, device)
    scores  = calculate_bleu(encoder, decoder, test_ref_loader, vocab, device)
    # scores == {"BLEU-1": 0.xx, "BLEU-2": 0.xx, "BLEU-3": 0.xx, "BLEU-4": 0.xx}
"""

from typing import List, Dict

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import tqdm

from data.vocabulary import Vocabulary, SOS_IDX, EOS_IDX, PAD_IDX
import config


# ─────────────────────────────────────────────────────────────────────────────
# Greedy caption generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_caption(
    encoder,
    decoder,
    image:      torch.Tensor,
    vocabulary: Vocabulary,
    device:     torch.device,
    max_length: int = None,
) -> str:
    """
    Greedily decode a single image into a caption string.

    At each step the token with the highest logit is selected as the next
    input, stopping when <EOS> is produced or ``max_length`` is reached.

    Parameters
    ----------
    encoder    : ImageEncoder  (eval mode expected)
    decoder    : CaptionDecoder (eval mode expected)
    image      : FloatTensor  (3, H, W) or (1, 3, H, W) — a single image
    vocabulary : Vocabulary instance with ``decode`` method
    device     : torch.device
    max_length : maximum number of tokens to generate
                 (defaults to config.MAX_CAPTION_LENGTH)

    Returns
    -------
    str
        Human-readable caption without special tokens.
    """
    max_length = max_length or config.MAX_CAPTION_LENGTH

    encoder.eval()
    decoder.eval()

    # Ensure image has a batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)          # (1, 3, H, W)
    image = image.to(device)

    # Encode image
    image_feature = encoder(image)          # (1, embed_dim)

    # Initialise LSTM hidden state
    hidden = decoder.init_hidden(batch_size=1, device=device)

    # Start token
    word_idx = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)  # (1, 1)

    generated = []

    for _ in range(max_length):
        logits, hidden = decoder.step(image_feature, word_idx, hidden)
        # logits : (1, vocab_size)
        word_idx = logits.argmax(dim=1, keepdim=True)  # (1, 1)
        predicted = word_idx.item()

        if predicted == EOS_IDX:
            break

        if predicted != PAD_IDX:
            generated.append(predicted)

    return vocabulary.decode(generated, skip_special=True)


# ─────────────────────────────────────────────────────────────────────────────
# Corpus BLEU evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calculate_bleu(
    encoder,
    decoder,
    test_ref_loader: torch.utils.data.DataLoader,
    vocabulary:      Vocabulary,
    device:          torch.device,
    max_length:      int = None,
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-4 on the test reference set.

    The test_ref_loader yields (image_tensor, list_of_reference_strings)
    one unique image at a time (batch_size=1).

    Parameters
    ----------
    encoder          : ImageEncoder
    decoder          : CaptionDecoder
    test_ref_loader  : DataLoader over TestReferenceDataset (batch_size=1)
    vocabulary       : Vocabulary instance
    device           : torch.device
    max_length       : max tokens to generate per image

    Returns
    -------
    dict
        {"BLEU-1": float, "BLEU-2": float, "BLEU-3": float, "BLEU-4": float}
    """
    max_length = max_length or config.MAX_CAPTION_LENGTH

    encoder.eval()
    decoder.eval()

    # NLTK corpus_bleu expects:
    #   hypotheses  : list of tokenised generated captions  [[w1, w2, ...], ...]
    #   references  : list of lists of tokenised references  [[[r1w1, ...], [r2w1, ...]], ...]
    hypotheses: List[List[str]] = []
    references: List[List[List[str]]] = []

    smoother = SmoothingFunction().method1

    for images, captions_list in tqdm.tqdm(
        test_ref_loader, leave=False, desc="BLEU "
    ):
        # images        : (1, 3, H, W)
        # captions_list : list of 5 strings (batch_size=1, so it's a list of lists)
        image = images[0]   # (3, H, W)

        # Generate caption
        hyp_str    = generate_caption(encoder, decoder, image, vocabulary, device, max_length)
        hyp_tokens = Vocabulary.tokenize(hyp_str)

        # Reference captions for this image
        # captions_list[i] is a list of strings for position i across the batch
        # Since batch_size=1, each element is a single string.
        ref_strings = [cap[0] if isinstance(cap, (list, tuple)) else cap
                       for cap in captions_list]
        ref_tokens  = [Vocabulary.tokenize(r) for r in ref_strings]

        hypotheses.append(hyp_tokens)
        references.append(ref_tokens)

    def _bleu(n: int) -> float:
        weights = tuple(1.0 / n for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
        return corpus_bleu(references, hypotheses,
                           weights=weights,
                           smoothing_function=smoother)

    return {
        "BLEU-1": round(_bleu(1), 4),
        "BLEU-2": round(_bleu(2), 4),
        "BLEU-3": round(_bleu(3), 4),
        "BLEU-4": round(_bleu(4), 4),
    }
