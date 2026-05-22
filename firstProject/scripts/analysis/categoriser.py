"""
categoriser.py
--------------
Responsibility: IoU-match predictions against ground truth and classify each
box as a False Negative, False Positive, or Poor Localisation case.

This module has no dependency on any model framework or visualisation library.
It only consumes plain dicts produced by ``inference.py`` and outputs plain
dicts consumed by ``visualiser.py`` and ``reporter.py``.

Public API
----------
  categorise_errors(per_image, ...)       → ErrorCases
  categorise_all_models(all_inferences, ...)→ dict[str, ErrorCases]

Data shapes
-----------
  FNCase   = {"path", "image_path", "gt_boxes", "missed_gt"}
  FPCase   = {"path", "image_path", "gt_boxes", "ghost_box", "score"}
  PoorCase = {"path", "image_path", "gt_box", "pred_box", "iou", "score"}
  ErrorCases = {"fn": list[FNCase], "fp": list[FPCase], "poor": list[PoorCase]}

Note: images are NOT loaded here — only paths are stored.  Loading is deferred
to the visualiser so the categoriser stays framework-free and memory-efficient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.ops import box_iou


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

IOU_MATCH_DEFAULT   = 0.5    # IoU ≥ this → correct detection
IOU_POOR_DEFAULT    = 0.3    # IOU_POOR ≤ IoU < IOU_MATCH → poor localisation
CONF_THRESH_DEFAULT = 0.3    # predictions below this are discarded


# ---------------------------------------------------------------------------
# Single-image matching helpers
# ---------------------------------------------------------------------------

def _match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_match: float,
    iou_poor: float,
) -> tuple[set[int], set[int], list[tuple[int, int, float]]]:
    """Greedy IoU matching: best-IoU GT for each prediction.

    Returns
    -------
    matched_pred  : set of prediction indices classified as TP (IoU ≥ iou_match)
    matched_gt    : set of GT indices matched by any prediction
    poor_pairs    : list of (pred_idx, gt_idx, iou) for poor-localisation cases
    """
    iou_mat = box_iou(pred_boxes, gt_boxes)   # [M, N]
    matched_gt   : set[int] = set()
    matched_pred : set[int] = set()
    poor_pairs   : list[tuple[int, int, float]] = []

    for pi in range(len(pred_boxes)):
        best_gt_idx = int(iou_mat[pi].argmax())
        best_iou    = float(iou_mat[pi, best_gt_idx])

        if best_iou >= iou_match and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pi)
        elif iou_poor <= best_iou < iou_match and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pi)
            poor_pairs.append((pi, best_gt_idx, best_iou))

    return matched_pred, matched_gt, poor_pairs


def _classify_one_image(
    item: dict[str, Any],
    iou_match: float,
    iou_poor: float,
    conf_thresh: float,
) -> tuple[list, list, list]:
    """Classify all boxes in one image into FN / FP / Poor cases.

    Returns
    -------
    (fn_cases, fp_cases, poor_cases) — each a list of case dicts.
    Image pixels are NOT loaded here; only the path is stored.
    """
    fn_cases:   list[dict] = []
    fp_cases:   list[dict] = []
    poor_cases: list[dict] = []

    gt = torch.tensor(item["gt"],          dtype=torch.float32)
    pb = torch.tensor(item["pred_boxes"],  dtype=torch.float32)
    ps = torch.tensor(item["pred_scores"], dtype=torch.float32)

    # Apply confidence threshold
    keep  = ps >= conf_thresh
    pb, ps = pb[keep], ps[keep]

    img_path = item["path"]

    # Edge cases
    if len(gt) == 0 and len(pb) == 0:
        return fn_cases, fp_cases, poor_cases

    if len(gt) == 0:
        for box, score in zip(pb.tolist(), ps.tolist()):
            fp_cases.append({"path": img_path, "gt_boxes": [],
                              "ghost_box": box, "score": score})
        return fn_cases, fp_cases, poor_cases

    if len(pb) == 0:
        for box in gt.tolist():
            fn_cases.append({"path": img_path, "gt_boxes": gt.tolist(),
                              "missed_gt": box})
        return fn_cases, fp_cases, poor_cases

    matched_pred, matched_gt, poor_pairs = _match_predictions_to_gt(
        pb, gt, iou_match, iou_poor
    )

    # False Positives — unmatched predictions
    for pi in range(len(pb)):
        if pi not in matched_pred:
            fp_cases.append({"path": img_path, "gt_boxes": gt.tolist(),
                              "ghost_box": pb[pi].tolist(),
                              "score": float(ps[pi])})

    # False Negatives — unmatched GT boxes
    for gi in range(len(gt)):
        if gi not in matched_gt:
            fn_cases.append({"path": img_path, "gt_boxes": gt.tolist(),
                              "missed_gt": gt[gi].tolist()})

    # Poor Localisation pairs
    for pi, gi, iou in poor_pairs:
        poor_cases.append({"path": img_path,
                           "gt_box":   gt[gi].tolist(),
                           "pred_box": pb[pi].tolist(),
                           "iou":      iou,
                           "score":    float(ps[pi])})

    return fn_cases, fp_cases, poor_cases


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def categorise_errors(
    per_image: list[dict[str, Any]],
    *,
    iou_match:   float = IOU_MATCH_DEFAULT,
    iou_poor:    float = IOU_POOR_DEFAULT,
    conf_thresh: float = CONF_THRESH_DEFAULT,
) -> dict[str, list]:
    """Categorise all predictions from one model's inference run.

    Parameters
    ----------
    per_image   : Output of any ``infer_*`` function from ``inference.py``.
    iou_match   : Minimum IoU to count a detection as correct (default 0.5).
    iou_poor    : Lower IoU bound for poor-localisation category (default 0.3).
    conf_thresh : Predictions below this confidence are ignored (default 0.3).

    Returns
    -------
    dict with keys ``"fn"``, ``"fp"``, ``"poor"`` — each a list of case dicts.
    """
    all_fn:   list[dict] = []
    all_fp:   list[dict] = []
    all_poor: list[dict] = []

    for item in per_image:
        fn, fp, poor = _classify_one_image(item, iou_match, iou_poor, conf_thresh)
        all_fn.extend(fn)
        all_fp.extend(fp)
        all_poor.extend(poor)

    return {"fn": all_fn, "fp": all_fp, "poor": all_poor}


def categorise_all_models(
    all_inferences: dict[str, list[dict[str, Any]]],
    *,
    iou_match:   float = IOU_MATCH_DEFAULT,
    iou_poor:    float = IOU_POOR_DEFAULT,
    conf_thresh: float = CONF_THRESH_DEFAULT,
) -> dict[str, dict[str, list]]:
    """Run :func:`categorise_errors` for every model in *all_inferences*.

    Parameters
    ----------
    all_inferences : Output of :func:`~inference.infer_all_models`.

    Returns
    -------
    dict mapping model name → ``{"fn": [...], "fp": [...], "poor": [...]}``.
    """
    return {
        name: categorise_errors(
            per_image,
            iou_match=iou_match,
            iou_poor=iou_poor,
            conf_thresh=conf_thresh,
        )
        for name, per_image in all_inferences.items()
    }
