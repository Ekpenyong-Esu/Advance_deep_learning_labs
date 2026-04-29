"""
evaluator.py
------------
Convert vendor results into EvalMetrics. Four responsibilities:
  parse_ultralytics_results  — ultralytics val() → EvalMetrics
  compute_precision_recall   — preds + targets → (P, R) at a fixed threshold
  parse_map_result           — torchmetrics dict + P/R/FPS → EvalMetrics
  measure_fps                — time single-image forward passes (shared by all trainers)
"""

import time

import torch

from .models import EvalMetrics


def parse_ultralytics_results(val_results, *, model_name, pretrain, fine_tuned, split) -> EvalMetrics:
    """Build EvalMetrics from an ultralytics model.val() return value."""
    rd  = val_results.results_dict
    fps = round(1000.0 / max(val_results.speed.get("inference", 1.0), 1e-6), 1)
    return EvalMetrics(
        model=model_name, pretrain=pretrain, fine_tuned=fine_tuned, split=split,
        map50=rd.get("metrics/mAP50(B)", 0.0),
        map50_95=rd.get("metrics/mAP50-95(B)", 0.0),
        precision=rd.get("metrics/precision(B)", 0.0),
        recall=rd.get("metrics/recall(B)", 0.0),
        fps=fps,
    )


def compute_precision_recall(preds, targets, *, iou_thresh=0.5, conf_thresh=0.5):
    """Greedy IoU matching → (precision, recall) at a fixed confidence threshold.

    torchmetrics MeanAveragePrecision does not expose a single-threshold P/R
    scalar, so we compute it here with torchvision.ops.box_iou.
    """
    from torchvision.ops import box_iou

    tp = fp = fn = 0
    for pred, target in zip(preds, targets):
        pb = pred["boxes"][pred["scores"] >= conf_thresh]
        gb = target["boxes"]
        if not len(gb):
            fp += len(pb); continue
        if not len(pb):
            fn += len(gb); continue
        ious = box_iou(pb, gb)
        matched: set[int] = set()
        for row in ious:
            best = int(row.argmax())
            if float(row[best]) >= iou_thresh and best not in matched:
                tp += 1; matched.add(best)
            else:
                fp += 1
        fn += len(gb) - len(matched)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return round(p, 4), round(r, 4)


def parse_map_result(metric_dict, *, precision, recall, fps, model_name, pretrain, fine_tuned, split, notes="") -> EvalMetrics:
    """Build EvalMetrics from a torchmetrics MeanAveragePrecision.compute() dict."""
    return EvalMetrics(
        model=model_name, pretrain=pretrain, fine_tuned=fine_tuned, split=split,
        map50=float(metric_dict.get("map_50", 0.0)),
        map50_95=float(metric_dict.get("map", 0.0)),
        precision=precision, recall=recall, fps=fps, notes=notes,
    )


def measure_fps(model_fn, dataset, *, device, n_samples=100) -> float:
    """Time n_samples single-image forward passes; return FPS.

    Parameters
    ----------
    model_fn : callable
        Accepts one image tensor and runs one forward pass (no grad needed).
        Example:  ``lambda img: model([img.to(device)])``
    dataset  : Dataset
        Items are ``(image_tensor, ...)``; only the first element is used.
    """
    n = min(n_samples, len(dataset))
    with torch.no_grad():
        model_fn(dataset[0][0])          # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(n):
            model_fn(dataset[i][0])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return round(n / (time.perf_counter() - start), 1)
