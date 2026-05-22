"""
inference.py
------------
Responsibility: load model checkpoints and run per-image prediction on the
test set.  Returns plain dicts so the caller has no dependency on any specific
model framework.

Public API
----------
  collect_test_items(test_split)             → list[TestItem]
  infer_yolo(ckpt, test_items, device_str)   → list[ImagePrediction]
  infer_detr(ckpt, imgsz, test_items, device)→ list[ImagePrediction]
  infer_frcnn(ckpt, imgsz, test_items, device)→ list[ImagePrediction]
  infer_model(name, info, test_items, ...)   → list[ImagePrediction] | None
  infer_all_models(checkpoint_map, ...)      → dict[str, list[ImagePrediction]]

Data shapes
-----------
  TestItem        = (image_path: Path, gt_boxes: list[list[float]])
  ImagePrediction = {
      "path":        Path,
      "gt":          list[list[float]],   # XYXY absolute pixels
      "pred_boxes":  list[list[float]],   # XYXY absolute pixels
      "pred_scores": list[float],
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from data_utils.models import DatasetSplit


# ---------------------------------------------------------------------------
# Type aliases (documentation only — plain dicts at runtime)
# ---------------------------------------------------------------------------
TestItem        = tuple[Path, list[list[float]]]
ImagePrediction = dict[str, Any]


# ---------------------------------------------------------------------------
# Collect ground-truth items from a DatasetSplit
# ---------------------------------------------------------------------------

def collect_test_items(test_split: DatasetSplit) -> list[TestItem]:
    """Extract (image_path, gt_boxes_xyxy_abs) pairs from a DatasetSplit.

    Parameters
    ----------
    test_split : DatasetSplit
        Loaded dataset split (e.g. from ``scripts/data_utils/loaders.py``).

    Returns
    -------
    list of (Path, list[list[float]])
        One tuple per image; gt_boxes may be empty for images with no cars.
    """
    items: list[TestItem] = []
    for ann in test_split.images:
        gt_boxes = [[b.x_min, b.y_min, b.x_max, b.y_max] for b in ann.bboxes]
        items.append((ann.image_path, gt_boxes))
    return items


# ---------------------------------------------------------------------------
# Per-framework inference functions (single responsibility each)
# ---------------------------------------------------------------------------

def infer_yolo(
    ckpt: Path,
    test_items: list[TestItem],
    device_str: str = "0",
    conf: float = 0.01,
) -> list[ImagePrediction]:
    """Run YOLOv9 inference on *test_items*.

    Parameters
    ----------
    ckpt       : Path to the ``.pt`` weights file.
    test_items : Output of :func:`collect_test_items`.
    device_str : Ultralytics device string (``"0"``, ``"cpu"``).
    conf       : Raw confidence threshold passed to predict(); keep low (0.01)
                 so the categoriser can apply its own threshold.
    """
    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    results: list[ImagePrediction] = []

    for img_path, gt in test_items:
        res = model.predict(str(img_path), device=device_str, verbose=False, conf=conf)[0]
        results.append({
            "path":        img_path,
            "gt":          gt,
            "pred_boxes":  res.boxes.xyxy.cpu().tolist() if res.boxes is not None else [],
            "pred_scores": res.boxes.conf.cpu().tolist() if res.boxes is not None else [],
        })

    del model
    return results


def infer_detr(
    ckpt: Path,
    imgsz: int,
    test_items: list[TestItem],
    device: torch.device,
    conf: float = 0.01,
) -> list[ImagePrediction]:
    """Run RT-DETR inference on *test_items*.

    The processor config is loaded from *ckpt* (saved by ``save_pretrained``
    during training), so image size and normalisation are guaranteed to match.

    Parameters
    ----------
    ckpt      : Path to the HuggingFace checkpoint directory.
    imgsz     : Fallback image size used only if the processor config is absent.
    test_items: Output of :func:`collect_test_items`.
    device    : ``torch.device`` to run inference on.
    conf      : Post-processing confidence threshold.
    """
    from PIL import Image
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    try:
        proc = RTDetrImageProcessor.from_pretrained(str(ckpt))
    except Exception:
        proc = RTDetrImageProcessor.from_pretrained(
            "PekingU/rtdetr_r50vd",
            size={"width": imgsz, "height": imgsz},
        )

    model = RTDetrForObjectDetection.from_pretrained(
        str(ckpt), ignore_mismatched_sizes=True
    ).to(device).eval()

    results: list[ImagePrediction] = []
    with torch.no_grad():
        for img_path, gt in test_items:
            img = Image.open(img_path).convert("RGB")
            enc = proc(images=img, return_tensors="pt").to(device)
            out = model(**enc)
            h, w = img.height, img.width
            res = proc.post_process_object_detection(
                out,
                threshold=conf,
                target_sizes=torch.tensor([[h, w]]).to(device),
                use_focal_loss=True,
            )[0]
            results.append({
                "path":        img_path,
                "gt":          gt,
                "pred_boxes":  res["boxes"].cpu().tolist(),
                "pred_scores": res["scores"].cpu().tolist(),
            })

    del model
    return results


def infer_frcnn(
    ckpt: Path,
    imgsz: int,
    test_items: list[TestItem],
    device: torch.device,
) -> list[ImagePrediction]:
    """Run Faster R-CNN inference on *test_items*.

    Uses ``_build_model`` from ``frcnn_trainer`` to guarantee the architecture
    matches the training configuration.

    Parameters
    ----------
    ckpt      : Path to the ``.pt`` state-dict file.
    imgsz     : GeneralizedRCNNTransform target size (must match training).
    test_items: Output of :func:`collect_test_items`.
    device    : ``torch.device`` to run inference on.
    """
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from training.frcnn_trainer import _build_model

    model = _build_model(freeze=None, imgsz=imgsz)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.to(device).eval()

    results: list[ImagePrediction] = []
    with torch.no_grad():
        for img_path, gt in test_items:
            img  = Image.open(img_path).convert("RGB")
            t    = to_tensor(img).to(device)
            out  = model([t])[0]
            mask = out["labels"] == 1   # 0 = background, 1 = car
            results.append({
                "path":        img_path,
                "gt":          gt,
                "pred_boxes":  out["boxes"][mask].cpu().tolist(),
                "pred_scores": out["scores"][mask].cpu().tolist(),
            })

    del model
    return results


# ---------------------------------------------------------------------------
# Dispatcher — selects the right inference function from checkpoint metadata
# ---------------------------------------------------------------------------

def infer_model(
    model_name: str,
    info: dict[str, Any],
    test_items: list[TestItem],
    device_str: str = "0",
) -> list[ImagePrediction] | None:
    """Dispatch inference to the correct framework based on *info["type"]*.

    Parameters
    ----------
    model_name  : Human-readable label (used only for logging).
    info        : Entry from ``_CHECKPOINT_MAP``:
                  ``{"type": "yolo"|"detr"|"frcnn", "imgsz": int, "path": Path}``.
    test_items  : Output of :func:`collect_test_items`.
    device_str  : Device string (``"0"``, ``"cpu"``).

    Returns
    -------
    list[ImagePrediction] on success, ``None`` if the checkpoint is missing.
    """
    ckpt  = info["path"]
    mtype = info["type"]
    imgsz = info["imgsz"]

    if not ckpt.exists():
        print(f"  SKIP {model_name} — checkpoint not found: {ckpt}")
        return None

    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

    if mtype == "yolo":
        result = infer_yolo(ckpt, test_items, device_str=device_str)
    elif mtype == "detr":
        result = infer_detr(ckpt, imgsz, test_items, device)
    elif mtype == "frcnn":
        result = infer_frcnn(ckpt, imgsz, test_items, device)
    else:
        raise ValueError(f"Unknown model type: {mtype!r}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Batch runner — iterates the full checkpoint map
# ---------------------------------------------------------------------------

def infer_all_models(
    checkpoint_map: dict[str, dict[str, Any]],
    test_items: list[TestItem],
    device_str: str = "0",
) -> dict[str, list[ImagePrediction]]:
    """Run inference for every entry in *checkpoint_map*.

    Parameters
    ----------
    checkpoint_map : ``_CHECKPOINT_MAP`` from the notebook (section 4.13).
    test_items     : Output of :func:`collect_test_items`.
    device_str     : Device string.

    Returns
    -------
    dict mapping model name → list[ImagePrediction].
    Missing checkpoints are silently excluded.
    """
    all_results: dict[str, list[ImagePrediction]] = {}

    for name, info in checkpoint_map.items():
        print(f"▶ {name}  [{info['type'].upper()}]")
        result = infer_model(name, info, test_items, device_str)
        if result is not None:
            all_results[name] = result
            print(f"  ✓ {len(result)} images processed")

    print(f"\nDone — {len(all_results)}/{len(checkpoint_map)} models inferred.")
    return all_results
