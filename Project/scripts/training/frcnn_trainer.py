"""
frcnn_trainer.py
----------------
Faster R-CNN (fasterrcnn_resnet50_fpn) fine-tuning and evaluation on NVD.

Public API: NVDDetectionDataset, run_fine_tuning, eval_checkpoint

Label note: torchvision reserves label 0 for background; car = label 1.
"""

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

from .evaluator import compute_precision_recall, measure_fps, parse_map_result
from .models import TrainingConfig
from .wandb_logger import finish, init_run, log, log_batch_loss, log_eval, log_eval_summary, log_model

_NUM_CLASSES = 2   # 0 = background, 1 = car

# Car label in torchvision's COCO 91-class pretrained model (1-indexed, 0=background).
# Confirmed: COCO category_id=3 → torchvision label 3 = "car".
# Used only in zero-shot eval to filter car predictions and remap to NVD label 1.
_COCO_CAR_LABEL = 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NVDDetectionDataset(Dataset):
    """YOLO-format NVD annotations as a torchvision-compatible detection dataset.

    Returns ``(image_tensor [C,H,W] float32 [0,1], target)`` per item.
    target has ``boxes`` (FloatTensor[N,4] XYXY abs pixels) and ``labels`` (int64).
    Labels are 1-indexed (car=1) to match torchvision conventions.

    Parameters
    ----------
    augment : albumentations.Compose | None
        Optional augmentation pipeline built with ``build_frcnn_pipeline``
        (bbox_format='pascal_voc', i.e. xyxy absolute).  Applied only during
        training — pass ``None`` for validation / eval datasets.
    """

    def __init__(self, image_paths, label_paths, augment=None):
        self._images  = list(image_paths)
        self._labels  = list(label_paths)
        self._augment = augment

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img = Image.open(self._images[idx]).convert("RGB")
        w, h = img.size
        boxes, labels = [], []
        lp = self._labels[idx]
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cx, cy, bw, bh = map(float, parts[1:])
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)

        if self._augment is not None:
            result = self._augment(
                image=np.array(img),
                bboxes=boxes,
                class_labels=labels,
            )
            img    = Image.fromarray(result["image"])
            boxes  = [list(b) for b in result["bboxes"]]
            labels = list(result["class_labels"])

        return (
            to_tensor(img),
            {
                "boxes":  torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.int64),
            },
        )


def _collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_split_paths(data_yaml_path, split):
    """Resolve image + label paths for one NVD split from data.yaml."""
    p   = Path(data_yaml_path).resolve()
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", str(p.parent)))
    if not root.is_absolute():
        root = p.parent / root
    lines = (root / cfg[split]).read_text(encoding="utf-8").splitlines()
    images, labels = [], []
    for line in lines:
        if not line.strip():
            continue
        img = Path(line.strip())
        if not img.is_absolute():
            img = root / img
        lbl = Path(
            str(img).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        ).with_suffix(".txt")
        images.append(img)
        labels.append(lbl)
    return images, labels


def _build_model(freeze=None, imgsz=1024):
    """Build fasterrcnn_resnet50_fpn with a 2-class head (background + car).

    freeze=N mirrors YOLOv9 freeze=N semantics — freezes the first N
    named children of the ResNet backbone body.
    Faster R-CNN backbone body layers: layer0 (stem), layer1, layer2, layer3, layer4
    freeze=10 → all 5 layers frozen (capped by min).

    imgsz controls GeneralizedRCNNTransform: shorter edge is scaled to imgsz,
    longer edge is capped at imgsz (square-ish, matching DETR/YOLO behaviour).
    Default torchvision values are min_size=800, max_size=1333.
    """
    model   = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=imgsz, max_size=imgsz)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, _NUM_CLASSES)
    if freeze is not None:
        backbone_layers = list(model.backbone.body.named_children())
        n_freeze = min(freeze, len(backbone_layers))
        for i, (name, layer) in enumerate(backbone_layers):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad_(False)
        print(f"[FRCNN] Froze {n_freeze}/{len(backbone_layers)} backbone layers "
              f"(requested {freeze})")
    return model


def _evaluate_loader(
    model,
    loader,
    device,
    remap_label: int | None = None,
):
    """One eval pass → (torchmetrics_result, all_preds, all_targets).

    Shared by both the per-epoch check in run_fine_tuning and eval_checkpoint
    so the inference loop is never duplicated.

    Parameters
    ----------
    remap_label : int | None
        When set (zero-shot only), keep only predictions whose label matches
        this COCO class index and remap them to label 1 so they align with
        NVD targets. None = no remapping (fine-tuned model already outputs 1).
    """
    metric = MeanAveragePrecision(iou_type="bbox")
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            raw    = [{k: v.cpu() for k, v in p.items()} for p in model(images)]
            if remap_label is not None:
                preds = []
                for r in raw:
                    mask = r["labels"] == remap_label
                    preds.append({
                        "boxes":  r["boxes"][mask],
                        "scores": r["scores"][mask],
                        "labels": torch.ones(mask.sum(), dtype=torch.long),
                    })
            else:
                preds = raw
            tgts = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds, tgts)
            all_preds.extend(preds)
            all_targets.extend(tgts)
    return metric.compute(), all_preds, all_targets


# ---------------------------------------------------------------------------
# Zero-shot baseline
# ---------------------------------------------------------------------------

def run_zero_shot_eval(
    data_yaml,
    *,
    split: str = "val",
    device_str: str = "0",
    batch_size: int = 4,
    workers: int = 4,
    conf_thresh: float = 0.3,
):
    """Evaluate COCO-pretrained Faster R-CNN on NVD with no fine-tuning. Returns EvalMetrics."""
    init_run(
        "nvd-car-detection",
        f"zero-shot-frcnn-{split}",
        config={"model": "fasterrcnn_resnet50_fpn", "split": split},
    )

    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)
    # Load with original 91-class COCO head (no head replacement)
    model  = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=640, max_size=640).to(device)

    ds     = NVDDetectionDataset(*_load_split_paths(data_yaml, split))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=_collate,
    )

    # Pass remap_label=_COCO_CAR_LABEL so predictions are filtered to car
    # detections only and remapped to label 1 to match NVD targets.
    map_result, all_preds, all_targets = _evaluate_loader(
        model, loader, device, remap_label=_COCO_CAR_LABEL
    )
    precision, recall = compute_precision_recall(
        all_preds, all_targets, conf_thresh=conf_thresh
    )
    fps = measure_fps(lambda img: model([img.to(device)]), ds, device=device)

    metrics = parse_map_result(
        map_result,
        precision=precision,
        recall=recall,
        fps=fps,
        model_name="fasterrcnn-resnet50-fpn",
        pretrain="COCO",
        fine_tuned=False,
        split=split,
    )
    log_eval(metrics)
    log_eval_summary(metrics)
    finish()
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_fine_tuning(config: TrainingConfig, data_yaml, aug_variant: str = "none") -> Path:
    """Fine-tune fasterrcnn_resnet50_fpn on NVD. Returns path to best.pt.

    Parameters
    ----------
    aug_variant : {"none", "snow", "full"}
        Snow-augmentation variant to inject into the training dataset.  Uses
        ``build_frcnn_pipeline`` (pascal_voc / xyxy-abs bbox format).
        Validation dataset is never augmented.
    """
    from scripts.augmentations import build_frcnn_pipeline  # avoid circular import

    device = torch.device(
        f"cuda:{config.device}" if config.device.isdigit() else config.device
    )

    aug_pipeline = build_frcnn_pipeline(aug_variant)
    if aug_pipeline is not None:
        print(f"[FRCNN] Augmentation variant: '{aug_variant}'")

    train_ds = NVDDetectionDataset(*_load_split_paths(data_yaml, "train"), augment=aug_pipeline)
    val_ds   = NVDDetectionDataset(*_load_split_paths(data_yaml, "val"))
    train_loader = DataLoader(
        train_ds, batch_size=config.batch, shuffle=True,
        num_workers=config.workers, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch, shuffle=False,
        num_workers=config.workers, collate_fn=_collate,
    )

    model = _build_model(freeze=config.freeze, imgsz=config.imgsz).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr0, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr0 * 0.01
    )

    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_pt = out_dir / "best.pt"

    best_map50       = -1.0
    patience_counter = 0

    init_run(
        config.project,
        config.run_name,
        config={
            "model":   "fasterrcnn-resnet50-fpn",
            "epochs":  config.epochs,
            "batch":   config.batch,
            "lr0":     config.lr0,
        },
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            log_batch_loss(loss.item(), global_step)

        scheduler.step()

        map_result, _, _ = _evaluate_loader(model, val_loader, device)
        # FIX: extract and log both mAP@0.5 and mAP@0.5:0.95
        map50   = float(map_result.get("map_50", 0.0))
        map5095 = float(map_result.get("map",    0.0))  # torchmetrics key for mAP@0.5:0.95
        avg_loss = total_loss / len(train_loader)

        print(f"[FRCNN] {epoch:>3}/{config.epochs}  loss={avg_loss:.4f}  "
              f"val_mAP@0.5={map50:.4f}  val_mAP@0.5:0.95={map5095:.4f}")
        # Use global_step to keep W&B steps monotonically increasing
        log({"train/loss": avg_loss, "val/map50": map50, "val/map50_95": map5095},
            step=global_step)

        if map50 > best_map50:
            best_map50       = map50
            patience_counter = 0
            torch.save(model.state_dict(), best_pt)
            print(f"[FRCNN] ✓ best={best_map50:.4f} → {best_pt}")
        else:
            patience_counter += 1
            if config.patience > 0:
                print(f"[FRCNN] No improvement ({patience_counter}/{config.patience})")
                if patience_counter >= config.patience:
                    print(f"[FRCNN] Early stopping at epoch {epoch} (patience={config.patience})")
                    break

    log_model(best_pt, name=f"{config.run_name}-best")
    finish()

    return best_pt.resolve()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_checkpoint(
    weights,
    data_yaml,
    *,
    split: str = "val",
    device_str: str = "0",
    batch_size: int = 4,
    workers: int = 4,
    conf_thresh: float = 0.3,
    imgsz: int = 1024,
):
    """Evaluate a saved Faster R-CNN checkpoint. Returns EvalMetrics."""
    run_label = Path(weights).stem
    init_run(
        "nvd-car-detection",
        f"eval-frcnn-{run_label}-{split}",
        config={"weights": str(weights), "split": split},
    )

    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)
    model  = _build_model(imgsz=imgsz).to(device)
    model.load_state_dict(
        torch.load(str(weights), map_location=device, weights_only=True)
    )

    ds     = NVDDetectionDataset(*_load_split_paths(data_yaml, split))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=_collate,
    )

    map_result, all_preds, all_targets = _evaluate_loader(model, loader, device)
    precision, recall = compute_precision_recall(
        all_preds, all_targets, conf_thresh=conf_thresh
    )
    fps = measure_fps(lambda img: model([img.to(device)]), ds, device=device)

    metrics = parse_map_result(
        map_result,
        precision=precision,
        recall=recall,
        fps=fps,
        model_name="fasterrcnn-resnet50-fpn",
        pretrain="COCO → NVD",
        fine_tuned=True,
        split=split,
    )
    log_eval(metrics)
    log_eval_summary(metrics)
    finish()

    return metrics