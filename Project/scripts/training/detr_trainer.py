"""
detr_trainer.py
---------------
RT-DETR fine-tuning and evaluation (PekingU/rtdetr_r50vd).

Public API: NVDCocoDataset, make_collate_fn, run_fine_tuning, eval_checkpoint

Label note: COCO JSON category IDs are 1-indexed; NVDCocoDataset remaps them
to 0-indexed model labels automatically (car category_id=1 → label 0).

RT-DETR notes:
- Uses RTDetrImageProcessor (fixed-size resize, no pixel_mask needed).
- RTDetrForObjectDetection replaces DetrForObjectDetection.
- Backbone path for freezing: model.model.backbone.
- Images resized to config.imgsz × config.imgsz (square) — no padding or pixel_mask required.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    get_linear_schedule_with_warmup,
)

from .evaluator import best_precision_recall, compute_precision_recall, measure_fps, parse_map_result
from .models import TrainingConfig
from .wandb_logger import finish, init_run, log, log_batch_loss, log_eval, log_eval_summary, log_model

# Suppress FutureWarning noise from the image processor
warnings.filterwarnings("ignore", category=FutureWarning)

# Car label index in PekingU/rtdetr_r50vd's id2label (0-indexed 80-class mapping).
# Confirmed from config.json: id2label["2"] = "car", label2id["car"] = 2.
# Used only in zero-shot eval to filter car predictions and remap to NVD label 0.
_COCO_CAR_LABEL = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_xyxy_abs(boxes, w, h):
    """Normalised CXCYWH → absolute XYXY (RT-DETR stores boxes normalised)."""
    if boxes.numel() == 0:
        return boxes
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack(
        [(cx - bw / 2) * w, (cy - bh / 2) * h,
         (cx + bw / 2) * w, (cy + bh / 2) * h],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NVDCocoDataset(Dataset):
    """COCO-format NVD dataset for RT-DETR. Category IDs are remapped to 0-indexed.

    Parameters
    ----------
    augment : albumentations.Compose | None
        Optional augmentation pipeline built with ``build_detr_pipeline``
        (bbox_format='coco', i.e. xywh absolute).  Applied only during
        training — pass ``None`` for validation / eval datasets.
    """

    def __init__(self, coco_json_path, images_dir, processor, augment=None):
        self._coco       = COCO(str(coco_json_path))
        self._image_ids  = list(self._coco.imgs.keys())
        self._images_dir = Path(images_dir)
        self._processor  = processor
        self._augment    = augment
        cat_ids = sorted(self._coco.getCatIds())
        self._cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}

    def __len__(self):
        return len(self._image_ids)

    @property
    def cat_id_to_label(self) -> dict:
        """Public access to category-ID → 0-indexed label mapping."""
        return self._cat_id_to_label

    @property
    def num_labels(self) -> int:
        return len(self._cat_id_to_label)

    @property
    def id2label(self) -> dict:
        return {i: self._coco.cats[cid]["name"]
                for cid, i in self._cat_id_to_label.items()}

    @property
    def label2id(self) -> dict:
        return {v: k for k, v in self.id2label.items()}

    def __getitem__(self, idx):
        img_id   = self._image_ids[idx]
        img_info = self._coco.imgs[img_id]
        image    = Image.open(self._images_dir / img_info["file_name"]).convert("RGB")
        anns     = self._coco.loadAnns(self._coco.getAnnIds(imgIds=img_id))

        if self._augment is not None and anns:
            # Use original ann indices as class_labels so we can recover
            # the full annotation dict for boxes that survive min_visibility.
            bboxes  = [ann["bbox"] for ann in anns]   # xywh absolute (COCO)
            indices = list(range(len(anns)))
            result  = self._augment(
                image=np.array(image),
                bboxes=bboxes,
                class_labels=indices,
            )
            image = Image.fromarray(result["image"])
            anns = [
                {**anns[int(orig_i)], "bbox": list(aug_box),
                 "area": aug_box[2] * aug_box[3]}
                for aug_box, orig_i in zip(result["bboxes"], result["class_labels"])
            ]
        elif self._augment is not None:
            # No annotations — still augment the image
            result = self._augment(image=np.array(image), bboxes=[], class_labels=[])
            image  = Image.fromarray(result["image"])

        encoding = self._processor(
            images=image,
            annotations={"image_id": img_id, "annotations": anns},
            return_tensors="pt",
        )
        labels = encoding["labels"][0]
        labels["class_labels"] = torch.tensor(
            [self._cat_id_to_label[int(c)] for c in labels["class_labels"]],
            dtype=torch.long,
        )
        return encoding["pixel_values"].squeeze(0), labels


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def make_collate_fn(processor):
    """Batch collate for RT-DETR.

    RTDetrImageProcessor resizes every image to a fixed square size (config.imgsz),
    so all tensors in a batch are already the same shape — no padding needed.
    pixel_mask is not used by RT-DETR.
    """
    def _collate(batch):
        pixel_values, labels = zip(*batch)
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels":       list(labels),
        }
    return _collate


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_loader(
    model,
    processor,
    loader,
    device,
    threshold: float = 0.01,
    remap_label: int | None = None,
):
    """One eval pass → (torchmetrics_result, all_preds, all_targets).

    Shared by both the per-epoch check in run_fine_tuning and eval_checkpoint
    so the inference loop is never duplicated.

    Parameters
    ----------
    threshold : float
        Post-processing confidence threshold. 0.0 keeps everything (very slow);
        0.01 filters near-zero scores without biasing mAP.
    remap_label : int | None
        When set (zero-shot only), keep only predictions whose label matches
        this COCO class index and remap them to label 0 so they align with
        NVD targets. None = no remapping (fine-tuned model already outputs 0).
    """
    metric = MeanAveragePrecision(iou_type="bbox")
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pv         = batch["pixel_values"].to(device)
            raw_labels = batch["labels"]
            # RT-DETR does not use pixel_mask
            outputs    = model(pixel_values=pv)
            sizes      = torch.stack([lbl["orig_size"] for lbl in raw_labels]).to(device)
            results    = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=sizes,
                use_focal_loss=True,   # RT-DETR: sigmoid per class, no background token
            )
            preds = []
            for r in results:
                if remap_label is not None:
                    # Zero-shot: keep only car detections and remap to label 0
                    mask = r["labels"] == remap_label
                    preds.append({
                        "boxes":  r["boxes"][mask].cpu(),
                        "scores": r["scores"][mask].cpu(),
                        "labels": torch.zeros(mask.sum(), dtype=torch.long),
                    })
                else:
                    preds.append({
                        "boxes":  r["boxes"].cpu(),
                        "scores": r["scores"].cpu(),
                        "labels": r["labels"].cpu(),
                    })

            targets = [
                {
                    "boxes": _to_xyxy_abs(
                        lbl["boxes"],
                        int(lbl["orig_size"][1]),
                        int(lbl["orig_size"][0]),
                    ).cpu(),
                    "labels": lbl["class_labels"].cpu(),
                }
                for lbl in raw_labels
            ]
            metric.update(preds, targets)
            all_preds.extend(preds)
            all_targets.extend(targets)
    return metric.compute(), all_preds, all_targets


# ---------------------------------------------------------------------------
# Zero-shot baseline
# ---------------------------------------------------------------------------

def run_zero_shot_eval(
    val_json,
    images_dir,
    *,
    model_name: str = "PekingU/rtdetr_r50vd",
    split: str = "val",
    device_str: str = "0",
    batch_size: int = 4,
    workers: int = 4,
    conf_thresh: float = 0.3,
    imgsz: int = 640,
):
    """Evaluate COCO-pretrained RT-DETR on NVD with no fine-tuning. Returns EvalMetrics."""
    init_run(
        "nvd-car-detection",
        f"zero-shot-rtdetr-{split}",
        config={"model": model_name, "split": split},
    )

    device    = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)
    processor = RTDetrImageProcessor.from_pretrained(
        model_name,
        size={"width": imgsz, "height": imgsz},
    )
    # Load with the original 80-class COCO head (no num_labels override)
    model = RTDetrForObjectDetection.from_pretrained(model_name).to(device)

    val_ds = NVDCocoDataset(val_json, images_dir, processor)
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=make_collate_fn(processor),
    )

    # Pass remap_label=_COCO_CAR_LABEL so predictions are filtered to car
    # detections only and remapped to label 0 to match NVD targets.
    # Use default threshold=0.01 so torchmetrics receives predictions at all
    # confidence levels — conf_thresh is only used for the P/R scalar below.
    map_result, all_preds, all_targets = _evaluate_loader(
        model, processor, loader, device,
        remap_label=_COCO_CAR_LABEL,
    )
    precision, recall = best_precision_recall(all_preds, all_targets)
    fps = measure_fps(
        lambda img: model(pixel_values=img.unsqueeze(0).to(device)),
        val_ds,
        device=device,
    )

    metrics = parse_map_result(
        map_result,
        precision=precision,
        recall=recall,
        fps=fps,
        model_name="rtdetr-r50vd",
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

def run_fine_tuning(config: TrainingConfig, train_json, val_json, images_dir, aug_variant: str = "none") -> Path:
    """Fine-tune PekingU/rtdetr_r50vd on NVD. Returns path to best checkpoint dir.

    Parameters
    ----------
    aug_variant : {"none", "snow", "full"}
        Snow-augmentation variant to inject into the training dataset.  Uses
        ``build_detr_pipeline`` (coco / xywh-abs bbox format).
        Validation dataset is never augmented.
    """
    from scripts.augmentations import build_detr_pipeline  # avoid circular import

    device = torch.device(
        f"cuda:{config.device}" if config.device.isdigit() else config.device
    )

    processor = RTDetrImageProcessor.from_pretrained(
        config.model_name,
        size={"width": config.imgsz, "height": config.imgsz},
        do_resize=True,
        do_normalize=True,
    )
    collate = make_collate_fn(processor)

    aug_pipeline = build_detr_pipeline(aug_variant, imgsz=config.imgsz)

    if aug_pipeline is not None:
        print(f"[RT-DETR] Augmentation variant: '{aug_variant}' (pre-resize to {config.imgsz}px)")

    train_ds = NVDCocoDataset(train_json, images_dir, processor, augment=aug_pipeline)
    val_ds   = NVDCocoDataset(val_json,   images_dir, processor)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch, shuffle=True,
        num_workers=config.workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch, shuffle=False,
        num_workers=config.workers, collate_fn=collate,
    )

    model = RTDetrForObjectDetection.from_pretrained(
        config.model_name,
        num_labels=train_ds.num_labels,
        id2label=train_ds.id2label,
        label2id=train_ds.label2id,
        ignore_mismatched_sizes=True,
    )

    # Freeze backbone if requested — mirrors YOLOv9 freeze=N semantics.
    # RT-DETR backbone has 8 logical blocks (indexed 0–7):
    #   [0] stem conv 1  [1] stem conv 2  [2] stem conv 3  [3] maxpool
    #   [4] ResNet stage 0  [5] stage 1  [6] stage 2  [7] stage 3
    # freeze=4  → stem only | freeze=8+ → full backbone (same as YOLOv9 freeze=10)
    if config.freeze is not None:
        try:
            bb = model.model.backbone.model
        except AttributeError:
            bb = model.model.backbone
        freeze_blocks = (
            list(bb.embedder.embedder.children())  # [0] [1] [2] stem convs
            + [bb.embedder.pooler]                 # [3] maxpool
            + list(bb.encoder.stages)              # [4] [5] [6] [7] ResNet stages
        )
        n_freeze = min(config.freeze, len(freeze_blocks))
        for block in freeze_blocks[:n_freeze]:
            for p in block.parameters():
                p.requires_grad_(False)
        print(f"[RT-DETR] Froze {n_freeze}/{len(freeze_blocks)} backbone blocks "
              f"(requested {config.freeze})")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr0, weight_decay=1e-4)
    total     = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total // 10),
        num_training_steps=total,
    )
    model.to(device)

    out_dir  = Path(config.output_dir) / config.run_name
    best_dir = out_dir / "best_checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_map50       = -1.0
    patience_counter = 0

    init_run(
        config.project,
        config.run_name,
        config={
            "model":  config.model_name,
            "epochs": config.epochs,
            "batch":  config.batch,
            "lr0":    config.lr0,
        },
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            labels = [
                {k: v.to(device) for k, v in lbl.items()}
                for lbl in batch["labels"]
            ]
            # RT-DETR does not use pixel_mask
            out = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=labels,
            )
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            log_batch_loss(out.loss.item(), global_step)

        map_result, _, _ = _evaluate_loader(
            model, processor, val_loader, device, threshold=0.01
        )
        # FIX: extract and log both mAP@0.5 and mAP@0.5:0.95
        map50   = float(map_result.get("map_50", 0.0))
        map5095 = float(map_result.get("map",    0.0))  # torchmetrics key for mAP@0.5:0.95
        avg_loss = total_loss / len(train_loader)

        print(f"[RT-DETR] {epoch:>3}/{config.epochs}  loss={avg_loss:.4f}  "
              f"val_mAP@0.5={map50:.4f}  val_mAP@0.5:0.95={map5095:.4f}")
        # Use global_step to keep W&B steps monotonically increasing
        log({"train/loss": avg_loss, "val/map50": map50, "val/map50_95": map5095},
            step=global_step)

        if map50 > best_map50:
            best_map50       = map50
            patience_counter = 0
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            print(f"[RT-DETR] ✓ best={best_map50:.4f} → {best_dir}")
        else:
            patience_counter += 1
            if config.patience > 0:
                print(f"[RT-DETR] No improvement ({patience_counter}/{config.patience})")
                if patience_counter >= config.patience:
                    print(f"[RT-DETR] Early stopping at epoch {epoch} (patience={config.patience})")
                    break

    log_model(best_dir, name=f"{config.run_name}-best")
    finish()

    return best_dir.resolve()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_checkpoint(
    checkpoint_dir,
    val_json,
    images_dir,
    *,
    model_name: str = "PekingU/rtdetr_r50vd",
    split: str = "val",
    device_str: str = "0",
    batch_size: int = 4,
    workers: int = 4,
    conf_thresh: float = 0.3,
    imgsz: int = 1024,
):
    """Evaluate a saved RT-DETR checkpoint. Returns EvalMetrics.

    Parameters
    ----------
    model_name : str
        Original HuggingFace model ID used for fine-tuning.  The processor is
        always loaded from here (not from the checkpoint) so preprocessing is
        identical to what was used during training.  Only the model *weights*
        come from ``checkpoint_dir``.
    """
    run_label = Path(checkpoint_dir).name
    init_run(
        "nvd-car-detection",
        f"eval-rtdetr-{run_label}-{split}",
        config={"checkpoint": str(checkpoint_dir), "split": split},
    )

    device    = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)
    # Load processor from the original Hub model, NOT the checkpoint directory.
    # This guarantees identical image preprocessing to training (same normalisation
    # parameters, same resize logic) and avoids any stale/modified attributes that
    # processor.save_pretrained may have written to preprocessor_config.json.
    processor = RTDetrImageProcessor.from_pretrained(
        model_name,
        size={"width": imgsz, "height": imgsz},
        do_resize=True,
        do_normalize=True,
    )
    model     = RTDetrForObjectDetection.from_pretrained(
        str(checkpoint_dir),
        ignore_mismatched_sizes=True,
    ).to(device)
    val_ds    = NVDCocoDataset(val_json, images_dir, processor)
    loader    = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=make_collate_fn(processor),
    )

    # Fine-tuned model outputs label 0 directly — no remap needed.
    # Use default threshold=0.01 so torchmetrics receives predictions at all
    # confidence levels — conf_thresh is only used for the P/R scalar below.
    map_result, all_preds, all_targets = _evaluate_loader(
        model, processor, loader, device
    )
    precision, recall = best_precision_recall(all_preds, all_targets)

    # RT-DETR does not use pixel_mask
    fps = measure_fps(
        lambda img: model(pixel_values=img.unsqueeze(0).to(device)),
        val_ds,
        device=device,
    )

    metrics = parse_map_result(
        map_result,
        precision=precision,
        recall=recall,
        fps=fps,
        model_name="rtdetr-r50vd",
        pretrain="COCO → NVD",
        fine_tuned=True,
        split=split,
    )
    log_eval(metrics)
    log_eval_summary(metrics)
    finish()

    return metrics