"""
models.py
---------
Dataclasses for Phase 2 training configurations and evaluation results.

These are plain data containers with no IO or computation — they are the
single source of truth for the shape of a training run and its outcomes.
"""

from dataclasses import dataclass

# Single source of truth for the W&B project name.
WANDB_PROJECT = "nvd-car-detection-refactor"

@dataclass
class TrainingConfig:
    """Hyperparameters for one training run."""

    model_name:  str                         # e.g. "yolov9c" or "facebook/detr-resnet-50"
    data_yaml:   str                         # path to configs/data.yaml (YOLO) or "" for DETR
    epochs:      int        = 50
    batch:       int        = 16
    imgsz:       int        = 640
    lr0:         float      = 0.01
    freeze:      int | None = None           # layers to freeze; None = no freezing
    project:     str        = WANDB_PROJECT
    output_dir:  str        = ""             # MUST be set explicitly — no silent cwd default
    run_name:    str        = "run"
    device:      str        = "0"            # "0" = first GPU, "cpu", or "mps"
    workers:     int        = 4
    mosaic:      float      = 1.0            # mosaic augmentation probability (YOLO only)
    close_mosaic: int       = 10             # disable mosaic for last N epochs (Ultralytics schedule)
    patience:    int        = 10             # early-stopping: epochs without val/map50 improvement
    num_classes: int        = 1              # number of foreground classes (excluding background)
    # Additional YOLO augmentation knobs for generalization
    mixup:       float      = 0.0            # mixup alpha (0 = disabled)
    copy_paste:  float      = 0.0            # copy-paste augmentation prob (0 = disabled)
    scale:       float      = 0.5            # image scale augmentation ±gain
    hsv_h:       float      = 0.015          # HSV-Hue augmentation
    hsv_s:       float      = 0.7            # HSV-Saturation augmentation
    hsv_v:       float      = 0.4            # HSV-Value augmentation
    # Scheduler / warmup
    cos_lr:      bool       = True           # cosine LR decay
    lrf:         float      = 0.01           # final LR factor (final_lr = lr0 * lrf)
    warmup_epochs: float    = 5.0            # warmup epochs (can be fractional)

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch <= 0:
            raise ValueError(f"batch must be > 0, got {self.batch}")
        if not (0.0 < self.lr0 < 1.0):
            raise ValueError(f"lr0 must be in (0, 1), got {self.lr0}")
        if self.freeze is not None and self.freeze < 0:
            raise ValueError(f"freeze must be >= 0 or None, got {self.freeze}")
        if not self.output_dir:
            raise ValueError(
                "output_dir must be set explicitly (e.g. '/Labs/Project/runs'). "
                "Leaving it empty causes Ultralytics to save to a default path "
                "like /runs/detect/ instead of your project directory."
            )


@dataclass
class EvalMetrics:
    """Per-model evaluation results on one split."""

    model:      str
    pretrain:   str           # e.g. "COCO" or "COCO → NVD"
    fine_tuned: bool
    split:      str           # "val" or "test"
    map50:      float         # mAP@0.5
    map50_95:   float         # mAP@0.5:0.95
    precision:  float
    recall:     float
    fps:        float         # single-image inference FPS (hardware noted in README)
    notes:      str = ""

    def __post_init__(self):
        for name, val in [
            ("map50",     self.map50),
            ("map50_95",  self.map50_95),
            ("precision", self.precision),
            ("recall",    self.recall),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if self.fps < 0:
            raise ValueError(f"fps must be >= 0, got {self.fps}")
        if not self.split:
            raise ValueError("split must be a non-empty string")