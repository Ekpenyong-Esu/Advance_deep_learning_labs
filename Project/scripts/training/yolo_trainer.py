"""
yolo_trainer.py
---------------
YOLOv9 training and evaluation logic.

Public API
----------
  run_zero_shot_eval  — evaluate COCO-pretrained weights on a dataset split
  run_fine_tuning     — fine-tune YOLOv9 and return the best checkpoint path
  eval_checkpoint     — evaluate a saved checkpoint on a dataset split
"""

from importlib import reload
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from .evaluator import parse_ultralytics_results
from .models import EvalMetrics, TrainingConfig, WANDB_PROJECT
from .wandb_logger import finish, init_run, log_eval, log_eval_summary, log_model
from scripts.augmentations import inject_into_yolo


def _enable_ultralytics_wandb() -> None:
    """Enable W&B integration in Ultralytics settings and reload the callback."""
    SETTINGS["wandb"] = True
    try:
        import ultralytics.utils.callbacks.wb as wb_cb
        reload(wb_cb)
    except Exception:
        pass


def run_zero_shot_eval(
    model_name: str,
    data_yaml: str,
    *,
    split: str = "val",
    device: str = "0",
    workers: int = 4,
    project: str = WANDB_PROJECT,
) -> EvalMetrics:
    """
    Run inference with COCO-pretrained YOLOv9 weights — no fine-tuning.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier including extension, e.g. ``"yolov9c.pt"``.
    data_yaml : str
        Path to ``configs/data.yaml``.
    split : str
        Dataset split to evaluate on (default ``"val"``).
    device : str
        ``"0"`` for first GPU, ``"cpu"`` for CPU, ``"mps"`` for Apple Silicon.
    project : str
        W&B project name.

    Returns
    -------
    EvalMetrics
        mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS.
    """
    _enable_ultralytics_wandb()
    init_run(
        project,
        f"zero-shot-{model_name.replace('.pt', '')}-{split}",
        config={"model": model_name, "split": split},
    )

    model = YOLO(model_name)
    results = model.val(
        data=data_yaml,
        split=split,
        device=device,
        workers=workers,
        verbose=True,
    )
    metrics = parse_ultralytics_results(
        results,
        model_name=model_name.replace(".pt", ""),
        pretrain="COCO",
        fine_tuned=False,
        split=split,
    )
    log_eval(metrics)
    log_eval_summary(metrics)
    finish()
    return metrics


def run_fine_tuning(config: TrainingConfig, aug_variant: str = "none") -> Path:
    """
    Fine-tune YOLOv9 using the supplied TrainingConfig.

    Ultralytics saves artefacts to ``{config.output_dir}/{config.run_name}/``.
    The W&B project is set to ``config.project`` (human label) while the
    filesystem root is ``config.output_dir`` — these are kept separate so
    the W&B dashboard name and the local folder name are both explicit.

    Parameters
    ----------
    config : TrainingConfig
        ``model_name`` should be the architecture string without ``.pt``,
        e.g. ``"yolov9c"``.
    aug_variant : str
        One of ``"none"`` (default, ultralytics built-in augmentation),
        ``"snow"`` (snow/fog/brightness pipeline), or
        ``"full"`` (snow pipeline + GaussNoise + MotionBlur).

    Returns
    -------
    Path
        Absolute path to the best checkpoint (``best.pt``) saved by ultralytics.
    """
    train_kwargs: dict = {
        "data":     config.data_yaml,
        "epochs":   config.epochs,
        "batch":    config.batch,
        "imgsz":    config.imgsz,
        "lr0":      config.lr0,
        "lrf":      config.lrf,
        "cos_lr":   config.cos_lr,
        "warmup_epochs": config.warmup_epochs,
        "device":   config.device,
        "workers":  config.workers,
        "project":  config.output_dir,  # filesystem root — controls WHERE files land
        "name":     config.run_name,    # subfolder name  — e.g. "yolov9_nvd"
        "exist_ok": True,
        # -------------------------
        # AUGMENTATION CONTROL
        # -------------------------
        "mosaic": config.mosaic,
        "close_mosaic": config.close_mosaic,
        "mixup": config.mixup,
        "copy_paste": config.copy_paste,
        "scale": config.scale,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        # -------------------------
        # EARLY STOPPING
        # -------------------------
        "patience": config.patience,
    }
    if config.freeze is not None:
        train_kwargs["freeze"] = config.freeze

    _enable_ultralytics_wandb()
    # W&B run uses config.project as the dashboard project name
    init_run(config.project, config.run_name, config=train_kwargs)

    model = YOLO(f"{config.model_name}.pt")

    if aug_variant != "none":
        
        inject_into_yolo(model, aug_variant)

    model.train(**train_kwargs)

    # Expected path: {output_dir}/{run_name}/weights/best.pt
    best_pt = Path(config.output_dir) / config.run_name / "weights" / "best.pt"

    if not best_pt.exists():
        # Ultralytics sometimes nests runs — search recursively as fallback
        candidates = list(Path(config.output_dir).rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"best.pt not found under '{config.output_dir}'. "
                f"Check that 'project' and 'name' in train_kwargs are correct.\n"
                f"Expected: {best_pt}"
            )
        best_pt = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"[WARNING] best.pt not at expected path, using: {best_pt}")

    best_pt = best_pt.resolve()
    print(f"[INFO] Best checkpoint: {best_pt}")

    log_model(best_pt, name=f"{config.run_name}-best")
    finish()

    return best_pt


def eval_checkpoint(
    weights: str | Path,
    data_yaml: str,
    *,
    split: str = "val",
    device: str = "0",
    workers: int = 4,
    model_label: str | None = None,
    project: str = WANDB_PROJECT,
) -> EvalMetrics:
    """
    Evaluate a saved YOLOv9 checkpoint on a dataset split.

    Parameters
    ----------
    weights : str | Path
        Path to a saved ``.pt`` checkpoint.
    data_yaml : str
        Path to ``configs/data.yaml``.
    split : str
        Split to evaluate (default ``"val"``).
    model_label : str | None
        Human-readable label for the model column in the results CSV.
        Defaults to the checkpoint filename stem.
    project : str
        W&B project name.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    label = model_label or weights.stem

    _enable_ultralytics_wandb()
    init_run(
        project,
        f"eval-{label}-{split}",
        config={"weights": str(weights), "split": split},
    )

    model = YOLO(str(weights))
    results = model.val(
        data=data_yaml,
        split=split,
        device=device,
        workers=workers,
        verbose=True,
    )
    metrics = parse_ultralytics_results(
        results,
        model_name=label,
        pretrain="COCO → NVD",
        fine_tuned=True,
        split=split,
    )
    log_eval(metrics)
    log_eval_summary(metrics)
    finish()

    return metrics