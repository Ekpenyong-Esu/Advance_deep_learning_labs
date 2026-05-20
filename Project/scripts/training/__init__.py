"""
training — Phase 2 model training and evaluation for the NVD car-detection project.

Public API
----------
TrainingConfig           — hyperparameter container for one training run
EvalMetrics              — evaluation results for one model × split

run_yolo_zero_shot_eval  — run COCO-pretrained YOLOv9 on a dataset split (no fine-tuning)
run_yolo_fine_tuning     — fine-tune YOLOv9 on NVD; returns best checkpoint path
eval_yolo_checkpoint     — evaluate a saved YOLOv9 checkpoint

run_detr_fine_tuning     — fine-tune RT-DETR on NVD COCO annotations; returns checkpoint dir
eval_detr_checkpoint     — evaluate a saved RT-DETR checkpoint
run_detr_zero_shot_eval  — evaluate COCO-pretrained RT-DETR on NVD without fine-tuning

run_frcnn_fine_tuning    — fine-tune Faster R-CNN on NVD; returns best checkpoint path
eval_frcnn_checkpoint    — evaluate a saved Faster R-CNN checkpoint
run_frcnn_zero_shot_eval — evaluate COCO-pretrained Faster R-CNN on NVD without fine-tuning

save_zero_shot_result    — write a single EvalMetrics result to a CSV file
append_comparison_row    — append one EvalMetrics row to results/model_comparison.csv
"""

from .models import EvalMetrics, TrainingConfig, WANDB_PROJECT

from .yolo_trainer import run_zero_shot_eval as run_yolo_zero_shot_eval
from .yolo_trainer import run_fine_tuning    as run_yolo_fine_tuning
from .yolo_trainer import eval_checkpoint    as eval_yolo_checkpoint

from .detr_trainer import run_fine_tuning    as run_detr_fine_tuning
from .detr_trainer import eval_checkpoint    as eval_detr_checkpoint
from .detr_trainer import run_zero_shot_eval as run_detr_zero_shot_eval

from .frcnn_trainer import run_fine_tuning    as run_frcnn_fine_tuning
from .frcnn_trainer import eval_checkpoint    as eval_frcnn_checkpoint
from .frcnn_trainer import run_zero_shot_eval as run_frcnn_zero_shot_eval

from .results_writer import save_zero_shot_result, append_comparison_row

__all__ = [
    "TrainingConfig",
    "EvalMetrics",
    "run_yolo_zero_shot_eval",
    "run_yolo_fine_tuning",
    "eval_yolo_checkpoint",
    "run_detr_fine_tuning",
    "eval_detr_checkpoint",
    "run_detr_zero_shot_eval",
    "run_frcnn_fine_tuning",
    "eval_frcnn_checkpoint",
    "run_frcnn_zero_shot_eval",
    "save_zero_shot_result",
    "append_comparison_row",
]
