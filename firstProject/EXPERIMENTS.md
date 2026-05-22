# Experiment Log

> One row per training run. Fill in immediately after each run completes.  
> Format: add newest runs at the top.

---

## Log

| # | Date | Model | Config / Changes | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS | wandb Run | Notes |
|---|------|-------|-----------------|---------|-------------|-----------|--------|-----|-----------|-------|
| — | — | — | — | — | — | — | — | — | — | Baseline row — fill when first run completes |

---

## Hyperparameter Sweep Results

| Run | lr0 | epochs | img_size | mAP@0.5 | Notes |
|-----|-----|--------|----------|---------|-------|
| — | — | — | — | — | — |

---

## Key Decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-14 | YOLOv9 via `ultralytics` (Option A) | Easier API, active maintenance, `model.train(freeze=10)` in Python |
| — | — | — |
