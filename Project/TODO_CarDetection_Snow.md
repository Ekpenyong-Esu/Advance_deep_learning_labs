# TODO — Car Detection in Snow Using Deep Learning

### D7047E Advanced Deep Learning | LTU VT2026
>
> Work through this file top to bottom. Check off each item as you complete it.
> Priority legend: 🔴 Critical · 🟡 Important · 🟢 Nice to have

---

## PHASE 0 — Setup (Day 1, before anything else)

- [x] 🔴 Create a GitHub repository: `car-detection-snow-nvd`
  - Add a `README.md` with project title, team names, and placeholder sections
  - Create folders: `data/`, `models/`, `results/`, `scripts/`, `presentation/`
  - Create `main.ipynb` at the project root (single entry-point notebook)
  - Commit a `.gitignore` for Python, large model weights, and dataset files
- [x] 🔴 **Decide which YOLOv9 implementation to use — do this before installing anything:** ✅ Option A (ultralytics) — documented in README.md and EXPERIMENTS.md
  - **Option A — `ultralytics` package** (`pip install ultralytics`): easier API, actively maintained, but `--freeze` and some training flags differ from the original repo. Use `model.train(freeze=10)` in Python, not a CLI flag.
  - **Option B — WongKinYiu/yolov9** (original repo, clone from GitHub): closer to the paper, uses the classic `train.py --freeze 10` CLI syntax, but requires manual dependency setup.
  - Recommendation: use **Option A (ultralytics)** unless your examiner specifies otherwise — it is better supported and easier to debug. Whichever you pick, document it in your README and report.
- [ ] 🔴 Set up a Python virtual environment (Python 3.10+)
  - Install: `torch`, `torchvision`, `ultralytics` (if using Option A), `transformers`, `albumentations`, `opencv-python`, `matplotlib`, `pandas`, `seaborn`, `tqdm`, `wandb`
- [x] 🔴 Create a `requirements.txt` and commit it
- [ ] 🔴 Set up Weights & Biases (wandb) for experiment tracking — free academic account

---

## PHASE 1 — Data (Week 1)

### Download & Inspect

- [x] 🔴 Download the Nordic Vehicle Dataset (NVD) from <https://nvd.ltu-ai.dev/>
- [x] 🔴 Confirm recordings and splits (verified 2026-04-18)
  - **Train** (4355 frames): 2022-12-02 Asjo 01_stabilized · 2022-12-03 Nyland 01_stabilized · 2022-12-04 Bjenberg 02 · 2022-12-23 Asjo 01_HD 5x stab
  - **Val** (2904 frames): same 4 recordings as train — frame-level split, not recording-level
  - **Test** (1191 frames): 2022-12-23 Bjenberg 02_stabilized (separate location, held out entirely)
  - nc=1, class_id 0 = car only (confirmed by scanning all 8450 label files)
- [x] 🔴 Count total frames and annotated bounding boxes per split — log these numbers in your report's Methodology section
  - `python scripts/inspect_dataset.py --data-yaml configs/data.yaml`
- [x] 🔴 Verify annotation format — NVD ships in YOLO format, no conversion needed

### Exploratory Data Analysis (EDA)

- [x] 🔴 Add an EDA section to `main.ipynb`
  - Visualise 20–30 random images with bounding boxes drawn on them
  - Plot class distribution (are there multiple vehicle types or just "car"?)
  - Plot bounding box size distribution (histogram of box width × height)
  - Plot bounding box aspect ratio distribution
  - Calculate and plot the number of objects per frame
- [x] 🟡 Manually label a sample of ~50 frames by snow severity: `light / medium / heavy`
  - Save this as a CSV: `data/snow_severity_sample.csv`
  - You will use this for per-condition analysis later
  - `python scripts/label_snow_severity.py --images data/raw/images/ --n 50 --output data/snow_severity_sample.csv`
- [x] 🟡 Check for and document any annotation quality issues (missing boxes, truncated objects, crowd scenes)
  - Quality check cell in `main.ipynb` reports: crowd boxes, tiny boxes (<32×32 px), empty images, missing dims
- [x] 🟢 Plot heatmap of bounding box centre positions across the frame — reveals camera bias

### Dataset Preparation

- [x] 🔴 Conversion script — N/A, NVD is already in YOLO format. `configs/data.yaml` points directly at the raw data.
  - ⚠️ **Verify the NVD class list before training.** Check label files in `data/raw/labels/` for class ids beyond 0. Update `nc:` and `names:` in `configs/data.yaml` if there are multiple vehicle types.
- [x] 🔴 Verify the annotations visually — draw boxes from `.txt` files on a sample of 10 images
  - `python scripts/verify_annotations.py --data-yaml configs/data.yaml --split train --n 10`
- [x] 🟡 Also prepare the dataset in COCO format (needed for DETR training)
  - `python scripts/prepare_coco_format.py --data-yaml configs/data.yaml`

---

## PHASE 2 — Baseline Models (Week 2)

### Zero-Shot Baseline (Critical — do this first)

- [ ] 🔴 Run COCO-pretrained YOLOv9 on the NVD **val set with zero fine-tuning**
  - Record: mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS
  - Save results to `results/baseline_zero_shot.csv`
  - This is your "before fine-tuning" anchor — it makes your fine-tuning results meaningful

### YOLOv9 Fine-tuning

- [ ] 🔴 Fine-tune YOLOv9 on the NVD training split
  - Starting config: `epochs=50`, `batch=16`, `img=640`, `lr0=0.01`
  - To freeze backbone layers: use `model.train(freeze=10)` if on **ultralytics** (Option A), or `--freeze 10` CLI flag if on **WongKinYiu/yolov9** (Option B) — use whichever matches your choice from Phase 0
  - Log all runs to wandb with descriptive run names
- [ ] 🔴 Evaluate on val set after training — record the same 5 metrics as above
- [ ] 🟡 Save the best checkpoint as `models/yolov9_nvd_best.pt`

### RT-DETR Baseline

- [ ] 🟡 Fine-tune `PekingU/rtdetr_r50vd` using HuggingFace `transformers` (≥4.39)
  - Using RT-DETR instead of standard DETR — faster inference, same transformer paradigm
  - `freeze=10` mirrors YOLOv9 setting for fair comparison (freezes `model.model.backbone`)
  - Training via `scripts/training/detr_trainer.py` · `run_detr_fine_tuning` / `eval_detr_checkpoint`
  - Requires COCO JSON annotations: run `scripts/prepare_coco_format.py` first
  - Evaluate on the same val set and record the same 5 metrics

### Faster R-CNN Baseline (optional but strengthens report)

- [ ] 🟢 Fine-tune `torchvision.models.detection.fasterrcnn_resnet50_fpn` on NVD
  - This gives you a classic two-stage detector to compare against
  - Evaluate on val set, same 5 metrics

### Results Table (fill this in as models complete)

- [ ] 🟡 Create `results/model_comparison.csv` with columns:
  `Model | Pretrain | Fine-tuned | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | FPS`

---

## PHASE 3 — Fine-tuning & Augmentation Experiments (Week 3, Days 1–4)

### Snow Augmentation Ablation (most important experiment)

- [ ] 🔴 Write augmentation pipeline using **Albumentations**: `scripts/augmentations.py`
  - Augmentations to implement and test:
    - `RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5)`
    - `RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3)`
    - `RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)`
    - `GaussNoise(var_limit=(10, 50))`
    - `MotionBlur(blur_limit=7)` — simulates UAV/drone motion blur
- [ ] 🔴 Train three variants and compare on val set:
  1. Fine-tuned YOLOv9 with **no augmentation** (already done in Phase 2)
  2. Fine-tuned YOLOv9 with **snow augmentation only**
  3. Fine-tuned YOLOv9 with **full augmentation stack**
- [ ] 🔴 Record results for all three — this is your **ablation study**
- [ ] 🟡 Visualise augmented images side-by-side with originals to include in your presentation slides

### Hyperparameter Sweep

- [ ] 🟡 Run a small sweep on your best model using wandb sweeps:
  - `lr0`: [0.001, 0.005, 0.01]
  - `epochs`: [30, 50, 100]
  - `img_size`: [416, 640]
- [ ] 🟡 Identify best hyperparameter combination and train final model with it

### Final Model Training

- [ ] 🔴 Train your **final best model** (best architecture + best augmentation + best hyperparams) on the full training split
- [ ] 🔴 Save final weights: `models/final_best_model.pt`

---

## PHASE 4 — Evaluation & Error Analysis (Week 4, Days 1–3)

### Test Set Evaluation

- [ ] 🔴 Run all trained models on the **held-out test set**
  - ⚠️ **Verify the exact recording name against the NVD official split documentation before locking this in.** The name used in this file ("2022-12-23 Bjenberg-02") is taken from the proposal — confirm it matches character-for-character (capitalisation, hyphens, spaces) with what appears in the actual dataset. A mismatch means you are testing on the wrong data.
  - Never use the test set during training or hyperparameter tuning — only here
- [ ] 🔴 Record final metrics for each model:
  - mAP@0.5, mAP@0.5:0.95, ____Precision, Recall, F1, FPS, also check IoU
  - ⚠️ **Always report FPS alongside the exact GPU used** (e.g. "47 FPS on NVIDIA RTX 3080 10GB"). FPS numbers without hardware context are meaningless to reviewers and graders. Also note batch size = 1 for inference benchmarking (that is the real-world condition).
- [ ] 🔴 Generate Precision-Recall curves for each model — save as figures for the report
- [ ] 🟡 Break down mAP by object size: small / medium / large (COCO standard area thresholds)
- [ ] 🟡 Break down mAP by the 50-frame snow severity sample you labelled in Phase 1

### Error Analysis (this separates a good project from a great one)

- [ ] 🔴 Add an error analysis section to `main.ipynb`
  - Find and save ~20 **false negatives** (missed cars) — what do they look like?
  - Find and save ~10 **false positives** (wrong detections) — what is the model confusing for cars?
  - Find and save ~10 **poor localisation** cases (detection exists but IoU < 0.5)
- [ ] 🔴 Categorise failure modes — create a 3×3 image grid for each category to include in the report
- [ ] 🟡 Write a short qualitative summary of failure patterns — include in presentation Discussion slide

### Visualisations for Presentation

- [ ] 🔴 Side-by-side detection examples: zero-shot baseline vs. fine-tuned best model
- [ ] 🔴 Augmentation ablation bar chart (mAP@0.5 for the 3 augmentation variants)
- [ ] 🔴 Model comparison table (all models, all metrics)
- [ ] 🟡 Confidence distribution histogram — where does the model place its scores?
- [ ] 🟡 Qualitative examples in different snow conditions (light / medium / heavy)

---

## PHASE 5 — ~~Report Writing~~ *(No written report required)*

> No written report is required for this project. The deliverables are the GitHub repository (source code + trained model + evaluation script) and the 10-minute presentation. Use the time saved here to polish your results and presentation slides.

---

## PHASE 6 — GitHub & Portfolio Packaging (Week 4, Days 4–5, parallel with report)

- [ ] 🔴 Write a complete `README.md`:
  - Project description + motivation
  - Dataset setup instructions (how to download NVD and prepare it)
  - One-command training: `python scripts/train.py --config configs/yolov9_nvd.yaml`
  - One-command inference: `python scripts/detect.py --source image.jpg --weights models/final_best_model.pt`
  - Results table embedded in README
  - Sample output images
- [ ] 🔴 Upload final model weights to a GitHub Release or Google Drive — link in README
- [ ] 🟡 Write a `model_card.md` (HuggingFace style):
  - Model description
  - Training data (NVD, split details)
  - Performance metrics on test set
  - Known failure modes (from error analysis)
  - How to use the model
- [ ] 🟢 Build a Gradio demo (`scripts/demo.py`):
  - Upload a snowy image → get bounding boxes drawn → download result
  - Deploy free on Hugging Face Spaces
- [ ] 🟢 Push to HuggingFace Hub if possible

---

## PHASE 7 — Presentation (Week 4, Day 5 / submission day)

> 10-minute presentation · **4–5 slides** · covers methods, implementation, results, learning outcome

- [ ] 🔴 Slides structure (4–5 slides):
  1. **Title + Team + Motivation** — problem statement, why snow is hard (show a hard example image)
  2. **Methods** — YOLOv9 architecture overview, NVD dataset summary, train/val/test split
  3. **Implementation** — training setup, augmentation strategy (show augmented examples side-by-side)
  4. **Results** — model comparison table, PR curves, ablation chart, qualitative detections
  5. **Learning Outcomes + Conclusion** — key findings, failure modes, what you would do differently
- [ ] 🔴 Prepare a 2-minute live demo or a screen-recorded video of inference on a snowy clip
- [ ] 🟡 Practice the presentation as a team — aim for equal speaking time

---

## Ongoing Throughout the Project

- [ ] Commit code to GitHub at least every 2 days — never lose work
- [ ] Log every experiment to wandb with a meaningful run name and notes
- [ ] Keep a short `EXPERIMENTS.md` file: one row per run, what you changed, what happened
- [ ] Back up model checkpoints to Google Drive after every successful training run
- [ ] Update presentation slides as you complete each phase — do not leave them to the last day

---

## Quick Reference — Key Numbers to Hit

| Metric | Acceptable | Good | Excellent |
|--------|-----------|------|-----------|
| mAP@0.5 (fine-tuned) | > 0.50 | > 0.65 | > 0.75 |
| mAP@0.5:0.95 | > 0.25 | > 0.40 | > 0.55 |
| FPS (YOLOv9, batch=1, **specify your GPU**) | > 15 | > 30 | > 60 |
| Zero-shot vs fine-tuned gap | Show improvement | +10 mAP | +20 mAP |
