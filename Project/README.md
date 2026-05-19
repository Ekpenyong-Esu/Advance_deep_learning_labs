# Car Detection in Snow Using Deep Learning

**Course:** D7047E Advanced Deep Learning | LTU VT2026  
**Team:** <!-- Add team member names here -->

---

## Motivation

Autonomous vehicles and surveillance systems must operate reliably in adverse weather conditions. Snow introduces occlusion, low contrast, and texture homogenisation that significantly degrades standard object detectors trained on clear-weather data. This project fine-tunes and compares deep learning object detectors (YOLOv9, DETR, Faster R-CNN) on the Nordic Vehicle Dataset (NVD) and evaluates the impact of synthetic snow/weather augmentation on detection performance.

---

## Dataset

The [Nordic Vehicle Dataset (NVD)](https://nvd.ltu-ai.dev/) is a UAV (drone) dataset — frames extracted from videos captured by unmanned aerial vehicles (UAVs) flying over northern Sweden under various winter conditions.

| Split | Recording(s) | Frames |
|-------|-------------|--------|
| Train | 2022-12-02 Asjo 01_stabilized, 2022-12-03 Nyland 01_stabilized, 2022-12-04 Bjenberg 02, 2022-12-23 Asjo 01_HD 5x stab | 4 355 |
| Val   | same 4 recordings as train (frame-level split) | 2 904 |
| Test  | 2022-12-23 Bjenberg 02_stabilized | 1 191 |

### Dataset Setup

1. Download NVD from https://nvd.ltu-ai.dev/ (requires registration)
2. Extract the "Labeled Frames (YOLO Format)" archive so `data/raw/` contains `images/`, `labels/`, `train.txt`, `val.txt`, `test.txt`
3. Verify the dataset loaded correctly:
   ```bash
   python scripts/inspect_dataset.py --data-yaml configs/data.yaml
   ```
4. `configs/data.yaml` is ready to pass directly to ultralytics — no conversion needed

---

## Installation

> **YOLOv9 Implementation: Option A — `ultralytics`**  
> Chosen for its maintained API, simpler training/inference calls, and active community support.

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Training scripts will be added in Phase 2.
# See TODO_CarDetection_Snow.md for the planned commands.
```

---

## Inference

```bash
python scripts/detect.py --source path/to/image.jpg --weights models/final_best_model.pt
```

---

## Results

<!-- Fill in after experiments are complete -->

| Model | Pretrain | Fine-tuned | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | FPS |
|-------|----------|-----------|---------|-------------|-----------|--------|----|-----|
| YOLOv9 (zero-shot) | COCO | ✗ | — | — | — | — | — | — |
| YOLOv9 | COCO | ✓ | — | — | — | — | — | — |
| YOLOv9 + snow aug | COCO | ✓ | — | — | — | — | — | — |
| YOLOv9 + full aug | COCO | ✓ | — | — | — | — | — | — |
| DETR (ResNet-50) | COCO | ✓ | — | — | — | — | — | — |
| Faster R-CNN | COCO | ✓ | — | — | — | — | — | — |

> FPS measured on <!-- specify GPU, e.g. NVIDIA RTX 3080 10GB --> at batch size 1.

---

## Sample Outputs

<!-- Add detection example images here once experiments are complete -->

---

## Model Weights

Final model weights are available at: <!-- Add GitHub Release or Google Drive link -->

---

## Project Structure

```
├── configs/          # Training config YAMLs
├── data/
│   ├── raw/          # Original NVD recordings (not tracked by git)
│   └── processed/    # YOLO-format annotations (not tracked by git)
├── models/           # Saved checkpoints (not tracked by git)
├── presentation/     # Slides for the 10-minute project presentation
├── results/
│   └── figures/      # PR curves, ablation charts, error grids
├── scripts/          # Training, inference, conversion scripts
├── main.ipynb        # Main entry point — EDA, training, evaluation, error analysis
├── EXPERIMENTS.md    # One row per run — what changed, what happened
└── requirements.txt
```

---

## Experiment Tracking

All runs are logged to [Weights & Biases](https://wandb.ai). See `EXPERIMENTS.md` for a human-readable summary.

---

## Citation

If you use NVD in your work, please cite:

```
Mokayed et al., "Nordic Vehicle Dataset", CVPR Workshops 2023.
```
