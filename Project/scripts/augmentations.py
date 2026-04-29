"""
augmentations.py
----------------
Albumentations pipelines for the Phase 3 snow-augmentation ablation study.

Three named variants:
  "none"  — no custom augmentation (ultralytics' built-in aug remains active)
  "snow"  — RandomSnow + RandomFog + RandomBrightnessContrast
  "full"  — snow pipeline + GaussNoise + MotionBlur

Public API
----------
  build_pipeline(variant, bbox_format)  → A.Compose | None
  build_frcnn_pipeline(variant)         → A.Compose | None  (pascal_voc / xyxy-abs)
  build_detr_pipeline(variant)          → A.Compose | None  (coco / xywh-abs)
  inject_into_yolo(model, variant)      → None   (call before model.train())
  visualize_augmentation(paths, pipeline, n, seed) → matplotlib Figure

bbox_format values (albumentations convention):
  "yolo"       — cx cy w h normalised  (ultralytics / YOLOv9)
  "pascal_voc" — x1 y1 x2 y2 absolute  (torchvision / Faster R-CNN)
  "coco"       — x y w h absolute      (pycocotools / RT-DETR)

Requirements: albumentations>=1.4.0
"""


import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def _snow_transforms() -> list:
    """Return a fresh list of snow-weather transform instances."""
    return [
        A.RandomSnow(
            snow_point_range=(0.1, 0.3),
            brightness_coeff=2.5,
            p=0.5,
        ),
        A.RandomFog(
            fog_coef_range=(0.1, 0.3),
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
    ]


def _full_transforms() -> list:
    """Return a fresh list of full-augmentation transform instances."""
    return _snow_transforms() + [
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.MotionBlur(blur_limit=7, p=0.3),
    ]


def _bbox_params(fmt: str) -> A.BboxParams:
    """Return BboxParams for the given albumentations bbox format string."""
    return A.BboxParams(
        format=fmt,
        label_fields=["class_labels"],
        min_visibility=0.2,
    )


def build_pipeline(variant: str, bbox_format: str = "yolo") -> "A.Compose | None":
    """
    Return an Albumentations Compose for the named variant.

    Parameters
    ----------
    variant : {"none", "snow", "full"}
    bbox_format : {"yolo", "pascal_voc", "coco"}
        Albumentations bounding-box format string.  Defaults to ``"yolo"``
        (normalised cx cy w h) for ultralytics compatibility.
        Use ``"pascal_voc"`` for torchvision (xyxy absolute) and
        ``"coco"`` for HuggingFace / pycocotools (xywh absolute).

    Returns
    -------
    A.Compose | None
        None for the "none" variant (no custom augmentation).
    """
    if variant == "none":
        return None
    params = _bbox_params(bbox_format)
    if variant == "snow":
        return A.Compose(_snow_transforms(), bbox_params=params)
    if variant == "full":
        return A.Compose(_full_transforms(), bbox_params=params)
    raise ValueError(
        f"Unknown augmentation variant '{variant}'. Choose: none, snow, full"
    )


def build_frcnn_pipeline(variant: str) -> "A.Compose | None":
    """Convenience wrapper: ``build_pipeline(variant, bbox_format='pascal_voc')``.

    Use with ``NVDDetectionDataset(augment=build_frcnn_pipeline(variant))``.
    Boxes must be in **xyxy absolute-pixel** format, which is the native
    format of ``NVDDetectionDataset``.
    """
    return build_pipeline(variant, bbox_format="pascal_voc")


def build_detr_pipeline(variant: str) -> "A.Compose | None":
    """Convenience wrapper: ``build_pipeline(variant, bbox_format='coco')``.

    Use with ``NVDCocoDataset(augment=build_detr_pipeline(variant))``.
    Boxes must be in **xywh absolute-pixel** format, which is the native
    format of COCO JSON annotations consumed by ``NVDCocoDataset``.
    """
    return build_pipeline(variant, bbox_format="coco")


# ---------------------------------------------------------------------------
# Ultralytics injection
# ---------------------------------------------------------------------------

def inject_into_yolo(model, variant: str) -> None:
    """
    Register an on_train_start callback that replaces the training dataset's
    albumentations pipeline with the named variant.

    Must be called **before** ``model.train()``.
    For the "none" variant this is a no-op — ultralytics' default aug remains.

    How it works
    ------------
    ultralytics >= 8.2 stores an ``Albumentations`` wrapper on the training
    ``YOLODataset`` (``trainer.train_loader.dataset.albumentations``).
    Its ``.transform`` attribute is an ``A.Compose`` that is called every
    iteration with the image and boxes in YOLO format.  Replacing it with our
    custom pipeline keeps the same calling convention while changing the
    augmentation content.

    Parameters
    ----------
    model : ultralytics.YOLO
    variant : {"none", "snow", "full"}
    """
    pipeline = build_pipeline(variant)
    if pipeline is None:
        return  # no-aug baseline — ultralytics defaults stay

    def _on_train_start(trainer) -> None:
        ds  = trainer.train_loader.dataset
        alb = getattr(ds, "albumentations", None)
        if alb is not None and getattr(alb, "transform", None) is not None:
            alb.transform = pipeline
            n = len(pipeline.transforms)
            print(
                f"[Augmentation] Injected '{variant}' pipeline "
                f"({n} transforms) into training dataset."
            )
        else:
            print(
                f"[Augmentation] WARNING: no active albumentations transform "
                "found on the training dataset.  Ensure albumentations>=1.4.0 "
                "is installed.  Falling back to ultralytics' default augmentation."
            )

    model.add_callback("on_train_start", _on_train_start)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_augmentation(
    image_paths,
    pipeline: A.Compose,
    n: int = 6,
    seed: int = 42,
) -> plt.Figure:
    """
    Return a matplotlib Figure showing *n* side-by-side original/augmented pairs.

    Parameters
    ----------
    image_paths : iterable of path-like
        Pool of images to sample from.
    pipeline : A.Compose
        The augmentation pipeline to preview.
    n : int
        Number of image pairs to show.
    seed : int
        Random seed for reproducible sampling.
    """
    rng    = random.Random(seed)
    paths  = list(image_paths)
    sample = rng.sample(paths, min(n, len(paths)))
    n      = len(sample)

    fig, axes = plt.subplots(n, 2, figsize=(10, n * 3))
    if n == 1:
        axes = [axes]

    for ax_row, img_path in zip(axes, sample):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        aug = pipeline(image=img, bboxes=[], class_labels=[])["image"]

        ax_row[0].imshow(img)
        ax_row[0].axis("off")
        ax_row[0].set_title("Original", fontsize=9)

        ax_row[1].imshow(aug)
        ax_row[1].axis("off")
        ax_row[1].set_title("Augmented", fontsize=9)

    fig.suptitle("Augmentation Preview", y=1.005, fontsize=11)
    fig.tight_layout()
    return fig
