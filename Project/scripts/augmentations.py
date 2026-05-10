"""
augmentations.py
----------------
Albumentations pipelines for the Phase 3 snow-augmentation ablation study.

Variants
--------
  "none"  — Ultralytics native augmentation only
  "snow"  — native YOLO + realistic snow/weather augmentation
  "full"  — native YOLO + weather + sensor degradation (noise + blur)
"""

import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def _snow_transforms() -> list:
    """
    Realistic snow/weather augmentations for aerial UAV footage.
    Separates weather effects from photometric (brightness/contrast) changes.
    """
    return [
        # Weather effects: Snow or Fog (not both at once)
        A.OneOf(
            [
                A.RandomSnow(
                    snow_point_range=(0.01, 0.10),
                    brightness_coeff=1.20,
                    p=1.0,
                ),
                A.RandomFog(
                    fog_coef_range=(0.03, 0.15),
                    p=1.0,
                ),
            ],
            p=0.40,                    # 40% chance of applying snow or fog
        ),
        
        # Independent brightness/contrast jitter (common in snowy conditions)
        A.RandomBrightnessContrast(
            brightness_limit=(-0.20, 0.20),
            contrast_limit=(-0.15, 0.20),
            p=0.35,
        ),
    ]


def _full_transforms() -> list:
    """Weather + sensor/motion degradation (UAV-realistic)."""
    return _snow_transforms() + [

        # Sensor noise (mild, realistic UAV conditions)
        A.GaussNoise(
            std_range=(0.005, 0.02),
            p=0.30,
        ),

        # Motion blur (reduced severity)
        A.MotionBlur(
            blur_limit=(3, 5),
            p=0.20,
        ),

        # Compression / transmission artifacts
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.ImageCompression(quality_range=(60, 95), p=1.0),
            ],
            p=0.25,
        ),
    ]


def _bbox_params(fmt: str) -> A.BboxParams:
    return A.BboxParams(
        format=fmt,
        label_fields=["class_labels"],
        min_visibility=0.08,      # Balanced for partially visible cars
        clip=True,
    )


def build_pipeline(variant: str, bbox_format: str = "yolo", imgsz: int | None = None) -> A.Compose | None:
    """
    Return an Albumentations Compose for the named variant.

    Parameters
    ----------
    imgsz : int | None
        When provided, prepends ``A.SmallestMaxSize(imgsz)`` so that all
        pixel-level transforms operate on already-downscaled images rather
        than full-resolution UAV frames.  This prevents the DataLoader from
        becoming a CPU bottleneck (e.g. RandomSnow on a 4K image is ~8×
        slower than on a 1024-px image).
    """
    if variant == "none":
        return None

    params = _bbox_params(bbox_format)
    pre = []  # ← always initialize first

    if imgsz is not None:
        pre = [A.SmallestMaxSize(max_size=imgsz)]  # ← ADD THIS LINE

    if variant == "snow":
        return A.Compose(pre + _snow_transforms(), bbox_params=params)

    if variant == "full":
        return A.Compose(pre + _full_transforms(), bbox_params=params)

    raise ValueError(f"Unknown augmentation variant '{variant}'. Choose from: none, snow, full")


def build_frcnn_pipeline(variant: str, imgsz: int | None = None):
    return build_pipeline(variant, bbox_format="pascal_voc", imgsz=imgsz)


def build_detr_pipeline(variant: str, imgsz: int | None = None):
    return build_pipeline(variant, bbox_format="coco", imgsz=imgsz)




# ---------------------------------------------------------------------------
# Ultralytics YOLO injection
# ---------------------------------------------------------------------------

def inject_into_yolo(model, variant: str) -> None:
    """
    Inject custom Albumentations pipeline into Ultralytics YOLO training.
    Native YOLO augmentations (Mosaic, HSV, geometric, etc.) remain active.

    Uses ``on_train_epoch_start`` (not ``on_train_start``) so the DataLoader
    and its dataset are guaranteed to exist.  Injection is attempted only once
    via the ``_injected`` flag.
    """
    pipeline = build_pipeline(variant)
    if pipeline is None:
        print("[Augmentation] Using 'none' variant → Ultralytics native augmentation only.")
        return

    _injected = [False]

    def _inject(trainer) -> None:
        if _injected[0]:
            return
        _injected[0] = True  # mark regardless — don't retry every epoch

        loader = getattr(trainer, "train_loader", None)
        if loader is None:
            print("[Augmentation] WARNING: train_loader not available. "
                  "Falling back to native Ultralytics augmentation.")
            return

        ds = loader.dataset

        # Path 1 — direct attribute (ultralytics < 8.3)
        alb = getattr(ds, "albumentations", None)
        if alb is not None and getattr(alb, "transform", None) is not None:
            alb.transform = pipeline
            if hasattr(alb, "contains_spatial"):
                alb.contains_spatial = True
            print(f"[Augmentation] Injected '{variant}' pipeline via ds.albumentations "
                  f"({len(pipeline.transforms)} transform groups).")
            return

        # Path 2 — embedded in ds.transforms.transforms (ultralytics ≥ 8.3)
        transform_list = getattr(getattr(ds, "transforms", None), "transforms", [])
        for t in transform_list:
            if type(t).__name__ == "Albumentations" and getattr(t, "transform", None) is not None:
                t.transform = pipeline
                if hasattr(t, "contains_spatial"):
                    t.contains_spatial = True
                print(f"[Augmentation] Injected '{variant}' pipeline via transforms pipeline "
                      f"({len(pipeline.transforms)} transform groups).")
                return

        print("[Augmentation] WARNING: Could not find Albumentations wrapper on dataset. "
              "Falling back to native Ultralytics augmentation.")

    model.add_callback("on_train_epoch_start", _inject)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_augmentation(
    image_paths,
    pipeline: A.Compose,
    n: int = 6,
    seed: int = 42,
):
    """Preview original vs augmented images."""
    if pipeline is None:
        raise ValueError("Pipeline is None. Use 'snow' or 'full' variant.")

    rng = random.Random(seed)
    paths = list(image_paths)
    if not paths:
        raise ValueError("No image paths provided.")

    sample = rng.sample(paths, min(n, len(paths)))

    fig, axes = plt.subplots(len(sample), 2, figsize=(10, len(sample) * 3))
    if len(sample) == 1:
        axes = [axes]

    for ax_row, img_path in zip(axes, sample):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Build a bare pipeline (no BboxParams) to suppress the
        # "no spatial transform for bboxes" warning during preview.
        bare = A.Compose(pipeline.transforms)
        aug = bare(image=img)["image"]

        ax_row[0].imshow(img)
        ax_row[0].axis("off")
        ax_row[0].set_title("Original", fontsize=9)

        ax_row[1].imshow(aug)
        ax_row[1].axis("off")
        ax_row[1].set_title("Augmented", fontsize=9)

    fig.suptitle("Augmentation Preview", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig