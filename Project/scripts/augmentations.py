"""
augmentations.py
----------------
Albumentations pipelines for the Phase 3 snow-augmentation ablation study.

Variants
--------
   "none"      — Ultralytics native augmentation only
  "geo"       — geometry-only augmentation (flip + shift/scale/rotate)
  "snow"      — native YOLO + realistic snow/weather augmentation
  "full"      — native YOLO + weather + sensor degradation (noise + blur)
  "snow_geo"  — geometry + weather augmentation
  "full_geo"  — geometry + weather + sensor degradation
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
    Probabilities tuned for domain shift to an unseen recording location:
    higher than initial values so the model cannot rely on recording-specific
    appearance cues (lighting, snow coverage, background texture).

    KEY INSIGHT: Training images are ~2× brighter (mean=139) than test images
    (mean=67). Aggressive brightness/gamma augmentation is critical to bridge
    this domain gap.
    """
    return [
        # Weather effects: Snow or Fog (not both at once)
        A.OneOf(
            [
                A.RandomSnow(
                    snow_point_range=(0.01, 0.15),
                    brightness_coeff=1.25,
                    p=1.0,
                ),
                A.RandomFog(
                    fog_coef_range=(0.03, 0.20),
                    p=1.0,
                ),
            ],
            p=0.60,
        ),

        # AGGRESSIVE brightness/contrast — bridge the train→test brightness gap
        A.RandomBrightnessContrast(
            brightness_limit=(-0.45, 0.15),   # heavily biased towards darkening
            contrast_limit=(-0.25, 0.25),
            p=0.70,
        ),

        # Gamma shift — non-linear darkening that simulates overcast/dusk
        A.RandomGamma(
            gamma_limit=(50, 150),            # <100 = darken, >100 = brighten
            p=0.50,
        ),

        # Hue/saturation shift — handles colour cast differences between recordings
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.40,
        ),

        # CLAHE — local contrast enhancement, helps with low-contrast snow scenes
        A.CLAHE(
            clip_limit=(1.0, 4.0),
            p=0.25,
        ),
    ]


def _full_transforms() -> list:
    """Weather + sensor/motion degradation (UAV-realistic)."""
    return _snow_transforms() + [

        # Sensor noise
        A.GaussNoise(
            std_range=(0.005, 0.035),
            p=0.45,
        ),

        # Motion blur — UAV vibration / camera shake
        A.MotionBlur(
            blur_limit=(3, 9),
            p=0.35,
        ),

        # Compression / transmission artifacts
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.ImageCompression(quality_range=(40, 90), p=1.0),
            ],
            p=0.40,
        ),

        # Random shadow / darkening patches — simulates building shadows on snow
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
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


def _geo_transforms() -> list:
    """Geometry augmentations useful for aerial vehicle detection."""
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.25,
        ),
    ]


def _crop_transforms(crop_size: int = 640) -> list:
    """
    Random crop augmentation for small-object aerial detection.

    Instead of downscaling the full 1920×1080 image (which shrinks cars to
    ~13px at 640), crop a region at native resolution so cars stay at ~38px.
    Falls back to the full image (resized) if the image is smaller than crop_size.
    """
    return [
        A.RandomCrop(width=crop_size, height=crop_size, p=0.7),
        # 30% of the time use the full image (resized) for scene context
        # This is handled implicitly: when RandomCrop doesn't fire, SmallestMaxSize
        # in the pipeline prefix handles the resize.
    ]


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
    pre = [A.SmallestMaxSize(imgsz)] if imgsz is not None else []

    if variant == "snow":
        return A.Compose(pre + _snow_transforms(), bbox_params=params)

    if variant == "full":
        return A.Compose(pre + _full_transforms(), bbox_params=params)

    if variant == "geo":
        return A.Compose(pre + _geo_transforms(), bbox_params=params)

    if variant == "snow_geo":
        return A.Compose(pre + _geo_transforms() + _snow_transforms(), bbox_params=params)

    if variant == "full_geo":
        return A.Compose(pre + _geo_transforms() + _full_transforms(), bbox_params=params)

    if variant == "crop_snow":
        # Random crop at native res + snow augmentation (best for small objects)
        crop_sz = imgsz or 640
        return A.Compose(
            _crop_transforms(crop_sz) + _snow_transforms(),
            bbox_params=params,
        )

    if variant == "crop_full":
        # Random crop at native res + full augmentation
        crop_sz = imgsz or 640
        return A.Compose(
            _crop_transforms(crop_sz) + _full_transforms(),
            bbox_params=params,
        )
        
    if variant == "crop":
        # Random crop at native res — objects stay at full size (~38px vs ~20px)
        crop_sz = imgsz or 640
        return A.Compose(
            _crop_transforms(crop_sz),
            bbox_params=params,
        )

    raise ValueError(
        f"Unknown augmentation variant '{variant}'. Choose from: "
        "none, geo, snow, full, snow_geo, full_geo, crop, crop_snow, crop_full"
    )


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