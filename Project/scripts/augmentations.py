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
import numpy as np


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

    if variant == "crop":
        # Random crop at native res — objects stay at full size (~38px vs ~20px)
        crop_sz = imgsz or 640
        return A.Compose(
            _crop_transforms(crop_sz),
            bbox_params=params,
        )

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

class _NativeResCrop:
    """
    Ultralytics-compatible transform that applies a random crop at NATIVE resolution.

    In ultralytics, ``load_image`` resizes the image to ``imgsz`` before any
    transforms fire.  This means Albumentations-based crops operate on the
    already-downscaled image — defeating the purpose of resolution preservation.

    This transform monkey-patches the dataset's ``load_image`` to return native-
    resolution images and applies the crop + resize here (before LetterBox).

    The image flow becomes:
        load_image (native 1920×1080) → _NativeResCrop (→ 640×640 crop → resize to imgsz)
        → standard pipeline (LetterBox, HSV, Flip, etc.)
    """

    def __init__(self, crop_size: int = 640, imgsz: int = 1024, p: float = 0.7):
        self.crop_size = crop_size
        self.imgsz = imgsz
        self.p = p

    def __call__(self, labels: dict) -> dict:
        img = labels["img"]
        h, w = img.shape[:2]

        # Decide whether to crop or just resize the full image
        if random.random() > self.p or h < self.crop_size or w < self.crop_size:
            # No crop — resize full image to imgsz (standard behavior)
            r = self.imgsz / max(h, w)
            if r != 1:
                new_w = min(int(round(w * r)), self.imgsz)
                new_h = min(int(round(h * r)), self.imgsz)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            labels["img"] = img
            labels["resized_shape"] = img.shape[:2]
            return labels

        # Random crop position
        y_max = h - self.crop_size
        x_max = w - self.crop_size
        y0 = random.randint(0, y_max)
        x0 = random.randint(0, x_max)
        x1 = x0 + self.crop_size
        y1 = y0 + self.crop_size

        # Crop the image
        img = img[y0:y1, x0:x1].copy()

        # Adjust bounding boxes (normalized [cx, cy, w, h] format)
        if "bboxes" in labels and len(labels["bboxes"]) > 0:
            bboxes = labels["bboxes"].copy()  # (N, 4) normalized

            # Convert to absolute pixels in original image
            cx_abs = bboxes[:, 0] * w
            cy_abs = bboxes[:, 1] * h
            bw_abs = bboxes[:, 2] * w
            bh_abs = bboxes[:, 3] * h

            # Box corners
            bx1 = cx_abs - bw_abs / 2
            by1 = cy_abs - bh_abs / 2
            bx2 = cx_abs + bw_abs / 2
            by2 = cy_abs + bh_abs / 2

            # Clip to crop region
            bx1_c = np.clip(bx1, x0, x1)
            by1_c = np.clip(by1, y0, y1)
            bx2_c = np.clip(bx2, x0, x1)
            by2_c = np.clip(by2, y0, y1)

            # Compute clipped area vs original area
            orig_area = bw_abs * bh_abs
            clip_w = bx2_c - bx1_c
            clip_h = by2_c - by1_c
            clip_area = clip_w * clip_h

            # Keep boxes with >= 30% area inside crop
            keep = (clip_area / (orig_area + 1e-8)) >= 0.3
            keep &= (clip_w > 2) & (clip_h > 2)  # minimum size

            if keep.any():
                # Convert to crop-local normalized coordinates
                new_cx = ((bx1_c[keep] + bx2_c[keep]) / 2 - x0) / self.crop_size
                new_cy = ((by1_c[keep] + by2_c[keep]) / 2 - y0) / self.crop_size
                new_w = clip_w[keep] / self.crop_size
                new_h = clip_h[keep] / self.crop_size

                labels["bboxes"] = np.column_stack([new_cx, new_cy, new_w, new_h])
                # Filter cls and other per-box arrays
                if "cls" in labels:
                    labels["cls"] = labels["cls"][keep]
                if "segments" in labels:
                    labels["segments"] = [s for s, k in zip(labels["segments"], keep) if k]
                if "keypoints" in labels:
                    labels["keypoints"] = labels["keypoints"][keep]
            else:
                # No boxes survived — return full image resized instead
                img = labels.get("_orig_img", img)
                r = self.imgsz / max(labels["ori_shape"])
                if r != 1:
                    oh, ow = labels["ori_shape"]
                    new_w_r = min(int(round(ow * r)), self.imgsz)
                    new_h_r = min(int(round(oh * r)), self.imgsz)
                    img = cv2.resize(labels.get("_orig_img", img),
                                     (new_w_r, new_h_r), interpolation=cv2.INTER_LINEAR)
                labels["img"] = img
                labels["resized_shape"] = img.shape[:2]
                return labels

        # Resize crop to imgsz (upscale: 640→1024 makes cars ~61px!)
        r = self.imgsz / self.crop_size
        if r != 1:
            new_sz = int(round(self.crop_size * r))
            img = cv2.resize(img, (new_sz, new_sz), interpolation=cv2.INTER_LINEAR)

        labels["img"] = img
        labels["ori_shape"] = (self.crop_size, self.crop_size)
        labels["resized_shape"] = img.shape[:2]
        return labels

    def __repr__(self):
        return f"{self.__class__.__name__}(crop={self.crop_size}, imgsz={self.imgsz}, p={self.p})"


def _patch_load_image_native(dataset) -> None:
    """
    Monkey-patch dataset.load_image to return NATIVE resolution images
    (skip the built-in resize to imgsz). This lets our _NativeResCrop
    transform apply the crop at full resolution.
    """
    import math
    from ultralytics.utils import imread

    original_load = dataset.load_image

    def _load_native(i, rect_mode=True):
        """Load image at native resolution (no resize)."""
        im = dataset.ims[i]
        f = dataset.im_files[i]
        fn = dataset.npy_files[i]

        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception:
                    im = imread(f, flags=dataset.cv2_flag)
            else:
                im = imread(f, flags=dataset.cv2_flag)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

        h0, w0 = im.shape[:2]
        if im.ndim == 2:
            im = im[..., None]
        # Return native resolution — no resize
        return im, (h0, w0), (h0, w0)

    dataset.load_image = _load_native


def inject_into_yolo(model, variant: str) -> None:
    """
    Inject custom augmentation into Ultralytics YOLO training.

    For **crop variants** (crop, crop_snow, crop_full):
        Monkey-patches load_image to return native-resolution images and
        inserts a NativeResCrop transform BEFORE the standard pipeline.
        This ensures cars stay at ~38px during cropping (not ~20px after resize).

    For **weather-only variants** (snow, full, geo, etc.):
        Replaces the Albumentations transform in the standard pipeline.
        These don't depend on resolution, so firing after resize is fine.

    Uses ``on_train_epoch_start`` so the DataLoader and dataset exist.
    """
    is_crop_variant = variant.startswith("crop")

    # For crop variants, build weather-only pipeline (applied via Albumentations slot)
    if is_crop_variant:
        # Extract the weather part (without the crop)
        weather_variant = variant.replace("crop_", "").replace("crop", "")
        weather_pipeline = build_pipeline(weather_variant) if weather_variant else None
    else:
        weather_pipeline = None

    # For non-crop variants, build the full pipeline for Albumentations injection
    full_pipeline = None if is_crop_variant else build_pipeline(variant)

    if not is_crop_variant and full_pipeline is None:
        print("[Augmentation] Using 'none' variant → Ultralytics native augmentation only.")
        return

    _injected = [False]

    def _inject(trainer) -> None:
        if _injected[0]:
            return
        _injected[0] = True

        loader = getattr(trainer, "train_loader", None)
        if loader is None:
            print("[Augmentation] WARNING: train_loader not available.")
            return

        ds = loader.dataset

        if is_crop_variant:
            # === CROP VARIANT: native-res crop before resize ===
            imgsz = ds.imgsz

            # 1. Monkey-patch load_image to skip resize
            _patch_load_image_native(ds)

            # 2. Insert NativeResCrop as FIRST transform
            crop_transform = _NativeResCrop(crop_size=640, imgsz=imgsz, p=0.7)
            if hasattr(ds, "transforms") and hasattr(ds.transforms, "transforms"):
                ds.transforms.transforms.insert(0, crop_transform)
                print(f"[Augmentation] Injected NativeResCrop(640→{imgsz}) as first transform. "
                      f"Cars at ~38px native → ~{int(38 * imgsz / 640)}px after upscale.")
            else:
                print("[Augmentation] WARNING: Could not insert NativeResCrop transform.")
                return

            # 3. Also inject weather augmentation if present (snow/full)
            if weather_pipeline is not None:
                _inject_weather(ds, weather_pipeline, weather_variant)
        else:
            # === NON-CROP VARIANT: standard Albumentations injection ===
            _inject_weather(ds, full_pipeline, variant)

    def _inject_weather(ds, pipeline, name):
        """Inject a pipeline into the Albumentations slot."""
        # Path 1 — direct attribute
        alb = getattr(ds, "albumentations", None)
        if alb is not None and getattr(alb, "transform", None) is not None:
            alb.transform = pipeline
            if hasattr(alb, "contains_spatial"):
                alb.contains_spatial = False  # weather transforms aren't spatial
            print(f"[Augmentation] Injected '{name}' weather pipeline via ds.albumentations.")
            return

        # Path 2 — embedded in transforms list
        transform_list = getattr(getattr(ds, "transforms", None), "transforms", [])
        for t in transform_list:
            if type(t).__name__ == "Albumentations" and getattr(t, "transform", None) is not None:
                t.transform = pipeline
                if hasattr(t, "contains_spatial"):
                    t.contains_spatial = False
                print(f"[Augmentation] Injected '{name}' weather pipeline via transforms.")
                return

        print(f"[Augmentation] WARNING: Could not inject '{name}' weather pipeline.")

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