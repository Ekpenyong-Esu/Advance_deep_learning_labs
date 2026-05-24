"""
tile_dataset.py
---------------
Slice full-resolution images into overlapping tiles for small-object training.

PROBLEM: Cars are ~38px in native 1920×1080 images.  When resized to imgsz=1024,
they shrink to ~20px — below the COCO "small object" threshold (32px).  Standard
augmentation-based crops are probabilistic (30% still see full image).

SOLUTION: Pre-slice every training/val image into 640×640 tiles with 20% overlap.
Each tile preserves native resolution, so cars stay at ~38px.  Tiles with no
annotations are discarded (background-only patches waste training time).

This is the standard approach used by VisDrone/DOTA competition winners.

Usage:
    python scripts/tile_dataset.py
    python scripts/tile_dataset.py --tile-size 640 --overlap 0.2 --min-area 0.3
    python scripts/tile_dataset.py --splits train val test

Output:
    data/tiled/images/   — sliced image tiles (PNG)
    data/tiled/labels/   — corresponding YOLO labels
    data/tiled/train.txt — tile list for training
    data/tiled/val.txt   — tile list for validation
    data/tiled/test.txt  — tile list for testing
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def compute_tile_positions(
    img_w: int, img_h: int, tile_size: int, overlap: float
) -> list[tuple[int, int]]:
    """Return top-left (x, y) positions for tiles covering the full image."""
    stride = int(tile_size * (1 - overlap))
    positions = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            # Clamp to image boundaries
            x = min(x, max(0, img_w - tile_size))
            y = min(y, max(0, img_h - tile_size))
            positions.append((x, y))
    # Remove duplicates (happens when image is smaller than tile)
    return list(set(positions))


def clip_boxes_to_tile(
    boxes: np.ndarray, tile_x: int, tile_y: int, tile_size: int,
    img_w: int, img_h: int, min_area_ratio: float = 0.3
) -> np.ndarray:
    """
    Convert YOLO boxes to tile-local coordinates, discarding boxes
    with less than min_area_ratio of their area inside the tile.

    Parameters
    ----------
    boxes : (N, 5) array — [class_id, cx, cy, w, h] normalized to full image
    tile_x, tile_y : top-left pixel of the tile in the full image
    tile_size : tile width/height in pixels
    img_w, img_h : full image dimensions
    min_area_ratio : minimum fraction of box area that must be inside tile

    Returns
    -------
    (M, 5) array — [class_id, cx, cy, w, h] normalized to tile
    """
    if len(boxes) == 0:
        return np.empty((0, 5))

    # Convert normalized YOLO to pixel coordinates
    cls = boxes[:, 0:1]
    cx_px = boxes[:, 1] * img_w
    cy_px = boxes[:, 2] * img_h
    w_px = boxes[:, 3] * img_w
    h_px = boxes[:, 4] * img_h

    # Box corners in full image
    x1 = cx_px - w_px / 2
    y1 = cy_px - h_px / 2
    x2 = cx_px + w_px / 2
    y2 = cy_px + h_px / 2

    # Tile boundaries
    tx1 = tile_x
    ty1 = tile_y
    tx2 = tile_x + tile_size
    ty2 = tile_y + tile_size

    # Intersection
    ix1 = np.maximum(x1, tx1)
    iy1 = np.maximum(y1, ty1)
    ix2 = np.minimum(x2, tx2)
    iy2 = np.minimum(y2, ty2)

    inter_w = np.maximum(ix2 - ix1, 0)
    inter_h = np.maximum(iy2 - iy1, 0)
    inter_area = inter_w * inter_h
    box_area = w_px * h_px

    # Keep boxes with sufficient area inside tile
    keep = (inter_area / (box_area + 1e-8)) >= min_area_ratio
    if not keep.any():
        return np.empty((0, 5))

    # Clip to tile and convert to tile-local normalized coords
    cx_tile = (np.clip(cx_px[keep], tx1, tx2) - tx1) / tile_size
    cy_tile = (np.clip(cy_px[keep], ty1, ty2) - ty1) / tile_size

    # Clipped width/height
    clipped_x1 = np.maximum(x1[keep], tx1)
    clipped_y1 = np.maximum(y1[keep], ty1)
    clipped_x2 = np.minimum(x2[keep], tx2)
    clipped_y2 = np.minimum(y2[keep], ty2)

    w_tile = (clipped_x2 - clipped_x1) / tile_size
    h_tile = (clipped_y2 - clipped_y1) / tile_size

    # Recompute center from clipped box
    cx_tile = ((clipped_x1 + clipped_x2) / 2 - tx1) / tile_size
    cy_tile = ((clipped_y1 + clipped_y2) / 2 - ty1) / tile_size

    result = np.column_stack([cls[keep], cx_tile, cy_tile, w_tile, h_tile])
    return result


def tile_single_image(
    img_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    tile_size: int,
    overlap: float,
    min_area_ratio: float,
    keep_empty: bool = False,
) -> list[str]:
    """Tile one image and return list of relative tile paths."""
    img = Image.open(img_path)
    img_w, img_h = img.size

    # Load YOLO labels
    boxes = np.empty((0, 5))
    if label_path.exists():
        lines = label_path.read_text().strip().splitlines()
        if lines:
            boxes = np.array([[float(v) for v in l.split()] for l in lines])

    positions = compute_tile_positions(img_w, img_h, tile_size, overlap)
    tile_paths = []

    stem = img_path.stem
    img_array = np.array(img)

    for i, (tx, ty) in enumerate(sorted(positions)):
        # Clip tile boxes
        tile_boxes = clip_boxes_to_tile(
            boxes, tx, ty, tile_size, img_w, img_h, min_area_ratio
        )

        # Skip empty tiles unless requested
        if len(tile_boxes) == 0 and not keep_empty:
            continue

        # Extract tile
        tile_img = img_array[ty:ty + tile_size, tx:tx + tile_size]
        tile_name = f"{stem}_tile{i:03d}"

        # Save tile image
        tile_img_path = out_img_dir / f"{tile_name}.png"
        Image.fromarray(tile_img).save(tile_img_path)

        # Save tile labels
        tile_lbl_path = out_lbl_dir / f"{tile_name}.txt"
        if len(tile_boxes) > 0:
            lines = []
            for box in tile_boxes:
                cls_id = int(box[0])
                lines.append(f"{cls_id} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}")
            tile_lbl_path.write_text("\n".join(lines) + "\n")
        else:
            tile_lbl_path.write_text("")

        tile_paths.append(f"./images/{tile_name}.png")

    return tile_paths


def main():
    parser = argparse.ArgumentParser(description="Tile dataset for small-object training")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile width/height in pixels")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap fraction between tiles")
    parser.add_argument("--min-area", type=float, default=0.3, help="Min fraction of box area in tile to keep")
    parser.add_argument("--splits", nargs="+", default=["train_rec_sub", "val_rec", "test"], help="Splits to process")
    parser.add_argument("--data-root", type=str, default="data/raw", help="Root of raw data")
    parser.add_argument("--output", type=str, default="data/tiled", help="Output directory")
    parser.add_argument("--keep-empty", action="store_true", help="Keep tiles with no objects")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output = Path(args.output)
    out_img_dir = output / "images"
    out_lbl_dir = output / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_file = data_root / f"{split}.txt"
        if not split_file.exists():
            print(f"⚠ {split_file} not found, skipping")
            continue

        lines = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        all_tile_paths = []

        print(f"\n{'='*60}")
        print(f"Processing {split} ({len(lines)} images)")
        print(f"{'='*60}")

        for idx, rel_path in enumerate(lines):
            img_path = (data_root / rel_path).resolve()
            # Derive label path
            lbl_name = img_path.stem + ".txt"
            label_path = data_root / "labels" / lbl_name

            if not img_path.exists():
                continue

            tiles = tile_single_image(
                img_path, label_path, out_img_dir, out_lbl_dir,
                args.tile_size, args.overlap, args.min_area, args.keep_empty,
            )
            all_tile_paths.extend(tiles)

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(lines)}] {len(all_tile_paths)} tiles so far")

        # Write split file
        out_split = output / f"{split}.txt"
        out_split.write_text("\n".join(all_tile_paths) + "\n")
        print(f"  ✓ {split}: {len(lines)} images → {len(all_tile_paths)} tiles")
        print(f"    Written to {out_split}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Done! Tiled dataset at: {output}")
    print(f"  Tile size: {args.tile_size}×{args.tile_size}")
    print(f"  Overlap: {args.overlap*100:.0f}%")
    print(f"  Min area ratio: {args.min_area}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
