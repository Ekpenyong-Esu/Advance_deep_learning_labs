"""
reporting.py
------------
Human-readable formatting of dataset statistics for CLI output.

Presentation concern kept separate from data_utils.dataset_stats (computation)
so that stats dictionaries can be consumed programmatically without side effects.
"""

def print_stats(stats: dict) -> None:
    """Pretty-print a statistics dictionary (from compute_stats) to stdout."""
    bpi = stats["boxes_per_image"]
    avg_bpi = sum(bpi) / len(bpi) if bpi else 0.0
    print(f"\n{'='*50}")
    print(f"  Split : {stats['split']}")
    print(f"  Images: {stats['num_images']}")
    print(f"  Boxes : {stats['num_boxes']}  (avg {avg_bpi:.1f} per image)")
    print(f"  Images without annotations: {stats['images_without_boxes']}")
    print("\n  Class counts:")
    for name, cnt in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        print(f"    {name:<20s} {cnt:>6d}")
    print("\n  Annotation quality:")
    print(f"    Crowd boxes (iscrowd=1) : {stats['crowd_boxes']}")
    print(f"    Tiny boxes  (<32×32 px) : {stats['tiny_boxes']}")
    print(f"    Images with missing dims: {stats['missing_dims']}")
    print(f"{'='*50}\n")
