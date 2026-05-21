"""
subsample_frames.py
-------------------
Reduce temporal redundancy by keeping only every Nth frame per recording.

PROBLEM: Adjacent video frames are nearly identical — training on 6056 frames
where avg gap is 1.3 gives massive redundancy.  The model memorises scenes
rather than learning generalisable car features.

SOLUTION: Subsample by frame index within each recording.  This:
  - Forces the model to learn from more diverse snapshots
  - Reduces overfitting to specific background/car positions
  - Speeds up each epoch (fewer images) allowing more epochs in same time

Usage:
    python scripts/subsample_frames.py --step 5
    python scripts/subsample_frames.py --step 3 --input data/raw/train_rec.txt

Output: data/raw/train_rec_sub.txt  (can be referenced in data_rec.yaml)
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def subsample(input_txt: Path, step: int) -> list[str]:
    """Keep every `step`-th frame per recording, sorted by frame number."""
    lines = [l.strip() for l in input_txt.read_text().splitlines() if l.strip()]

    # Group by recording
    recordings: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for line in lines:
        m = re.search(r"images/(.+?)-frame(\d+)", line)
        if m:
            recordings[m.group(1)].append((int(m.group(2)), line))
        else:
            recordings["_unknown"].append((0, line))

    # Subsample: sort by frame number, keep every Nth
    result: list[str] = []
    for rec, frames in recordings.items():
        frames.sort(key=lambda x: x[0])
        result.extend(line for i, (_, line) in enumerate(frames) if i % step == 0)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="Input split file (default: <project_root>/data/raw/train_rec.txt)")
    parser.add_argument("--step", type=int, default=5, help="Keep every Nth frame (default: 5)")
    parser.add_argument("--output", default=None, help="Output file (default: input_sub.txt)")
    args = parser.parse_args()

    # Resolve paths relative to project root regardless of working directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"

    input_path = Path(args.input) if args.input else data_dir / "train_rec.txt"
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_sub" + input_path.suffix
    )

    result = subsample(input_path, args.step)
    output_path.write_text("\n".join(result) + "\n", encoding="utf-8")
    original_count = len(input_path.read_text().splitlines())
    print(f"Subsampled: {original_count} → {len(result)} frames (step={args.step})")
    print(f"Written: {output_path}")

if __name__ == "__main__":
    main()
