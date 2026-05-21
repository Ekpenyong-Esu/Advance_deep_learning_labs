"""
make_recording_splits.py
------------------------
Generate recording-level train/val splits for NVD to prevent temporal leakage.

PROBLEM WITH THE EXISTING SPLITS
---------------------------------
The original train.txt and val.txt both contain frames from the same 4
recordings.  Because consecutive video frames are visually near-identical, a
random frame-level split gives the model a val set that is effectively a
near-duplicate of its training data.  Val mAP of ~0.99 is a consequence of
this leakage, NOT of genuine generalisation.

The test set comes from a 5th recording (Bjenberg 02_stabilized) that never
appears in train or val — so the train→test gap is large by design.

SOLUTION: RECORDING-LEVEL VAL SPLIT
--------------------------------------
Hold out one complete recording as val; train on the remaining ones.

  Train  : Bjenberg 02 + Asjo 01_stabilized + Asjo 01_HD 5x stab
           (all frames from these recordings across the original train+val txts)
  Val    : Nyland 01_stabilized
           (all frames from this recording across original train+val txts)
  Test   : Bjenberg 02_stabilized  ← unchanged

Recording frame counts (train+val pool):
  Bjenberg 02            : 4003 frames  → train
  Asjo 01_stabilized     :  801 frames  → train
  Asjo 01_HD 5x stab     : 1252 frames  → train
  Nyland 01_stabilized   : 1203 frames  → val  ← geographically distinct location
  --------------------------------------------------
  Total train            : 6056 frames
  Total val              : 1203 frames
  Total test             : 1191 frames  (unchanged)

Why Nyland for val?
  Nyland 01_stabilized is from a different geographic location than the other
  three training recordings, making it the most informative held-out proxy for
  test-time generalisation.

Output files
------------
  data/raw/train_rec.txt  — recording-level training split
  data/raw/val_rec.txt    — recording-level validation split
  data/raw/test.txt       — unchanged; leave it as-is

These files can be used by updating configs/data.yaml to point to
train_rec.txt and val_rec.txt instead of train.txt and val.txt.

Usage
-----
  python scripts/make_recording_splits.py
  python scripts/make_recording_splits.py --val-recording "Nyland 01_stabilized"
"""

import argparse
import re
from pathlib import Path

# Project root = parent of the scripts/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default recording to hold out as val
DEFAULT_VAL_RECORDING = "2022-12-03 Nyland 01_stabilized"


def extract_recording(line: str) -> str | None:
    """Extract the recording name from a frame path line in a split .txt file."""
    m = re.search(r"images/(.+?)-frame", line)
    return m.group(1) if m else None


def build_recording_splits(
    data_dir: Path,
    val_recording: str = DEFAULT_VAL_RECORDING,
) -> tuple[list[str], list[str]]:
    """
    Read the original train.txt + val.txt and redistribute by recording.

    Returns (train_lines, val_lines) where all lines from `val_recording`
    go to val and everything else goes to train.
    """
    all_lines: list[str] = []
    for split_name in ("train", "val"):
        txt = data_dir / f"{split_name}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"Expected split file not found: {txt}")
        all_lines.extend(
            line for line in txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    train_lines: list[str] = []
    val_lines: list[str] = []
    unmatched: list[str] = []

    for line in all_lines:
        rec = extract_recording(line)
        if rec is None:
            unmatched.append(line)
            continue
        if rec == val_recording:
            val_lines.append(line)
        else:
            train_lines.append(line)

    if unmatched:
        print(f"[WARNING] {len(unmatched)} lines had no recognisable recording name — skipped:")
        for l in unmatched[:5]:
            print(f"  {l}")

    return train_lines, val_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--val-recording",
        default=DEFAULT_VAL_RECORDING,
        help=(
            "Full recording name to hold out as val "
            f"(default: '{DEFAULT_VAL_RECORDING}')"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to the directory containing train.txt / val.txt (default: <project_root>/data/raw)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing any files",
    )
    args = parser.parse_args()

    # Resolve data_dir relative to the project root (one level above scripts/)
    # so the script works regardless of which directory you run it from.
    project_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "raw"

    train_lines, val_lines = build_recording_splits(data_dir, args.val_recording)

    # --- Statistics -----------------------------------------------------------
    from collections import Counter

    def recording_counts(lines: list[str]) -> Counter:
        c: Counter = Counter()
        for l in lines:
            rec = extract_recording(l)
            if rec:
                c[rec] += 1
        return c

    print("\nRecording-level split statistics")
    print("=" * 60)
    print(f"\nTRAIN  ({len(train_lines)} frames):")
    for rec, n in sorted(recording_counts(train_lines).items()):
        print(f"  {rec}: {n}")

    print(f"\nVAL  ({len(val_lines)} frames):")
    for rec, n in sorted(recording_counts(val_lines).items()):
        print(f"  {rec}: {n}")

    test_txt = data_dir / "test.txt"
    if test_txt.exists():
        test_lines = [l for l in test_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\nTEST  ({len(test_lines)} frames — unchanged):")
        for rec, n in sorted(recording_counts(test_lines).items()):
            print(f"  {rec}: {n}")

    print()

    if args.dry_run:
        print("[DRY RUN] No files written.")
        return

    # --- Write output files ---------------------------------------------------
    out_train = data_dir / "train_rec.txt"
    out_val   = data_dir / "val_rec.txt"

    out_train.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    out_val.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"Written: {out_train}  ({len(train_lines)} lines)")
    print(f"Written: {out_val}  ({len(val_lines)} lines)")
    print()
    print("To use the recording-level splits, update configs/data.yaml:")
    print("  train: train_rec.txt")
    print("  val:   val_rec.txt")

if __name__ == "__main__":
    main()
