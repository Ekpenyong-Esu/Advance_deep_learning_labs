"""
results_writer.py
-----------------
Persist EvalMetrics to CSV files.

Responsibilities
----------------
  save_zero_shot_result  — write a single result to results/baseline_zero_shot.csv
  append_comparison_row  — append one row to results/model_comparison.csv
  init_comparison_csv    — create model_comparison.csv with headers if absent

All output directories are created on first write; the caller never needs
to manage filesystem state.
"""



import csv
from pathlib import Path

from .models import EvalMetrics

_FIELDNAMES = [
    "Model",
    "Pretrain",
    "Fine-tuned",
    "Split",
    "mAP@0.5",
    "mAP@0.5:0.95",
    "Precision",
    "Recall",
    "FPS",
    "Notes",
]


def _metrics_to_row(m: EvalMetrics) -> dict:
    return {
        "Model":        m.model,
        "Pretrain":     m.pretrain,
        "Fine-tuned":   str(m.fine_tuned),
        "Split":        m.split,
        "mAP@0.5":      f"{m.map50:.4f}",
        "mAP@0.5:0.95": f"{m.map50_95:.4f}",
        "Precision":    f"{m.precision:.4f}",
        "Recall":       f"{m.recall:.4f}",
        "FPS":          f"{m.fps:.1f}",
        "Notes":        m.notes,
    }


def save_zero_shot_result(metrics: EvalMetrics, output_path: str | Path) -> None:
    """Write a single EvalMetrics entry to *output_path* (overwrites if it exists)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerow(_metrics_to_row(metrics))
    print(f"[results_writer] Zero-shot result saved → {out}")


def init_comparison_csv(csv_path: str | Path) -> None:
    """Create *csv_path* with column headers if it does not already exist."""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with open(out, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_FIELDNAMES).writeheader()
        print(f"[results_writer] Initialised comparison CSV → {out}")


def append_comparison_row(metrics: EvalMetrics, csv_path: str | Path) -> None:
    """Append one row for *metrics* to *csv_path*, creating the file if needed."""
    out = Path(csv_path)
    init_comparison_csv(out)
    with open(out, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=_FIELDNAMES).writerow(_metrics_to_row(metrics))
    print(f"[results_writer] Appended row for '{metrics.model}' → {out}")
