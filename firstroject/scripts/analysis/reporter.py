"""
reporter.py
-----------
Responsibility: produce summary outputs from categorised error data.

This module knows about pandas, matplotlib, and CSV files.  It has no
knowledge of model frameworks or OpenCV.

Public API
----------
  build_summary_df(all_errors)              → pd.DataFrame
  save_summary_csv(df, path)                → None
  plot_error_bar_chart(all_errors, save_path) → matplotlib.figure.Figure
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_df(all_errors: dict[str, dict[str, list]]) -> pd.DataFrame:
    """Build a tidy DataFrame summarising error counts per model.

    Parameters
    ----------
    all_errors : Output of :func:`~categoriser.categorise_all_models`.

    Returns
    -------
    pd.DataFrame with columns:
        ``Model``, ``False Negatives``, ``False Positives``, ``Poor Localisation``.
    """
    rows = [
        {
            "Model":             name,
            "False Negatives":   len(errors["fn"]),
            "False Positives":   len(errors["fp"]),
            "Poor Localisation": len(errors["poor"]),
        }
        for name, errors in all_errors.items()
    ]
    return pd.DataFrame(rows)


def save_summary_csv(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to a CSV file at *path*, creating parent directories if needed.

    Parameters
    ----------
    df   : Output of :func:`build_summary_df`.
    path : Absolute path for the output CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def plot_error_bar_chart(
    all_errors: dict[str, dict[str, list]],
    save_path: Path | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Plot a grouped bar chart comparing error counts across all models.

    Parameters
    ----------
    all_errors : Output of :func:`~categoriser.categorise_all_models`.
    save_path  : If given, the figure is saved as a PNG at this path.
    figsize    : Matplotlib figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels     = list(all_errors.keys())
    short_labels = [n.replace(" · ", "\n") for n in labels]
    fn_vals    = [len(all_errors[n]["fn"])   for n in labels]
    fp_vals    = [len(all_errors[n]["fp"])   for n in labels]
    poor_vals  = [len(all_errors[n]["poor"]) for n in labels]

    x     = np.arange(len(labels))
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - bar_w, fn_vals,   bar_w, label="False Negatives",   color="tomato",     alpha=0.85)
    ax.bar(x,         fp_vals,   bar_w, label="False Positives",   color="steelblue",  alpha=0.85)
    ax.bar(x + bar_w, poor_vals, bar_w, label="Poor Localisation", color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("Error count")
    ax.set_title("Error Analysis — All Models  (test set · conf ≥ 0.3)")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig
