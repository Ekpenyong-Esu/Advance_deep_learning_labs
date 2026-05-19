"""
analysis package
----------------
Error analysis for the NVD car detection project.

Four sub-modules, each with a single responsibility:

  inference    — load checkpoints and run per-image prediction
  categoriser  — IoU-match predictions → GT and classify FN / FP / Poor
  visualiser   — render cropped failure panels and save image grids
  reporter     — build summary DataFrames, bar charts, and CSV exports

Typical usage (see project.ipynb Section 6):

    from analysis.inference   import collect_test_items, infer_all_models
    from analysis.categoriser import categorise_all_models
    from analysis.visualiser  import save_all_grids
    from analysis.reporter    import build_summary_df, plot_error_bar_chart, save_summary_csv
"""

from .inference   import collect_test_items, infer_all_models
from .categoriser import categorise_all_models
from .visualiser  import save_all_grids
from .reporter    import build_summary_df, plot_error_bar_chart, save_summary_csv

__all__ = [
    "collect_test_items",
    "infer_all_models",
    "categorise_all_models",
    "save_all_grids",
    "build_summary_df",
    "plot_error_bar_chart",
    "save_summary_csv",
]
