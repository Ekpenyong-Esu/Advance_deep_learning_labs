"""
data/__init__.py — Lab 1 Data Package
======================================
Re-exports all public loader functions so that the rest of the codebase
can import from the package root without caring which sub-module owns each.

    from data.ann_loader         import get_ann_loaders
    from data.lstm_loader        import get_lstm_loaders
    from data.transformer_loader import get_transformer_loaders
    from data.base_loader        import get_raw_splits
"""

from data.ann_loader         import get_ann_loaders
from data.lstm_loader        import get_lstm_loaders
from data.transformer_loader import get_transformer_loaders
from data.base_loader        import get_raw_splits

__all__ = [
    "get_ann_loaders",
    "get_lstm_loaders",
    "get_transformer_loaders",
    "get_raw_splits",
]
