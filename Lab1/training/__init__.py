"""
training/__init__.py — Lab 1 Training Package
===============================================
Re-exports the public entry point so experiments only need:

    from training.trainer import train_model
"""

from training.trainer import train_model

__all__ = ["train_model"]
