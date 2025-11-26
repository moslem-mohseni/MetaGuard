"""
MetaGuard Models Module

Contains model definitions and training utilities.

Author: Moslem Mohseni
"""

from __future__ import annotations

from .base import BaseModel
from .ensemble import EnsembleModel, create_default_ensemble
from .random_forest import RandomForestModel

# Conditional XGBoost import
try:
    from .xgboost_model import XGBoostModel, is_xgboost_available
except ImportError:
    XGBoostModel = None  # type: ignore
    is_xgboost_available = lambda: False  # noqa: E731

__all__: list[str] = [
    "BaseModel",
    "EnsembleModel",
    "RandomForestModel",
    "XGBoostModel",
    "create_default_ensemble",
    "is_xgboost_available",
]
