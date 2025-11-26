"""
Base Model Abstract Class for MetaGuard

Author: Moslem Mohseni

This module defines the abstract base class for all MetaGuard ML models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all MetaGuard models.

    All model implementations must inherit from this class and implement
    the required abstract methods.

    Attributes:
        model: The underlying ML model instance.
        model_path: Path to the saved model file.
        feature_names: List of feature names used by the model.
        is_fitted: Whether the model has been trained.
    """

    # Default feature names for transaction data
    FEATURE_NAMES: list[str] = [
        "amount",
        "hour",
        "user_age_days",
        "transaction_count",
    ]

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Initialize the base model.

        Args:
            model_path: Optional path to load a pre-trained model from.
        """
        self._model: Any = None
        self._model_path: Path | None = None
        self._is_fitted: bool = False
        self.feature_names: list[str] = self.FEATURE_NAMES.copy()

        if model_path is not None:
            self.load(model_path)

    @property
    def model(self) -> Any:
        """Get the underlying model instance."""
        return self._model

    @property
    def model_path(self) -> Path | None:
        """Get the path to the loaded model."""
        return self._model_path

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been trained."""
        return self._is_fitted

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Train the model on the provided data.

        Args:
            X: Training features array of shape (n_samples, n_features).
            y: Training labels array of shape (n_samples,).
            validation_data: Optional tuple of (X_val, y_val) for validation.

        Returns:
            Dictionary containing training metrics (accuracy, precision, etc.).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Args:
            X: Features array of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Features array of shape (n_samples, n_features).

        Returns:
            Predicted probabilities of shape (n_samples, n_classes).
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model to.
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load a model from disk.

        Args:
            path: Path to load the model from.
        """
        pass

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.

        Raises:
            NotImplementedError: If the model doesn't support feature importance.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support feature importance"
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary containing model information.
        """
        return {
            "model_type": self.__class__.__name__,
            "model_path": str(self._model_path) if self._model_path else None,
            "is_fitted": self._is_fitted,
            "feature_names": self.feature_names,
        }

    def _validate_features(self, X: np.ndarray) -> None:
        """Validate that input features have correct shape.

        Args:
            X: Features array to validate.

        Raises:
            ValueError: If features don't match expected dimensions.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array")

        expected_features = len(self.feature_names)
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {X.shape[1]}"
            )

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"is_fitted={self._is_fitted}, "
            f"model_path={self._model_path})"
        )
