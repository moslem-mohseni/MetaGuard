"""
Random Forest Model for MetaGuard

Author: Moslem Mohseni

This module implements the Random Forest classifier for fraud detection.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..utils.exceptions import ModelLoadError, ModelNotFoundError
from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest model for fraud detection.

    This model uses scikit-learn's RandomForestClassifier with optimized
    hyperparameters for fraud detection tasks.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of the trees.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        random_state: Random seed for reproducibility.
        class_weight: Weights for handling imbalanced classes.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_estimators: int = 100,
        max_depth: int | None = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        class_weight: str | dict[int, float] | None = "balanced",
        n_jobs: int = -1,
    ) -> None:
        """Initialize the Random Forest model.

        Args:
            model_path: Optional path to load a pre-trained model.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees (None for unlimited).
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples at leaf nodes.
            random_state: Random seed for reproducibility.
            class_weight: Class weights ('balanced' for imbalanced data).
            n_jobs: Number of parallel jobs (-1 for all CPUs).
        """
        # Store hyperparameters before calling parent init
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs

        # Call parent init (may load model)
        super().__init__(model_path)

    def _create_model(self) -> RandomForestClassifier:
        """Create a new RandomForestClassifier with current hyperparameters.

        Returns:
            Configured RandomForestClassifier instance.
        """
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Train the Random Forest model.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).
            validation_data: Optional (X_val, y_val) for validation metrics.

        Returns:
            Dictionary with training metrics.
        """
        self._validate_features(X)

        logger.info(
            f"Training RandomForest with {X.shape[0]} samples, "
            f"{self.n_estimators} trees"
        )

        # Create and train model
        self._model = self._create_model()
        self._model.fit(X, y)
        self._is_fitted = True

        # Calculate training metrics
        y_pred = self._model.predict(X)
        y_proba = self._model.predict_proba(X)[:, 1]

        metrics = {
            "train_accuracy": accuracy_score(y, y_pred),
            "train_precision": precision_score(y, y_pred, zero_division=0),
            "train_recall": recall_score(y, y_pred, zero_division=0),
            "train_f1": f1_score(y, y_pred, zero_division=0),
            "train_auc_roc": roc_auc_score(y, y_proba),
        }

        # Calculate validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            self._validate_features(X_val)

            y_val_pred = self._model.predict(X_val)
            y_val_proba = self._model.predict_proba(X_val)[:, 1]

            metrics.update({
                "val_accuracy": accuracy_score(y_val, y_val_pred),
                "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
                "val_auc_roc": roc_auc_score(y_val, y_val_proba),
            })

        logger.info(
            f"Training complete - Accuracy: {metrics['train_accuracy']:.4f}, "
            f"AUC-ROC: {metrics['train_auc_roc']:.4f}"
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        self._validate_features(X)
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, n_classes).
        """
        self._validate_features(X)
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self._model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model.
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model with metadata
        model_data = {
            "model": self._model,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
                "class_weight": self.class_weight,
            },
            "feature_names": self.feature_names,
            "model_type": "RandomForestModel",
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        self._model_path = path
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load a model from disk.

        Args:
            path: Path to the model file.

        Raises:
            ModelNotFoundError: If model file doesn't exist.
            ModelLoadError: If model file cannot be loaded.
        """
        path = Path(path)

        if not path.exists():
            raise ModelNotFoundError(str(path))

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            # Handle both old format (just model) and new format (dict with metadata)
            if isinstance(model_data, dict) and "model" in model_data:
                self._model = model_data["model"]
                if "feature_names" in model_data:
                    self.feature_names = model_data["feature_names"]
                if "hyperparameters" in model_data:
                    hp = model_data["hyperparameters"]
                    self.n_estimators = hp.get("n_estimators", self.n_estimators)
                    self.max_depth = hp.get("max_depth", self.max_depth)
            else:
                # Old format: just the sklearn model
                self._model = model_data

            self._model_path = path
            self._is_fitted = True
            logger.info(f"Model loaded from {path}")

        except Exception as e:
            raise ModelLoadError(str(path), e) from e

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        importances = self._model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def get_model_info(self) -> dict[str, Any]:
        """Get detailed model information.

        Returns:
            Dictionary with model details.
        """
        info = super().get_model_info()
        info.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
        })

        if self._is_fitted:
            info["feature_importance"] = self.get_feature_importance()

        return info
