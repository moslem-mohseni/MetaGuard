"""
Ensemble Model for MetaGuard

Author: Moslem Mohseni

This module implements an ensemble model that combines multiple classifiers
for improved fraud detection accuracy.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
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
from .random_forest import RandomForestModel

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple classifiers.

    This model combines predictions from multiple base models using
    weighted averaging or voting strategies for improved accuracy.

    Attributes:
        models: List of base models.
        weights: Weights for each model's predictions.
        voting: Voting strategy ('soft' or 'hard').
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        models: list[BaseModel] | None = None,
        weights: list[float] | None = None,
        voting: str = "soft",
    ) -> None:
        """Initialize the Ensemble model.

        Args:
            model_path: Optional path to load a pre-trained ensemble.
            models: List of base models to combine.
            weights: Weights for each model (equal weights if None).
            voting: Voting strategy ('soft' for probability averaging,
                   'hard' for majority voting).
        """
        self.models: list[BaseModel] = models or []
        self.voting = voting

        # Set weights
        if weights is not None:
            if len(weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            self.weights = np.array(weights) / np.sum(weights)  # Normalize
        else:
            n = max(len(self.models), 1)
            self.weights = np.ones(n) / n

        # Don't call parent init if loading from path
        self._model = None
        self._model_path = None
        self._is_fitted = False
        self.feature_names = self.FEATURE_NAMES.copy()

        if model_path is not None:
            self.load(model_path)

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble.

        Args:
            model: Model to add.
            weight: Weight for this model's predictions.
        """
        self.models.append(model)
        # Recalculate weights
        new_weights = list(self.weights * (len(self.models) - 1)) + [weight]
        self.weights = np.array(new_weights) / np.sum(new_weights)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Train all models in the ensemble.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).
            validation_data: Optional (X_val, y_val) for validation metrics.

        Returns:
            Dictionary with ensemble training metrics.
        """
        self._validate_features(X)

        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")

        logger.info(f"Training ensemble with {len(self.models)} models")

        # Train each model
        all_metrics = []
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            metrics = model.train(X, y, validation_data)
            all_metrics.append(metrics)

        self._is_fitted = True

        # Calculate ensemble metrics
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        ensemble_metrics = {
            "train_accuracy": accuracy_score(y, y_pred),
            "train_precision": precision_score(y, y_pred, zero_division=0),
            "train_recall": recall_score(y, y_pred, zero_division=0),
            "train_f1": f1_score(y, y_pred, zero_division=0),
            "train_auc_roc": roc_auc_score(y, y_proba),
        }

        # Add validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val_pred = self.predict(X_val)
            y_val_proba = self.predict_proba(X_val)[:, 1]

            ensemble_metrics.update(
                {
                    "val_accuracy": accuracy_score(y_val, y_val_pred),
                    "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                    "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                    "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
                    "val_auc_roc": roc_auc_score(y_val, y_val_proba),
                }
            )

        # Include individual model metrics
        for i, metrics in enumerate(all_metrics):
            for key, value in metrics.items():
                ensemble_metrics[f"model_{i}_{key}"] = value

        logger.info(
            f"Ensemble training complete - Accuracy: {ensemble_metrics['train_accuracy']:.4f}, "
            f"AUC-ROC: {ensemble_metrics['train_auc_roc']:.4f}"
        )

        return ensemble_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using ensemble.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        self._validate_features(X)
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        if self.voting == "hard":
            # Majority voting
            predictions = np.array([model.predict(X) for model in self.models])
            # Weighted voting
            weighted_preds = predictions * self.weights[:, np.newaxis]
            return (weighted_preds.sum(axis=0) >= 0.5).astype(int)
        else:
            # Soft voting (threshold on probabilities)
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using weighted averaging.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, n_classes).
        """
        self._validate_features(X)
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get probabilities from each model
        all_proba = np.array([model.predict_proba(X) for model in self.models])

        # Weighted average
        weighted_proba = np.average(all_proba, axis=0, weights=self.weights)

        return weighted_proba

    def save(self, path: str | Path) -> None:
        """Save the ensemble model to disk.

        Args:
            path: Path to save the model.
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted ensemble")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save ensemble configuration and all models
        model_data = {
            "models": self.models,
            "weights": self.weights.tolist(),
            "voting": self.voting,
            "feature_names": self.feature_names,
            "model_type": "EnsembleModel",
        }

        with path.open("wb") as f:
            pickle.dump(model_data, f)

        self._model_path = path
        logger.info(f"Ensemble saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load an ensemble model from disk.

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
            with path.open("rb") as f:
                model_data = pickle.load(f)  # nosec B301 - Loading trusted model files

            self.models = model_data["models"]
            self.weights = np.array(model_data["weights"])
            self.voting = model_data.get("voting", "soft")
            if "feature_names" in model_data:
                self.feature_names = model_data["feature_names"]

            self._model_path = path
            self._is_fitted = all(m.is_fitted for m in self.models)
            logger.info(f"Ensemble loaded from {path} with {len(self.models)} models")

        except Exception as e:
            raise ModelLoadError(str(path), e) from e

    def get_feature_importance(self) -> dict[str, float]:
        """Get averaged feature importance from all models.

        Returns:
            Dictionary mapping feature names to average importance scores.
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted to get feature importance")

        # Collect importances from models that support it
        all_importances = []
        model_weights = []

        for i, model in enumerate(self.models):
            try:
                importance = model.get_feature_importance()
                all_importances.append(importance)
                model_weights.append(self.weights[i])
            except NotImplementedError:
                continue

        if not all_importances:
            raise NotImplementedError("No models in ensemble support feature importance")

        # Weighted average of importances
        model_weights = np.array(model_weights) / np.sum(model_weights)
        avg_importance = {}

        for feature in self.feature_names:
            weighted_sum = sum(imp[feature] * w for imp, w in zip(all_importances, model_weights))
            avg_importance[feature] = weighted_sum

        return avg_importance

    def get_model_info(self) -> dict[str, Any]:
        """Get detailed ensemble information.

        Returns:
            Dictionary with ensemble details.
        """
        info = super().get_model_info()
        info.update(
            {
                "n_models": len(self.models),
                "voting": self.voting,
                "weights": self.weights.tolist(),
                "model_types": [m.__class__.__name__ for m in self.models],
            }
        )

        if self._is_fitted:
            try:
                info["feature_importance"] = self.get_feature_importance()
            except NotImplementedError:
                pass

        return info


def create_default_ensemble() -> EnsembleModel:
    """Create a default ensemble with common models.

    Returns:
        Configured EnsembleModel with RandomForest models using
        different hyperparameters.
    """
    models = [
        RandomForestModel(n_estimators=100, max_depth=10),
        RandomForestModel(n_estimators=100, max_depth=15),
        RandomForestModel(n_estimators=200, max_depth=12),
    ]

    return EnsembleModel(
        models=models,
        weights=[1.0, 1.0, 1.0],
        voting="soft",
    )
