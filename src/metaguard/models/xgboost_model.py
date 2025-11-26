"""
XGBoost Model for MetaGuard

Author: Moslem Mohseni

This module implements the XGBoost classifier for fraud detection.
Requires the optional xgboost dependency: pip install metaguard[xgboost]
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

logger = get_logger(__name__)

# Try to import xgboost, set flag if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


class XGBoostModel(BaseModel):
    """XGBoost model for fraud detection.

    This model uses XGBoost's gradient boosting classifier with optimized
    hyperparameters for fraud detection tasks.

    Note:
        Requires xgboost package: pip install metaguard[xgboost]

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        subsample: Subsample ratio for training.
        colsample_bytree: Column subsample ratio.
        scale_pos_weight: Balance positive/negative weights.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """Initialize the XGBoost model.

        Args:
            model_path: Optional path to load a pre-trained model.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate (eta).
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns.
            scale_pos_weight: Balance positive/negative weights.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 for all CPUs).

        Raises:
            ImportError: If xgboost is not installed.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install metaguard[xgboost]"
            )

        # Store hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Call parent init
        super().__init__(model_path)

    def _create_model(self) -> Any:
        """Create a new XGBClassifier with current hyperparameters.

        Returns:
            Configured XGBClassifier instance.
        """
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }

        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight

        return xgb.XGBClassifier(**params)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Train the XGBoost model.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).
            validation_data: Optional (X_val, y_val) for validation metrics.

        Returns:
            Dictionary with training metrics.
        """
        self._validate_features(X)

        # Calculate scale_pos_weight if not set (for imbalanced data)
        if self.scale_pos_weight is None:
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            if pos_count > 0:
                self.scale_pos_weight = neg_count / pos_count

        logger.info(
            f"Training XGBoost with {X.shape[0]} samples, "
            f"{self.n_estimators} rounds, lr={self.learning_rate}"
        )

        # Create and train model
        self._model = self._create_model()

        # Prepare eval set if validation data provided
        eval_set = None
        if validation_data is not None:
            eval_set = [validation_data]

        self._model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
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
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "scale_pos_weight": self.scale_pos_weight,
                "random_state": self.random_state,
            },
            "feature_names": self.feature_names,
            "model_type": "XGBoostModel",
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

            if isinstance(model_data, dict) and "model" in model_data:
                self._model = model_data["model"]
                if "feature_names" in model_data:
                    self.feature_names = model_data["feature_names"]
                if "hyperparameters" in model_data:
                    hp = model_data["hyperparameters"]
                    self.n_estimators = hp.get("n_estimators", self.n_estimators)
                    self.max_depth = hp.get("max_depth", self.max_depth)
                    self.learning_rate = hp.get("learning_rate", self.learning_rate)
            else:
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
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": self.scale_pos_weight,
        })

        if self._is_fitted:
            info["feature_importance"] = self.get_feature_importance()

        return info


def is_xgboost_available() -> bool:
    """Check if XGBoost is installed.

    Returns:
        True if xgboost is available, False otherwise.
    """
    return XGBOOST_AVAILABLE
