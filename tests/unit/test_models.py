"""
Unit tests for MetaGuard Models Module

Author: Moslem Mohseni
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metaguard.models import (
    BaseModel,
    RandomForestModel,
    EnsembleModel,
    create_default_ensemble,
    is_xgboost_available,
)


class TestRandomForestModel:
    """Tests for RandomForestModel class."""

    @pytest.fixture
    def model(self) -> RandomForestModel:
        """Create a RandomForestModel instance."""
        return RandomForestModel(n_estimators=10, max_depth=5)

    @pytest.fixture
    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create synthetic training data."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_model_initialization(self, model: RandomForestModel) -> None:
        """Test model initializes correctly."""
        assert model is not None
        assert model.n_estimators == 10
        assert model.max_depth == 5
        assert model.is_fitted == False  # noqa: E712

    def test_model_training(
        self, model: RandomForestModel, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model training."""
        X, y = training_data
        metrics = model.train(X, y)

        assert model.is_fitted == True  # noqa: E712
        assert "train_accuracy" in metrics
        assert "train_auc_roc" in metrics
        assert 0 <= metrics["train_accuracy"] <= 1
        assert 0 <= metrics["train_auc_roc"] <= 1

    def test_model_training_with_validation(
        self, model: RandomForestModel, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model training with validation data."""
        X, y = training_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        metrics = model.train(X_train, y_train, validation_data=(X_val, y_val))

        assert "val_accuracy" in metrics
        assert "val_auc_roc" in metrics

    def test_model_predict(
        self, model: RandomForestModel, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model prediction."""
        X, y = training_data
        model.train(X, y)

        predictions = model.predict(X)

        assert predictions.shape == (len(X),)
        assert set(predictions).issubset({0, 1})

    def test_model_predict_proba(
        self, model: RandomForestModel, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model probability prediction."""
        X, y = training_data
        model.train(X, y)

        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_model_predict_unfitted_raises(self, model: RandomForestModel) -> None:
        """Test prediction on unfitted model raises error."""
        X = np.random.randn(10, 4)

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_model_save_and_load(
        self,
        model: RandomForestModel,
        training_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Test model save and load."""
        X, y = training_data
        model.train(X, y)

        # Save
        model_path = tmp_path / "test_model.pkl"
        model.save(model_path)
        assert model_path.exists()

        # Load
        loaded_model = RandomForestModel(model_path=model_path)
        assert loaded_model.is_fitted == True  # noqa: E712

        # Predictions should match
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_model_feature_importance(
        self, model: RandomForestModel, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test feature importance extraction."""
        X, y = training_data
        model.train(X, y)

        importance = model.get_feature_importance()

        assert len(importance) == 4
        assert all(isinstance(v, float) for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 0.01  # Should sum to ~1

    def test_model_info(
        self, model: RandomForestModel, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test model info retrieval."""
        X, y = training_data
        model.train(X, y)

        info = model.get_model_info()

        assert info["model_type"] == "RandomForestModel"
        assert info["is_fitted"] == True  # noqa: E712
        assert info["n_estimators"] == 10
        assert "feature_importance" in info

    def test_model_validate_features(self, model: RandomForestModel) -> None:
        """Test feature validation."""
        # Wrong number of features
        X_wrong = np.random.randn(10, 3)  # Should be 4

        with pytest.raises(ValueError, match="Expected 4 features"):
            model._validate_features(X_wrong)

        # 1D array
        X_1d = np.random.randn(10)

        with pytest.raises(ValueError, match="Expected 2D array"):
            model._validate_features(X_1d)


class TestEnsembleModel:
    """Tests for EnsembleModel class."""

    @pytest.fixture
    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create synthetic training data."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_ensemble_initialization(self) -> None:
        """Test ensemble initializes correctly."""
        ensemble = EnsembleModel()
        assert ensemble.models == []
        assert ensemble.voting == "soft"

    def test_ensemble_with_models(self) -> None:
        """Test ensemble with models."""
        models = [
            RandomForestModel(n_estimators=5),
            RandomForestModel(n_estimators=10),
        ]
        ensemble = EnsembleModel(models=models)

        assert len(ensemble.models) == 2
        np.testing.assert_array_almost_equal(ensemble.weights, [0.5, 0.5])

    def test_ensemble_add_model(self) -> None:
        """Test adding model to ensemble."""
        ensemble = EnsembleModel()
        ensemble.add_model(RandomForestModel(n_estimators=5))
        ensemble.add_model(RandomForestModel(n_estimators=10), weight=2.0)

        assert len(ensemble.models) == 2
        # Weights should be normalized

    def test_ensemble_training(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test ensemble training."""
        X, y = training_data
        ensemble = create_default_ensemble()

        metrics = ensemble.train(X, y)

        assert ensemble.is_fitted == True  # noqa: E712
        assert "train_accuracy" in metrics
        assert "train_auc_roc" in metrics

    def test_ensemble_predict(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test ensemble prediction."""
        X, y = training_data
        ensemble = create_default_ensemble()
        ensemble.train(X, y)

        predictions = ensemble.predict(X)

        assert predictions.shape == (len(X),)
        assert set(predictions).issubset({0, 1})

    def test_ensemble_predict_proba(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test ensemble probability prediction."""
        X, y = training_data
        ensemble = create_default_ensemble()
        ensemble.train(X, y)

        proba = ensemble.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_ensemble_hard_voting(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test ensemble with hard voting."""
        X, y = training_data
        models = [
            RandomForestModel(n_estimators=5),
            RandomForestModel(n_estimators=10),
        ]
        ensemble = EnsembleModel(models=models, voting="hard")
        ensemble.train(X, y)

        predictions = ensemble.predict(X)
        assert predictions.shape == (len(X),)

    def test_ensemble_save_and_load(
        self, training_data: tuple[np.ndarray, np.ndarray], tmp_path: Path
    ) -> None:
        """Test ensemble save and load."""
        X, y = training_data
        ensemble = create_default_ensemble()
        ensemble.train(X, y)

        # Save
        model_path = tmp_path / "ensemble.pkl"
        ensemble.save(model_path)
        assert model_path.exists()

        # Load
        loaded = EnsembleModel(model_path=model_path)
        assert loaded.is_fitted == True  # noqa: E712
        assert len(loaded.models) == 3

    def test_ensemble_no_models_raises(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test training empty ensemble raises error."""
        X, y = training_data
        ensemble = EnsembleModel()

        with pytest.raises(ValueError, match="No models"):
            ensemble.train(X, y)

    def test_create_default_ensemble(self) -> None:
        """Test default ensemble creation."""
        ensemble = create_default_ensemble()

        assert len(ensemble.models) == 3
        assert all(isinstance(m, RandomForestModel) for m in ensemble.models)


class TestXGBoostModel:
    """Tests for XGBoostModel class (if available)."""

    @pytest.fixture
    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create synthetic training data."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.mark.skipif(
        not is_xgboost_available(),
        reason="XGBoost not installed"
    )
    def test_xgboost_training(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test XGBoost model training."""
        from metaguard.models import XGBoostModel

        X, y = training_data
        model = XGBoostModel(n_estimators=10)
        metrics = model.train(X, y)

        assert model.is_fitted == True  # noqa: E712
        assert "train_accuracy" in metrics
        assert "train_auc_roc" in metrics

    @pytest.mark.skipif(
        not is_xgboost_available(),
        reason="XGBoost not installed"
    )
    def test_xgboost_predict(
        self, training_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test XGBoost prediction."""
        from metaguard.models import XGBoostModel

        X, y = training_data
        model = XGBoostModel(n_estimators=10)
        model.train(X, y)

        predictions = model.predict(X)
        assert predictions.shape == (len(X),)

    def test_xgboost_availability(self) -> None:
        """Test XGBoost availability check."""
        result = is_xgboost_available()
        assert isinstance(result, bool)


class TestBaseModel:
    """Tests for BaseModel abstract class."""

    def test_cannot_instantiate_base(self) -> None:
        """Test BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore

    def test_feature_names(self) -> None:
        """Test default feature names."""
        assert BaseModel.FEATURE_NAMES == [
            "amount", "hour", "user_age_days", "transaction_count"
        ]
