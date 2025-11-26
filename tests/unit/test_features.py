"""
Unit tests for MetaGuard Features Module

Author: Moslem Mohseni
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metaguard.features import (
    FeatureEngineer,
    extract_features,
    create_risk_features,
    normalize_features,
    apply_normalization,
)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    @pytest.fixture
    def engineer(self) -> FeatureEngineer:
        """Create a FeatureEngineer instance."""
        return FeatureEngineer(include_derived=True)

    @pytest.fixture
    def sample_transactions(self) -> list[dict]:
        """Create sample transaction data."""
        return [
            {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
            {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50},
            {"amount": 50, "hour": 10, "user_age_days": 365, "transaction_count": 3},
        ]

    def test_engineer_initialization(self, engineer: FeatureEngineer) -> None:
        """Test engineer initializes correctly."""
        assert engineer.include_derived == True  # noqa: E712
        assert "amount" in engineer.feature_names
        assert "log_amount" in engineer.feature_names

    def test_engineer_base_only(self) -> None:
        """Test engineer with base features only."""
        engineer = FeatureEngineer(include_derived=False)

        assert engineer.include_derived == False  # noqa: E712
        assert len(engineer.feature_names) == 4
        assert "log_amount" not in engineer.feature_names

    def test_transform_list(
        self, engineer: FeatureEngineer, sample_transactions: list[dict]
    ) -> None:
        """Test transform with list input."""
        features = engineer.transform(sample_transactions)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 3
        assert features.shape[1] == len(engineer.feature_names)

    def test_transform_dataframe(
        self, engineer: FeatureEngineer, sample_transactions: list[dict]
    ) -> None:
        """Test transform with DataFrame input."""
        df = pd.DataFrame(sample_transactions)
        features = engineer.transform(df)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 3

    def test_transform_single(
        self, engineer: FeatureEngineer, sample_transactions: list[dict]
    ) -> None:
        """Test transform_single method."""
        features = engineer.transform_single(sample_transactions[0])

        assert features.shape == (1, len(engineer.feature_names))

    def test_derived_features_calculation(
        self, engineer: FeatureEngineer
    ) -> None:
        """Test derived feature calculations."""
        transaction = {"amount": 100, "hour": 3, "user_age_days": 10, "transaction_count": 20}
        features = engineer.transform_single(transaction)

        # Get feature indices
        names = engineer.get_feature_names()

        # log_amount = log(101) ~ 4.62
        log_idx = names.index("log_amount")
        assert abs(features[0, log_idx] - np.log1p(100)) < 0.01

        # is_night_hour should be 1 (hour=3)
        night_idx = names.index("is_night_hour")
        assert features[0, night_idx] == 1.0

        # is_new_account should be 1 (age < 30)
        new_idx = names.index("is_new_account")
        assert features[0, new_idx] == 1.0

        # transactions_per_day = 20/10 = 2
        velocity_idx = names.index("transactions_per_day")
        assert features[0, velocity_idx] == 2.0

    def test_get_feature_names(self, engineer: FeatureEngineer) -> None:
        """Test get_feature_names method."""
        names = engineer.get_feature_names()

        assert isinstance(names, list)
        assert names == engineer.feature_names
        # Should return a copy
        names.append("test")
        assert "test" not in engineer.feature_names

    def test_get_feature_descriptions(self, engineer: FeatureEngineer) -> None:
        """Test get_feature_descriptions method."""
        descriptions = engineer.get_feature_descriptions()

        assert isinstance(descriptions, dict)
        assert "amount" in descriptions
        assert "log_amount" in descriptions
        assert len(descriptions) == 11  # All features


class TestExtractFeatures:
    """Tests for extract_features function."""

    @pytest.fixture
    def sample_transactions(self) -> list[dict]:
        """Create sample transaction data."""
        return [
            {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
            {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50},
        ]

    def test_extract_with_derived(self, sample_transactions: list[dict]) -> None:
        """Test extract_features with derived features."""
        features, names = extract_features(sample_transactions, include_derived=True)

        assert features.shape[0] == 2
        assert features.shape[1] == len(names)
        assert "log_amount" in names

    def test_extract_without_derived(self, sample_transactions: list[dict]) -> None:
        """Test extract_features without derived features."""
        features, names = extract_features(sample_transactions, include_derived=False)

        assert features.shape[1] == 4
        assert "log_amount" not in names


class TestCreateRiskFeatures:
    """Tests for create_risk_features function."""

    def test_high_risk_transaction(self) -> None:
        """Test risk features for high-risk transaction."""
        transaction = {
            "amount": 5000,
            "hour": 3,
            "user_age_days": 5,
            "transaction_count": 50,
        }
        features = create_risk_features(transaction)

        assert features["is_high_amount"] == True  # noqa: E712
        assert features["is_new_account"] == True  # noqa: E712
        assert features["is_high_frequency"] == True  # noqa: E712
        assert features["is_unusual_hour"] == True  # noqa: E712
        assert features["combined_risk"] > 0.5

    def test_low_risk_transaction(self) -> None:
        """Test risk features for low-risk transaction."""
        transaction = {
            "amount": 100,
            "hour": 14,
            "user_age_days": 365,
            "transaction_count": 5,
        }
        features = create_risk_features(transaction)

        assert features["is_high_amount"] == False  # noqa: E712
        assert features["is_new_account"] == False  # noqa: E712
        assert features["is_high_frequency"] == False  # noqa: E712
        assert features["is_unusual_hour"] == False  # noqa: E712
        assert features["combined_risk"] < 0.3

    def test_risk_score_bounds(self) -> None:
        """Test risk scores are bounded correctly."""
        transaction = {
            "amount": 100000,
            "hour": 3,
            "user_age_days": 1,
            "transaction_count": 500,
        }
        features = create_risk_features(transaction)

        # All risk components should be <= 1
        assert features["amount_risk"] <= 1.0
        assert features["frequency_risk"] <= 1.0
        assert features["combined_risk"] <= 1.0

    def test_missing_fields_handled(self) -> None:
        """Test handling of missing fields."""
        transaction = {"amount": 100}  # Missing other fields
        features = create_risk_features(transaction)

        assert "combined_risk" in features
        assert isinstance(features["combined_risk"], float)


class TestNormalizeFeatures:
    """Tests for normalize_features function."""

    @pytest.fixture
    def sample_features(self) -> np.ndarray:
        """Create sample feature array."""
        return np.array([
            [100, 14, 30, 5],
            [5000, 3, 5, 50],
            [50, 10, 365, 3],
        ], dtype=float)

    def test_standard_normalization(self, sample_features: np.ndarray) -> None:
        """Test standard (z-score) normalization."""
        normalized, params = normalize_features(sample_features, method="standard")

        # Mean should be ~0, std ~1
        assert np.allclose(normalized.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(normalized.std(axis=0), 1, atol=1e-10)

        # Params should contain mean and std
        assert "mean" in params
        assert "std" in params

    def test_minmax_normalization(self, sample_features: np.ndarray) -> None:
        """Test min-max normalization."""
        normalized, params = normalize_features(sample_features, method="minmax")

        # Values should be in [0, 1]
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

        # Params should contain min and max
        assert "min" in params
        assert "max" in params

    def test_robust_normalization(self, sample_features: np.ndarray) -> None:
        """Test robust normalization."""
        normalized, params = normalize_features(sample_features, method="robust")

        # Params should contain median and iqr
        assert "median" in params
        assert "iqr" in params

    def test_invalid_method_raises(self, sample_features: np.ndarray) -> None:
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalize_features(sample_features, method="invalid")


class TestApplyNormalization:
    """Tests for apply_normalization function."""

    def test_apply_standard(self) -> None:
        """Test applying standard normalization."""
        features = np.array([[100, 200], [300, 400]], dtype=float)
        _, params = normalize_features(features, method="standard")

        # Apply to new data
        new_features = np.array([[150, 250]], dtype=float)
        normalized = apply_normalization(new_features, params, method="standard")

        # Should be normalized using original params
        expected = (new_features - params["mean"]) / params["std"]
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_apply_minmax(self) -> None:
        """Test applying min-max normalization."""
        features = np.array([[0, 0], [100, 100]], dtype=float)
        _, params = normalize_features(features, method="minmax")

        # Apply to value at 50%
        new_features = np.array([[50, 50]], dtype=float)
        normalized = apply_normalization(new_features, params, method="minmax")

        # Should be 0.5
        np.testing.assert_array_almost_equal(normalized, [[0.5, 0.5]])
