"""
Feature Engineering Module for MetaGuard

Author: Moslem Mohseni

This module provides feature engineering utilities for fraud detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Feature engineering for transaction data.

    This class provides methods to create derived features from raw
    transaction data to improve model performance.

    Attributes:
        feature_names: List of feature names after transformation.
        include_derived: Whether to include derived features.
    """

    # Base feature names
    BASE_FEATURES: list[str] = [
        "amount",
        "hour",
        "user_age_days",
        "transaction_count",
    ]

    # Derived feature names
    DERIVED_FEATURES: list[str] = [
        "log_amount",
        "amount_per_transaction",
        "transactions_per_day",
        "is_night_hour",
        "is_new_account",
        "hour_sin",
        "hour_cos",
    ]

    def __init__(self, include_derived: bool = True) -> None:
        """Initialize the feature engineer.

        Args:
            include_derived: Whether to include derived features.
        """
        self.include_derived = include_derived
        self.feature_names = self.BASE_FEATURES.copy()
        if include_derived:
            self.feature_names.extend(self.DERIVED_FEATURES)

    def transform(
        self,
        transactions: list[dict[str, Any]] | pd.DataFrame,
    ) -> np.ndarray:
        """Transform raw transactions into feature array.

        Args:
            transactions: List of transaction dicts or DataFrame.

        Returns:
            Feature array of shape (n_samples, n_features).
        """
        # Convert to DataFrame if needed
        if isinstance(transactions, list):
            df = pd.DataFrame(transactions)
        else:
            df = transactions.copy()

        # Extract base features
        features = df[self.BASE_FEATURES].values.astype(float)

        if self.include_derived:
            # Create derived features
            derived = self._create_derived_features(df)
            features = np.hstack([features, derived])

        return features

    def _create_derived_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create derived features from base features.

        Args:
            df: DataFrame with base features.

        Returns:
            Array of derived features.
        """
        derived = []

        # Log amount (for handling large values)
        log_amount = np.log1p(df["amount"].values)
        derived.append(log_amount)

        # Amount per transaction
        tx_count = df["transaction_count"].values.clip(min=1)
        amount_per_tx = df["amount"].values / tx_count
        derived.append(amount_per_tx)

        # Transactions per day (velocity)
        user_age = df["user_age_days"].values.clip(min=1)
        tx_per_day = df["transaction_count"].values / user_age
        derived.append(tx_per_day)

        # Is night hour (0-5 or 22-23)
        hour = df["hour"].values
        is_night = ((hour >= 0) & (hour <= 5)) | (hour >= 22)
        derived.append(is_night.astype(float))

        # Is new account (< 30 days)
        is_new = (df["user_age_days"].values < 30).astype(float)
        derived.append(is_new)

        # Cyclical encoding of hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        derived.append(hour_sin)
        derived.append(hour_cos)

        return np.column_stack(derived)

    def transform_single(self, transaction: dict[str, Any]) -> np.ndarray:
        """Transform a single transaction.

        Args:
            transaction: Transaction dictionary.

        Returns:
            Feature array of shape (1, n_features).
        """
        return self.transform([transaction])

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names in order.
        """
        return self.feature_names.copy()

    def get_feature_descriptions(self) -> dict[str, str]:
        """Get descriptions for all features.

        Returns:
            Dictionary mapping feature names to descriptions.
        """
        return {
            "amount": "Transaction amount in currency units",
            "hour": "Hour of day when transaction occurred (0-23)",
            "user_age_days": "Account age in days",
            "transaction_count": "Number of prior transactions",
            "log_amount": "Natural log of (amount + 1)",
            "amount_per_transaction": "Average amount per transaction",
            "transactions_per_day": "Transaction velocity (count / age)",
            "is_night_hour": "1 if transaction is during night (0-5 or 22-23)",
            "is_new_account": "1 if account age < 30 days",
            "hour_sin": "Sine encoding of hour (cyclical)",
            "hour_cos": "Cosine encoding of hour (cyclical)",
        }


def extract_features(
    transactions: list[dict[str, Any]] | pd.DataFrame,
    include_derived: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Extract features from transaction data.

    Convenience function for one-off feature extraction.

    Args:
        transactions: Transaction data.
        include_derived: Whether to include derived features.

    Returns:
        Tuple of (feature_array, feature_names).
    """
    engineer = FeatureEngineer(include_derived=include_derived)
    features = engineer.transform(transactions)
    return features, engineer.get_feature_names()


def create_risk_features(transaction: dict[str, Any]) -> dict[str, Any]:
    """Create risk-related features for a transaction.

    Args:
        transaction: Transaction dictionary.

    Returns:
        Dictionary with risk features.
    """
    amount = transaction.get("amount", 0)
    hour = transaction.get("hour", 12)
    user_age = transaction.get("user_age_days", 1)
    tx_count = transaction.get("transaction_count", 0)

    # Risk indicators
    risk_features = {
        "is_high_amount": amount > 1000,
        "is_very_high_amount": amount > 5000,
        "is_new_account": user_age < 30,
        "is_very_new_account": user_age < 7,
        "is_high_frequency": tx_count > 20,
        "is_very_high_frequency": tx_count > 50,
        "is_unusual_hour": 0 <= hour <= 5,
        "is_late_night": hour >= 22 or hour <= 3,
    }

    # Risk score components
    risk_features["amount_risk"] = min(amount / 10000, 1.0)
    risk_features["age_risk"] = max(0, 1 - user_age / 365)
    risk_features["frequency_risk"] = min(tx_count / 100, 1.0)
    risk_features["hour_risk"] = 1.0 if 0 <= hour <= 5 else 0.0

    # Combined risk score
    risk_features["combined_risk"] = (
        risk_features["amount_risk"] * 0.3
        + risk_features["age_risk"] * 0.3
        + risk_features["frequency_risk"] * 0.2
        + risk_features["hour_risk"] * 0.2
    )

    return risk_features


def normalize_features(
    features: np.ndarray,
    method: str = "standard",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Normalize feature values.

    Args:
        features: Feature array of shape (n_samples, n_features).
        method: Normalization method ('standard', 'minmax', 'robust').

    Returns:
        Tuple of (normalized_features, normalization_params).
    """
    params: dict[str, np.ndarray] = {}

    if method == "standard":
        # Z-score normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = (features - mean) / std
        params["mean"] = mean
        params["std"] = std

    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        normalized = (features - min_val) / range_val
        params["min"] = min_val
        params["max"] = max_val

    elif method == "robust":
        # Robust normalization using median and IQR
        median = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1
        normalized = (features - median) / iqr
        params["median"] = median
        params["iqr"] = iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def apply_normalization(
    features: np.ndarray,
    params: dict[str, np.ndarray],
    method: str = "standard",
) -> np.ndarray:
    """Apply pre-computed normalization to features.

    Args:
        features: Feature array to normalize.
        params: Normalization parameters from normalize_features().
        method: Normalization method used.

    Returns:
        Normalized feature array.
    """
    if method == "standard":
        return (features - params["mean"]) / params["std"]
    elif method == "minmax":
        range_val = params["max"] - params["min"]
        return (features - params["min"]) / range_val
    elif method == "robust":
        return (features - params["median"]) / params["iqr"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")
