"""
MetaGuard Fraud Detector

ML-based fraud detection for metaverse transactions.

Author: Moslem Mohseni
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from metaguard.risk import get_risk_level
from metaguard.utils.config import MetaGuardConfig, get_config, get_default_model_path
from metaguard.utils.exceptions import (
    InvalidTransactionError,
    ModelLoadError,
    ModelNotFoundError,
)
from metaguard.utils.logging import get_logger
from metaguard.utils.validators import validate_transaction

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier

logger = get_logger(__name__)


class SimpleDetector:
    """
    Simple fraud detector using pre-trained ML model.

    This class provides fraud detection capabilities using a pre-trained
    machine learning model (RandomForest by default).

    Author: Moslem Mohseni

    Attributes:
        model: The loaded ML model
        config: Configuration settings

    Example:
        >>> detector = SimpleDetector()
        >>> result = detector.detect({
        ...     "amount": 5000,
        ...     "hour": 3,
        ...     "user_age_days": 5,
        ...     "transaction_count": 50
        ... })
        >>> print(f"Suspicious: {result['is_suspicious']}")
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: MetaGuardConfig | None = None,
    ) -> None:
        """
        Initialize detector with pre-trained model.

        Author: Moslem Mohseni

        Args:
            model_path: Path to the trained model file. If None, uses default.
            config: Optional configuration settings. If None, uses global config.

        Raises:
            ModelNotFoundError: If model file doesn't exist
            ModelLoadError: If model fails to load
        """
        self.config = config or get_config()
        self._model_path = self._resolve_model_path(model_path)
        self._model: RandomForestClassifier | None = None
        self._load_model()
        logger.info(f"SimpleDetector initialized with model: {self._model_path}")

    def _resolve_model_path(self, model_path: str | Path | None) -> Path:
        """
        Resolve the model path from various sources.

        Author: Moslem Mohseni

        Args:
            model_path: Explicit model path or None

        Returns:
            Resolved Path to the model file
        """
        if model_path is not None:
            return Path(model_path)

        if self.config.model_path:
            return Path(self.config.model_path)

        return get_default_model_path()

    def _load_model(self) -> None:
        """
        Load the pre-trained model from file.

        Author: Moslem Mohseni

        Raises:
            ModelNotFoundError: If model file doesn't exist
            ModelLoadError: If model fails to load
        """
        if not self._model_path.exists():
            logger.error(f"Model file not found: {self._model_path}")
            raise ModelNotFoundError(str(self._model_path))

        try:
            with self._model_path.open("rb") as f:
                self._model = pickle.load(f)  # nosec B301 - Loading trusted model files
            logger.debug(f"Model loaded successfully from {self._model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(str(self._model_path), e) from e

    @property
    def model(self) -> RandomForestClassifier:
        """Get the loaded model."""
        if self._model is None:
            raise ModelLoadError(str(self._model_path))
        return self._model

    @property
    def model_path(self) -> Path:
        """Get the model path."""
        return self._model_path

    def detect(self, transaction: dict[str, Any]) -> dict[str, Any]:
        """
        Detect if a transaction is suspicious.

        Author: Moslem Mohseni

        Args:
            transaction: Dictionary with transaction data containing:
                - amount: Transaction amount (positive number)
                - hour: Hour of transaction (0-23)
                - user_age_days: User account age in days (>= 1)
                - transaction_count: Number of previous transactions (>= 0)

        Returns:
            Dictionary with detection results:
                - is_suspicious: Boolean indicating if transaction is suspicious
                - risk_score: Probability of fraud (0-1)
                - risk_level: Categorical level ('Low', 'Medium', 'High')

        Raises:
            InvalidTransactionError: If transaction data is invalid

        Example:
            >>> detector = SimpleDetector()
            >>> result = detector.detect({
            ...     "amount": 5000,
            ...     "hour": 3,
            ...     "user_age_days": 5,
            ...     "transaction_count": 50
            ... })
            >>> result
            {'is_suspicious': True, 'risk_score': 0.85, 'risk_level': 'High'}
        """
        # Validate input
        try:
            validated = validate_transaction(transaction)
        except Exception as e:
            logger.warning(f"Invalid transaction: {e}")
            raise InvalidTransactionError(
                message=str(e),
                field=None,
                value=transaction,
            ) from e

        # Convert to DataFrame with correct feature order
        df = pd.DataFrame(
            [
                {
                    "amount": validated.amount,
                    "hour": validated.hour,
                    "user_age_days": validated.user_age_days,
                    "transaction_count": validated.transaction_count,
                }
            ]
        )

        # Get prediction probability
        prob = self.model.predict_proba(df)[0][1]

        # Determine if suspicious based on threshold
        is_suspicious = bool(prob > self.config.risk_threshold)

        result = {
            "is_suspicious": is_suspicious,
            "risk_score": float(prob),
            "risk_level": get_risk_level(prob * 100),
        }

        logger.debug(
            f"Detection result: suspicious={is_suspicious}, "
            f"score={prob:.4f}, level={result['risk_level']}"
        )

        return result

    def batch_detect(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Detect fraud in multiple transactions.

        Author: Moslem Mohseni

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of detection results, one for each transaction

        Raises:
            InvalidTransactionError: If any transaction is invalid

        Example:
            >>> detector = SimpleDetector()
            >>> transactions = [
            ...     {"amount": 100, "hour": 10, "user_age_days": 100, "transaction_count": 5},
            ...     {"amount": 5000, "hour": 3, "user_age_days": 2, "transaction_count": 50},
            ... ]
            >>> results = detector.batch_detect(transactions)
            >>> len(results)
            2
        """
        if not transactions:
            logger.debug("Empty transaction list provided")
            return []

        logger.info(f"Processing batch of {len(transactions)} transactions")
        results = [self.detect(tx) for tx in transactions]

        suspicious_count = sum(1 for r in results if r["is_suspicious"])
        logger.info(f"Batch complete: {suspicious_count}/{len(results)} suspicious transactions")

        return results

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Author: Moslem Mohseni

        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": str(self._model_path),
            "model_type": type(self.model).__name__,
            "risk_threshold": self.config.risk_threshold,
        }

        # Add model-specific info if available
        if hasattr(self.model, "n_estimators"):
            info["n_estimators"] = self.model.n_estimators
        if hasattr(self.model, "max_depth"):
            info["max_depth"] = self.model.max_depth
        if hasattr(self.model, "feature_importances_"):
            feature_names = ["amount", "hour", "user_age_days", "transaction_count"]
            info["feature_importance"] = dict(
                zip(feature_names, self.model.feature_importances_.tolist())
            )

        return info


def check_transaction(
    transaction: dict[str, Any],
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Quick helper function for single transaction checking.

    This function creates a SimpleDetector instance and checks a single
    transaction. For multiple transactions, create a SimpleDetector
    instance directly to avoid repeated model loading.

    Author: Moslem Mohseni

    Args:
        transaction: Transaction dictionary with amount, hour, user_age_days,
                    and transaction_count
        model_path: Optional path to a custom model file

    Returns:
        Detection result dictionary

    Example:
        >>> from metaguard import check_transaction
        >>> result = check_transaction({
        ...     "amount": 5000,
        ...     "hour": 3,
        ...     "user_age_days": 5,
        ...     "transaction_count": 50
        ... })
        >>> if result['is_suspicious']:
        ...     print(f"Suspicious! Risk: {result['risk_score']:.2%}")
    """
    detector = SimpleDetector(model_path=model_path)
    return detector.detect(transaction)
