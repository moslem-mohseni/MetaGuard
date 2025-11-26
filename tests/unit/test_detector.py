"""
Unit tests for MetaGuard Detector Module

Author: Moslem Mohseni
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from metaguard import SimpleDetector, check_transaction
from metaguard.utils.exceptions import (
    InvalidTransactionError,
    ModelNotFoundError,
)


class TestSimpleDetector:
    """Tests for SimpleDetector class."""

    def test_detector_initialization(self, model_path: Path) -> None:
        """Test detector initializes correctly."""
        detector = SimpleDetector(model_path=str(model_path))
        assert detector is not None
        assert detector.model is not None

    def test_detector_with_path_object(self, model_path: Path) -> None:
        """Test detector accepts Path object."""
        detector = SimpleDetector(model_path=model_path)
        assert detector is not None

    def test_detector_with_invalid_path(self, temp_dir: Path) -> None:
        """Test detector raises error for non-existent model."""
        fake_path = temp_dir / "nonexistent.pkl"
        with pytest.raises(ModelNotFoundError) as exc_info:
            SimpleDetector(model_path=str(fake_path))
        assert "nonexistent.pkl" in str(exc_info.value)

    def test_detect_returns_required_keys(
        self, detector: SimpleDetector, sample_transaction: dict[str, Any]
    ) -> None:
        """Test detect returns all required keys."""
        result = detector.detect(sample_transaction)
        assert "is_suspicious" in result
        assert "risk_score" in result
        assert "risk_level" in result

    def test_detect_types(
        self, detector: SimpleDetector, sample_transaction: dict[str, Any]
    ) -> None:
        """Test detect returns correct types."""
        result = detector.detect(sample_transaction)
        assert result["is_suspicious"] in (True, False)
        assert isinstance(result["risk_score"], (int, float))
        assert isinstance(result["risk_level"], str)

    def test_detect_risk_score_bounds(
        self, detector: SimpleDetector, sample_transaction: dict[str, Any]
    ) -> None:
        """Test risk score is within valid bounds."""
        result = detector.detect(sample_transaction)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_detect_risk_level_values(
        self, detector: SimpleDetector, sample_transaction: dict[str, Any]
    ) -> None:
        """Test risk level is a valid category."""
        result = detector.detect(sample_transaction)
        assert result["risk_level"] in ["Low", "Medium", "High"]

    def test_detect_suspicious_transaction(
        self, detector: SimpleDetector, suspicious_transaction: dict[str, Any]
    ) -> None:
        """Test detection of suspicious transaction."""
        result = detector.detect(suspicious_transaction)
        assert result["is_suspicious"] == True  # noqa: E712
        assert result["risk_score"] > 0.5

    def test_detect_normal_transaction(
        self, detector: SimpleDetector, normal_transaction: dict[str, Any]
    ) -> None:
        """Test detection of normal transaction."""
        result = detector.detect(normal_transaction)
        assert result["is_suspicious"] == False  # noqa: E712
        assert result["risk_score"] < 0.5

    def test_detect_invalid_negative_amount(self, detector: SimpleDetector) -> None:
        """Test detection fails for negative amount."""
        invalid = {
            "amount": -100,
            "hour": 14,
            "user_age_days": 30,
            "transaction_count": 5,
        }
        with pytest.raises(InvalidTransactionError):
            detector.detect(invalid)

    def test_detect_invalid_hour(self, detector: SimpleDetector) -> None:
        """Test detection fails for invalid hour."""
        invalid = {
            "amount": 100,
            "hour": 25,
            "user_age_days": 30,
            "transaction_count": 5,
        }
        with pytest.raises(InvalidTransactionError):
            detector.detect(invalid)

    def test_detect_invalid_user_age(self, detector: SimpleDetector) -> None:
        """Test detection fails for invalid user age."""
        invalid = {
            "amount": 100,
            "hour": 14,
            "user_age_days": 0,
            "transaction_count": 5,
        }
        with pytest.raises(InvalidTransactionError):
            detector.detect(invalid)

    def test_detect_missing_fields(self, detector: SimpleDetector) -> None:
        """Test detection fails for missing fields."""
        incomplete = {"amount": 100}
        with pytest.raises(InvalidTransactionError):
            detector.detect(incomplete)

    def test_batch_detect_empty_list(self, detector: SimpleDetector) -> None:
        """Test batch detect with empty list."""
        results = detector.batch_detect([])
        assert results == []

    def test_batch_detect_multiple(
        self, detector: SimpleDetector, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test batch detection with multiple transactions."""
        results = detector.batch_detect(batch_transactions)
        assert len(results) == len(batch_transactions)
        for result in results:
            assert "is_suspicious" in result
            assert "risk_score" in result
            assert "risk_level" in result

    def test_batch_detect_mixed_results(
        self, detector: SimpleDetector, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test batch detection returns mixed results."""
        results = detector.batch_detect(batch_transactions)
        suspicious_count = sum(1 for r in results if r["is_suspicious"] == True)  # noqa: E712
        normal_count = sum(1 for r in results if r["is_suspicious"] == False)  # noqa: E712
        # Should have at least one of each in our test data
        assert suspicious_count >= 1
        assert normal_count >= 1

    def test_get_model_info(self, detector: SimpleDetector) -> None:
        """Test get_model_info returns expected keys."""
        info = detector.get_model_info()
        assert "model_path" in info
        assert "model_type" in info
        assert "risk_threshold" in info

    def test_model_property(self, detector: SimpleDetector) -> None:
        """Test model property returns the model."""
        model = detector.model
        assert model is not None
        assert hasattr(model, "predict_proba")

    def test_model_path_property(self, detector: SimpleDetector) -> None:
        """Test model_path property returns Path."""
        path = detector.model_path
        assert isinstance(path, Path)
        assert path.exists()


class TestCheckTransaction:
    """Tests for check_transaction helper function."""

    def test_check_transaction_basic(self, sample_transaction: dict[str, Any]) -> None:
        """Test check_transaction returns valid result."""
        result = check_transaction(sample_transaction)
        assert "is_suspicious" in result
        assert "risk_score" in result
        assert "risk_level" in result

    def test_check_transaction_suspicious(
        self, suspicious_transaction: dict[str, Any]
    ) -> None:
        """Test check_transaction detects suspicious."""
        result = check_transaction(suspicious_transaction)
        assert result["is_suspicious"] == True  # noqa: E712

    def test_check_transaction_normal(self, normal_transaction: dict[str, Any]) -> None:
        """Test check_transaction detects normal."""
        result = check_transaction(normal_transaction)
        assert result["is_suspicious"] == False  # noqa: E712

    def test_check_transaction_with_custom_model(
        self, model_path: Path, sample_transaction: dict[str, Any]
    ) -> None:
        """Test check_transaction with custom model path."""
        result = check_transaction(sample_transaction, model_path=str(model_path))
        assert "is_suspicious" in result

    def test_check_transaction_invalid_raises(self) -> None:
        """Test check_transaction raises for invalid input."""
        with pytest.raises(InvalidTransactionError):
            check_transaction({"amount": -100})


class TestEdgeCases:
    """Edge case tests for detector."""

    def test_minimum_valid_transaction(self, detector: SimpleDetector) -> None:
        """Test with minimum valid values."""
        tx = {
            "amount": 0.01,
            "hour": 0,
            "user_age_days": 1,
            "transaction_count": 0,
        }
        result = detector.detect(tx)
        assert "is_suspicious" in result

    def test_maximum_hour(self, detector: SimpleDetector) -> None:
        """Test with maximum hour value."""
        tx = {
            "amount": 100,
            "hour": 23,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = detector.detect(tx)
        assert "is_suspicious" in result

    def test_large_amount(self, detector: SimpleDetector) -> None:
        """Test with large transaction amount."""
        tx = {
            "amount": 9999999,
            "hour": 12,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = detector.detect(tx)
        assert "is_suspicious" in result

    def test_old_account(self, detector: SimpleDetector) -> None:
        """Test with very old account."""
        tx = {
            "amount": 100,
            "hour": 12,
            "user_age_days": 36500,
            "transaction_count": 5,
        }
        result = detector.detect(tx)
        assert "is_suspicious" in result

    def test_float_values(self, detector: SimpleDetector) -> None:
        """Test with float values for integer fields."""
        tx = {
            "amount": 100.5,
            "hour": 12,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = detector.detect(tx)
        assert "is_suspicious" in result
