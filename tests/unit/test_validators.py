"""
Unit tests for MetaGuard Validators Module

Author: Moslem Mohseni
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from metaguard.utils.validators import (
    BatchTransactionInput,
    DetectionResult,
    RiskAnalysisResult,
    TransactionInput,
    validate_transaction,
    validate_transactions,
)


class TestTransactionInput:
    """Tests for TransactionInput model."""

    def test_valid_transaction(self) -> None:
        """Test valid transaction creation."""
        tx = TransactionInput(
            amount=100.0,
            hour=14,
            user_age_days=30,
            transaction_count=5,
        )
        assert tx.amount == 100.0
        assert tx.hour == 14
        assert tx.user_age_days == 30
        assert tx.transaction_count == 5

    def test_amount_must_be_positive(self) -> None:
        """Test amount validation - must be positive."""
        with pytest.raises(ValidationError):
            TransactionInput(
                amount=-100,
                hour=14,
                user_age_days=30,
                transaction_count=5,
            )

    def test_amount_must_not_be_zero(self) -> None:
        """Test amount validation - must not be zero."""
        with pytest.raises(ValidationError):
            TransactionInput(
                amount=0,
                hour=14,
                user_age_days=30,
                transaction_count=5,
            )

    def test_amount_maximum(self) -> None:
        """Test amount validation - maximum limit."""
        with pytest.raises(ValidationError):
            TransactionInput(
                amount=10_000_001,
                hour=14,
                user_age_days=30,
                transaction_count=5,
            )

    def test_hour_minimum(self) -> None:
        """Test hour validation - minimum is 0."""
        tx = TransactionInput(
            amount=100,
            hour=0,
            user_age_days=30,
            transaction_count=5,
        )
        assert tx.hour == 0

    def test_hour_maximum(self) -> None:
        """Test hour validation - maximum is 23."""
        tx = TransactionInput(
            amount=100,
            hour=23,
            user_age_days=30,
            transaction_count=5,
        )
        assert tx.hour == 23

    def test_hour_out_of_range(self) -> None:
        """Test hour validation - out of range."""
        with pytest.raises(ValidationError):
            TransactionInput(
                amount=100,
                hour=24,
                user_age_days=30,
                transaction_count=5,
            )

        with pytest.raises(ValidationError):
            TransactionInput(
                amount=100,
                hour=-1,
                user_age_days=30,
                transaction_count=5,
            )

    def test_user_age_minimum(self) -> None:
        """Test user_age_days validation - minimum is 1."""
        tx = TransactionInput(
            amount=100,
            hour=14,
            user_age_days=1,
            transaction_count=5,
        )
        assert tx.user_age_days == 1

    def test_user_age_zero_invalid(self) -> None:
        """Test user_age_days validation - zero is invalid."""
        with pytest.raises(ValidationError):
            TransactionInput(
                amount=100,
                hour=14,
                user_age_days=0,
                transaction_count=5,
            )

    def test_transaction_count_minimum(self) -> None:
        """Test transaction_count validation - minimum is 0."""
        tx = TransactionInput(
            amount=100,
            hour=14,
            user_age_days=30,
            transaction_count=0,
        )
        assert tx.transaction_count == 0

    def test_transaction_count_negative_invalid(self) -> None:
        """Test transaction_count validation - negative is invalid."""
        with pytest.raises(ValidationError):
            TransactionInput(
                amount=100,
                hour=14,
                user_age_days=30,
                transaction_count=-1,
            )

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "amount": 100.0,
            "hour": 14,
            "user_age_days": 30,
            "transaction_count": 5,
        }
        tx = TransactionInput.from_dict(data)
        assert tx.amount == 100.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        tx = TransactionInput(
            amount=100.0,
            hour=14,
            user_age_days=30,
            transaction_count=5,
        )
        data = tx.to_dict()
        assert data["amount"] == 100.0
        assert data["hour"] == 14

    def test_to_features(self) -> None:
        """Test conversion to feature list."""
        tx = TransactionInput(
            amount=100.0,
            hour=14,
            user_age_days=30,
            transaction_count=5,
        )
        features = tx.to_features()
        assert features == [100.0, 14.0, 30.0, 5.0]

    def test_extra_fields_forbidden(self) -> None:
        """Test extra fields are not allowed."""
        with pytest.raises(ValidationError):
            TransactionInput(
                amount=100.0,
                hour=14,
                user_age_days=30,
                transaction_count=5,
                extra_field="value",
            )

    def test_amount_rounded(self) -> None:
        """Test amount is rounded to 2 decimal places."""
        tx = TransactionInput(
            amount=100.123456,
            hour=14,
            user_age_days=30,
            transaction_count=5,
        )
        assert tx.amount == 100.12


class TestDetectionResult:
    """Tests for DetectionResult model."""

    def test_valid_result(self) -> None:
        """Test valid detection result."""
        result = DetectionResult(
            is_suspicious=True,
            risk_score=0.85,
            risk_level="High",
        )
        assert result.is_suspicious is True
        assert result.risk_score == 0.85
        assert result.risk_level == "High"

    def test_risk_score_bounds(self) -> None:
        """Test risk_score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            DetectionResult(
                is_suspicious=True,
                risk_score=1.5,
                risk_level="High",
            )

        with pytest.raises(ValidationError):
            DetectionResult(
                is_suspicious=True,
                risk_score=-0.1,
                risk_level="High",
            )

    def test_risk_level_pattern(self) -> None:
        """Test risk_level must match pattern."""
        with pytest.raises(ValidationError):
            DetectionResult(
                is_suspicious=True,
                risk_score=0.5,
                risk_level="Invalid",
            )

    def test_valid_risk_levels(self) -> None:
        """Test all valid risk levels."""
        for level in ["Low", "Medium", "High"]:
            result = DetectionResult(
                is_suspicious=False,
                risk_score=0.5,
                risk_level=level,
            )
            assert result.risk_level == level

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = DetectionResult(
            is_suspicious=True,
            risk_score=0.85,
            risk_level="High",
        )
        data = result.to_dict()
        assert data["is_suspicious"] is True
        assert data["risk_score"] == 0.85


class TestBatchTransactionInput:
    """Tests for BatchTransactionInput model."""

    def test_valid_batch(self) -> None:
        """Test valid batch creation."""
        transactions = [
            TransactionInput(
                amount=100, hour=14, user_age_days=30, transaction_count=5
            ),
            TransactionInput(
                amount=200, hour=10, user_age_days=60, transaction_count=10
            ),
        ]
        batch = BatchTransactionInput(transactions=transactions)
        assert len(batch.transactions) == 2

    def test_empty_batch_invalid(self) -> None:
        """Test empty batch is invalid."""
        with pytest.raises(ValidationError):
            BatchTransactionInput(transactions=[])

    def test_from_list(self) -> None:
        """Test creation from list of dictionaries."""
        data = [
            {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
            {"amount": 200, "hour": 10, "user_age_days": 60, "transaction_count": 10},
        ]
        batch = BatchTransactionInput.from_list(data)
        assert len(batch.transactions) == 2


class TestRiskAnalysisResult:
    """Tests for RiskAnalysisResult model."""

    def test_valid_result(self) -> None:
        """Test valid risk analysis result."""
        result = RiskAnalysisResult(
            risk_score=75.5,
            risk_level="High",
            factors={"high_amount": True, "new_account": False},
        )
        assert result.risk_score == 75.5
        assert result.risk_level == "High"
        assert result.factors["high_amount"] is True

    def test_risk_score_bounds(self) -> None:
        """Test risk_score must be between 0 and 100."""
        with pytest.raises(ValidationError):
            RiskAnalysisResult(
                risk_score=101,
                risk_level="High",
            )


class TestValidateTransaction:
    """Tests for validate_transaction function."""

    def test_valid_transaction(self, sample_transaction: dict[str, Any]) -> None:
        """Test validation of valid transaction."""
        result = validate_transaction(sample_transaction)
        assert isinstance(result, TransactionInput)

    def test_invalid_transaction_raises(self) -> None:
        """Test invalid transaction raises error."""
        from metaguard.utils.exceptions import InvalidTransactionError

        with pytest.raises(InvalidTransactionError):
            validate_transaction({"amount": -100})


class TestValidateTransactions:
    """Tests for validate_transactions function."""

    def test_valid_transactions(
        self, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test validation of valid transactions."""
        results = validate_transactions(batch_transactions)
        assert len(results) == len(batch_transactions)
        for result in results:
            assert isinstance(result, TransactionInput)

    def test_empty_list(self) -> None:
        """Test validation of empty list."""
        results = validate_transactions([])
        assert results == []
