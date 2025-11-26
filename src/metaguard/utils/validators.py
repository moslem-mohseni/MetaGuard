"""
MetaGuard Input Validators

Provides input validation using Pydantic models.

Author: Moslem Mohseni
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """
    Validated transaction input model.

    This model ensures all transaction data is valid before processing.

    Author: Moslem Mohseni

    Attributes:
        amount: Transaction amount in currency units (must be positive)
        hour: Hour of the transaction (0-23)
        user_age_days: Age of the user account in days (must be at least 1)
        transaction_count: Number of prior transactions (non-negative)

    Example:
        >>> tx = TransactionInput(
        ...     amount=1000.0,
        ...     hour=14,
        ...     user_age_days=30,
        ...     transaction_count=5
        ... )
        >>> tx.amount
        1000.0
    """

    amount: float = Field(
        ...,
        gt=0,
        le=10_000_000,
        description="Transaction amount (must be positive, max 10M)",
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the transaction (0-23)",
    )
    user_age_days: int = Field(
        ...,
        ge=1,
        le=36500,  # ~100 years max
        description="Age of the user account in days (1-36500)",
    )
    transaction_count: int = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Number of prior transactions (non-negative)",
    )

    model_config = {"extra": "forbid", "validate_default": True}

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        """Validate amount is reasonable."""
        if v <= 0:
            raise ValueError("amount must be positive")
        if v > 10_000_000:
            raise ValueError("amount exceeds maximum allowed (10,000,000)")
        return round(v, 2)  # Round to 2 decimal places

    @field_validator("hour")
    @classmethod
    def validate_hour(cls, v: int) -> int:
        """Validate hour is within valid range."""
        if not 0 <= v <= 23:
            raise ValueError("hour must be between 0 and 23")
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransactionInput:
        """
        Create a TransactionInput from a dictionary.

        Author: Moslem Mohseni

        Args:
            data: Dictionary containing transaction data

        Returns:
            Validated TransactionInput instance

        Raises:
            ValidationError: If data is invalid

        Example:
            >>> tx = TransactionInput.from_dict({
            ...     "amount": 100,
            ...     "hour": 10,
            ...     "user_age_days": 30,
            ...     "transaction_count": 5
            ... })
        """
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Author: Moslem Mohseni

        Returns:
            Dictionary representation
        """
        return self.model_dump()

    def to_features(self) -> list[float]:
        """
        Convert to feature list for model input.

        Author: Moslem Mohseni

        Returns:
            List of features in correct order for the model
        """
        return [
            float(self.amount),
            float(self.hour),
            float(self.user_age_days),
            float(self.transaction_count),
        ]


class DetectionResult(BaseModel):
    """
    Detection result model.

    Represents the result of a fraud detection operation.

    Author: Moslem Mohseni

    Attributes:
        is_suspicious: Whether the transaction is flagged as suspicious
        risk_score: Probability of fraud (0-1)
        risk_level: Categorical risk level (Low, Medium, High)
    """

    is_suspicious: bool = Field(
        ...,
        description="Whether the transaction is flagged as suspicious",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of fraud (0-1)",
    )
    risk_level: str = Field(
        ...,
        pattern="^(Low|Medium|High)$",
        description="Categorical risk level",
    )

    model_config = {"extra": "forbid"}

    @field_validator("risk_score")
    @classmethod
    def round_risk_score(cls, v: float) -> float:
        """Round risk score to 4 decimal places."""
        return round(v, 4)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Author: Moslem Mohseni

        Returns:
            Dictionary representation
        """
        return self.model_dump()


class BatchTransactionInput(BaseModel):
    """
    Batch of transactions for processing.

    Author: Moslem Mohseni

    Attributes:
        transactions: List of transactions to process
    """

    transactions: list[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of transactions to process (1-10000)",
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> BatchTransactionInput:
        """
        Create from a list of dictionaries.

        Author: Moslem Mohseni

        Args:
            data: List of transaction dictionaries

        Returns:
            BatchTransactionInput instance
        """
        transactions = [TransactionInput.from_dict(tx) for tx in data]
        return cls(transactions=transactions)


class RiskAnalysisResult(BaseModel):
    """
    Detailed risk analysis result.

    Author: Moslem Mohseni

    Attributes:
        risk_score: Numeric risk score (0-100)
        risk_level: Categorical risk level
        factors: Dictionary of risk factors
    """

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Risk score (0-100)",
    )
    risk_level: str = Field(
        ...,
        pattern="^(Low|Medium|High)$",
        description="Categorical risk level",
    )
    factors: dict[str, bool] = Field(
        default_factory=dict,
        description="Dictionary of risk factors",
    )

    model_config = {"extra": "forbid"}


def validate_transaction(transaction: dict[str, Any]) -> TransactionInput:
    """
    Validate a transaction dictionary.

    Author: Moslem Mohseni

    Args:
        transaction: Dictionary containing transaction data

    Returns:
        Validated TransactionInput

    Raises:
        InvalidTransactionError: If validation fails
    """
    from metaguard.utils.exceptions import InvalidTransactionError

    try:
        return TransactionInput.from_dict(transaction)
    except Exception as e:
        # Extract field and value info from Pydantic errors if available
        error_msg = str(e)
        raise InvalidTransactionError(
            message=error_msg,
            field=None,
            value=transaction,
        ) from e


def validate_transactions(transactions: list[dict[str, Any]]) -> list[TransactionInput]:
    """
    Validate a list of transactions.

    Author: Moslem Mohseni

    Args:
        transactions: List of transaction dictionaries

    Returns:
        List of validated TransactionInput objects

    Raises:
        InvalidTransactionError: If any transaction is invalid
    """
    return [validate_transaction(tx) for tx in transactions]
