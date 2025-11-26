"""
MetaGuard Custom Exceptions

Defines custom exception classes for better error handling and debugging.

Author: Moslem Mohseni
"""

from __future__ import annotations

from typing import Any


class MetaGuardError(Exception):
    """
    Base exception class for all MetaGuard errors.

    All custom exceptions in MetaGuard inherit from this class,
    making it easy to catch all MetaGuard-specific errors.

    Author: Moslem Mohseni

    Example:
        >>> try:
        ...     result = detector.detect(invalid_transaction)
        ... except MetaGuardError as e:
        ...     print(f"MetaGuard error: {e}")
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """
        Initialize MetaGuardError.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ModelNotFoundError(MetaGuardError):
    """
    Raised when the model file cannot be found.

    This error occurs when attempting to load a model from a path
    that doesn't exist or is inaccessible.

    Author: Moslem Mohseni

    Example:
        >>> detector = SimpleDetector(model_path="nonexistent.pkl")
        ModelNotFoundError: Model file not found at 'nonexistent.pkl'
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize ModelNotFoundError.

        Args:
            model_path: Path where the model was expected to be found
        """
        message = (
            f"Model file not found at '{model_path}'. "
            "Please train the model first by running: python scripts/train.py"
        )
        super().__init__(message, details={"model_path": model_path})
        self.model_path = model_path


class ModelLoadError(MetaGuardError):
    """
    Raised when the model fails to load.

    This error occurs when the model file exists but cannot be
    loaded due to corruption, version mismatch, or other issues.

    Author: Moslem Mohseni

    Example:
        >>> detector = SimpleDetector(model_path="corrupted.pkl")
        ModelLoadError: Failed to load model from 'corrupted.pkl'
    """

    def __init__(self, model_path: str, original_error: Exception | None = None) -> None:
        """
        Initialize ModelLoadError.

        Args:
            model_path: Path to the model file that failed to load
            original_error: The original exception that caused the load failure
        """
        message = f"Failed to load model from '{model_path}'"
        if original_error:
            message += f": {original_error}"

        details = {"model_path": model_path}
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__

        super().__init__(message, details=details)
        self.model_path = model_path
        self.original_error = original_error


class InvalidTransactionError(MetaGuardError):
    """
    Raised when transaction data is invalid.

    This error occurs when the transaction dictionary is missing
    required fields or contains invalid values.

    Author: Moslem Mohseni

    Example:
        >>> detector.detect({"amount": -100})
        InvalidTransactionError: Invalid transaction: amount must be positive
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
    ) -> None:
        """
        Initialize InvalidTransactionError.

        Args:
            message: Description of what's wrong with the transaction
            field: The field that contains the invalid value
            value: The invalid value that was provided
            expected: Description of what was expected
        """
        details: dict[str, Any] = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if expected:
            details["expected"] = expected

        super().__init__(f"Invalid transaction: {message}", details=details)
        self.field = field
        self.value = value
        self.expected = expected


class ValidationError(MetaGuardError):
    """
    Raised when input validation fails.

    This is a general validation error for any input that doesn't
    meet the expected criteria.

    Author: Moslem Mohseni

    Example:
        >>> config = MetaGuardConfig(risk_threshold=1.5)
        ValidationError: risk_threshold must be between 0 and 1
    """

    def __init__(
        self,
        message: str,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize ValidationError.

        Args:
            message: Description of the validation failure
            errors: List of individual validation errors
        """
        details = {"errors": errors} if errors else {}
        super().__init__(f"Validation failed: {message}", details=details)
        self.errors = errors or []


class ConfigurationError(MetaGuardError):
    """
    Raised when configuration is invalid.

    Author: Moslem Mohseni
    """

    def __init__(self, message: str, config_key: str | None = None) -> None:
        """
        Initialize ConfigurationError.

        Args:
            message: Description of the configuration error
            config_key: The configuration key that has an issue
        """
        details = {"config_key": config_key} if config_key else {}
        super().__init__(f"Configuration error: {message}", details=details)
        self.config_key = config_key
