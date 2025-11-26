"""
Unit tests for MetaGuard Exceptions Module

Author: Moslem Mohseni
"""

from __future__ import annotations

import pytest

from metaguard.utils.exceptions import (
    ConfigurationError,
    InvalidTransactionError,
    MetaGuardError,
    ModelLoadError,
    ModelNotFoundError,
    ValidationError,
)


class TestMetaGuardError:
    """Tests for base MetaGuardError class."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        error = MetaGuardError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"

    def test_with_details(self) -> None:
        """Test exception with details."""
        error = MetaGuardError("Test error", details={"key": "value"})
        assert error.details == {"key": "value"}
        assert "key" in str(error)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        error = MetaGuardError("Test error", details={"key": "value"})
        data = error.to_dict()
        assert data["error_type"] == "MetaGuardError"
        assert data["message"] == "Test error"
        assert data["details"]["key"] == "value"

    def test_is_exception(self) -> None:
        """Test it's a proper exception."""
        error = MetaGuardError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised(self) -> None:
        """Test exception can be raised and caught."""
        with pytest.raises(MetaGuardError) as exc_info:
            raise MetaGuardError("Test error")
        assert "Test error" in str(exc_info.value)


class TestModelNotFoundError:
    """Tests for ModelNotFoundError class."""

    def test_creation(self) -> None:
        """Test exception creation."""
        error = ModelNotFoundError("/path/to/model.pkl")
        assert error.model_path == "/path/to/model.pkl"
        assert "model.pkl" in str(error)

    def test_message_includes_path(self) -> None:
        """Test message includes model path."""
        error = ModelNotFoundError("/path/to/model.pkl")
        assert "/path/to/model.pkl" in str(error)

    def test_inherits_from_base(self) -> None:
        """Test inherits from MetaGuardError."""
        error = ModelNotFoundError("/path")
        assert isinstance(error, MetaGuardError)

    def test_details_include_path(self) -> None:
        """Test details include model path."""
        error = ModelNotFoundError("/path/to/model.pkl")
        assert error.details["model_path"] == "/path/to/model.pkl"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        error = ModelNotFoundError("/path/to/model.pkl")
        data = error.to_dict()
        assert data["error_type"] == "ModelNotFoundError"


class TestModelLoadError:
    """Tests for ModelLoadError class."""

    def test_creation(self) -> None:
        """Test exception creation."""
        error = ModelLoadError("/path/to/model.pkl")
        assert error.model_path == "/path/to/model.pkl"

    def test_with_original_error(self) -> None:
        """Test exception with original error."""
        original = ValueError("Corrupt file")
        error = ModelLoadError("/path/to/model.pkl", original_error=original)
        assert error.original_error is original
        assert "Corrupt file" in str(error)

    def test_details_include_original_error(self) -> None:
        """Test details include original error info."""
        original = ValueError("Corrupt file")
        error = ModelLoadError("/path/to/model.pkl", original_error=original)
        assert "original_error" in error.details
        assert error.details["original_error_type"] == "ValueError"

    def test_inherits_from_base(self) -> None:
        """Test inherits from MetaGuardError."""
        error = ModelLoadError("/path")
        assert isinstance(error, MetaGuardError)


class TestInvalidTransactionError:
    """Tests for InvalidTransactionError class."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        error = InvalidTransactionError("Amount must be positive")
        assert "Amount must be positive" in str(error)

    def test_with_field_info(self) -> None:
        """Test exception with field information."""
        error = InvalidTransactionError(
            "Must be positive",
            field="amount",
            value=-100,
            expected="positive number",
        )
        assert error.field == "amount"
        assert error.value == -100
        assert error.expected == "positive number"

    def test_details_include_field_info(self) -> None:
        """Test details include field information."""
        error = InvalidTransactionError(
            "Invalid",
            field="amount",
            value=-100,
        )
        assert error.details["field"] == "amount"
        assert error.details["value"] == -100

    def test_inherits_from_base(self) -> None:
        """Test inherits from MetaGuardError."""
        error = InvalidTransactionError("Invalid")
        assert isinstance(error, MetaGuardError)


class TestValidationError:
    """Tests for ValidationError class."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        error = ValidationError("Validation failed")
        assert "Validation failed" in str(error)

    def test_with_errors_list(self) -> None:
        """Test exception with errors list."""
        errors = [
            {"field": "amount", "message": "Must be positive"},
            {"field": "hour", "message": "Must be 0-23"},
        ]
        error = ValidationError("Multiple errors", errors=errors)
        assert error.errors == errors
        assert len(error.errors) == 2

    def test_inherits_from_base(self) -> None:
        """Test inherits from MetaGuardError."""
        error = ValidationError("Invalid")
        assert isinstance(error, MetaGuardError)


class TestConfigurationError:
    """Tests for ConfigurationError class."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        error = ConfigurationError("Invalid config")
        assert "Invalid config" in str(error)

    def test_with_config_key(self) -> None:
        """Test exception with config key."""
        error = ConfigurationError("Invalid value", config_key="risk_threshold")
        assert error.config_key == "risk_threshold"
        assert error.details["config_key"] == "risk_threshold"

    def test_inherits_from_base(self) -> None:
        """Test inherits from MetaGuardError."""
        error = ConfigurationError("Invalid")
        assert isinstance(error, MetaGuardError)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_inherit_from_base(self) -> None:
        """Test all custom exceptions inherit from MetaGuardError."""
        exceptions = [
            ModelNotFoundError("/path"),
            ModelLoadError("/path"),
            InvalidTransactionError("msg"),
            ValidationError("msg"),
            ConfigurationError("msg"),
        ]
        for exc in exceptions:
            assert isinstance(exc, MetaGuardError)

    def test_can_catch_all_with_base(self) -> None:
        """Test all exceptions can be caught with base class."""
        exceptions_to_test = [
            lambda: (_ for _ in ()).throw(ModelNotFoundError("/path")),
            lambda: (_ for _ in ()).throw(ModelLoadError("/path")),
            lambda: (_ for _ in ()).throw(InvalidTransactionError("msg")),
            lambda: (_ for _ in ()).throw(ValidationError("msg")),
            lambda: (_ for _ in ()).throw(ConfigurationError("msg")),
        ]
        for exc_func in exceptions_to_test:
            with pytest.raises(MetaGuardError):
                next(exc_func())
