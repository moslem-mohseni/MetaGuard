"""
MetaGuard Test Fixtures

Shared fixtures for all tests.

Author: Moslem Mohseni
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pytest


@pytest.fixture
def sample_transaction() -> dict[str, Any]:
    """A valid sample transaction."""
    return {
        "amount": 100.0,
        "hour": 14,
        "user_age_days": 30,
        "transaction_count": 5,
    }


@pytest.fixture
def suspicious_transaction() -> dict[str, Any]:
    """A suspicious transaction for testing."""
    return {
        "amount": 5000.0,
        "hour": 3,
        "user_age_days": 2,
        "transaction_count": 50,
    }


@pytest.fixture
def normal_transaction() -> dict[str, Any]:
    """A clearly normal transaction."""
    return {
        "amount": 50.0,
        "hour": 10,
        "user_age_days": 200,
        "transaction_count": 3,
    }


@pytest.fixture
def batch_transactions() -> list[dict[str, Any]]:
    """A batch of transactions for testing."""
    return [
        {"amount": 100, "hour": 10, "user_age_days": 100, "transaction_count": 5},
        {"amount": 5000, "hour": 2, "user_age_days": 3, "transaction_count": 50},
        {"amount": 25, "hour": 15, "user_age_days": 365, "transaction_count": 1},
    ]


@pytest.fixture
def invalid_transactions() -> list[dict[str, Any]]:
    """Invalid transactions for testing validation."""
    return [
        {"amount": -100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
        {"amount": 100, "hour": 25, "user_age_days": 30, "transaction_count": 5},
        {"amount": 100, "hour": 14, "user_age_days": 0, "transaction_count": 5},
        {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": -1},
    ]


@pytest.fixture
def missing_field_transactions() -> list[dict[str, Any]]:
    """Transactions with missing fields."""
    return [
        {"amount": 100},
        {"hour": 14, "user_age_days": 30},
        {"amount": 100, "hour": 14},
        {},
    ]


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "amount": [100, 200, 5000, 50],
            "hour": [10, 14, 3, 12],
            "user_age_days": [100, 50, 5, 200],
            "transaction_count": [5, 10, 50, 2],
            "is_fraud": [0, 0, 1, 0],
        }
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_model_dir(temp_dir: Path) -> Path:
    """Temporary directory for model files."""
    model_dir = temp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Temporary directory for data files."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def edge_case_transactions() -> list[dict[str, Any]]:
    """Edge case transactions for boundary testing."""
    return [
        # Minimum valid values
        {"amount": 0.01, "hour": 0, "user_age_days": 1, "transaction_count": 0},
        # Maximum hour
        {"amount": 100, "hour": 23, "user_age_days": 100, "transaction_count": 5},
        # Very high amount
        {"amount": 9999999, "hour": 12, "user_age_days": 100, "transaction_count": 5},
        # Very old account
        {"amount": 100, "hour": 12, "user_age_days": 36500, "transaction_count": 5},
        # Many transactions
        {"amount": 100, "hour": 12, "user_age_days": 100, "transaction_count": 999999},
    ]


@pytest.fixture
def risk_score_test_cases() -> list[tuple[float, str]]:
    """Test cases for risk level determination."""
    return [
        (0, "Low"),
        (20, "Low"),
        (39.9, "Low"),
        (40, "Medium"),
        (40.1, "Medium"),
        (55, "Medium"),
        (69.9, "Medium"),
        (70, "High"),
        (70.1, "High"),
        (85, "High"),
        (100, "High"),
    ]


@pytest.fixture(scope="session")
def model_path() -> Path:
    """Path to the trained model."""
    return Path(__file__).parent.parent / "src" / "metaguard" / "models" / "model.pkl"


@pytest.fixture
def detector(model_path: Path):
    """SimpleDetector instance for testing."""
    from metaguard import SimpleDetector

    return SimpleDetector(model_path=str(model_path))
