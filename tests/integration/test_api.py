"""
API Integration Tests

Author: Moslem Mohseni

Tests for FastAPI REST API endpoints.
"""

from __future__ import annotations

from typing import Any

import pytest

# Skip all tests if httpx is not installed
httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from metaguard.api.rest import app, get_detector


@pytest.fixture
def client() -> TestClient:
    """Create test client for API."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_health_check(self, client: TestClient) -> None:
        """Test root endpoint returns health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["model_loaded"] is True

    def test_health_endpoint(self, client: TestClient) -> None:
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDetectionEndpoint:
    """Tests for /detect endpoint."""

    def test_detect_normal_transaction(
        self, client: TestClient, normal_transaction: dict[str, Any]
    ) -> None:
        """Test detection of normal transaction."""
        response = client.post("/detect", json=normal_transaction)
        assert response.status_code == 200
        data = response.json()
        assert "is_suspicious" in data
        assert "risk_score" in data
        assert "risk_level" in data
        assert "processing_time_ms" in data
        assert data["is_suspicious"] is False
        assert data["risk_level"] == "Low"

    def test_detect_suspicious_transaction(
        self, client: TestClient, suspicious_transaction: dict[str, Any]
    ) -> None:
        """Test detection of suspicious transaction."""
        response = client.post("/detect", json=suspicious_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["is_suspicious"] is True
        assert data["risk_level"] == "High"

    def test_detect_invalid_amount(self, client: TestClient) -> None:
        """Test detection with invalid amount."""
        transaction = {
            "amount": -100,
            "hour": 14,
            "user_age_days": 30,
            "transaction_count": 5,
        }
        response = client.post("/detect", json=transaction)
        assert response.status_code == 422  # Validation error

    def test_detect_invalid_hour(self, client: TestClient) -> None:
        """Test detection with invalid hour."""
        transaction = {
            "amount": 100,
            "hour": 25,
            "user_age_days": 30,
            "transaction_count": 5,
        }
        response = client.post("/detect", json=transaction)
        assert response.status_code == 422

    def test_detect_missing_field(self, client: TestClient) -> None:
        """Test detection with missing field."""
        transaction = {
            "amount": 100,
            "hour": 14,
        }
        response = client.post("/detect", json=transaction)
        assert response.status_code == 422

    def test_detect_processing_time(
        self, client: TestClient, sample_transaction: dict[str, Any]
    ) -> None:
        """Test that processing time is reported."""
        response = client.post("/detect", json=sample_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["processing_time_ms"] >= 0


class TestBatchDetectionEndpoint:
    """Tests for /detect/batch endpoint."""

    def test_batch_detect(
        self, client: TestClient, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test batch detection."""
        response = client.post("/detect/batch", json={"transactions": batch_transactions})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == len(batch_transactions)
        assert "total_transactions" in data
        assert data["total_transactions"] == len(batch_transactions)
        assert "suspicious_count" in data
        assert "processing_time_ms" in data

    def test_batch_detect_single(
        self, client: TestClient, sample_transaction: dict[str, Any]
    ) -> None:
        """Test batch detection with single transaction."""
        response = client.post("/detect/batch", json={"transactions": [sample_transaction]})
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1

    def test_batch_detect_empty(self, client: TestClient) -> None:
        """Test batch detection with empty list."""
        response = client.post("/detect/batch", json={"transactions": []})
        assert response.status_code == 422  # Validation error for min_length=1

    def test_batch_detect_mixed(self, client: TestClient) -> None:
        """Test batch with mix of normal and suspicious."""
        transactions = [
            {"amount": 50, "hour": 10, "user_age_days": 200, "transaction_count": 3},
            {"amount": 5000, "hour": 3, "user_age_days": 2, "transaction_count": 50},
        ]
        response = client.post("/detect/batch", json={"transactions": transactions})
        assert response.status_code == 200
        data = response.json()
        assert data["suspicious_count"] >= 1


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""

    def test_analyze_transaction(
        self, client: TestClient, sample_transaction: dict[str, Any]
    ) -> None:
        """Test risk analysis endpoint."""
        response = client.post("/analyze", json=sample_transaction)
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_level" in data
        assert "factors" in data
        assert "active_factor_count" in data
        assert "processing_time_ms" in data

    def test_analyze_factors(
        self, client: TestClient, suspicious_transaction: dict[str, Any]
    ) -> None:
        """Test risk factors are returned."""
        response = client.post("/analyze", json=suspicious_transaction)
        assert response.status_code == 200
        data = response.json()
        factors = data["factors"]
        assert "high_amount" in factors
        assert "new_account" in factors
        assert "high_frequency" in factors
        assert "unusual_hour" in factors
        # Suspicious transaction should have active factors
        assert data["active_factor_count"] > 0

    def test_analyze_normal_transaction(
        self, client: TestClient, normal_transaction: dict[str, Any]
    ) -> None:
        """Test analysis of normal transaction."""
        response = client.post("/analyze", json=normal_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] == "Low"
        assert data["active_factor_count"] == 0


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_get_model_info(self, client: TestClient) -> None:
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "risk_threshold" in data
        assert "feature_names" in data
        assert len(data["feature_names"]) == 4


class TestAPIEdgeCases:
    """Tests for API edge cases."""

    def test_detect_edge_values(self, client: TestClient) -> None:
        """Test detection with edge case values."""
        # Minimum valid values
        transaction = {
            "amount": 0.01,
            "hour": 0,
            "user_age_days": 1,
            "transaction_count": 0,
        }
        response = client.post("/detect", json=transaction)
        assert response.status_code == 200

    def test_detect_max_values(self, client: TestClient) -> None:
        """Test detection with maximum values."""
        transaction = {
            "amount": 9999999,
            "hour": 23,
            "user_age_days": 36500,
            "transaction_count": 999999,
        }
        response = client.post("/detect", json=transaction)
        assert response.status_code == 200

    def test_detect_float_amount(self, client: TestClient) -> None:
        """Test detection with float amount."""
        transaction = {
            "amount": 123.45,
            "hour": 14,
            "user_age_days": 30,
            "transaction_count": 5,
        }
        response = client.post("/detect", json=transaction)
        assert response.status_code == 200

    def test_detect_zero_amount_invalid(self, client: TestClient) -> None:
        """Test that zero amount is invalid."""
        transaction = {
            "amount": 0,
            "hour": 14,
            "user_age_days": 30,
            "transaction_count": 5,
        }
        response = client.post("/detect", json=transaction)
        assert response.status_code == 422


class TestAPIResponseFormat:
    """Tests for API response format consistency."""

    def test_detect_response_types(
        self, client: TestClient, sample_transaction: dict[str, Any]
    ) -> None:
        """Test response field types."""
        response = client.post("/detect", json=sample_transaction)
        data = response.json()
        assert isinstance(data["is_suspicious"], bool)
        assert isinstance(data["risk_score"], float)
        assert isinstance(data["risk_level"], str)
        assert isinstance(data["processing_time_ms"], float)

    def test_batch_response_types(
        self, client: TestClient, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test batch response field types."""
        response = client.post("/detect/batch", json={"transactions": batch_transactions})
        data = response.json()
        assert isinstance(data["results"], list)
        assert isinstance(data["total_transactions"], int)
        assert isinstance(data["suspicious_count"], int)
        assert isinstance(data["processing_time_ms"], float)

    def test_health_response_types(self, client: TestClient) -> None:
        """Test health response field types."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["model_loaded"], bool)
