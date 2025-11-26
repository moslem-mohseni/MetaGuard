"""
Integration tests for MetaGuard Pipeline

Author: Moslem Mohseni
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from metaguard import SimpleDetector, check_transaction
from metaguard.utils.config import MetaGuardConfig, reset_config, set_config


class TestDetectionPipeline:
    """Integration tests for the full detection pipeline."""

    def test_single_detection_pipeline(
        self, sample_transaction: dict[str, Any]
    ) -> None:
        """Test full pipeline for single transaction detection."""
        # Use check_transaction which creates detector internally
        result = check_transaction(sample_transaction)

        # Verify complete result
        assert "is_suspicious" in result
        assert "risk_score" in result
        assert "risk_level" in result
        assert result["is_suspicious"] in (True, False)
        assert 0 <= result["risk_score"] <= 1
        assert result["risk_level"] in ["Low", "Medium", "High"]

    def test_batch_detection_pipeline(
        self, batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Test full pipeline for batch detection."""
        detector = SimpleDetector()
        results = detector.batch_detect(batch_transactions)

        assert len(results) == len(batch_transactions)
        for result in results:
            assert "is_suspicious" in result
            assert "risk_score" in result
            assert "risk_level" in result

    def test_detector_with_custom_config(
        self, sample_transaction: dict[str, Any]
    ) -> None:
        """Test detection with custom configuration."""
        reset_config()

        # Create custom config with different threshold
        custom_config = MetaGuardConfig(risk_threshold=0.9)

        detector = SimpleDetector(config=custom_config)
        result = detector.detect(sample_transaction)

        # Result should use custom threshold
        assert "is_suspicious" in result

    def test_multiple_detectors_same_model(
        self, sample_transaction: dict[str, Any]
    ) -> None:
        """Test multiple detector instances produce consistent results."""
        detector1 = SimpleDetector()
        detector2 = SimpleDetector()

        result1 = detector1.detect(sample_transaction)
        result2 = detector2.detect(sample_transaction)

        # Results should be identical
        assert result1["is_suspicious"] == result2["is_suspicious"]
        assert result1["risk_score"] == result2["risk_score"]
        assert result1["risk_level"] == result2["risk_level"]

    def test_detection_consistency(self) -> None:
        """Test detection produces consistent results for same input."""
        detector = SimpleDetector()
        transaction = {
            "amount": 1000,
            "hour": 12,
            "user_age_days": 50,
            "transaction_count": 10,
        }

        results = [detector.detect(transaction) for _ in range(5)]

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]


class TestModelPipeline:
    """Integration tests for model operations."""

    def test_model_info_accuracy(self, detector: SimpleDetector) -> None:
        """Test model info matches actual model."""
        info = detector.get_model_info()

        # Verify info matches model
        assert info["model_type"] == type(detector.model).__name__

        if hasattr(detector.model, "n_estimators"):
            assert info["n_estimators"] == detector.model.n_estimators

    def test_load_custom_model(self, temp_model_dir: Path) -> None:
        """Test loading a custom trained model."""
        # Create and save a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Train on minimal data
        X = [[100, 12, 30, 5], [5000, 3, 2, 50], [50, 14, 200, 3]]
        y = [0, 1, 0]
        model.fit(X, y)

        # Save model
        model_path = temp_model_dir / "custom_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Load with detector
        detector = SimpleDetector(model_path=str(model_path))
        result = detector.detect(
            {"amount": 100, "hour": 12, "user_age_days": 30, "transaction_count": 5}
        )

        assert "is_suspicious" in result


class TestConfigIntegration:
    """Integration tests for configuration."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        reset_config()

    def test_global_config_affects_detector(self) -> None:
        """Test global config is used by detector."""
        custom_config = MetaGuardConfig(risk_threshold=0.1)
        set_config(custom_config)

        detector = SimpleDetector()
        # Detector should use global config
        assert detector.config.risk_threshold == 0.1

    def test_explicit_config_overrides_global(self) -> None:
        """Test explicit config overrides global."""
        global_config = MetaGuardConfig(risk_threshold=0.1)
        set_config(global_config)

        explicit_config = MetaGuardConfig(risk_threshold=0.9)
        detector = SimpleDetector(config=explicit_config)

        assert detector.config.risk_threshold == 0.9


class TestDataFlow:
    """Integration tests for data flow through the system."""

    def test_transaction_validation_to_detection(self) -> None:
        """Test data flows correctly from validation to detection."""
        transaction = {
            "amount": 100.5,  # Float
            "hour": 12,
            "user_age_days": 30,
            "transaction_count": 5,
        }

        detector = SimpleDetector()
        result = detector.detect(transaction)

        # Should complete without error
        assert result is not None

    def test_batch_processing_order(self) -> None:
        """Test batch results maintain order."""
        transactions = [
            {"amount": 100, "hour": 10, "user_age_days": 100, "transaction_count": 5},
            {"amount": 5000, "hour": 3, "user_age_days": 2, "transaction_count": 50},
            {"amount": 25, "hour": 15, "user_age_days": 365, "transaction_count": 1},
        ]

        detector = SimpleDetector()
        results = detector.batch_detect(transactions)

        # Results should be in same order
        # Second transaction should be suspicious (high risk)
        assert results[1]["is_suspicious"] == True  # noqa: E712
        # First and third should be normal
        assert results[0]["risk_score"] < results[1]["risk_score"]
