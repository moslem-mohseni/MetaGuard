"""
End-to-End tests for MetaGuard

Author: Moslem Mohseni
"""

from __future__ import annotations

import subprocess
import sys

import pytest


class TestPackageImport:
    """E2E tests for package import and basic usage."""

    def test_import_metaguard(self) -> None:
        """Test package can be imported."""
        import metaguard

        assert metaguard is not None

    def test_import_version(self) -> None:
        """Test version is accessible."""
        from metaguard import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        # Should be semver format
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_import_author(self) -> None:
        """Test author is accessible."""
        from metaguard import __author__

        assert __author__ == "Moslem Mohseni"

    def test_import_all_public_api(self) -> None:
        """Test all public API items can be imported."""
        from metaguard import (
            SimpleDetector,
            analyze_transaction_risk,
            calculate_risk,
            check_transaction,
            get_risk_level,
        )

        assert SimpleDetector is not None
        assert callable(check_transaction)
        assert callable(calculate_risk)
        assert callable(get_risk_level)
        assert callable(analyze_transaction_risk)


class TestBasicUsage:
    """E2E tests for basic usage patterns."""

    def test_three_line_usage(self) -> None:
        """Test the advertised 3-line usage."""
        from metaguard import check_transaction

        result = check_transaction(
            {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50}
        )
        is_suspicious = result["is_suspicious"]

        assert is_suspicious in (True, False)

    def test_detector_usage(self) -> None:
        """Test SimpleDetector usage pattern."""
        from metaguard import SimpleDetector

        detector = SimpleDetector()

        # Single detection
        result = detector.detect(
            {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5}
        )
        assert "is_suspicious" in result

        # Batch detection
        results = detector.batch_detect(
            [
                {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
                {"amount": 200, "hour": 10, "user_age_days": 60, "transaction_count": 10},
            ]
        )
        assert len(results) == 2

    def test_risk_analysis_usage(self) -> None:
        """Test risk analysis usage pattern."""
        from metaguard import analyze_transaction_risk

        result = analyze_transaction_risk(
            {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50}
        )

        assert "risk_score" in result
        assert "risk_level" in result
        assert "factors" in result


class TestRealWorldScenarios:
    """E2E tests for real-world scenarios."""

    def test_normal_user_transactions(self) -> None:
        """Test normal user transaction pattern."""
        from metaguard import SimpleDetector

        detector = SimpleDetector()

        # Normal user - small purchases during business hours
        transactions = [
            {"amount": 25, "hour": 10, "user_age_days": 180, "transaction_count": 5},
            {"amount": 50, "hour": 14, "user_age_days": 180, "transaction_count": 6},
            {"amount": 100, "hour": 16, "user_age_days": 180, "transaction_count": 7},
        ]

        results = detector.batch_detect(transactions)

        # All should be normal
        for result in results:
            assert result["is_suspicious"] == False  # noqa: E712

    def test_suspicious_pattern_new_account(self) -> None:
        """Test suspicious pattern - new account, high activity."""
        from metaguard import SimpleDetector

        detector = SimpleDetector()

        # Suspicious - new account with high value transactions
        transaction = {
            "amount": 5000,
            "hour": 2,
            "user_age_days": 3,
            "transaction_count": 50,
        }

        result = detector.detect(transaction)
        assert result["is_suspicious"] == True  # noqa: E712
        assert result["risk_level"] == "High"

    def test_edge_case_boundary_transaction(self) -> None:
        """Test transaction at risk threshold boundary."""
        from metaguard import SimpleDetector

        detector = SimpleDetector()

        # Transaction designed to be near threshold
        transaction = {
            "amount": 500,
            "hour": 12,
            "user_age_days": 60,
            "transaction_count": 15,
        }

        result = detector.detect(transaction)
        # Should complete without error
        assert "is_suspicious" in result
        assert result["risk_level"] in ["Low", "Medium", "High"]


class TestErrorHandling:
    """E2E tests for error handling."""

    def test_invalid_transaction_error(self) -> None:
        """Test error handling for invalid transaction."""
        from metaguard import check_transaction
        from metaguard.utils.exceptions import InvalidTransactionError

        with pytest.raises(InvalidTransactionError):
            check_transaction({"amount": -100, "hour": 14})

    def test_model_not_found_error(self) -> None:
        """Test error handling for missing model."""
        from metaguard import SimpleDetector
        from metaguard.utils.exceptions import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            SimpleDetector(model_path="/nonexistent/path/model.pkl")


class TestCommandLineUsage:
    """E2E tests for command-line usage."""

    def test_version_import(self) -> None:
        """Test version can be retrieved via Python."""
        result = subprocess.run(
            [sys.executable, "-c", "import metaguard; print(metaguard.__version__)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "1." in result.stdout  # Version starts with 1.

    def test_basic_detection_script(self) -> None:
        """Test basic detection can be run as script."""
        script = """
from metaguard import check_transaction
result = check_transaction({
    'amount': 100,
    'hour': 14,
    'user_age_days': 30,
    'transaction_count': 5
})
print('OK' if 'is_suspicious' in result else 'FAIL')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout


class TestPerformance:
    """E2E tests for performance characteristics."""

    def test_single_detection_speed(self) -> None:
        """Test single detection completes in reasonable time."""
        import time

        from metaguard import SimpleDetector

        detector = SimpleDetector()
        transaction = {
            "amount": 100,
            "hour": 14,
            "user_age_days": 30,
            "transaction_count": 5,
        }

        start = time.time()
        for _ in range(100):
            detector.detect(transaction)
        elapsed = time.time() - start

        # 100 detections should complete in < 10 seconds (increased for slower CI runners)
        assert elapsed < 10.0

    def test_batch_detection_speed(self) -> None:
        """Test batch detection completes in reasonable time."""
        import time

        from metaguard import SimpleDetector

        detector = SimpleDetector()
        transactions = [
            {"amount": 100 + i, "hour": i % 24, "user_age_days": 30, "transaction_count": 5}
            for i in range(100)  # Reduced for faster tests
        ]

        start = time.time()
        results = detector.batch_detect(transactions)
        elapsed = time.time() - start

        assert len(results) == 100
        # 100 transactions should complete in < 30 seconds
        assert elapsed < 30.0
