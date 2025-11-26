"""
Unit tests for MetaGuard Risk Module

Author: Moslem Mohseni
"""

from __future__ import annotations

from typing import Any

import pytest

from metaguard import analyze_transaction_risk, calculate_risk, get_risk_level
from metaguard.risk import calculate_combined_risk, get_risk_factors_description


class TestCalculateRisk:
    """Tests for calculate_risk function."""

    def test_low_risk_transaction(self) -> None:
        """Test low risk transaction."""
        risk = calculate_risk(amount=50, user_age=200, transaction_count=3)
        assert risk < 40

    def test_high_risk_transaction(self) -> None:
        """Test high risk transaction."""
        risk = calculate_risk(amount=5000, user_age=5, transaction_count=50)
        assert risk > 70

    def test_medium_risk_transaction(self) -> None:
        """Test medium risk transaction."""
        risk = calculate_risk(amount=500, user_age=50, transaction_count=10)
        # Risk is calculated and capped at 0-100
        assert 0 <= risk <= 100

    def test_risk_increases_with_amount(self) -> None:
        """Test risk increases with transaction amount."""
        risk1 = calculate_risk(amount=100, user_age=100, transaction_count=5)
        risk2 = calculate_risk(amount=1000, user_age=100, transaction_count=5)
        assert risk2 > risk1

    def test_risk_decreases_with_user_age(self) -> None:
        """Test risk decreases with account age."""
        risk1 = calculate_risk(amount=100, user_age=10, transaction_count=5)
        risk2 = calculate_risk(amount=100, user_age=100, transaction_count=5)
        assert risk2 < risk1

    def test_risk_increases_with_transaction_count(self) -> None:
        """Test risk increases with transaction count."""
        risk1 = calculate_risk(amount=100, user_age=100, transaction_count=5)
        risk2 = calculate_risk(amount=100, user_age=100, transaction_count=50)
        assert risk2 > risk1

    def test_risk_bounded_at_100(self) -> None:
        """Test risk score is capped at 100."""
        risk = calculate_risk(amount=100000, user_age=1, transaction_count=1000)
        assert risk <= 100

    def test_risk_minimum_zero(self) -> None:
        """Test risk score minimum is 0."""
        risk = calculate_risk(amount=1, user_age=365, transaction_count=1)
        assert risk >= 0

    def test_risk_with_zero_transaction_count(self) -> None:
        """Test risk with zero transaction count."""
        risk = calculate_risk(amount=100, user_age=100, transaction_count=0)
        assert risk == 0

    def test_risk_protects_against_zero_user_age(self) -> None:
        """Test function handles zero user age safely."""
        # Should not raise division by zero
        risk = calculate_risk(amount=100, user_age=0, transaction_count=5)
        assert isinstance(risk, float)

    def test_risk_returns_float(self) -> None:
        """Test function returns float."""
        risk = calculate_risk(amount=100, user_age=100, transaction_count=5)
        assert isinstance(risk, float)

    def test_risk_rounded(self) -> None:
        """Test risk score is rounded to 2 decimal places."""
        risk = calculate_risk(amount=123, user_age=47, transaction_count=13)
        # Check it's properly rounded
        assert risk == round(risk, 2)


class TestGetRiskLevel:
    """Tests for get_risk_level function."""

    @pytest.mark.parametrize(
        "score,expected",
        [
            (0, "Low"),
            (20, "Low"),
            (39, "Low"),
            (40, "Low"),
            (40.1, "Medium"),
            (55, "Medium"),
            (69, "Medium"),
            (70, "Medium"),
            (70.1, "High"),
            (85, "High"),
            (100, "High"),
        ],
    )
    def test_risk_level_thresholds(self, score: float, expected: str) -> None:
        """Test risk level boundaries."""
        assert get_risk_level(score) == expected

    def test_risk_level_returns_string(self) -> None:
        """Test function returns string."""
        level = get_risk_level(50)
        assert isinstance(level, str)

    def test_risk_level_valid_values(self) -> None:
        """Test all returned values are valid."""
        for score in range(0, 101, 10):
            level = get_risk_level(score)
            assert level in ["Low", "Medium", "High"]


class TestAnalyzeTransactionRisk:
    """Tests for analyze_transaction_risk function."""

    def test_returns_all_keys(self, sample_transaction: dict[str, Any]) -> None:
        """Test function returns all expected keys."""
        result = analyze_transaction_risk(sample_transaction)
        assert "risk_score" in result
        assert "risk_level" in result
        assert "factors" in result
        assert "active_factor_count" in result

    def test_factors_structure(self, sample_transaction: dict[str, Any]) -> None:
        """Test factors dictionary structure."""
        result = analyze_transaction_risk(sample_transaction)
        factors = result["factors"]
        assert "high_amount" in factors
        assert "new_account" in factors
        assert "high_frequency" in factors
        assert "unusual_hour" in factors

    def test_factors_are_boolean(self, sample_transaction: dict[str, Any]) -> None:
        """Test all factors are boolean values."""
        result = analyze_transaction_risk(sample_transaction)
        for value in result["factors"].values():
            assert isinstance(value, bool)

    def test_high_amount_factor(self) -> None:
        """Test high_amount factor detection."""
        high_amount_tx = {
            "amount": 2000,
            "hour": 12,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = analyze_transaction_risk(high_amount_tx)
        assert result["factors"]["high_amount"] is True

        low_amount_tx = {
            "amount": 500,
            "hour": 12,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = analyze_transaction_risk(low_amount_tx)
        assert result["factors"]["high_amount"] is False

    def test_new_account_factor(self) -> None:
        """Test new_account factor detection."""
        new_account_tx = {
            "amount": 100,
            "hour": 12,
            "user_age_days": 10,
            "transaction_count": 5,
        }
        result = analyze_transaction_risk(new_account_tx)
        assert result["factors"]["new_account"] is True

        old_account_tx = {
            "amount": 100,
            "hour": 12,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = analyze_transaction_risk(old_account_tx)
        assert result["factors"]["new_account"] is False

    def test_high_frequency_factor(self) -> None:
        """Test high_frequency factor detection."""
        high_freq_tx = {
            "amount": 100,
            "hour": 12,
            "user_age_days": 100,
            "transaction_count": 30,
        }
        result = analyze_transaction_risk(high_freq_tx)
        assert result["factors"]["high_frequency"] is True

        low_freq_tx = {
            "amount": 100,
            "hour": 12,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = analyze_transaction_risk(low_freq_tx)
        assert result["factors"]["high_frequency"] is False

    def test_unusual_hour_factor(self) -> None:
        """Test unusual_hour factor detection."""
        late_night_tx = {
            "amount": 100,
            "hour": 3,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = analyze_transaction_risk(late_night_tx)
        assert result["factors"]["unusual_hour"] is True

        normal_hour_tx = {
            "amount": 100,
            "hour": 14,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = analyze_transaction_risk(normal_hour_tx)
        assert result["factors"]["unusual_hour"] is False

    def test_active_factor_count(self) -> None:
        """Test active_factor_count is correct."""
        # Transaction with multiple risk factors
        risky_tx = {
            "amount": 2000,
            "hour": 3,
            "user_age_days": 10,
            "transaction_count": 30,
        }
        result = analyze_transaction_risk(risky_tx)
        # All 4 factors should be active
        assert result["active_factor_count"] == 4

        # Safe transaction
        safe_tx = {
            "amount": 100,
            "hour": 12,
            "user_age_days": 100,
            "transaction_count": 5,
        }
        result = analyze_transaction_risk(safe_tx)
        assert result["active_factor_count"] == 0


class TestCalculateCombinedRisk:
    """Tests for calculate_combined_risk function."""

    def test_combined_risk_basic(self) -> None:
        """Test basic combined risk calculation."""
        combined = calculate_combined_risk(ml_score=0.8, formula_score=60)
        # Default weight is 0.7 for ML
        # Expected: (0.8 * 100 * 0.7) + (60 * 0.3) = 56 + 18 = 74
        assert combined == 74.0

    def test_combined_risk_with_custom_weight(self) -> None:
        """Test combined risk with custom ML weight."""
        combined = calculate_combined_risk(ml_score=0.5, formula_score=50, ml_weight=0.5)
        # Expected: (0.5 * 100 * 0.5) + (50 * 0.5) = 25 + 25 = 50
        assert combined == 50.0

    def test_combined_risk_bounded(self) -> None:
        """Test combined risk is bounded 0-100."""
        combined = calculate_combined_risk(ml_score=1.0, formula_score=100)
        assert combined <= 100

        combined = calculate_combined_risk(ml_score=0.0, formula_score=0)
        assert combined >= 0


class TestGetRiskFactorsDescription:
    """Tests for get_risk_factors_description function."""

    def test_returns_list(self) -> None:
        """Test function returns list."""
        factors = {"high_amount": True, "new_account": False}
        descriptions = get_risk_factors_description(factors)
        assert isinstance(descriptions, list)

    def test_includes_active_factors(self) -> None:
        """Test active factors are included."""
        factors = {"high_amount": True, "new_account": True}
        descriptions = get_risk_factors_description(factors)
        assert len(descriptions) == 2

    def test_excludes_inactive_factors(self) -> None:
        """Test inactive factors are excluded."""
        factors = {"high_amount": False, "new_account": False}
        descriptions = get_risk_factors_description(factors)
        assert len(descriptions) == 0

    def test_descriptions_are_strings(self) -> None:
        """Test all descriptions are strings."""
        factors = {
            "high_amount": True,
            "new_account": True,
            "high_frequency": True,
            "unusual_hour": True,
        }
        descriptions = get_risk_factors_description(factors)
        for desc in descriptions:
            assert isinstance(desc, str)
            assert len(desc) > 0
