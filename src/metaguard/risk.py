"""
MetaGuard Risk Calculator

Simple risk calculation for metaverse transactions.

Author: Moslem Mohseni
"""

from __future__ import annotations

from typing import Any

from metaguard.utils.config import get_config
from metaguard.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_risk(
    amount: float,
    user_age: int,
    transaction_count: int,
) -> float:
    """
    Calculate risk score for a transaction using a simple formula.

    The risk score is calculated based on:
    - Transaction amount (higher = more risk)
    - User account age (newer = more risk)
    - Transaction count (more transactions = more risk)

    Formula:
        Risk = (Amount / 1000) * (5 / User_Age) * (Transaction_Count / 10)
        Normalized to 0-100 scale

    Author: Moslem Mohseni

    Args:
        amount: Transaction amount in currency units
        user_age: User account age in days (minimum 1)
        transaction_count: Number of previous transactions

    Returns:
        Risk score normalized to 0-100 scale

    Example:
        >>> calculate_risk(amount=1000, user_age=30, transaction_count=5)
        8.33
        >>> calculate_risk(amount=5000, user_age=5, transaction_count=50)
        100.0
    """
    # Ensure minimum values to avoid division by zero
    safe_user_age = max(user_age, 1)
    safe_amount = max(amount, 0)
    safe_tx_count = max(transaction_count, 0)

    # Calculate raw risk score
    risk = (safe_amount / 1000) * (5 / safe_user_age) * (safe_tx_count / 10)

    # Normalize to 0-100 and cap at 100
    normalized_risk = min(100.0, risk * 10)

    logger.debug(
        f"Risk calculated: amount={safe_amount}, user_age={safe_user_age}, "
        f"tx_count={safe_tx_count} -> risk={normalized_risk:.2f}"
    )

    return round(normalized_risk, 2)


def get_risk_level(risk_score: float) -> str:
    """
    Convert numeric risk score to categorical level.

    Risk levels are determined by configurable thresholds:
    - Low: score <= low_risk_threshold (default 40)
    - Medium: low_risk_threshold < score <= high_risk_threshold (default 70)
    - High: score > high_risk_threshold

    Author: Moslem Mohseni

    Args:
        risk_score: Numeric risk score (0-100)

    Returns:
        Risk level string: 'Low', 'Medium', or 'High'

    Example:
        >>> get_risk_level(25)
        'Low'
        >>> get_risk_level(55)
        'Medium'
        >>> get_risk_level(85)
        'High'
    """
    config = get_config()

    if risk_score > config.high_risk_threshold:
        return "High"
    elif risk_score > config.low_risk_threshold:
        return "Medium"
    else:
        return "Low"


def analyze_transaction_risk(transaction: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze a transaction and return detailed risk information.

    This function provides a comprehensive risk analysis including:
    - Numeric risk score
    - Categorical risk level
    - Individual risk factors

    Author: Moslem Mohseni

    Args:
        transaction: Dictionary with transaction details:
            - amount: Transaction amount
            - user_age_days: User account age in days
            - transaction_count: Number of previous transactions
            - hour: Hour of transaction (optional, for factor analysis)

    Returns:
        Dictionary containing:
            - risk_score: Numeric score (0-100)
            - risk_level: Categorical level ('Low', 'Medium', 'High')
            - factors: Dictionary of individual risk factors

    Example:
        >>> result = analyze_transaction_risk({
        ...     "amount": 5000,
        ...     "hour": 3,
        ...     "user_age_days": 5,
        ...     "transaction_count": 30
        ... })
        >>> result['risk_level']
        'High'
        >>> result['factors']['high_amount']
        True
    """
    # Extract values with defaults
    amount = transaction.get("amount", 0)
    user_age = transaction.get("user_age_days", 1)
    tx_count = transaction.get("transaction_count", 0)
    hour = transaction.get("hour", 12)

    # Calculate base risk score
    risk_score = calculate_risk(amount, user_age, tx_count)
    risk_level = get_risk_level(risk_score)

    # Analyze individual risk factors
    factors = {
        "high_amount": amount > 1000,
        "new_account": user_age < 30,
        "high_frequency": tx_count > 20,
        "unusual_hour": hour < 6 or hour > 22,
    }

    # Count active risk factors
    active_factors = sum(1 for v in factors.values() if v)

    result = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "factors": factors,
        "active_factor_count": active_factors,
    }

    logger.debug(
        f"Risk analysis: score={risk_score:.2f}, level={risk_level}, "
        f"active_factors={active_factors}"
    )

    return result


def calculate_combined_risk(
    ml_score: float,
    formula_score: float,
    ml_weight: float = 0.7,
) -> float:
    """
    Combine ML-based risk score with formula-based risk score.

    This function provides a weighted combination of the ML model's
    prediction and the rule-based formula calculation.

    Author: Moslem Mohseni

    Args:
        ml_score: Risk score from ML model (0-1)
        formula_score: Risk score from formula (0-100)
        ml_weight: Weight for ML score (0-1), formula gets (1 - ml_weight)

    Returns:
        Combined risk score (0-100)

    Example:
        >>> calculate_combined_risk(ml_score=0.8, formula_score=60, ml_weight=0.7)
        74.0
    """
    # Normalize ML score to 0-100
    ml_normalized = ml_score * 100

    # Calculate weighted average
    combined = (ml_normalized * ml_weight) + (formula_score * (1 - ml_weight))

    return round(min(100.0, max(0.0, combined)), 2)


def get_risk_factors_description(factors: dict[str, bool]) -> list[str]:
    """
    Get human-readable descriptions of active risk factors.

    Author: Moslem Mohseni

    Args:
        factors: Dictionary of risk factors and their boolean values

    Returns:
        List of descriptions for active (True) risk factors

    Example:
        >>> factors = {"high_amount": True, "new_account": False, "unusual_hour": True}
        >>> get_risk_factors_description(factors)
        ['High transaction amount (>$1000)', 'Unusual transaction hour']
    """
    descriptions = {
        "high_amount": "High transaction amount (>$1000)",
        "new_account": "New account (<30 days old)",
        "high_frequency": "High transaction frequency (>20 prior transactions)",
        "unusual_hour": "Unusual transaction hour (before 6am or after 10pm)",
    }

    return [descriptions[key] for key, value in factors.items() if value and key in descriptions]
