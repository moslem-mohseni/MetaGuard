"""
MetaGuard - Metaverse Fraud Detection Library

A Python library for detecting fraudulent transactions in metaverse environments
using machine learning algorithms.

Author: Moslem Mohseni
License: MIT
Repository: https://github.com/moslem-mohseni/MetaGuard

Example usage:
    >>> from metaguard import check_transaction
    >>> result = check_transaction({
    ...     "amount": 1000,
    ...     "hour": 3,
    ...     "user_age_days": 5,
    ...     "transaction_count": 50
    ... })
    >>> if result['is_suspicious']:
    ...     print(f"Suspicious! Risk: {result['risk_score']:.2%}")
"""

from __future__ import annotations

__version__ = "2.3.0"
__author__ = "Moslem Mohseni"
__email__ = "moslem.mohseni@example.com"
__license__ = "MIT"

from metaguard.detector import SimpleDetector, check_transaction
from metaguard.risk import analyze_transaction_risk, calculate_risk, get_risk_level

__all__ = [
    "SimpleDetector",
    "__author__",
    "__version__",
    "analyze_transaction_risk",
    "calculate_risk",
    "check_transaction",
    "get_risk_level",
]
