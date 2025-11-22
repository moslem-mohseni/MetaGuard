"""
MetaGuard - Metaverse Fraud Detection Library
Author: Moslem Mohseni
Version: 1.0.0 (MVP)

Simple fraud detection for metaverse transactions with just 3 lines of code.

Example usage:
    from metaguard import check_transaction

    result = check_transaction({"amount": 1000, "user_age_days": 5})
    if result['is_suspicious']:
        print(f"⚠️ Suspicious! Risk: {result['risk_score']}")
"""

__version__ = "1.0.0"
__author__ = "Moslem Mohseni"

from .detector import SimpleDetector, check_transaction
from .risk import calculate_risk, get_risk_level, analyze_transaction_risk

__all__ = [
    'SimpleDetector',
    'check_transaction',
    'calculate_risk',
    'get_risk_level',
    'analyze_transaction_risk'
]
