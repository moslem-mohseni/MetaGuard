#!/usr/bin/env python
"""
Basic MetaGuard Usage Example

Author: Moslem Mohseni

This example demonstrates the basic usage of MetaGuard for fraud detection.
"""

from metaguard import check_transaction, SimpleDetector


def main():
    print("=" * 60)
    print("MetaGuard - Basic Usage Example")
    print("=" * 60)

    # Example 1: Quick check with check_transaction()
    print("\n1. Quick Transaction Check")
    print("-" * 40)

    transaction = {
        "amount": 100,
        "hour": 14,
        "user_age_days": 30,
        "transaction_count": 5,
    }

    result = check_transaction(transaction)

    print(f"Transaction: {transaction}")
    print(f"Is Suspicious: {result['is_suspicious']}")
    print(f"Risk Score: {result['risk_score']:.2%}")
    print(f"Risk Level: {result['risk_level']}")

    # Example 2: Check a suspicious transaction
    print("\n2. Suspicious Transaction Check")
    print("-" * 40)

    suspicious = {
        "amount": 5000,
        "hour": 3,
        "user_age_days": 5,
        "transaction_count": 50,
    }

    result = check_transaction(suspicious)

    print(f"Transaction: {suspicious}")
    print(f"Is Suspicious: {result['is_suspicious']}")
    print(f"Risk Score: {result['risk_score']:.2%}")
    print(f"Risk Level: {result['risk_level']}")

    # Example 3: Using SimpleDetector class
    print("\n3. Using SimpleDetector Class")
    print("-" * 40)

    detector = SimpleDetector()

    # Get model info
    info = detector.get_model_info()
    print(f"Model Type: {info['model_type']}")
    print(f"Risk Threshold: {info['risk_threshold']}")

    # Detect single transaction
    result = detector.detect({
        "amount": 250,
        "hour": 10,
        "user_age_days": 60,
        "transaction_count": 8,
    })
    print(f"Detection Result: {result['risk_level']}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
