#!/usr/bin/env python
"""
Batch Processing Example

Author: Moslem Mohseni

This example demonstrates batch processing of multiple transactions.
"""

from metaguard import SimpleDetector


def main():
    print("=" * 60)
    print("MetaGuard - Batch Processing Example")
    print("=" * 60)

    # Create detector instance
    detector = SimpleDetector()

    # Define multiple transactions
    transactions = [
        # Normal transactions
        {"amount": 50, "hour": 10, "user_age_days": 200, "transaction_count": 3},
        {"amount": 100, "hour": 14, "user_age_days": 365, "transaction_count": 5},
        {"amount": 25, "hour": 16, "user_age_days": 180, "transaction_count": 2},
        # Suspicious transactions
        {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50},
        {"amount": 3000, "hour": 2, "user_age_days": 10, "transaction_count": 40},
        # Borderline transactions
        {"amount": 500, "hour": 12, "user_age_days": 60, "transaction_count": 15},
        {"amount": 800, "hour": 8, "user_age_days": 45, "transaction_count": 12},
    ]

    print(f"\nProcessing {len(transactions)} transactions...")
    print("-" * 60)

    # Batch detection
    results = detector.batch_detect(transactions)

    # Display results
    print(f"\n{'#':<3} {'Amount':>8} {'Hour':>5} {'Age':>5} {'Count':>6} {'Status':>12} {'Level':>8}")
    print("-" * 60)

    suspicious_count = 0
    for i, (tx, result) in enumerate(zip(transactions, results), 1):
        status = "SUSPICIOUS" if result["is_suspicious"] else "OK"
        if result["is_suspicious"]:
            suspicious_count += 1

        print(
            f"{i:<3} ${tx['amount']:>7} {tx['hour']:>5} {tx['user_age_days']:>5}d "
            f"{tx['transaction_count']:>6} {status:>12} {result['risk_level']:>8}"
        )

    # Summary
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total Transactions: {len(transactions)}")
    print(f"  Suspicious: {suspicious_count}")
    print(f"  Normal: {len(transactions) - suspicious_count}")
    print(f"  Suspicious Rate: {suspicious_count/len(transactions):.1%}")

    print("\n" + "=" * 60)
    print("Batch processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
