#!/usr/bin/env python
"""
Risk Analysis Example

Author: Moslem Mohseni

This example demonstrates detailed risk analysis with factor breakdown.
"""

from metaguard import analyze_transaction_risk, calculate_risk, get_risk_level
from metaguard.risk import get_risk_factors_description


def main():
    print("=" * 60)
    print("MetaGuard - Risk Analysis Example")
    print("=" * 60)

    # Example 1: Full risk analysis
    print("\n1. Detailed Risk Analysis")
    print("-" * 40)

    transaction = {
        "amount": 3000,
        "hour": 2,
        "user_age_days": 10,
        "transaction_count": 35,
    }

    result = analyze_transaction_risk(transaction)

    print(f"Transaction:")
    print(f"  Amount: ${transaction['amount']}")
    print(f"  Hour: {transaction['hour']}:00")
    print(f"  Account Age: {transaction['user_age_days']} days")
    print(f"  Transaction Count: {transaction['transaction_count']}")

    print(f"\nRisk Assessment:")
    print(f"  Risk Score: {result['risk_score']:.1f}/100")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Active Factors: {result['active_factor_count']}/4")

    print(f"\nRisk Factors:")
    for factor, active in result['factors'].items():
        status = "[X]" if active else "[ ]"
        print(f"  {status} {factor}")

    # Get human-readable descriptions
    descriptions = get_risk_factors_description(result['factors'])
    if descriptions:
        print(f"\nRisk Factor Descriptions:")
        for desc in descriptions:
            print(f"  - {desc}")

    # Example 2: Using individual risk functions
    print("\n2. Individual Risk Calculation")
    print("-" * 40)

    test_cases = [
        (100, 200, 5),     # Low risk
        (1000, 50, 15),    # Medium risk
        (5000, 5, 50),     # High risk
    ]

    print(f"{'Amount':>8} {'Age':>6} {'Count':>6} {'Score':>8} {'Level':>8}")
    print("-" * 40)

    for amount, age, count in test_cases:
        score = calculate_risk(amount=amount, user_age=age, transaction_count=count)
        level = get_risk_level(score)
        print(f"${amount:>7} {age:>5}d {count:>6} {score:>7.1f} {level:>8}")

    # Example 3: Compare transactions
    print("\n3. Transaction Comparison")
    print("-" * 40)

    transactions = [
        {"name": "Normal", "amount": 100, "hour": 14, "user_age_days": 100, "transaction_count": 5},
        {"name": "New User", "amount": 100, "hour": 14, "user_age_days": 5, "transaction_count": 5},
        {"name": "High Amount", "amount": 5000, "hour": 14, "user_age_days": 100, "transaction_count": 5},
        {"name": "Late Night", "amount": 100, "hour": 3, "user_age_days": 100, "transaction_count": 5},
        {"name": "High Freq", "amount": 100, "hour": 14, "user_age_days": 100, "transaction_count": 50},
        {"name": "All Flags", "amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50},
    ]

    print(f"{'Type':<12} {'Score':>8} {'Level':>8} {'Factors':>8}")
    print("-" * 40)

    for tx in transactions:
        tx_data = {k: v for k, v in tx.items() if k != "name"}
        result = analyze_transaction_risk(tx_data)
        print(
            f"{tx['name']:<12} "
            f"{result['risk_score']:>7.1f} "
            f"{result['risk_level']:>8} "
            f"{result['active_factor_count']:>8}"
        )

    print("\n" + "=" * 60)
    print("Risk analysis completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
