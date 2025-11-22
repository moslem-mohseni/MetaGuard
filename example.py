"""
MetaGuard - Example Usage
Author: Moslem Mohseni
Description: Complete end-to-end example of fraud detection
"""

import pandas as pd
from metaguard.detector import SimpleDetector

def main():
    """
    Demonstrate MetaGuard fraud detection capabilities

    Author: Moslem Mohseni
    """
    print("=" * 60)
    print("MetaGuard - Fraud Detection Example")
    print("Author: Moslem Mohseni")
    print("=" * 60)

    # Test transactions
    test_transactions = [
        {
            'amount': 100,
            'hour': 14,
            'user_age_days': 30,
            'transaction_count': 5,
            'label': 'Normal - Small amount, old account'
        },
        {
            'amount': 5000,
            'hour': 3,
            'user_age_days': 1,
            'transaction_count': 50,
            'label': 'Suspicious - Large amount, new account, late night'
        },
        {
            'amount': 200,
            'hour': 20,
            'user_age_days': 100,
            'transaction_count': 10,
            'label': 'Normal - Moderate amount, old account'
        },
        {
            'amount': 3000,
            'hour': 2,
            'user_age_days': 3,
            'transaction_count': 35,
            'label': 'Suspicious - High amount, very new account'
        },
        {
            'amount': 50,
            'hour': 12,
            'user_age_days': 200,
            'transaction_count': 3,
            'label': 'Normal - Low amount, established account'
        },
    ]

    # Initialize detector
    print("\nInitializing detector...")
    try:
        detector = SimpleDetector()
        print("✓ Detector loaded successfully\n")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train the model first:")
        print("  1. python scripts/generate_data.py")
        print("  2. python scripts/train.py")
        return

    # Analyze transactions
    print("\nAnalyzing Transactions:")
    print("-" * 60)

    results = []
    for i, tx in enumerate(test_transactions):
        # Get label and remove from transaction dict
        label = tx.pop('label')

        # Detect fraud
        result = detector.detect(tx)

        # Store result
        results.append({
            'tx_id': i + 1,
            'expected': label,
            'suspicious': result['is_suspicious'],
            'risk_score': result['risk_score'],
            'risk_level': result['risk_level']
        })

        # Print result
        status = "🚨 SUSPICIOUS" if result['is_suspicious'] else "✓ Normal"
        print(f"\nTransaction {i+1}: {status}")
        print(f"  Expected: {label}")
        print(f"  Amount: ${tx['amount']}")
        print(f"  Hour: {tx['hour']}:00")
        print(f"  User Age: {tx['user_age_days']} days")
        print(f"  Transaction Count: {tx['transaction_count']}")
        print(f"  Risk Score: {result['risk_score']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    df_results = pd.DataFrame(results)
    suspicious_count = df_results['suspicious'].sum()
    total_count = len(df_results)

    print(f"\nTotal Transactions: {total_count}")
    print(f"Flagged as Suspicious: {suspicious_count}")
    print(f"Flagged as Normal: {total_count - suspicious_count}")

    print("\nRisk Level Distribution:")
    print(df_results['risk_level'].value_counts().to_string())

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("Developed by: Moslem Mohseni")
    print("=" * 60)

    # Quick usage example
    print("\n" + "=" * 60)
    print("QUICK USAGE EXAMPLE")
    print("=" * 60)
    print("""
from metaguard import check_transaction

result = check_transaction({
    'amount': 1000,
    'hour': 14,
    'user_age_days': 5,
    'transaction_count': 10
})

if result['is_suspicious']:
    print(f"⚠️ Suspicious! Risk: {result['risk_score']:.2%}")
else:
    print(f"✓ Normal transaction")
    """)

if __name__ == "__main__":
    main()
