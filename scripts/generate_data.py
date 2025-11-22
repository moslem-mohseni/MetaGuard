"""
MetaGuard - Data Generator
Author: Moslem Mohseni
Description: Generate synthetic transaction data for training
"""

import pandas as pd
import numpy as np
import os

def generate_simple_data(n=10000):
    """
    Generate synthetic metaverse transaction data

    Features:
    - amount: Transaction amount (log-normal distribution)
    - hour: Hour of day (0-23)
    - user_age_days: User account age in days
    - transaction_count: Number of previous transactions
    - is_fraud: Label (5% fraud rate)

    Author: Moslem Mohseni
    """

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate base features
    data = {
        'amount': np.random.lognormal(3, 2, n),
        'hour': np.random.randint(0, 24, n),
        'user_age_days': np.random.randint(1, 365, n),
        'transaction_count': np.random.poisson(3, n),
    }

    # Create fraud labels (5% fraud rate)
    is_fraud = np.random.binomial(1, 0.05, n)

    # Make fraud cases more obvious
    for i in range(n):
        if is_fraud[i] == 1:
            # Fraudulent transactions tend to:
            # - Have higher amounts
            # - Occur at unusual hours (late night)
            # - Come from newer accounts
            # - Have more transactions
            data['amount'][i] *= np.random.uniform(3, 10)
            data['hour'][i] = np.random.choice([0, 1, 2, 3, 4, 22, 23])
            data['user_age_days'][i] = np.random.randint(1, 30)
            data['transaction_count'][i] = np.random.randint(10, 100)

    data['is_fraud'] = is_fraud

    return pd.DataFrame(data)

def main():
    """
    Generate and save training data
    Author: Moslem Mohseni
    """
    print("Generating synthetic data...")
    print("Author: Moslem Mohseni")

    # Generate 10,000 transactions
    df = generate_simple_data(10000)

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    output_path = 'data/transactions.csv'
    df.to_csv(output_path, index=False)

    # Print statistics
    print(f"\n✓ Generated {len(df)} transactions")
    print(f"✓ Saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  - Total transactions: {len(df)}")
    print(f"  - Fraudulent: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"  - Normal: {(~df['is_fraud'].astype(bool)).sum()} ({(~df['is_fraud'].astype(bool)).mean()*100:.1f}%)")
    print(f"\nFeature ranges:")
    print(f"  - Amount: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
    print(f"  - Hour: {df['hour'].min()} - {df['hour'].max()}")
    print(f"  - User age: {df['user_age_days'].min()} - {df['user_age_days'].max()} days")
    print(f"  - Transaction count: {df['transaction_count'].min()} - {df['transaction_count'].max()}")

if __name__ == "__main__":
    main()
