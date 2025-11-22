"""
MetaGuard - Model Training
Author: Moslem Mohseni
Description: Train fraud detection model using Random Forest
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

def train_model():
    """
    Train fraud detection model

    Author: Moslem Mohseni
    """
    print("MetaGuard Model Training")
    print("Author: Moslem Mohseni")
    print("=" * 50)

    # Check if data exists
    data_path = 'data/transactions.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Data file not found at {data_path}")
        print("Please run: python scripts/generate_data.py first")
        return

    # Load data
    print("\n1. Loading data...")
    data = pd.read_csv(data_path)
    print(f"   ✓ Loaded {len(data)} transactions")

    # Prepare features and labels
    print("\n2. Preparing features...")
    features = ['amount', 'hour', 'user_age_days', 'transaction_count']
    X = data[features]
    y = data['is_fraud']
    print(f"   ✓ Features: {', '.join(features)}")
    print(f"   ✓ Fraud cases: {y.sum()} ({y.mean()*100:.1f}%)")

    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")

    # Train model
    print("\n4. Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   ✓ Training complete")

    # Evaluate
    print("\n5. Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"ACCURACY: {accuracy:.2%}")
    print(f"{'='*50}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(features, model.feature_importances_):
        print(f"  {feature:20s}: {importance:.4f}")

    # Save model
    print("\n6. Saving model...")
    os.makedirs('metaguard/models', exist_ok=True)
    model_path = 'metaguard/models/model.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"   ✓ Model saved to: {model_path}")

    # Success message
    print(f"\n{'='*50}")
    print("✓ Training completed successfully!")
    print("✓ Model is ready to use")
    print(f"✓ Accuracy: {accuracy:.2%}")
    print(f"{'='*50}")
    print("\nNext steps:")
    print("  1. Test the model: python example.py")
    print("  2. Use in your code: from metaguard import check_transaction")
    print(f"\nDeveloped by: Moslem Mohseni")

if __name__ == "__main__":
    train_model()
