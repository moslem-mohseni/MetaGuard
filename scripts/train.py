"""
MetaGuard - Model Training

Train fraud detection model using Random Forest.

Author: Moslem Mohseni
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def train_model(
    data_path: str = "data/transactions.csv",
    output_path: str = "src/metaguard/models/model.pkl",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 15,
    random_state: int = 42,
) -> dict:
    """
    Train fraud detection model.

    Author: Moslem Mohseni

    Args:
        data_path: Path to training data CSV
        output_path: Path to save trained model
        test_size: Fraction of data for testing
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with training metrics
    """
    print("MetaGuard Model Training")
    print("Author: Moslem Mohseni")
    print("=" * 50)

    # Check if data exists
    if not os.path.exists(data_path):
        print(f"\n[ERROR] Data file not found at {data_path}")
        print("Please run: python scripts/generate_data.py first")
        return {}

    # Load data
    print("\n1. Loading data...")
    data = pd.read_csv(data_path)
    print(f"   [OK] Loaded {len(data)} transactions")

    # Prepare features and labels
    print("\n2. Preparing features...")
    features = ["amount", "hour", "user_age_days", "transaction_count"]
    X = data[features]
    y = data["is_fraud"]
    print(f"   [OK] Features: {', '.join(features)}")
    print(f"   [OK] Fraud cases: {y.sum()} ({y.mean()*100:.1f}%)")

    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   [OK] Training set: {len(X_train)} samples")
    print(f"   [OK] Test set: {len(X_test)} samples")

    # Train model
    print("\n4. Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print("   [OK] Training complete")

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
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    # Feature importance
    print("\nFeature Importance:")
    importance_dict = {}
    for feature, importance in zip(features, model.feature_importances_):
        print(f"  {feature:20s}: {importance:.4f}")
        importance_dict[feature] = float(importance)

    # Save model
    print("\n6. Saving model...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"   [OK] Model saved to: {output_path}")

    # Success message
    print(f"\n{'='*50}")
    print("[OK] Training completed successfully!")
    print(f"[OK] Accuracy: {accuracy:.2%}")
    print(f"{'='*50}")
    print("\nNext steps:")
    print("  1. Test the model: python example.py")
    print("  2. Use in your code: from metaguard import check_transaction")
    print("\nDeveloped by: Moslem Mohseni")

    return {
        "accuracy": float(accuracy),
        "n_samples": len(data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": importance_dict,
        "model_path": output_path,
    }


if __name__ == "__main__":
    train_model()
