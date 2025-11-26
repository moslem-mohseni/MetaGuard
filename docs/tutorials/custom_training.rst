Custom Model Training
=====================

Learn how to train MetaGuard with your own data.

Overview
--------

MetaGuard comes with a pre-trained model, but you can train your own:

1. Prepare your transaction data
2. Generate features
3. Train a model
4. Use the custom model

Prerequisites
-------------

- MetaGuard with dev dependencies: ``pip install metaguard[dev]``
- Transaction data in CSV format

Step 1: Prepare Your Data
-------------------------

Your data needs these columns:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - amount
     - float
     - Transaction amount
   * - hour
     - int
     - Hour of day (0-23)
   * - user_age_days
     - int
     - Account age in days
   * - transaction_count
     - int
     - Recent transactions
   * - is_fraud
     - int
     - Target: 1 for fraud, 0 for normal

Example CSV:

.. code-block:: text

   amount,hour,user_age_days,transaction_count,is_fraud
   100,14,30,5,0
   5000,3,5,50,1
   50,10,365,3,0
   3000,2,10,40,1

Step 2: Load and Explore Data
-----------------------------

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Load your data
   df = pd.read_csv("transactions.csv")

   # Check the data
   print(f"Total transactions: {len(df)}")
   print(f"Fraud cases: {df['is_fraud'].sum()}")
   print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

   # View distribution
   print("\nFeature statistics:")
   print(df.describe())

Step 3: Split the Data
----------------------

.. code-block:: python

   from sklearn.model_selection import train_test_split

   # Features and target
   X = df[['amount', 'hour', 'user_age_days', 'transaction_count']]
   y = df['is_fraud']

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   print(f"Training samples: {len(X_train)}")
   print(f"Test samples: {len(X_test)}")

Step 4: Train the Model
-----------------------

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report, roc_auc_score

   # Create and train model
   model = RandomForestClassifier(
       n_estimators=100,
       max_depth=10,
       min_samples_split=5,
       min_samples_leaf=2,
       random_state=42,
       n_jobs=-1
   )

   model.fit(X_train, y_train)

   # Evaluate
   y_pred = model.predict(X_test)
   y_proba = model.predict_proba(X_test)[:, 1]

   print("Classification Report:")
   print(classification_report(y_test, y_pred))
   print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

Step 5: Save the Model
----------------------

.. code-block:: python

   import pickle
   from pathlib import Path

   # Create models directory
   model_dir = Path("models")
   model_dir.mkdir(exist_ok=True)

   # Save model
   model_path = model_dir / "custom_model.pkl"
   with open(model_path, "wb") as f:
       pickle.dump(model, f)

   print(f"Model saved to: {model_path}")

Step 6: Use Your Custom Model
-----------------------------

.. code-block:: python

   from metaguard import SimpleDetector

   # Load with custom model
   detector = SimpleDetector(model_path="models/custom_model.pkl")

   # Verify
   info = detector.get_model_info()
   print(f"Model type: {info['model_type']}")
   print(f"Model path: {info['model_path']}")

   # Test detection
   result = detector.detect({
       "amount": 1000,
       "hour": 3,
       "user_age_days": 10,
       "transaction_count": 25
   })
   print(f"Result: {result}")

Using the Training Script
-------------------------

MetaGuard includes a training script:

.. code-block:: bash

   # Generate synthetic data
   python scripts/generate_data.py --output data/transactions.csv --count 10000

   # Train model
   python scripts/train.py --data data/transactions.csv --output models/model.pkl

Complete Training Script
------------------------

Here's a complete training script:

.. code-block:: python

   #!/usr/bin/env python
   """Train a custom MetaGuard model."""

   import argparse
   import pickle
   from pathlib import Path

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split, cross_val_score
   from sklearn.metrics import classification_report, roc_auc_score


   def load_data(path: str) -> pd.DataFrame:
       """Load transaction data from CSV."""
       df = pd.read_csv(path)
       required = ['amount', 'hour', 'user_age_days', 'transaction_count', 'is_fraud']
       missing = set(required) - set(df.columns)
       if missing:
           raise ValueError(f"Missing columns: {missing}")
       return df


   def train_model(df: pd.DataFrame) -> RandomForestClassifier:
       """Train the fraud detection model."""
       X = df[['amount', 'hour', 'user_age_days', 'transaction_count']]
       y = df['is_fraud']

       # Split
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.2, random_state=42, stratify=y
       )

       # Train
       model = RandomForestClassifier(
           n_estimators=100,
           max_depth=10,
           random_state=42,
           n_jobs=-1
       )
       model.fit(X_train, y_train)

       # Evaluate
       y_pred = model.predict(X_test)
       y_proba = model.predict_proba(X_test)[:, 1]

       print("\n=== Model Evaluation ===")
       print(classification_report(y_test, y_pred))
       print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

       # Cross-validation
       cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
       print(f"CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

       return model


   def save_model(model: RandomForestClassifier, path: str) -> None:
       """Save model to pickle file."""
       Path(path).parent.mkdir(parents=True, exist_ok=True)
       with open(path, "wb") as f:
           pickle.dump(model, f)
       print(f"\nModel saved to: {path}")


   def main():
       parser = argparse.ArgumentParser(description="Train MetaGuard model")
       parser.add_argument("--data", required=True, help="Path to training data CSV")
       parser.add_argument("--output", default="models/custom_model.pkl",
                           help="Output model path")
       args = parser.parse_args()

       print("Loading data...")
       df = load_data(args.data)
       print(f"Loaded {len(df)} transactions")

       print("\nTraining model...")
       model = train_model(df)

       save_model(model, args.output)


   if __name__ == "__main__":
       main()

Tips for Better Models
----------------------

1. **Balance Your Data**: Use oversampling or class weights for imbalanced data
2. **Feature Engineering**: Add derived features
3. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
4. **Regular Retraining**: Retrain periodically with new data

Example with class weights:

.. code-block:: python

   model = RandomForestClassifier(
       n_estimators=100,
       class_weight='balanced',  # Handle imbalanced data
       random_state=42
   )

Next Steps
----------

- :doc:`deployment` - Deploy your trained model
- :doc:`../user_guide/advanced` - Advanced configuration
