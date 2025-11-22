"""
MetaGuard - Fraud Detector
Author: Moslem Mohseni
Description: ML-based fraud detection for metaverse transactions
"""

import pickle
import pandas as pd
import os
from .risk import get_risk_level

class SimpleDetector:
    """
    Simple fraud detector using pre-trained ML model

    Author: Moslem Mohseni
    """

    def __init__(self, model_path='metaguard/models/model.pkl'):
        """
        Initialize detector with pre-trained model

        Parameters:
        - model_path: Path to the trained model file

        Author: Moslem Mohseni
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load the pre-trained model from file

        Author: Moslem Mohseni
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Please train the model first by running: python scripts/train.py"
            )

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def detect(self, transaction):
        """
        Detect if a transaction is suspicious

        Parameters:
        - transaction: Dictionary with keys:
            - amount: Transaction amount
            - hour: Hour of transaction (0-23)
            - user_age_days: User account age in days
            - transaction_count: Number of previous transactions

        Returns:
        - Dictionary with detection results:
            - is_suspicious: Boolean
            - risk_score: Probability of fraud (0-1)
            - risk_level: 'Low', 'Medium', or 'High'

        Author: Moslem Mohseni
        """
        # Convert to DataFrame with correct feature order
        df = pd.DataFrame([{
            'amount': transaction.get('amount', 0),
            'hour': transaction.get('hour', 0),
            'user_age_days': transaction.get('user_age_days', 1),
            'transaction_count': transaction.get('transaction_count', 0)
        }])

        # Get prediction probability
        prob = self.model.predict_proba(df)[0][1]

        # Return results
        return {
            'is_suspicious': prob > 0.5,
            'risk_score': float(prob),
            'risk_level': get_risk_level(prob * 100)
        }

    def batch_detect(self, transactions):
        """
        Detect fraud in multiple transactions

        Parameters:
        - transactions: List of transaction dictionaries

        Returns:
        - List of detection results

        Author: Moslem Mohseni
        """
        return [self.detect(tx) for tx in transactions]


def check_transaction(transaction):
    """
    Quick helper function for single transaction checking

    Parameters:
    - transaction: Transaction dictionary

    Returns:
    - Detection result dictionary

    Author: Moslem Mohseni
    """
    detector = SimpleDetector()
    return detector.detect(transaction)
