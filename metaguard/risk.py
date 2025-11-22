"""
MetaGuard - Risk Calculator
Author: Moslem Mohseni
Description: Simple risk calculation for metaverse transactions
"""

def calculate_risk(amount, user_age, transaction_count):
    """
    Calculate risk score for a transaction using a simple formula

    Formula:
    Risk = (Amount / 1000) * (5 / User_Age) * (Transaction_Count / 10)
    Normalized to 0-100 scale

    Parameters:
    - amount: Transaction amount in dollars
    - user_age: User account age in days
    - transaction_count: Number of previous transactions

    Returns:
    - Risk score (0-100)

    Author: Moslem Mohseni
    """
    risk = (amount / 1000) * (5 / max(user_age, 1)) * (transaction_count / 10)
    return min(100, risk * 10)  # Normalize to 0-100


def get_risk_level(risk_score):
    """
    Convert numeric risk score to categorical level

    Parameters:
    - risk_score: Numeric risk score (0-100)

    Returns:
    - Risk level: 'Low', 'Medium', or 'High'

    Author: Moslem Mohseni
    """
    if risk_score > 70:
        return 'High'
    elif risk_score > 40:
        return 'Medium'
    else:
        return 'Low'


def analyze_transaction_risk(transaction):
    """
    Analyze a transaction and return detailed risk information

    Parameters:
    - transaction: Dictionary with transaction details

    Returns:
    - Dictionary with risk analysis

    Author: Moslem Mohseni
    """
    amount = transaction.get('amount', 0)
    user_age = transaction.get('user_age_days', 1)
    tx_count = transaction.get('transaction_count', 0)

    risk_score = calculate_risk(amount, user_age, tx_count)
    risk_level = get_risk_level(risk_score)

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'factors': {
            'high_amount': amount > 1000,
            'new_account': user_age < 30,
            'high_frequency': tx_count > 20
        }
    }
