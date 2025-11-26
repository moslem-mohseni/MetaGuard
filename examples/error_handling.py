#!/usr/bin/env python
"""
Error Handling Example

Author: Moslem Mohseni

This example demonstrates proper error handling with MetaGuard.
"""

from metaguard import SimpleDetector, check_transaction
from metaguard.utils.exceptions import (
    MetaGuardError,
    InvalidTransactionError,
    ModelNotFoundError,
)


def main():
    print("=" * 60)
    print("MetaGuard - Error Handling Example")
    print("=" * 60)

    # Example 1: Invalid transaction - negative amount
    print("\n1. Invalid Transaction: Negative Amount")
    print("-" * 40)

    try:
        result = check_transaction({
            "amount": -100,  # Invalid!
            "hour": 14,
            "user_age_days": 30,
            "transaction_count": 5,
        })
    except InvalidTransactionError as e:
        print(f"Error Type: {type(e).__name__}")
        print(f"Message: {e}")
        print(f"Field: {e.field}")
        print(f"Value: {e.value}")
        print(f"Reason: {e.reason}")

    # Example 2: Invalid transaction - invalid hour
    print("\n2. Invalid Transaction: Invalid Hour")
    print("-" * 40)

    try:
        result = check_transaction({
            "amount": 100,
            "hour": 25,  # Invalid! (0-23)
            "user_age_days": 30,
            "transaction_count": 5,
        })
    except InvalidTransactionError as e:
        print(f"Caught InvalidTransactionError")
        print(f"Field: {e.field}, Value: {e.value}")

    # Example 3: Invalid transaction - missing fields
    print("\n3. Invalid Transaction: Missing Fields")
    print("-" * 40)

    try:
        result = check_transaction({
            "amount": 100,
            # Missing: hour, user_age_days, transaction_count
        })
    except InvalidTransactionError as e:
        print(f"Caught InvalidTransactionError")
        print(f"Message: {e}")

    # Example 4: Model not found
    print("\n4. Model Not Found")
    print("-" * 40)

    try:
        detector = SimpleDetector(model_path="/nonexistent/path/model.pkl")
    except ModelNotFoundError as e:
        print(f"Error Type: {type(e).__name__}")
        print(f"Message: {e}")
        print(f"Model Path: {e.model_path}")

    # Example 5: Catch base exception
    print("\n5. Catching Base MetaGuardError")
    print("-" * 40)

    invalid_transactions = [
        {"amount": -100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
        {"amount": 100, "hour": 25, "user_age_days": 30, "transaction_count": 5},
        {"amount": 100, "hour": 14, "user_age_days": 0, "transaction_count": 5},
    ]

    for i, tx in enumerate(invalid_transactions, 1):
        try:
            result = check_transaction(tx)
        except MetaGuardError as e:
            print(f"Transaction {i}: Caught {type(e).__name__}")

    # Example 6: Safe detection function
    print("\n6. Safe Detection Function")
    print("-" * 40)

    def safe_check(transaction):
        """Safely check a transaction, returning error info on failure."""
        try:
            result = check_transaction(transaction)
            return {"success": True, "result": result}
        except InvalidTransactionError as e:
            return {
                "success": False,
                "error": "invalid_transaction",
                "field": e.field,
                "reason": e.reason,
            }
        except MetaGuardError as e:
            return {
                "success": False,
                "error": "metaguard_error",
                "message": str(e),
            }
        except Exception as e:
            return {
                "success": False,
                "error": "unexpected_error",
                "message": str(e),
            }

    test_transactions = [
        {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
        {"amount": -100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
    ]

    for tx in test_transactions:
        result = safe_check(tx)
        if result["success"]:
            print(f"Success: {result['result']['risk_level']}")
        else:
            print(f"Failed: {result['error']} - {result.get('reason', result.get('message'))}")

    # Example 7: Batch processing with error handling
    print("\n7. Batch Processing with Error Handling")
    print("-" * 40)

    transactions = [
        {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
        {"amount": -100, "hour": 14, "user_age_days": 30, "transaction_count": 5},  # Invalid
        {"amount": 200, "hour": 10, "user_age_days": 60, "transaction_count": 8},
        {"amount": 300, "hour": 25, "user_age_days": 90, "transaction_count": 3},  # Invalid
        {"amount": 400, "hour": 16, "user_age_days": 120, "transaction_count": 12},
    ]

    results = []
    errors = []

    for i, tx in enumerate(transactions):
        try:
            result = check_transaction(tx)
            results.append({"index": i, "result": result})
        except InvalidTransactionError as e:
            errors.append({"index": i, "field": e.field, "reason": e.reason})

    print(f"Successful: {len(results)}")
    print(f"Failed: {len(errors)}")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  Transaction {err['index']}: {err['field']} - {err['reason']}")

    print("\n" + "=" * 60)
    print("Error handling example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
