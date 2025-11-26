Getting Started Tutorial
========================

This tutorial will walk you through using MetaGuard for fraud detection.

Prerequisites
-------------

- Python 3.9 or later
- MetaGuard installed (``pip install metaguard``)

Step 1: Import MetaGuard
------------------------

.. code-block:: python

   from metaguard import check_transaction, SimpleDetector

Step 2: Check a Single Transaction
----------------------------------

The quickest way to check a transaction:

.. code-block:: python

   # Define transaction data
   transaction = {
       "amount": 100,           # Transaction amount
       "hour": 14,              # Hour of day (0-23)
       "user_age_days": 30,     # Account age in days
       "transaction_count": 5   # Recent transaction count
   }

   # Check for fraud
   result = check_transaction(transaction)

   # View results
   print(f"Is Suspicious: {result['is_suspicious']}")
   print(f"Risk Score: {result['risk_score']:.2%}")
   print(f"Risk Level: {result['risk_level']}")

Expected output:

.. code-block:: text

   Is Suspicious: False
   Risk Score: 12.50%
   Risk Level: Low

Step 3: Check a Suspicious Transaction
--------------------------------------

Let's check a transaction with suspicious characteristics:

.. code-block:: python

   suspicious = {
       "amount": 5000,          # High amount
       "hour": 3,               # Late night
       "user_age_days": 5,      # New account
       "transaction_count": 50  # High frequency
   }

   result = check_transaction(suspicious)

   print(f"Is Suspicious: {result['is_suspicious']}")
   print(f"Risk Score: {result['risk_score']:.2%}")
   print(f"Risk Level: {result['risk_level']}")

Expected output:

.. code-block:: text

   Is Suspicious: True
   Risk Score: 95.00%
   Risk Level: High

Step 4: Use the Detector Class
------------------------------

For more control, use ``SimpleDetector``:

.. code-block:: python

   # Create detector instance
   detector = SimpleDetector()

   # Check transaction
   result = detector.detect({
       "amount": 500,
       "hour": 10,
       "user_age_days": 60,
       "transaction_count": 8
   })

   print(result)

Step 5: Batch Processing
------------------------

Process multiple transactions at once:

.. code-block:: python

   transactions = [
       {"amount": 50, "hour": 10, "user_age_days": 200, "transaction_count": 3},
       {"amount": 3000, "hour": 2, "user_age_days": 10, "transaction_count": 40},
       {"amount": 100, "hour": 15, "user_age_days": 365, "transaction_count": 5},
   ]

   results = detector.batch_detect(transactions)

   for i, result in enumerate(results):
       status = "ALERT" if result["is_suspicious"] else "OK"
       print(f"Transaction {i+1}: [{status}] - {result['risk_level']}")

Expected output:

.. code-block:: text

   Transaction 1: [OK] - Low
   Transaction 2: [ALERT] - High
   Transaction 3: [OK] - Low

Step 6: Get Detailed Risk Analysis
----------------------------------

Use ``analyze_transaction_risk`` for factor breakdown:

.. code-block:: python

   from metaguard import analyze_transaction_risk

   result = analyze_transaction_risk({
       "amount": 3000,
       "hour": 2,
       "user_age_days": 10,
       "transaction_count": 40
   })

   print(f"Risk Score: {result['risk_score']:.1f}")
   print(f"Risk Level: {result['risk_level']}")
   print(f"\nRisk Factors:")
   for factor, active in result['factors'].items():
       status = "[X]" if active else "[ ]"
       print(f"  {status} {factor}")

Expected output:

.. code-block:: text

   Risk Score: 82.5
   Risk Level: High

   Risk Factors:
     [X] high_amount
     [X] new_account
     [X] high_frequency
     [X] unusual_hour

Step 7: Handle Errors
---------------------

Always handle potential errors:

.. code-block:: python

   from metaguard import check_transaction
   from metaguard.utils.exceptions import InvalidTransactionError

   try:
       result = check_transaction({
           "amount": -100,  # Invalid!
           "hour": 14,
           "user_age_days": 30,
           "transaction_count": 5
       })
   except InvalidTransactionError as e:
       print(f"Error: {e}")
       print(f"Field: {e.field}")
       print(f"Reason: {e.reason}")

Complete Example
----------------

Here's a complete script:

.. code-block:: python

   #!/usr/bin/env python
   """MetaGuard Getting Started Example"""

   from metaguard import (
       check_transaction,
       SimpleDetector,
       analyze_transaction_risk
   )
   from metaguard.utils.exceptions import InvalidTransactionError


   def main():
       print("=== MetaGuard Getting Started ===\n")

       # 1. Quick check
       print("1. Quick Transaction Check:")
       result = check_transaction({
           "amount": 100,
           "hour": 14,
           "user_age_days": 30,
           "transaction_count": 5
       })
       print(f"   Result: {result['risk_level']}\n")

       # 2. Detector class
       print("2. Using SimpleDetector:")
       detector = SimpleDetector()
       info = detector.get_model_info()
       print(f"   Model: {info['model_type']}\n")

       # 3. Batch processing
       print("3. Batch Processing:")
       transactions = [
           {"amount": 50, "hour": 10, "user_age_days": 200, "transaction_count": 3},
           {"amount": 3000, "hour": 2, "user_age_days": 10, "transaction_count": 40},
       ]
       results = detector.batch_detect(transactions)
       for i, r in enumerate(results):
           print(f"   Transaction {i+1}: {r['risk_level']}")
       print()

       # 4. Detailed analysis
       print("4. Detailed Risk Analysis:")
       analysis = analyze_transaction_risk({
           "amount": 3000,
           "hour": 2,
           "user_age_days": 10,
           "transaction_count": 40
       })
       print(f"   Active Factors: {analysis['active_factor_count']}")


   if __name__ == "__main__":
       main()

Next Steps
----------

- :doc:`custom_training` - Train your own model
- :doc:`deployment` - Deploy in production
- :doc:`../user_guide/advanced` - Advanced features
