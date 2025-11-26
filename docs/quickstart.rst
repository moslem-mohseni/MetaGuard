Quick Start Guide
=================

This guide will get you started with MetaGuard in 5 minutes.

Basic Usage
-----------

The simplest way to use MetaGuard is with the ``check_transaction`` function:

.. code-block:: python

   from metaguard import check_transaction

   # Check a single transaction
   result = check_transaction({
       "amount": 5000,
       "hour": 3,
       "user_age_days": 5,
       "transaction_count": 50
   })

   print(f"Is Suspicious: {result['is_suspicious']}")
   print(f"Risk Score: {result['risk_score']:.2f}")
   print(f"Risk Level: {result['risk_level']}")

Output:

.. code-block:: text

   Is Suspicious: True
   Risk Score: 0.85
   Risk Level: High

Transaction Fields
------------------

Each transaction requires these fields:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``amount``
     - float
     - Transaction amount (must be > 0)
   * - ``hour``
     - int
     - Hour of day (0-23)
   * - ``user_age_days``
     - int
     - Account age in days (must be >= 1)
   * - ``transaction_count``
     - int
     - Number of transactions in period (>= 0)

Using the Detector Class
------------------------

For more control, use the ``SimpleDetector`` class:

.. code-block:: python

   from metaguard import SimpleDetector

   # Create detector instance
   detector = SimpleDetector()

   # Single detection
   transaction = {
       "amount": 100,
       "hour": 14,
       "user_age_days": 30,
       "transaction_count": 5
   }
   result = detector.detect(transaction)
   print(f"Result: {result}")

   # Batch detection
   transactions = [
       {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
       {"amount": 5000, "hour": 3, "user_age_days": 2, "transaction_count": 50},
       {"amount": 50, "hour": 10, "user_age_days": 365, "transaction_count": 3},
   ]
   results = detector.batch_detect(transactions)

   for i, result in enumerate(results):
       print(f"Transaction {i+1}: {result['risk_level']}")

Risk Analysis
-------------

Get detailed risk analysis with factor breakdown:

.. code-block:: python

   from metaguard import analyze_transaction_risk

   result = analyze_transaction_risk({
       "amount": 5000,
       "hour": 3,
       "user_age_days": 5,
       "transaction_count": 50
   })

   print(f"Risk Score: {result['risk_score']}")
   print(f"Risk Level: {result['risk_level']}")
   print(f"Active Factors: {result['active_factor_count']}")
   print("Factors:")
   for factor, active in result['factors'].items():
       status = "Yes" if active else "No"
       print(f"  - {factor}: {status}")

Output:

.. code-block:: text

   Risk Score: 85.5
   Risk Level: High
   Active Factors: 4
   Factors:
     - high_amount: Yes
     - new_account: Yes
     - high_frequency: Yes
     - unusual_hour: Yes

Understanding Results
---------------------

Risk Levels
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Level
     - Score Range
     - Meaning
   * - Low
     - 0-40
     - Transaction appears safe
   * - Medium
     - 40-70
     - Some risk indicators present
   * - High
     - 70-100
     - High probability of fraud

Risk Factors
^^^^^^^^^^^^

MetaGuard analyzes these risk factors:

- **high_amount**: Transaction amount > $1000
- **new_account**: Account age < 30 days
- **high_frequency**: > 20 transactions in period
- **unusual_hour**: Transaction between midnight and 6 AM

Error Handling
--------------

MetaGuard raises specific exceptions for invalid input:

.. code-block:: python

   from metaguard import check_transaction
   from metaguard.utils.exceptions import InvalidTransactionError

   try:
       result = check_transaction({
           "amount": -100,  # Invalid: negative amount
           "hour": 14,
           "user_age_days": 30,
           "transaction_count": 5
       })
   except InvalidTransactionError as e:
       print(f"Invalid transaction: {e}")

Next Steps
----------

- :doc:`user_guide/index` - Detailed usage guide
- :doc:`api/index` - API reference
- :doc:`tutorials/index` - Step-by-step tutorials
