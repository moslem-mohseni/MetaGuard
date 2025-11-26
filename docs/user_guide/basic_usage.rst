Basic Usage
===========

This section covers the basic usage patterns for MetaGuard.

Simple Transaction Check
------------------------

The easiest way to check a transaction:

.. code-block:: python

   from metaguard import check_transaction

   result = check_transaction({
       "amount": 100,
       "hour": 14,
       "user_age_days": 30,
       "transaction_count": 5
   })

   if result["is_suspicious"]:
       print(f"Warning! Risk level: {result['risk_level']}")
   else:
       print("Transaction appears safe")

Using SimpleDetector
--------------------

For more control over detection:

.. code-block:: python

   from metaguard import SimpleDetector

   # Initialize detector
   detector = SimpleDetector()

   # Detect single transaction
   result = detector.detect({
       "amount": 100,
       "hour": 14,
       "user_age_days": 30,
       "transaction_count": 5
   })

   print(f"Suspicious: {result['is_suspicious']}")
   print(f"Risk Score: {result['risk_score']:.2%}")
   print(f"Risk Level: {result['risk_level']}")

Batch Processing
----------------

Process multiple transactions efficiently:

.. code-block:: python

   from metaguard import SimpleDetector

   detector = SimpleDetector()

   transactions = [
       {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
       {"amount": 5000, "hour": 3, "user_age_days": 2, "transaction_count": 50},
       {"amount": 50, "hour": 10, "user_age_days": 365, "transaction_count": 3},
   ]

   results = detector.batch_detect(transactions)

   for i, result in enumerate(results):
       status = "SUSPICIOUS" if result["is_suspicious"] else "OK"
       print(f"Transaction {i+1}: [{status}] Risk: {result['risk_level']}")

Result Fields
-------------

Every detection result contains:

.. code-block:: python

   {
       "is_suspicious": bool,    # True if transaction is suspicious
       "risk_score": float,      # Score from 0.0 to 1.0
       "risk_level": str,        # "Low", "Medium", or "High"
   }

Risk Score Interpretation
^^^^^^^^^^^^^^^^^^^^^^^^^

- **0.0 - 0.4**: Low risk (normal transaction)
- **0.4 - 0.7**: Medium risk (monitor closely)
- **0.7 - 1.0**: High risk (likely fraudulent)

Getting Model Information
-------------------------

Inspect the loaded model:

.. code-block:: python

   from metaguard import SimpleDetector

   detector = SimpleDetector()
   info = detector.get_model_info()

   print(f"Model Type: {info['model_type']}")
   print(f"Model Path: {info['model_path']}")
   print(f"Risk Threshold: {info['risk_threshold']}")

Practical Examples
------------------

E-commerce Transaction Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from metaguard import check_transaction

   def process_purchase(user, cart):
       transaction = {
           "amount": cart.total,
           "hour": datetime.now().hour,
           "user_age_days": (datetime.now() - user.created_at).days,
           "transaction_count": user.transaction_count_today
       }

       result = check_transaction(transaction)

       if result["is_suspicious"]:
           # Require additional verification
           return {"status": "pending_verification", "reason": result["risk_level"]}

       return {"status": "approved"}

Real-time Monitoring
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from metaguard import SimpleDetector

   detector = SimpleDetector()

   def monitor_transactions(stream):
       for transaction in stream:
           result = detector.detect(transaction)

           if result["risk_score"] > 0.8:
               alert_security_team(transaction, result)
           elif result["risk_score"] > 0.5:
               log_warning(transaction, result)
