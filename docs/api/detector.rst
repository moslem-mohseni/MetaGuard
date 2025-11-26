Detector Module
===============

.. module:: metaguard.detector
   :synopsis: Fraud detection classes and functions

The detector module contains the main fraud detection functionality.

SimpleDetector Class
--------------------

.. autoclass:: metaguard.SimpleDetector
   :members:
   :undoc-members:
   :show-inheritance:

   The main class for detecting fraudulent transactions.

   **Example:**

   .. code-block:: python

      from metaguard import SimpleDetector

      detector = SimpleDetector()

      result = detector.detect({
          "amount": 100,
          "hour": 14,
          "user_age_days": 30,
          "transaction_count": 5
      })

      print(f"Suspicious: {result['is_suspicious']}")

   **Attributes:**

   .. attribute:: model
      :type: sklearn.base.BaseEstimator

      The loaded ML model used for predictions.

   .. attribute:: model_path
      :type: pathlib.Path

      Path to the loaded model file.

   .. attribute:: config
      :type: MetaGuardConfig

      Current configuration settings.

   **Methods:**

   .. method:: detect(transaction: dict) -> dict

      Detect if a single transaction is fraudulent.

      :param transaction: Transaction data dictionary
      :type transaction: dict
      :returns: Detection result with is_suspicious, risk_score, risk_level
      :rtype: dict
      :raises InvalidTransactionError: If transaction data is invalid

   .. method:: batch_detect(transactions: list) -> list

      Detect fraud in multiple transactions.

      :param transactions: List of transaction dictionaries
      :type transactions: list[dict]
      :returns: List of detection results
      :rtype: list[dict]

   .. method:: get_model_info() -> dict

      Get information about the loaded model.

      :returns: Model information dictionary
      :rtype: dict

check_transaction Function
--------------------------

.. autofunction:: metaguard.check_transaction

   Quick check for a single transaction.

   **Example:**

   .. code-block:: python

      from metaguard import check_transaction

      result = check_transaction({
          "amount": 5000,
          "hour": 3,
          "user_age_days": 5,
          "transaction_count": 50
      })

      if result["is_suspicious"]:
          print("Warning: Suspicious transaction!")

   :param transaction: Transaction data dictionary
   :type transaction: dict
   :param model_path: Optional path to custom model
   :type model_path: str, optional
   :returns: Detection result
   :rtype: dict
   :raises InvalidTransactionError: If transaction data is invalid

Transaction Data Format
-----------------------

All transaction dictionaries must contain:

.. code-block:: python

   {
       "amount": float,          # Transaction amount (> 0)
       "hour": int,              # Hour of day (0-23)
       "user_age_days": int,     # Account age in days (>= 1)
       "transaction_count": int  # Transaction count (>= 0)
   }

Detection Result Format
-----------------------

All detection functions return:

.. code-block:: python

   {
       "is_suspicious": bool,   # True if flagged as fraud
       "risk_score": float,     # Probability score (0.0-1.0)
       "risk_level": str        # "Low", "Medium", or "High"
   }
