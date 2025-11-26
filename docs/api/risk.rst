Risk Module
===========

.. module:: metaguard.risk
   :synopsis: Risk calculation and analysis functions

The risk module provides functions for calculating and analyzing transaction risk.

Risk Calculation
----------------

calculate_risk
^^^^^^^^^^^^^^

.. autofunction:: metaguard.calculate_risk

   Calculate a risk score based on transaction parameters.

   **Example:**

   .. code-block:: python

      from metaguard import calculate_risk

      score = calculate_risk(
          amount=5000,
          user_age=5,
          transaction_count=50
      )
      print(f"Risk Score: {score}")  # Output: ~85.5

   :param amount: Transaction amount
   :type amount: float
   :param user_age: Account age in days
   :type user_age: int
   :param transaction_count: Number of transactions
   :type transaction_count: int
   :returns: Risk score (0-100)
   :rtype: float

get_risk_level
^^^^^^^^^^^^^^

.. autofunction:: metaguard.get_risk_level

   Convert a risk score to a categorical level.

   **Example:**

   .. code-block:: python

      from metaguard import get_risk_level

      level = get_risk_level(75)
      print(level)  # Output: "High"

   :param score: Risk score (0-100)
   :type score: float
   :returns: Risk level category
   :rtype: str

   **Risk Level Thresholds:**

   .. list-table::
      :header-rows: 1
      :widths: 20 30 50

      * - Level
        - Score Range
        - Description
      * - Low
        - 0 - 40
        - Transaction appears safe
      * - Medium
        - 40.1 - 70
        - Some risk indicators
      * - High
        - 70.1 - 100
        - High fraud probability

Risk Analysis
-------------

analyze_transaction_risk
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: metaguard.analyze_transaction_risk

   Perform detailed risk analysis with factor breakdown.

   **Example:**

   .. code-block:: python

      from metaguard import analyze_transaction_risk

      result = analyze_transaction_risk({
          "amount": 5000,
          "hour": 3,
          "user_age_days": 5,
          "transaction_count": 50
      })

      print(f"Score: {result['risk_score']}")
      print(f"Level: {result['risk_level']}")
      print(f"Factors: {result['factors']}")

   :param transaction: Transaction data dictionary
   :type transaction: dict
   :returns: Risk analysis result
   :rtype: dict

   **Return Format:**

   .. code-block:: python

      {
          "risk_score": float,         # Score (0-100)
          "risk_level": str,           # "Low", "Medium", "High"
          "factors": {
              "high_amount": bool,     # Amount > 1000
              "new_account": bool,     # Age < 30 days
              "high_frequency": bool,  # Count > 20
              "unusual_hour": bool     # Hour 0-5
          },
          "active_factor_count": int   # Number of active factors
      }

Advanced Functions
------------------

calculate_combined_risk
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: metaguard.risk.calculate_combined_risk

   Combine ML and formula-based risk scores.

   **Example:**

   .. code-block:: python

      from metaguard.risk import calculate_combined_risk

      combined = calculate_combined_risk(
          ml_score=0.8,
          formula_score=60,
          ml_weight=0.7
      )
      print(f"Combined: {combined}")  # Output: 74.0

   :param ml_score: ML model probability (0-1)
   :type ml_score: float
   :param formula_score: Formula-based score (0-100)
   :type formula_score: float
   :param ml_weight: Weight for ML score (default 0.7)
   :type ml_weight: float
   :returns: Combined risk score (0-100)
   :rtype: float

get_risk_factors_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: metaguard.risk.get_risk_factors_description

   Get human-readable descriptions for active risk factors.

   **Example:**

   .. code-block:: python

      from metaguard.risk import get_risk_factors_description

      factors = {"high_amount": True, "new_account": True}
      descriptions = get_risk_factors_description(factors)

      for desc in descriptions:
          print(f"- {desc}")

   :param factors: Dictionary of risk factors
   :type factors: dict
   :returns: List of descriptions for active factors
   :rtype: list[str]

Risk Factors
------------

MetaGuard analyzes these risk factors:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Factor
     - Threshold
     - Description
   * - high_amount
     - > $1,000
     - Large transaction amounts
   * - new_account
     - < 30 days
     - Recently created accounts
   * - high_frequency
     - > 20
     - High transaction frequency
   * - unusual_hour
     - 0-5 AM
     - Late night/early morning
