Advanced Usage
==============

This section covers advanced MetaGuard features.

Custom Model Path
-----------------

Load a custom trained model:

.. code-block:: python

   from metaguard import SimpleDetector

   # Load custom model
   detector = SimpleDetector(model_path="/path/to/custom_model.pkl")

   result = detector.detect({
       "amount": 100,
       "hour": 14,
       "user_age_days": 30,
       "transaction_count": 5
   })

Risk Factor Analysis
--------------------

Get detailed breakdown of risk factors:

.. code-block:: python

   from metaguard import analyze_transaction_risk

   result = analyze_transaction_risk({
       "amount": 5000,
       "hour": 3,
       "user_age_days": 5,
       "transaction_count": 50
   })

   print(f"Overall Risk Score: {result['risk_score']}")
   print(f"Risk Level: {result['risk_level']}")
   print(f"Number of Active Factors: {result['active_factor_count']}")
   print("\nRisk Factors:")
   for factor, is_active in result['factors'].items():
       status = "[X]" if is_active else "[ ]"
       print(f"  {status} {factor}")

Output:

.. code-block:: text

   Overall Risk Score: 85.5
   Risk Level: High
   Number of Active Factors: 4

   Risk Factors:
     [X] high_amount
     [X] new_account
     [X] high_frequency
     [X] unusual_hour

Using Risk Calculation Functions
--------------------------------

Calculate risk scores directly:

.. code-block:: python

   from metaguard import calculate_risk, get_risk_level

   # Calculate risk score
   score = calculate_risk(
       amount=5000,
       user_age=5,
       transaction_count=50
   )
   print(f"Risk Score: {score}")

   # Get risk level from score
   level = get_risk_level(score)
   print(f"Risk Level: {level}")

Combined Risk Scoring
---------------------

Combine ML and formula-based risk scores:

.. code-block:: python

   from metaguard.risk import calculate_combined_risk

   # Combine ML prediction with formula score
   combined = calculate_combined_risk(
       ml_score=0.8,          # ML model probability
       formula_score=60,      # Formula-based score
       ml_weight=0.7          # Weight for ML (0.3 for formula)
   )
   print(f"Combined Risk: {combined}")

Custom Risk Thresholds
----------------------

Configure detection thresholds via configuration:

.. code-block:: python

   from metaguard import SimpleDetector
   from metaguard.utils.config import MetaGuardConfig

   # Create custom configuration
   config = MetaGuardConfig(
       risk_threshold=0.6,    # Lower threshold = more sensitive
       ml_weight=0.8,         # Higher ML weight
       log_level="DEBUG"
   )

   # Create detector with custom config
   detector = SimpleDetector(config=config)

   result = detector.detect({
       "amount": 500,
       "hour": 12,
       "user_age_days": 60,
       "transaction_count": 15
   })

Global Configuration
--------------------

Set global configuration for all detectors:

.. code-block:: python

   from metaguard import SimpleDetector
   from metaguard.utils.config import set_config, reset_config, MetaGuardConfig

   # Set global config
   set_config(MetaGuardConfig(risk_threshold=0.5))

   # All detectors will use this config
   detector1 = SimpleDetector()  # Uses global config
   detector2 = SimpleDetector()  # Uses global config

   # Reset to defaults
   reset_config()

Environment Variables
---------------------

Configure MetaGuard via environment variables:

.. code-block:: bash

   export METAGUARD_MODEL_PATH=/path/to/model.pkl
   export METAGUARD_RISK_THRESHOLD=0.6
   export METAGUARD_ML_WEIGHT=0.8
   export METAGUARD_LOG_LEVEL=DEBUG

.. code-block:: python

   from metaguard import SimpleDetector

   # Detector automatically picks up environment variables
   detector = SimpleDetector()

Error Handling
--------------

Handle specific exceptions:

.. code-block:: python

   from metaguard import SimpleDetector, check_transaction
   from metaguard.utils.exceptions import (
       InvalidTransactionError,
       ModelNotFoundError,
       ModelLoadError,
       ValidationError,
   )

   # Handle invalid transaction data
   try:
       result = check_transaction({
           "amount": -100,  # Invalid
           "hour": 25,      # Invalid
       })
   except InvalidTransactionError as e:
       print(f"Invalid transaction: {e}")

   # Handle model errors
   try:
       detector = SimpleDetector(model_path="/nonexistent/model.pkl")
   except ModelNotFoundError as e:
       print(f"Model not found: {e}")
   except ModelLoadError as e:
       print(f"Failed to load model: {e}")

Logging
-------

Enable detailed logging for debugging:

.. code-block:: python

   from metaguard.utils.logging import setup_logging

   # Setup logging with JSON output
   logger = setup_logging(level="DEBUG", json_format=True)

   # Or setup with file output
   logger = setup_logging(
       level="INFO",
       log_file="/var/log/metaguard.log"
   )

   # Now all MetaGuard operations will be logged
   from metaguard import check_transaction
   result = check_transaction({...})

Custom Logger
^^^^^^^^^^^^^

Get a logger for your application:

.. code-block:: python

   from metaguard.utils.logging import get_logger, LoggerAdapter

   # Get a namespaced logger
   logger = get_logger("myapp")

   # Add context to log messages
   adapter = LoggerAdapter(logger, {"request_id": "abc123"})
   adapter.info("Processing transaction")

Performance Optimization
------------------------

For high-throughput scenarios:

.. code-block:: python

   from metaguard import SimpleDetector

   # Create detector once, reuse for all detections
   detector = SimpleDetector()

   def process_batch(transactions):
       # Use batch_detect for efficiency
       return detector.batch_detect(transactions)

   # Process in batches
   all_transactions = [...]  # Large list
   batch_size = 100

   for i in range(0, len(all_transactions), batch_size):
       batch = all_transactions[i:i+batch_size]
       results = process_batch(batch)
       # Process results...
