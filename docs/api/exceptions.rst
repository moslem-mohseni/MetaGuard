Exceptions
==========

.. module:: metaguard.utils.exceptions
   :synopsis: Custom exceptions for MetaGuard

MetaGuard defines a hierarchy of exceptions for error handling.

Exception Hierarchy
-------------------

.. code-block:: text

   Exception
   └── MetaGuardError (base)
       ├── ModelNotFoundError
       ├── ModelLoadError
       ├── InvalidTransactionError
       ├── ValidationError
       └── ConfigurationError

Base Exception
--------------

MetaGuardError
^^^^^^^^^^^^^^

.. autoexception:: metaguard.utils.exceptions.MetaGuardError
   :show-inheritance:

   Base exception for all MetaGuard errors.

   **Example:**

   .. code-block:: python

      from metaguard.utils.exceptions import MetaGuardError

      try:
          # MetaGuard operations
          pass
      except MetaGuardError as e:
          print(f"MetaGuard error: {e}")

Model Exceptions
----------------

ModelNotFoundError
^^^^^^^^^^^^^^^^^^

.. autoexception:: metaguard.utils.exceptions.ModelNotFoundError
   :show-inheritance:

   Raised when the model file cannot be found.

   **Example:**

   .. code-block:: python

      from metaguard import SimpleDetector
      from metaguard.utils.exceptions import ModelNotFoundError

      try:
          detector = SimpleDetector(model_path="/nonexistent/model.pkl")
      except ModelNotFoundError as e:
          print(f"Model not found: {e.model_path}")

   **Attributes:**

   - ``model_path``: The path that was not found

ModelLoadError
^^^^^^^^^^^^^^

.. autoexception:: metaguard.utils.exceptions.ModelLoadError
   :show-inheritance:

   Raised when the model file exists but cannot be loaded.

   **Example:**

   .. code-block:: python

      from metaguard import SimpleDetector
      from metaguard.utils.exceptions import ModelLoadError

      try:
          detector = SimpleDetector(model_path="/path/to/corrupted.pkl")
      except ModelLoadError as e:
          print(f"Failed to load model: {e}")
          if e.original_error:
              print(f"Cause: {e.original_error}")

   **Attributes:**

   - ``model_path``: Path to the model file
   - ``original_error``: The underlying exception

Transaction Exceptions
----------------------

InvalidTransactionError
^^^^^^^^^^^^^^^^^^^^^^^

.. autoexception:: metaguard.utils.exceptions.InvalidTransactionError
   :show-inheritance:

   Raised when transaction data is invalid.

   **Example:**

   .. code-block:: python

      from metaguard import check_transaction
      from metaguard.utils.exceptions import InvalidTransactionError

      try:
          result = check_transaction({
              "amount": -100,  # Invalid: negative
              "hour": 25,      # Invalid: > 23
          })
      except InvalidTransactionError as e:
          print(f"Invalid transaction: {e}")
          print(f"Field: {e.field}")
          print(f"Value: {e.value}")
          print(f"Reason: {e.reason}")

   **Attributes:**

   - ``field``: The field that caused the error
   - ``value``: The invalid value
   - ``reason``: Description of why it's invalid

ValidationError
^^^^^^^^^^^^^^^

.. autoexception:: metaguard.utils.exceptions.ValidationError
   :show-inheritance:

   Raised for general validation errors.

   **Example:**

   .. code-block:: python

      from metaguard.utils.exceptions import ValidationError

      try:
          # Validation operation
          pass
      except ValidationError as e:
          for error in e.errors:
              print(f"- {error}")

   **Attributes:**

   - ``errors``: List of validation error messages

Configuration Exceptions
------------------------

ConfigurationError
^^^^^^^^^^^^^^^^^^

.. autoexception:: metaguard.utils.exceptions.ConfigurationError
   :show-inheritance:

   Raised when configuration is invalid.

   **Example:**

   .. code-block:: python

      from metaguard.utils.config import MetaGuardConfig
      from metaguard.utils.exceptions import ConfigurationError

      try:
          config = MetaGuardConfig(risk_threshold=1.5)  # Invalid: > 1.0
      except ConfigurationError as e:
          print(f"Configuration error: {e}")
          print(f"Key: {e.config_key}")

   **Attributes:**

   - ``config_key``: The configuration key that caused the error

Error Handling Best Practices
-----------------------------

Catch Specific Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from metaguard import SimpleDetector, check_transaction
   from metaguard.utils.exceptions import (
       MetaGuardError,
       ModelNotFoundError,
       InvalidTransactionError,
   )

   def safe_detection(transaction):
       try:
           return check_transaction(transaction)
       except InvalidTransactionError as e:
           # Handle bad input
           return {"error": f"Invalid input: {e.field}"}
       except MetaGuardError as e:
           # Handle other MetaGuard errors
           return {"error": str(e)}

Log Errors with Context
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import logging
   from metaguard import check_transaction
   from metaguard.utils.exceptions import InvalidTransactionError

   logger = logging.getLogger(__name__)

   def process_transaction(transaction):
       try:
           return check_transaction(transaction)
       except InvalidTransactionError as e:
           logger.error(
               "Invalid transaction",
               extra={
                   "field": e.field,
                   "value": e.value,
                   "reason": e.reason,
               }
           )
           raise
