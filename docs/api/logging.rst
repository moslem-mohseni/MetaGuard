Logging Module
==============

.. module:: metaguard.utils.logging
   :synopsis: Logging utilities for MetaGuard

The logging module provides structured logging support.

Setup Functions
---------------

setup_logging
^^^^^^^^^^^^^

.. autofunction:: metaguard.utils.logging.setup_logging

   Configure logging for MetaGuard.

   **Example:**

   .. code-block:: python

      from metaguard.utils.logging import setup_logging

      # Basic setup
      logger = setup_logging()

      # With custom level
      logger = setup_logging(level="DEBUG")

      # With JSON format
      logger = setup_logging(json_format=True)

      # With file output
      logger = setup_logging(log_file="/var/log/metaguard.log")

   :param level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   :type level: str
   :param json_format: Use JSON formatter
   :type json_format: bool
   :param log_file: Path to log file
   :type log_file: str, optional
   :returns: Configured logger
   :rtype: logging.Logger

get_logger
^^^^^^^^^^

.. autofunction:: metaguard.utils.logging.get_logger

   Get a logger with MetaGuard namespace.

   **Example:**

   .. code-block:: python

      from metaguard.utils.logging import get_logger

      logger = get_logger("mymodule")
      # Logger name: "metaguard.mymodule"

      logger.info("Processing transaction")
      logger.error("Transaction failed", exc_info=True)

   :param name: Logger name (will be prefixed with "metaguard.")
   :type name: str
   :param level: Optional log level
   :type level: str, optional
   :returns: Logger instance
   :rtype: logging.Logger

Formatters
----------

JSONFormatter
^^^^^^^^^^^^^

.. autoclass:: metaguard.utils.logging.JSONFormatter
   :show-inheritance:

   Formats log records as JSON for structured logging.

   **Example:**

   .. code-block:: python

      import logging
      from metaguard.utils.logging import JSONFormatter

      handler = logging.StreamHandler()
      handler.setFormatter(JSONFormatter())

      logger = logging.getLogger("myapp")
      logger.addHandler(handler)
      logger.info("Test message")

   **Output:**

   .. code-block:: json

      {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "level": "INFO",
        "logger": "myapp",
        "message": "Test message",
        "module": "mymodule",
        "function": "myfunction",
        "line": 42
      }

ColoredFormatter
^^^^^^^^^^^^^^^^

.. autoclass:: metaguard.utils.logging.ColoredFormatter
   :show-inheritance:

   Formats log records with ANSI color codes.

   **Example:**

   .. code-block:: python

      import logging
      from metaguard.utils.logging import ColoredFormatter

      handler = logging.StreamHandler()
      handler.setFormatter(ColoredFormatter(
          fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
      ))

      logger = logging.getLogger("myapp")
      logger.addHandler(handler)

   **Color Scheme:**

   - DEBUG: Cyan
   - INFO: Green
   - WARNING: Yellow
   - ERROR: Red
   - CRITICAL: Bold Red

LoggerAdapter
-------------

.. autoclass:: metaguard.utils.logging.LoggerAdapter
   :show-inheritance:

   Adds extra context to log messages.

   **Example:**

   .. code-block:: python

      from metaguard.utils.logging import get_logger, LoggerAdapter

      base_logger = get_logger("api")
      adapter = LoggerAdapter(base_logger, {
          "request_id": "abc123",
          "user_id": "user456"
      })

      adapter.info("Processing request")
      # Log includes: request_id=abc123, user_id=user456

   :param logger: Base logger
   :type logger: logging.Logger
   :param extra: Extra context to add
   :type extra: dict

Usage Examples
--------------

Basic Logging
^^^^^^^^^^^^^

.. code-block:: python

   from metaguard.utils.logging import setup_logging, get_logger

   # Setup once at application start
   setup_logging(level="INFO")

   # Get logger for each module
   logger = get_logger(__name__)

   logger.debug("Debug message")
   logger.info("Info message")
   logger.warning("Warning message")
   logger.error("Error message")

JSON Logging for Production
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from metaguard.utils.logging import setup_logging

   setup_logging(
       level="INFO",
       json_format=True,
       log_file="/var/log/metaguard/app.log"
   )

Request Tracing
^^^^^^^^^^^^^^^

.. code-block:: python

   from metaguard.utils.logging import get_logger, LoggerAdapter

   def process_request(request):
       logger = get_logger("api")
       adapter = LoggerAdapter(logger, {
           "request_id": request.id,
           "user_id": request.user.id,
       })

       adapter.info("Request started")
       try:
           result = do_work()
           adapter.info("Request completed")
           return result
       except Exception as e:
           adapter.error(f"Request failed: {e}")
           raise

Integration with Detector
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from metaguard import SimpleDetector
   from metaguard.utils.logging import setup_logging

   # Enable debug logging
   setup_logging(level="DEBUG")

   detector = SimpleDetector()
   # Now all internal operations are logged
   result = detector.detect({...})
