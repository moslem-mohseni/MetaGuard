Configuration Module
====================

.. module:: metaguard.utils.config
   :synopsis: Configuration management for MetaGuard

The config module provides configuration management for MetaGuard.

MetaGuardConfig Class
---------------------

.. autoclass:: metaguard.utils.config.MetaGuardConfig
   :members:
   :undoc-members:
   :show-inheritance:

   Configuration settings for MetaGuard.

   **Example:**

   .. code-block:: python

      from metaguard.utils.config import MetaGuardConfig

      config = MetaGuardConfig(
          risk_threshold=0.6,
          ml_weight=0.8,
          log_level="DEBUG"
      )

   **Parameters:**

   .. attribute:: model_path
      :type: str | None

      Path to the model file. If None, uses built-in model.

   .. attribute:: risk_threshold
      :type: float

      Threshold for marking transactions suspicious (0.0-1.0).
      Default: 0.5

   .. attribute:: ml_weight
      :type: float

      Weight for ML score in combined calculation (0.0-1.0).
      Default: 0.7

   .. attribute:: log_level
      :type: str

      Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.
      Default: "INFO"

   .. attribute:: batch_size
      :type: int

      Maximum batch size for batch processing.
      Default: 1000

Global Configuration Functions
------------------------------

get_config
^^^^^^^^^^

.. autofunction:: metaguard.utils.config.get_config

   Get the current global configuration.

   **Example:**

   .. code-block:: python

      from metaguard.utils.config import get_config

      config = get_config()
      print(f"Risk threshold: {config.risk_threshold}")

   :returns: Current configuration
   :rtype: MetaGuardConfig

set_config
^^^^^^^^^^

.. autofunction:: metaguard.utils.config.set_config

   Set the global configuration.

   **Example:**

   .. code-block:: python

      from metaguard.utils.config import set_config, MetaGuardConfig

      set_config(MetaGuardConfig(
          risk_threshold=0.4,
          log_level="DEBUG"
      ))

   :param config: Configuration to set
   :type config: MetaGuardConfig

reset_config
^^^^^^^^^^^^

.. autofunction:: metaguard.utils.config.reset_config

   Reset configuration to defaults.

   **Example:**

   .. code-block:: python

      from metaguard.utils.config import reset_config

      reset_config()

Environment Variables
---------------------

MetaGuardConfig automatically reads from environment variables:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Type
     - Description
   * - ``METAGUARD_MODEL_PATH``
     - str
     - Path to model file
   * - ``METAGUARD_RISK_THRESHOLD``
     - float
     - Risk threshold (0.0-1.0)
   * - ``METAGUARD_ML_WEIGHT``
     - float
     - ML weight (0.0-1.0)
   * - ``METAGUARD_LOG_LEVEL``
     - str
     - Log level
   * - ``METAGUARD_BATCH_SIZE``
     - int
     - Batch size

**Example:**

.. code-block:: bash

   export METAGUARD_RISK_THRESHOLD=0.6
   export METAGUARD_LOG_LEVEL=DEBUG

.. code-block:: python

   from metaguard.utils.config import get_config

   config = get_config()
   # config.risk_threshold == 0.6
   # config.log_level == "DEBUG"

Validation
----------

Configuration values are validated using Pydantic:

.. code-block:: python

   from metaguard.utils.config import MetaGuardConfig
   from pydantic import ValidationError

   try:
       config = MetaGuardConfig(
           risk_threshold=1.5,  # Invalid: > 1.0
           ml_weight=-0.1,      # Invalid: < 0.0
       )
   except ValidationError as e:
       print(e)

**Valid Ranges:**

- ``risk_threshold``: 0.0 to 1.0
- ``ml_weight``: 0.0 to 1.0
- ``batch_size``: 1 to 100000
- ``log_level``: DEBUG, INFO, WARNING, ERROR, CRITICAL
