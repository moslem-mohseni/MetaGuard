Configuration
=============

MetaGuard supports multiple configuration methods.

Configuration Options
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Option
     - Type
     - Default
     - Description
   * - ``model_path``
     - str
     - None
     - Path to model file (uses built-in if not set)
   * - ``risk_threshold``
     - float
     - 0.5
     - Threshold for marking transactions suspicious
   * - ``ml_weight``
     - float
     - 0.7
     - Weight for ML score in combined calculation
   * - ``log_level``
     - str
     - "INFO"
     - Logging level (DEBUG, INFO, WARNING, ERROR)
   * - ``batch_size``
     - int
     - 1000
     - Maximum batch size for batch processing

Configuration Methods
---------------------

1. Direct Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Pass configuration to the detector:

.. code-block:: python

   from metaguard import SimpleDetector
   from metaguard.utils.config import MetaGuardConfig

   config = MetaGuardConfig(
       risk_threshold=0.6,
       ml_weight=0.8,
       log_level="DEBUG"
   )

   detector = SimpleDetector(config=config)

2. Global Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Set configuration globally:

.. code-block:: python

   from metaguard.utils.config import set_config, get_config, reset_config, MetaGuardConfig

   # Set global configuration
   set_config(MetaGuardConfig(risk_threshold=0.4))

   # Get current configuration
   config = get_config()
   print(f"Current threshold: {config.risk_threshold}")

   # Reset to defaults
   reset_config()

3. Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^

Configure via environment variables:

.. code-block:: bash

   # Linux/macOS
   export METAGUARD_MODEL_PATH=/path/to/model.pkl
   export METAGUARD_RISK_THRESHOLD=0.6
   export METAGUARD_ML_WEIGHT=0.8
   export METAGUARD_LOG_LEVEL=DEBUG
   export METAGUARD_BATCH_SIZE=500

.. code-block:: powershell

   # Windows PowerShell
   $env:METAGUARD_MODEL_PATH = "C:\path\to\model.pkl"
   $env:METAGUARD_RISK_THRESHOLD = "0.6"

Environment variables are automatically loaded:

.. code-block:: python

   from metaguard import SimpleDetector

   # Detector reads from environment
   detector = SimpleDetector()

Configuration Priority
----------------------

When multiple configuration sources exist, priority is:

1. **Explicit config** passed to ``SimpleDetector``
2. **Global config** set via ``set_config()``
3. **Environment variables**
4. **Default values**

Example:

.. code-block:: python

   import os
   from metaguard import SimpleDetector
   from metaguard.utils.config import set_config, MetaGuardConfig

   # Environment variable
   os.environ["METAGUARD_RISK_THRESHOLD"] = "0.3"

   # Global config
   set_config(MetaGuardConfig(risk_threshold=0.5))

   # Explicit config
   explicit = MetaGuardConfig(risk_threshold=0.7)

   # This detector uses threshold=0.7 (explicit wins)
   detector1 = SimpleDetector(config=explicit)

   # This detector uses threshold=0.5 (global wins over env)
   detector2 = SimpleDetector()

Validation
----------

Configuration values are validated:

.. code-block:: python

   from metaguard.utils.config import MetaGuardConfig
   from metaguard.utils.exceptions import ConfigurationError

   try:
       # Invalid threshold
       config = MetaGuardConfig(risk_threshold=1.5)
   except ValueError as e:
       print(f"Invalid config: {e}")

Valid ranges:

- ``risk_threshold``: 0.0 to 1.0
- ``ml_weight``: 0.0 to 1.0
- ``batch_size``: 1 to 100000
- ``log_level``: DEBUG, INFO, WARNING, ERROR, CRITICAL

Best Practices
--------------

Development vs Production
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   from metaguard.utils.config import MetaGuardConfig

   if os.getenv("ENV") == "production":
       config = MetaGuardConfig(
           risk_threshold=0.5,
           log_level="WARNING"
       )
   else:
       config = MetaGuardConfig(
           risk_threshold=0.3,  # More sensitive in dev
           log_level="DEBUG"
       )

Configuration File
^^^^^^^^^^^^^^^^^^

Create a configuration file for your project:

.. code-block:: python

   # config.py
   from metaguard.utils.config import MetaGuardConfig

   METAGUARD_CONFIG = MetaGuardConfig(
       risk_threshold=0.5,
       ml_weight=0.7,
       log_level="INFO"
   )

   # main.py
   from config import METAGUARD_CONFIG
   from metaguard import SimpleDetector

   detector = SimpleDetector(config=METAGUARD_CONFIG)
