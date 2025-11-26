API Reference
=============

This section contains the complete API reference for MetaGuard.

.. toctree::
   :maxdepth: 2

   detector
   risk
   exceptions
   config
   logging
   rest_api
   cli

Quick Reference
---------------

Main Functions
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - :func:`metaguard.check_transaction`
     - Quick check for a single transaction
   * - :func:`metaguard.calculate_risk`
     - Calculate risk score from parameters
   * - :func:`metaguard.get_risk_level`
     - Get risk level from score
   * - :func:`metaguard.analyze_transaction_risk`
     - Detailed risk analysis with factors

Main Classes
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - :class:`metaguard.SimpleDetector`
     - Main fraud detection class
   * - :class:`metaguard.utils.config.MetaGuardConfig`
     - Configuration management

Exceptions
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Exception
     - Description
   * - :exc:`MetaGuardError`
     - Base exception for all errors
   * - :exc:`InvalidTransactionError`
     - Invalid transaction data
   * - :exc:`ModelNotFoundError`
     - Model file not found
   * - :exc:`ModelLoadError`
     - Failed to load model
