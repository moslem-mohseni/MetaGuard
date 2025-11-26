User Guide
==========

This guide covers all aspects of using MetaGuard for fraud detection.

.. toctree::
   :maxdepth: 2

   basic_usage
   advanced
   configuration

Overview
--------

MetaGuard provides several ways to detect fraud:

1. **Quick Check**: Use ``check_transaction()`` for simple one-off checks
2. **Detector Class**: Use ``SimpleDetector`` for batch processing and configuration
3. **Risk Analysis**: Use ``analyze_transaction_risk()`` for detailed factor analysis

Choosing the Right Approach
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use Case
     - Recommended Approach
   * - Single transaction check
     - ``check_transaction()``
   * - Multiple transactions
     - ``SimpleDetector.batch_detect()``
   * - Custom model path
     - ``SimpleDetector(model_path=...)``
   * - Detailed risk breakdown
     - ``analyze_transaction_risk()``
   * - Custom risk thresholds
     - ``SimpleDetector(config=...)``
