Changelog
=========

All notable changes to MetaGuard will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[1.1.0] - 2024-XX-XX
--------------------

Added
^^^^^
- Type hints throughout codebase
- Pydantic validation for transaction inputs
- Custom exception hierarchy (MetaGuardError, InvalidTransactionError, etc.)
- Comprehensive logging module with JSON and colored formatters
- Configuration management with environment variable support
- ``analyze_transaction_risk()`` function for detailed risk analysis
- ``get_risk_factors_description()`` for human-readable factor descriptions
- ``calculate_combined_risk()`` for combining ML and formula scores
- Full test suite with >90% code coverage
- Sphinx documentation with API reference
- Pre-commit hooks for code quality

Changed
^^^^^^^
- Migrated to src layout structure
- Updated to modern pyproject.toml packaging
- Improved batch detection performance
- Enhanced error messages with detailed context

Fixed
^^^^^
- Model compatibility with latest scikit-learn
- Handling of edge cases in risk calculation

[1.0.0] - 2024-XX-XX
--------------------

Added
^^^^^
- Initial release
- SimpleDetector class for fraud detection
- ``check_transaction()`` quick check function
- Random Forest based ML model
- Basic risk calculation functions
- Training script for custom models
- Data generation script

---

Version History
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Version
     - Date
     - Highlights
   * - 1.1.0
     - TBD
     - Type hints, validation, logging, tests
   * - 1.0.0
     - TBD
     - Initial release
