Testing Guide
=============

This guide covers testing practices for MetaGuard.

Test Structure
--------------

.. code-block:: text

   tests/
   ├── conftest.py          # Shared fixtures
   ├── unit/                # Unit tests
   │   ├── test_detector.py
   │   ├── test_risk.py
   │   ├── test_validators.py
   │   ├── test_config.py
   │   ├── test_exceptions.py
   │   └── test_logging.py
   ├── integration/         # Integration tests
   │   └── test_pipeline.py
   └── e2e/                 # End-to-end tests
       └── test_full_workflow.py

Running Tests
-------------

Basic Commands
^^^^^^^^^^^^^^

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Verbose output
   pytest tests/ -v

   # Run specific directory
   pytest tests/unit/

   # Run specific file
   pytest tests/unit/test_detector.py

   # Run specific test
   pytest tests/unit/test_detector.py::TestSimpleDetector::test_detect

   # Stop on first failure
   pytest tests/ -x

   # Show local variables on failure
   pytest tests/ -l

Coverage
^^^^^^^^

.. code-block:: bash

   # With terminal report
   pytest tests/ --cov=src/metaguard --cov-report=term-missing

   # With HTML report
   pytest tests/ --cov=src/metaguard --cov-report=html
   # Open htmlcov/index.html

   # With minimum coverage requirement
   pytest tests/ --cov=src/metaguard --cov-fail-under=90

Writing Tests
-------------

Unit Test Example
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # tests/unit/test_detector.py
   import pytest
   from metaguard import SimpleDetector
   from metaguard.utils.exceptions import InvalidTransactionError


   class TestSimpleDetector:
       """Tests for SimpleDetector class."""

       def test_detect_normal_transaction(self, detector, sample_transaction):
           """Test detection of normal transaction."""
           result = detector.detect(sample_transaction)

           assert result["is_suspicious"] == False
           assert 0 <= result["risk_score"] <= 1
           assert result["risk_level"] in ["Low", "Medium", "High"]

       def test_detect_invalid_amount(self, detector):
           """Test detection fails for invalid amount."""
           invalid = {
               "amount": -100,
               "hour": 14,
               "user_age_days": 30,
               "transaction_count": 5,
           }
           with pytest.raises(InvalidTransactionError):
               detector.detect(invalid)

Fixtures
^^^^^^^^

Shared fixtures in ``conftest.py``:

.. code-block:: python

   # tests/conftest.py
   import pytest
   from pathlib import Path
   from metaguard import SimpleDetector


   @pytest.fixture
   def detector():
       """Create a SimpleDetector instance."""
       return SimpleDetector()


   @pytest.fixture
   def sample_transaction():
       """Normal transaction data."""
       return {
           "amount": 100,
           "hour": 14,
           "user_age_days": 30,
           "transaction_count": 5,
       }


   @pytest.fixture
   def suspicious_transaction():
       """Suspicious transaction data."""
       return {
           "amount": 5000,
           "hour": 3,
           "user_age_days": 5,
           "transaction_count": 50,
       }

Parametrized Tests
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pytest
   from metaguard import get_risk_level


   @pytest.mark.parametrize("score,expected", [
       (0, "Low"),
       (20, "Low"),
       (40, "Low"),
       (50, "Medium"),
       (70, "Medium"),
       (80, "High"),
       (100, "High"),
   ])
   def test_risk_levels(score, expected):
       """Test risk level thresholds."""
       assert get_risk_level(score) == expected

Integration Tests
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # tests/integration/test_pipeline.py
   from metaguard import SimpleDetector, check_transaction


   class TestDetectionPipeline:
       """Integration tests for detection pipeline."""

       def test_full_pipeline(self, sample_transaction):
           """Test complete detection pipeline."""
           # Method 1: check_transaction
           result1 = check_transaction(sample_transaction)

           # Method 2: SimpleDetector
           detector = SimpleDetector()
           result2 = detector.detect(sample_transaction)

           # Results should match
           assert result1["is_suspicious"] == result2["is_suspicious"]
           assert result1["risk_level"] == result2["risk_level"]

Test Configuration
------------------

pytest configuration in ``pyproject.toml``:

.. code-block:: toml

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   python_classes = ["Test*"]
   python_functions = ["test_*"]
   addopts = "-v --tb=short"
   filterwarnings = [
       "ignore::DeprecationWarning",
   ]

   [tool.coverage.run]
   source = ["src/metaguard"]
   branch = true

   [tool.coverage.report]
   exclude_lines = [
       "pragma: no cover",
       "if TYPE_CHECKING:",
       "raise NotImplementedError",
   ]
   fail_under = 80

Best Practices
--------------

1. **Test One Thing**: Each test should verify one behavior
2. **Clear Names**: Test names should describe what's being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use Fixtures**: Share setup code via fixtures
5. **Test Edge Cases**: Include boundary conditions
6. **Fast Tests**: Keep unit tests fast (<100ms each)

Example following best practices:

.. code-block:: python

   def test_detect_transaction_at_threshold_boundary(self, detector):
       """Test detection at exact risk threshold."""
       # Arrange
       transaction = {
           "amount": 1000,  # Exactly at high_amount threshold
           "hour": 6,       # Just outside unusual_hour
           "user_age_days": 30,  # Exactly at new_account threshold
           "transaction_count": 20,  # Exactly at high_frequency threshold
       }

       # Act
       result = detector.detect(transaction)

       # Assert
       assert "is_suspicious" in result
       assert isinstance(result["risk_score"], float)
       assert result["risk_level"] in ["Low", "Medium", "High"]

Continuous Integration
----------------------

Tests run automatically on:

- Pull requests
- Pushes to main branch
- Scheduled (nightly)

GitHub Actions workflow:

.. code-block:: yaml

   # .github/workflows/tests.yml
   name: Tests

   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ['3.9', '3.10', '3.11', '3.12']

       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: ${{ matrix.python-version }}
         - run: pip install -e ".[dev]"
         - run: pytest tests/ --cov=src/metaguard --cov-report=xml
         - uses: codecov/codecov-action@v4
