Contributing Guide
==================

Thank you for your interest in contributing to MetaGuard!

Getting Started
---------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/MetaGuard.git
      cd MetaGuard

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Development Workflow
--------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes

3. Run tests:

   .. code-block:: bash

      pytest tests/ -v

4. Run linting:

   .. code-block:: bash

      ruff check src/ tests/
      ruff format src/ tests/

5. Run type checking:

   .. code-block:: bash

      mypy src/

6. Commit your changes:

   .. code-block:: bash

      git commit -m "feat: add your feature"

7. Push and create a pull request

Code Style
----------

We follow these conventions:

- **Python**: PEP 8 with ruff
- **Docstrings**: Google style
- **Type hints**: Required for all public functions
- **Line length**: 88 characters (Black default)

Example:

.. code-block:: python

   def calculate_risk(
       amount: float,
       user_age: int,
       transaction_count: int
   ) -> float:
       """Calculate risk score for a transaction.

       Args:
           amount: Transaction amount in dollars.
           user_age: Account age in days.
           transaction_count: Number of recent transactions.

       Returns:
           Risk score between 0 and 100.

       Raises:
           ValueError: If any parameter is negative.

       Example:
           >>> calculate_risk(100, 30, 5)
           25.5
       """
       ...

Testing
-------

Requirements:

- All new features must have tests
- Maintain >90% code coverage
- Tests should be in ``tests/`` directory

Run tests:

.. code-block:: bash

   # All tests
   pytest tests/ -v

   # With coverage
   pytest tests/ --cov=src/metaguard --cov-report=term-missing

   # Specific test file
   pytest tests/unit/test_detector.py -v

   # Specific test
   pytest tests/unit/test_detector.py::TestSimpleDetector::test_detect -v

Documentation
-------------

Update documentation for any API changes:

1. Edit RST files in ``docs/``
2. Build and check locally:

   .. code-block:: bash

      cd docs
      make html
      # Open _build/html/index.html

3. Add docstrings to new functions/classes

Commit Messages
---------------

We use conventional commits:

- ``feat:`` New feature
- ``fix:`` Bug fix
- ``docs:`` Documentation only
- ``style:`` Code style (formatting)
- ``refactor:`` Code change (no feature/fix)
- ``test:`` Adding tests
- ``chore:`` Maintenance

Examples:

.. code-block:: text

   feat: add batch processing support
   fix: handle negative amounts gracefully
   docs: update quickstart guide
   test: add edge case tests for detector

Pull Request Process
--------------------

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

PR Template:

.. code-block:: markdown

   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation
   - [ ] Refactoring

   ## Testing
   - [ ] Tests added/updated
   - [ ] All tests pass

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-reviewed code
   - [ ] Documentation updated

Issue Reporting
---------------

When reporting issues:

1. Check existing issues first
2. Use the issue template
3. Include:

   - MetaGuard version
   - Python version
   - OS
   - Steps to reproduce
   - Expected vs actual behavior

Questions?
----------

- Open a GitHub discussion
- Email: moslem.mohseni@example.com

Thank you for contributing!
