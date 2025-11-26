# Contributing to MetaGuard

Thank you for your interest in contributing to MetaGuard! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/MetaGuard.git
cd MetaGuard
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/metaguard --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_detector.py -v

# Run specific test
pytest tests/unit/test_detector.py::TestSimpleDetector::test_detect -v
```

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

### Building Documentation

```bash
cd docs
pip install -r ../requirements-docs.txt
make html
# Open _build/html/index.html
```

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use ruff for linting and formatting
- Maximum line length: 88 characters
- Use type hints for all public functions

### Docstrings

Use Google-style docstrings:

```python
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
```

### Commit Messages

Use conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding/updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add batch processing support
fix: handle negative amounts gracefully
docs: update installation guide
test: add edge case tests for detector
```

## Pull Request Process

### Before Submitting

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Rebase on latest main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Coverage maintained >90%

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Submit PR against `main` branch
2. Wait for CI checks to pass
3. Request review from maintainers
4. Address feedback
5. Merge after approval

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- MetaGuard version (`python -c "import metaguard; print(metaguard.__version__)"`)
- Python version (`python --version`)
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/stack traces

### Feature Requests

When requesting features, include:

- Use case description
- Proposed solution
- Alternative solutions considered
- Additional context

## Project Structure

```
MetaGuard/
├── src/metaguard/          # Main package
│   ├── __init__.py
│   ├── detector.py         # SimpleDetector class
│   ├── risk.py             # Risk calculation
│   └── utils/              # Utilities
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── docs/                   # Sphinx documentation
├── scripts/                # Training scripts
└── examples/               # Example code
```

## Testing Guidelines

### Test Requirements

- All new features must have tests
- Maintain >90% code coverage
- Use pytest fixtures for setup
- Test edge cases and error conditions

### Test Structure

```python
class TestFeatureName:
    """Tests for FeatureName."""

    def test_basic_functionality(self, fixture):
        """Test basic use case."""
        # Arrange
        input_data = {...}

        # Act
        result = function(input_data)

        # Assert
        assert result == expected

    def test_edge_case(self, fixture):
        """Test edge case."""
        ...

    def test_error_handling(self, fixture):
        """Test error conditions."""
        with pytest.raises(ExpectedError):
            function(invalid_input)
```

## Questions?

- Open a GitHub issue
- Start a discussion
- Contact: moslem.mohseni@example.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MetaGuard!
