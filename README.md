# MetaGuard

[![PyPI version](https://badge.fury.io/py/metaguard.svg)](https://badge.fury.io/py/metaguard)
[![Python Version](https://img.shields.io/pypi/pyversions/metaguard.svg)](https://pypi.org/project/metaguard/)
[![License](https://img.shields.io/github/license/moslem-mohseni/MetaGuard.svg)](https://github.com/moslem-mohseni/MetaGuard/blob/main/LICENSE)
[![Tests](https://github.com/moslem-mohseni/MetaGuard/workflows/Tests/badge.svg)](https://github.com/moslem-mohseni/MetaGuard/actions)
[![Coverage](https://codecov.io/gh/moslem-mohseni/MetaGuard/branch/main/graph/badge.svg)](https://codecov.io/gh/moslem-mohseni/MetaGuard)
[![Documentation](https://readthedocs.org/projects/metaguard/badge/?version=latest)](https://metaguard.readthedocs.io)

> Fraud detection for metaverse transactions with just 3 lines of Python code.

**Author:** Moslem Mohseni
**Version:** 1.1.0

## Why MetaGuard?

In 2024, over **$3 billion** was lost to fraud in the metaverse. MetaGuard provides a simple, effective solution to detect suspicious transactions using machine learning.

## Features

- **Simple API** - Detect fraud with just 3 lines of code
- **ML-Powered** - Pre-trained Random Forest model included
- **Batch Processing** - Process multiple transactions efficiently
- **Risk Analysis** - Get detailed risk factors and scores
- **Configurable** - Customize thresholds and model parameters
- **Type-Safe** - Full type hints for IDE support
- **Well-Tested** - >90% test coverage

## Installation

```bash
pip install metaguard
```

For development:

```bash
pip install metaguard[dev]
```

## Quick Start

### 3-Line Fraud Detection

```python
from metaguard import check_transaction

result = check_transaction({
    "amount": 5000,
    "hour": 3,
    "user_age_days": 5,
    "transaction_count": 50
})

print(f"Suspicious: {result['is_suspicious']}")  # True
print(f"Risk Level: {result['risk_level']}")     # High
```

### Using the Detector Class

```python
from metaguard import SimpleDetector

detector = SimpleDetector()

# Single detection
result = detector.detect({
    "amount": 100,
    "hour": 14,
    "user_age_days": 30,
    "transaction_count": 5
})

# Batch detection
results = detector.batch_detect([
    {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
    {"amount": 5000, "hour": 3, "user_age_days": 2, "transaction_count": 50},
])
```

### Risk Analysis

```python
from metaguard import analyze_transaction_risk

result = analyze_transaction_risk({
    "amount": 5000,
    "hour": 3,
    "user_age_days": 5,
    "transaction_count": 50
})

print(f"Risk Score: {result['risk_score']}")
print(f"Active Factors: {result['active_factor_count']}")
# Factors: high_amount, new_account, high_frequency, unusual_hour
```

## Transaction Format

| Field | Type | Description |
|-------|------|-------------|
| `amount` | float | Transaction amount (> 0) |
| `hour` | int | Hour of day (0-23) |
| `user_age_days` | int | Account age in days (>= 1) |
| `transaction_count` | int | Recent transactions (>= 0) |

## Result Format

| Field | Type | Description |
|-------|------|-------------|
| `is_suspicious` | bool | True if flagged as fraud |
| `risk_score` | float | Probability score (0.0-1.0) |
| `risk_level` | str | "Low", "Medium", or "High" |

## Risk Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| Low | 0 - 40 | Transaction appears safe |
| Medium | 40 - 70 | Some risk indicators |
| High | 70 - 100 | High fraud probability |

## Configuration

### Environment Variables

```bash
export METAGUARD_RISK_THRESHOLD=0.5
export METAGUARD_ML_WEIGHT=0.7
export METAGUARD_LOG_LEVEL=INFO
```

### Programmatic Configuration

```python
from metaguard import SimpleDetector
from metaguard.utils.config import MetaGuardConfig

config = MetaGuardConfig(
    risk_threshold=0.6,
    ml_weight=0.8,
    log_level="DEBUG"
)

detector = SimpleDetector(config=config)
```

## CLI Usage

Install with CLI support:

```bash
pip install metaguard[cli]
```

### Commands

```bash
# Detect fraud
metaguard detect -a 5000 -h 3 -u 5 -t 50

# Detailed analysis
metaguard analyze -a 5000 -h 3 -u 5 -t 50

# Batch processing
metaguard batch transactions.json --output results.json

# Show model info
metaguard info

# Start API server
metaguard serve --port 8000
```

## REST API

Install with API support:

```bash
pip install metaguard[api]
```

### Start Server

```bash
metaguard serve
# or
uvicorn metaguard.api.rest:app --reload
```

### Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Single detection
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50}'

# Batch detection
curl -X POST http://localhost:8000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"amount": 100, "hour": 10, "user_age_days": 100, "transaction_count": 5}]}'

# Risk analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50}'
```

API documentation: http://localhost:8000/docs

## Docker

```bash
# Build
docker build -t metaguard .

# Run
docker run -p 8000:8000 metaguard

# Docker Compose
docker-compose up
```

## Error Handling

```python
from metaguard import check_transaction
from metaguard.utils.exceptions import InvalidTransactionError

try:
    result = check_transaction({"amount": -100})
except InvalidTransactionError as e:
    print(f"Invalid: {e.field} - {e.reason}")
```

## Project Structure

```
MetaGuard/
├── src/metaguard/          # Main package
│   ├── __init__.py
│   ├── detector.py         # SimpleDetector class
│   ├── risk.py             # Risk calculation
│   └── utils/              # Utilities
│       ├── config.py       # Configuration
│       ├── exceptions.py   # Custom exceptions
│       ├── logging.py      # Logging utilities
│       └── validators.py   # Input validation
├── tests/                  # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/                   # Sphinx documentation
├── scripts/                # Training scripts
└── examples/               # Example code
```

## Development

```bash
# Clone repository
git clone https://github.com/moslem-mohseni/MetaGuard.git
cd MetaGuard

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/metaguard --cov-report=term-missing

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

## Documentation

Full documentation is available at [metaguard.readthedocs.io](https://metaguard.readthedocs.io).

Build locally:

```bash
cd docs
pip install -r requirements-docs.txt
make html
```

## Performance

| Metric | Value |
|--------|-------|
| Test Coverage | >90% |
| Speed | <100ms per transaction |
| Batch (1000) | <30 seconds |

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Author

**Moslem Mohseni**

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## Citation

```bibtex
@software{metaguard2024,
  author = {Moslem Mohseni},
  title = {MetaGuard: Fraud Detection for Metaverse Transactions},
  year = {2024},
  version = {1.1.0},
  url = {https://github.com/moslem-mohseni/MetaGuard}
}
```

---

**MetaGuard** - Protecting the Metaverse, one transaction at a time.
