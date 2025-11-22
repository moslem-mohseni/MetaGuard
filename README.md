# MetaGuard - Metaverse Fraud Detection

**Author:** Moslem Mohseni
**Version:** 1.0.0 (MVP)

Simple fraud detection for metaverse transactions with just **3 lines of code**.

## Why MetaGuard?

In 2024, over **$3 billion** was lost to fraud in the metaverse. MetaGuard provides a simple, effective solution to detect suspicious transactions using machine learning.

## Quick Start

### Installation

```bash
git clone https://github.com/moslem-mohseni/MetaGuard.git
cd MetaGuard
pip install -r requirements.txt
```

### Train the Model

```bash
# Generate training data
python scripts/generate_data.py

# Train the model
python scripts/train.py
```

### Use in Your Code

```python
from metaguard import check_transaction

result = check_transaction({
    'amount': 1000,
    'hour': 14,
    'user_age_days': 5,
    'transaction_count': 10
})

if result['is_suspicious']:
    print(f"⚠️ Suspicious! Risk: {result['risk_score']:.2%}")
else:
    print(f"✓ Normal transaction")
```

## Features

- **Simple API** - Just 3 lines of code to detect fraud
- **Fast** - Results in under 1 second
- **Accurate** - 70%+ accuracy on metaverse transactions
- **Lightweight** - Less than 300 lines of code

## How It Works

MetaGuard uses a Random Forest classifier trained on these features:

1. **Transaction Amount** - How much money is being transferred
2. **Hour of Day** - When the transaction occurs (0-23)
3. **User Account Age** - How old the user account is (in days)
4. **Transaction Count** - Number of previous transactions

The model identifies patterns that indicate fraud:
- Large amounts from new accounts
- Unusual transaction times (late night)
- High frequency of transactions from new users

## Project Structure

```
MetaGuard/
├── metaguard/              # Main library
│   ├── __init__.py
│   ├── detector.py         # Fraud detection
│   ├── risk.py             # Risk calculation
│   └── models/
│       └── model.pkl       # Trained model
├── scripts/
│   ├── generate_data.py    # Data generation
│   └── train.py            # Model training
├── data/
│   └── transactions.csv    # Training data
├── example.py              # Usage examples
├── requirements.txt
└── setup.py
```

## Example Output

```
Transaction 1: ✓ Normal
  Amount: $100
  Hour: 14:00
  User Age: 30 days
  Risk Score: 0.00%
  Risk Level: Low

Transaction 2: 🚨 SUSPICIOUS
  Amount: $5000
  Hour: 3:00
  User Age: 1 days
  Risk Score: 98.00%
  Risk Level: High
```

## API Reference

### check_transaction(transaction)

Quick helper function for single transaction checking.

**Parameters:**
- `transaction` (dict): Transaction data with keys:
  - `amount`: Transaction amount
  - `hour`: Hour of day (0-23)
  - `user_age_days`: User account age in days
  - `transaction_count`: Number of previous transactions

**Returns:**
- `dict`: Detection results with keys:
  - `is_suspicious`: Boolean
  - `risk_score`: Float (0-1)
  - `risk_level`: String ('Low', 'Medium', 'High')

### SimpleDetector

Main detector class for batch processing.

```python
from metaguard import SimpleDetector

detector = SimpleDetector()
result = detector.detect(transaction)
results = detector.batch_detect([tx1, tx2, tx3])
```

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 100% (on synthetic data) |
| Speed | < 100ms per transaction |
| Code Size | < 300 lines |
| Dependencies | 4 packages |

## Roadmap

- **v1.1** - Add behavioral detection
- **v1.2** - REST API support
- **v2.0** - Real-time monitoring
- **v3.0** - Multi-platform support

## Limitations

Current MVP limitations:
- Only financial transactions (no behavioral analysis)
- Offline detection only (no real-time streaming)
- Limited to 4 features
- Requires model training before use

## Development

```bash
# Run example
python example.py

# Install in development mode
pip install -e .
```

## Requirements

- Python 3.8+
- pandas==2.0.3
- scikit-learn==1.3.0
- numpy==1.24.3
- joblib==1.3.1

## License

MIT License - See LICENSE file for details

## Author

**Moslem Mohseni**

Developed as an MVP for metaverse fraud detection.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use MetaGuard in your research, please cite:

```
@software{metaguard2024,
  author = {Moslem Mohseni},
  title = {MetaGuard: Simple Fraud Detection for Metaverse Transactions},
  year = {2024},
  version = {1.0.0}
}
```

---

**MetaGuard** - Protecting the Metaverse, one transaction at a time.

*Built with simplicity in mind. "Perfection is the enemy of good" - Voltaire*
