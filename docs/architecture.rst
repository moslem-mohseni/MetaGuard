Architecture
============

**Author:** Moslem Mohseni

This document describes the high-level architecture of MetaGuard, a fraud detection library for metaverse transactions.

System Overview
---------------

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────┐
    │                           MetaGuard                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
    │  │   CLI       │     │  REST API   │     │   Python Library    │   │
    │  │  (Typer)    │     │  (FastAPI)  │     │   (Direct Import)   │   │
    │  └──────┬──────┘     └──────┬──────┘     └──────────┬──────────┘   │
    │         │                   │                       │               │
    │         └───────────────────┼───────────────────────┘               │
    │                             │                                        │
    │                     ┌───────▼───────┐                               │
    │                     │ SimpleDetector │                               │
    │                     └───────┬───────┘                               │
    │                             │                                        │
    │         ┌───────────────────┼───────────────────┐                   │
    │         │                   │                   │                   │
    │  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐            │
    │  │  Validator  │    │  ML Model   │    │    Risk     │            │
    │  │             │    │ (sklearn)   │    │  Analyzer   │            │
    │  └─────────────┘    └─────────────┘    └─────────────┘            │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Component Architecture
----------------------

Core Layer
~~~~~~~~~~

.. code-block:: text

    src/metaguard/
    ├── __init__.py          # Public API exports
    ├── detector.py          # SimpleDetector class
    ├── risk.py              # Risk calculation functions
    └── utils/
        ├── validators.py    # Input validation
        ├── config.py        # Configuration management
        ├── exceptions.py    # Custom exceptions
        └── logging.py       # Logging utilities

**SimpleDetector** (``detector.py``)

The main entry point for fraud detection. Responsibilities:

- Load and manage ML model
- Validate transaction data
- Predict fraud probability
- Determine risk levels

.. code-block:: python

    class SimpleDetector:
        def __init__(self, model_path=None, config=None):
            self._load_model()

        def detect(self, transaction: dict) -> dict:
            validated = validate_transaction(transaction)
            prob = self.model.predict_proba(features)[0][1]
            return {
                "is_suspicious": prob > threshold,
                "risk_score": prob,
                "risk_level": get_risk_level(prob * 100)
            }

**Risk Module** (``risk.py``)

Pure functions for risk calculation:

.. code-block:: python

    def calculate_risk(amount, user_age, tx_count) -> float:
        """Calculate risk score (0-100)."""

    def get_risk_level(score) -> str:
        """Map score to Low/Medium/High."""

    def analyze_transaction_risk(transaction) -> dict:
        """Detailed analysis with factors."""

ML Models Layer
~~~~~~~~~~~~~~~

.. code-block:: text

    src/metaguard/models/
    ├── __init__.py
    ├── base.py              # BaseModel ABC
    ├── random_forest.py     # RandomForestModel
    ├── xgboost_model.py     # XGBoostModel (optional)
    ├── ensemble.py          # EnsembleModel
    └── model.pkl            # Pre-trained model

**Model Hierarchy**

.. code-block:: text

    BaseModel (ABC)
    ├── RandomForestModel
    ├── XGBoostModel
    └── EnsembleModel
        ├── contains: RandomForestModel
        └── contains: XGBoostModel (optional)

**Feature Engineering** (``features/engineering.py``)

.. code-block:: python

    class FeatureEngineer:
        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            # Base features
            # + log_amount
            # + amount_per_transaction
            # + transactions_per_day
            # + hour_sin, hour_cos (cyclical encoding)

API Layer
~~~~~~~~~

.. code-block:: text

    src/metaguard/api/
    ├── __init__.py
    ├── rest.py              # FastAPI application
    └── schemas.py           # Pydantic models

**REST API Design**

Follows RESTful principles:

- Resource-based URLs
- HTTP methods (GET, POST)
- JSON request/response
- Proper status codes

.. code-block:: text

    GET  /               Health check
    GET  /health         Health check (alias)
    POST /detect         Single transaction
    POST /detect/batch   Batch processing
    POST /analyze        Detailed analysis
    GET  /model/info     Model information

**Request/Response Flow**

.. code-block:: text

    Client Request
         │
         ▼
    ┌────────────┐
    │  FastAPI   │ ← Pydantic validation
    └────────────┘
         │
         ▼
    ┌────────────┐
    │  Detector  │ ← ML inference
    └────────────┘
         │
         ▼
    ┌────────────┐
    │  Response  │ → JSON
    └────────────┘

CLI Layer
~~~~~~~~~

.. code-block:: text

    src/metaguard/cli/
    ├── __init__.py
    └── main.py              # Typer application

Built with Typer and Rich for beautiful output:

- Commands: detect, analyze, batch, info, serve
- Rich tables and panels
- JSON output option

Data Flow
---------

Detection Flow
~~~~~~~~~~~~~~

.. code-block:: text

    1. Input Transaction
           │
           ▼
    2. Validation (validators.py)
           │ ← InvalidTransactionError if invalid
           ▼
    3. Feature Extraction
           │
           ▼
    4. ML Prediction (model.predict_proba)
           │
           ▼
    5. Risk Assessment (risk.py)
           │
           ▼
    6. Response Generation

Batch Processing Flow
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Input: [tx1, tx2, ..., txN]
           │
           ▼
    ┌──────────────────┐
    │ Parallel/Serial  │
    │ Detection Loop   │
    └────────┬─────────┘
             │
    ┌────────▼────────┐
    │ Aggregate Stats │
    │ (count, %)      │
    └────────┬────────┘
             │
             ▼
    Output: {results, summary}

Security Architecture
---------------------

Input Validation
~~~~~~~~~~~~~~~~

All inputs validated before processing:

.. code-block:: python

    @dataclass
    class Transaction:
        amount: float        # Must be > 0
        hour: int            # Must be 0-23
        user_age_days: int   # Must be >= 1
        transaction_count: int  # Must be >= 0

Model Security
~~~~~~~~~~~~~~

- Models loaded only from configured paths
- Pickle security: trust only known sources
- Model versioning supported

API Security
~~~~~~~~~~~~

- CORS configurable
- Rate limiting (via proxy)
- No authentication (add externally)

Deployment Architecture
-----------------------

Docker Deployment
~~~~~~~~~~~~~~~~~

.. code-block:: text

    ┌─────────────────────────────────────────────┐
    │               Docker Host                    │
    ├─────────────────────────────────────────────┤
    │                                              │
    │  ┌─────────────────────────────────────┐    │
    │  │       metaguard-api container       │    │
    │  │                                     │    │
    │  │  ┌─────────────────────────────┐   │    │
    │  │  │    Uvicorn (ASGI Server)    │   │    │
    │  │  │         Port 8000           │   │    │
    │  │  └──────────────┬──────────────┘   │    │
    │  │                 │                   │    │
    │  │  ┌──────────────▼──────────────┐   │    │
    │  │  │        FastAPI App          │   │    │
    │  │  └─────────────────────────────┘   │    │
    │  │                                     │    │
    │  │  Non-root user: metaguard          │    │
    │  │  Memory limit: 1GB                 │    │
    │  └─────────────────────────────────────┘    │
    │                                              │
    └─────────────────────────────────────────────┘

Production Architecture
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Client    │────▶│  Load       │────▶│  MetaGuard  │
    │             │     │  Balancer   │     │  Instance 1 │
    └─────────────┘     │  (Nginx)    │     └─────────────┘
                        │             │     ┌─────────────┐
                        │             │────▶│  MetaGuard  │
                        │             │     │  Instance 2 │
                        └─────────────┘     └─────────────┘

Performance Considerations
--------------------------

Model Loading
~~~~~~~~~~~~~

- Model loaded once at startup
- Kept in memory for fast inference
- ~50MB memory footprint

Inference Speed
~~~~~~~~~~~~~~~

- Single transaction: ~5-15ms
- Batch of 1000: ~500ms
- Main bottleneck: feature engineering

Optimization Tips
~~~~~~~~~~~~~~~~~

1. Use batch API for multiple transactions
2. Deploy behind load balancer for scale
3. Consider model quantization for edge deployment
4. Use connection pooling for high-throughput

Extensibility
-------------

Custom Models
~~~~~~~~~~~~~

Implement ``BaseModel`` interface:

.. code-block:: python

    from metaguard.models.base import BaseModel

    class CustomModel(BaseModel):
        def train(self, X, y):
            ...

        def predict(self, X):
            ...

        def predict_proba(self, X):
            ...

Custom Features
~~~~~~~~~~~~~~~

Extend ``FeatureEngineer``:

.. code-block:: python

    class CustomFeatureEngineer(FeatureEngineer):
        def transform(self, df):
            df = super().transform(df)
            df["custom_feature"] = ...
            return df
