REST API Reference
==================

**Author:** Moslem Mohseni

MetaGuard provides a production-ready REST API built with FastAPI for fraud detection services.

Starting the Server
-------------------

Using CLI
~~~~~~~~~

.. code-block:: bash

    # Default (localhost:8000)
    metaguard serve

    # Custom host and port
    metaguard serve --host 0.0.0.0 --port 8080

    # With hot reload (development)
    metaguard serve --reload

Using Docker
~~~~~~~~~~~~

.. code-block:: bash

    # Build and run
    docker-compose up

    # Or directly with Docker
    docker run -p 8000:8000 ghcr.io/moslem-mohseni/metaguard:latest

API Endpoints
-------------

Health Check
~~~~~~~~~~~~

**GET /**

Returns API health status.

.. code-block:: bash

    curl http://localhost:8000/

Response:

.. code-block:: json

    {
        "status": "healthy",
        "version": "1.1.0",
        "model_loaded": true
    }

Single Transaction Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**POST /detect**

Analyze a single transaction for fraud.

Request:

.. code-block:: bash

    curl -X POST http://localhost:8000/detect \
        -H "Content-Type: application/json" \
        -d '{
            "amount": 5000,
            "hour": 3,
            "user_age_days": 5,
            "transaction_count": 50
        }'

Response:

.. code-block:: json

    {
        "is_suspicious": true,
        "risk_score": 0.95,
        "risk_level": "High",
        "processing_time_ms": 12.5
    }

Parameters:

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``amount``
     - float
     - Yes
     - Transaction amount (> 0)
   * - ``hour``
     - int
     - Yes
     - Hour of day (0-23)
   * - ``user_age_days``
     - int
     - Yes
     - Account age in days (>= 1)
   * - ``transaction_count``
     - int
     - Yes
     - Number of prior transactions (>= 0)

Batch Detection
~~~~~~~~~~~~~~~

**POST /detect/batch**

Analyze multiple transactions (up to 1000).

Request:

.. code-block:: bash

    curl -X POST http://localhost:8000/detect/batch \
        -H "Content-Type: application/json" \
        -d '{
            "transactions": [
                {"amount": 100, "hour": 10, "user_age_days": 100, "transaction_count": 5},
                {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50}
            ]
        }'

Response:

.. code-block:: json

    {
        "results": [
            {"is_suspicious": false, "risk_score": 0.15, "risk_level": "Low", "processing_time_ms": 5.0},
            {"is_suspicious": true, "risk_score": 0.95, "risk_level": "High", "processing_time_ms": 5.0}
        ],
        "total_transactions": 2,
        "suspicious_count": 1,
        "processing_time_ms": 10.0
    }

Risk Analysis
~~~~~~~~~~~~~

**POST /analyze**

Get detailed risk breakdown.

Request:

.. code-block:: bash

    curl -X POST http://localhost:8000/analyze \
        -H "Content-Type: application/json" \
        -d '{
            "amount": 5000,
            "hour": 3,
            "user_age_days": 5,
            "transaction_count": 50
        }'

Response:

.. code-block:: json

    {
        "risk_score": 100.0,
        "risk_level": "High",
        "factors": {
            "high_amount": true,
            "new_account": true,
            "high_frequency": true,
            "unusual_hour": true
        },
        "active_factor_count": 4,
        "processing_time_ms": 8.2
    }

Model Information
~~~~~~~~~~~~~~~~~

**GET /model/info**

Get information about the loaded model.

.. code-block:: bash

    curl http://localhost:8000/model/info

Response:

.. code-block:: json

    {
        "model_type": "RandomForestClassifier",
        "model_path": "/path/to/model.pkl",
        "risk_threshold": 0.5,
        "feature_names": ["amount", "hour", "user_age_days", "transaction_count"]
    }

Error Handling
--------------

Validation Error (422)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "detail": [
            {
                "type": "greater_than",
                "loc": ["body", "amount"],
                "msg": "Input should be greater than 0",
                "input": -100
            }
        ]
    }

Invalid Transaction (400)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

    {
        "detail": "Invalid transaction: hour must be between 0 and 23",
        "error_type": "InvalidTransactionError"
    }

Python Client Example
---------------------

Using ``httpx``:

.. code-block:: python

    import httpx

    client = httpx.Client(base_url="http://localhost:8000")

    # Single detection
    response = client.post("/detect", json={
        "amount": 5000,
        "hour": 3,
        "user_age_days": 5,
        "transaction_count": 50
    })
    result = response.json()
    print(f"Suspicious: {result['is_suspicious']}")

    # Batch detection
    transactions = [
        {"amount": 100, "hour": 10, "user_age_days": 100, "transaction_count": 5},
        {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50},
    ]
    response = client.post("/detect/batch", json={"transactions": transactions})
    batch_result = response.json()
    print(f"Suspicious: {batch_result['suspicious_count']}/{batch_result['total_transactions']}")

Async Client
~~~~~~~~~~~~

.. code-block:: python

    import asyncio
    import httpx

    async def check_transactions(transactions):
        async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
            response = await client.post("/detect/batch", json={"transactions": transactions})
            return response.json()

    # Usage
    result = asyncio.run(check_transactions([...]))

OpenAPI Documentation
---------------------

The API provides auto-generated documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
