Deployment Guide
================

Deploy MetaGuard in production environments.

Deployment Options
------------------

1. **Python Library**: Direct integration in Python applications
2. **REST API**: HTTP service using FastAPI
3. **Docker Container**: Containerized deployment
4. **Serverless**: AWS Lambda, Google Cloud Functions

Option 1: Python Library Integration
------------------------------------

Simplest deployment - import and use directly:

.. code-block:: python

   # your_app.py
   from metaguard import SimpleDetector

   # Create detector once at startup
   detector = SimpleDetector()

   def check_fraud(transaction_data):
       return detector.detect(transaction_data)

Production Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   from metaguard import SimpleDetector
   from metaguard.utils.config import MetaGuardConfig
   from metaguard.utils.logging import setup_logging

   # Configure logging
   setup_logging(
       level=os.getenv("LOG_LEVEL", "INFO"),
       json_format=True,
       log_file="/var/log/metaguard/app.log"
   )

   # Create production config
   config = MetaGuardConfig(
       model_path=os.getenv("MODEL_PATH"),
       risk_threshold=float(os.getenv("RISK_THRESHOLD", "0.5")),
   )

   # Initialize detector
   detector = SimpleDetector(config=config)

Option 2: REST API with FastAPI
-------------------------------

Create a REST API for fraud detection:

.. code-block:: python

   # api.py
   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel, Field
   from metaguard import SimpleDetector
   from metaguard.utils.exceptions import InvalidTransactionError

   app = FastAPI(
       title="MetaGuard API",
       description="Fraud detection API",
       version="1.0.0"
   )

   # Initialize detector at startup
   detector = SimpleDetector()


   class TransactionRequest(BaseModel):
       amount: float = Field(..., gt=0)
       hour: int = Field(..., ge=0, le=23)
       user_age_days: int = Field(..., ge=1)
       transaction_count: int = Field(..., ge=0)


   class DetectionResponse(BaseModel):
       is_suspicious: bool
       risk_score: float
       risk_level: str


   @app.post("/detect", response_model=DetectionResponse)
   async def detect_fraud(transaction: TransactionRequest):
       try:
           result = detector.detect(transaction.dict())
           return result
       except InvalidTransactionError as e:
           raise HTTPException(status_code=400, detail=str(e))


   @app.get("/health")
   async def health_check():
       return {"status": "healthy"}


   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8000)

Run the API:

.. code-block:: bash

   pip install metaguard[api]
   uvicorn api:app --host 0.0.0.0 --port 8000

Test the API:

.. code-block:: bash

   curl -X POST http://localhost:8000/detect \
     -H "Content-Type: application/json" \
     -d '{"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50}'

Option 3: Docker Deployment
---------------------------

Create a Dockerfile:

.. code-block:: dockerfile

   # Dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application
   COPY . .

   # Install MetaGuard
   RUN pip install metaguard[api]

   # Run API
   EXPOSE 8000
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

Docker Compose:

.. code-block:: yaml

   # docker-compose.yml
   version: '3.8'

   services:
     metaguard:
       build: .
       ports:
         - "8000:8000"
       environment:
         - LOG_LEVEL=INFO
         - RISK_THRESHOLD=0.5
       volumes:
         - ./models:/app/models
         - ./logs:/var/log/metaguard
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3

Build and run:

.. code-block:: bash

   docker-compose up -d

Option 4: Serverless (AWS Lambda)
---------------------------------

Create a Lambda handler:

.. code-block:: python

   # handler.py
   import json
   from metaguard import check_transaction
   from metaguard.utils.exceptions import InvalidTransactionError

   def lambda_handler(event, context):
       try:
           # Parse request
           body = json.loads(event.get('body', '{}'))

           # Detect fraud
           result = check_transaction(body)

           return {
               'statusCode': 200,
               'body': json.dumps(result)
           }
       except InvalidTransactionError as e:
           return {
               'statusCode': 400,
               'body': json.dumps({'error': str(e)})
           }
       except Exception as e:
           return {
               'statusCode': 500,
               'body': json.dumps({'error': 'Internal error'})
           }

Package for Lambda:

.. code-block:: bash

   pip install metaguard -t package/
   cd package && zip -r ../deployment.zip .
   cd .. && zip deployment.zip handler.py

Performance Optimization
------------------------

Model Caching
^^^^^^^^^^^^^

Cache the detector instance:

.. code-block:: python

   from functools import lru_cache
   from metaguard import SimpleDetector

   @lru_cache(maxsize=1)
   def get_detector():
       return SimpleDetector()

   def detect_fraud(transaction):
       return get_detector().detect(transaction)

Batch Processing
^^^^^^^^^^^^^^^^

Use batch detection for multiple transactions:

.. code-block:: python

   detector = SimpleDetector()

   # More efficient than individual calls
   results = detector.batch_detect(transactions)

Connection Pooling (for API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from fastapi import FastAPI
   from contextlib import asynccontextmanager
   from metaguard import SimpleDetector

   detector = None

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       global detector
       detector = SimpleDetector()  # Load on startup
       yield
       detector = None  # Cleanup on shutdown

   app = FastAPI(lifespan=lifespan)

Monitoring
----------

Add metrics and logging:

.. code-block:: python

   import time
   import logging
   from metaguard import SimpleDetector
   from metaguard.utils.logging import setup_logging

   setup_logging(level="INFO", json_format=True)
   logger = logging.getLogger("metaguard.api")

   detector = SimpleDetector()

   def detect_with_metrics(transaction):
       start = time.time()
       try:
           result = detector.detect(transaction)
           elapsed = time.time() - start

           logger.info(
               "Detection completed",
               extra={
                   "latency_ms": elapsed * 1000,
                   "is_suspicious": result["is_suspicious"],
                   "risk_level": result["risk_level"],
               }
           )
           return result
       except Exception as e:
           logger.error(f"Detection failed: {e}")
           raise

Security Considerations
-----------------------

1. **Input Validation**: Always validate inputs
2. **Rate Limiting**: Protect against abuse
3. **Authentication**: Secure API endpoints
4. **Logging**: Audit all requests
5. **Model Security**: Protect model files

Example with rate limiting:

.. code-block:: python

   from fastapi import FastAPI, Request
   from slowapi import Limiter
   from slowapi.util import get_remote_address

   limiter = Limiter(key_func=get_remote_address)
   app = FastAPI()
   app.state.limiter = limiter

   @app.post("/detect")
   @limiter.limit("100/minute")
   async def detect(request: Request, transaction: dict):
       return detector.detect(transaction)

Next Steps
----------

- Set up monitoring dashboards
- Configure alerting
- Implement A/B testing for model updates
- Plan for model retraining
