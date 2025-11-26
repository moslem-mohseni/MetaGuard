"""
REST API for MetaGuard

Author: Moslem Mohseni

FastAPI-based REST API for fraud detection.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__, SimpleDetector, analyze_transaction_risk
from ..utils.exceptions import InvalidTransactionError, MetaGuardError
from ..utils.logging import get_logger
from .schemas import (
    TransactionRequest,
    DetectionResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    RiskAnalysisResponse,
    ModelInfoResponse,
    ErrorResponse,
    RiskFactors,
)

logger = get_logger(__name__)

# Global detector instance
_detector: SimpleDetector | None = None


def get_detector() -> SimpleDetector:
    """Get or create the detector instance."""
    global _detector
    if _detector is None:
        _detector = SimpleDetector()
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MetaGuard API...")
    global _detector
    _detector = SimpleDetector()
    logger.info("Model loaded successfully")
    yield
    # Shutdown
    logger.info("Shutting down MetaGuard API...")
    _detector = None


# Create FastAPI app
app = FastAPI(
    title="MetaGuard API",
    description="Fraud detection for metaverse transactions",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(InvalidTransactionError)
async def invalid_transaction_handler(
    request: Request, exc: InvalidTransactionError
) -> JSONResponse:
    """Handle invalid transaction errors."""
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "error_type": "InvalidTransactionError",
        },
    )


@app.exception_handler(MetaGuardError)
async def metaguard_error_handler(
    request: Request, exc: MetaGuardError
) -> JSONResponse:
    """Handle MetaGuard errors."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": exc.__class__.__name__,
        },
    )


# Endpoints
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns the current status of the API and whether the model is loaded.
    """
    try:
        detector = get_detector()
        return HealthResponse(
            status="healthy",
            version=__version__,
            model_loaded=detector.model is not None,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            model_loaded=False,
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Alias for health check endpoint."""
    return await health_check()


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Detection"],
)
async def detect_fraud(transaction: TransactionRequest) -> DetectionResponse:
    """Detect fraud in a single transaction.

    Analyzes the provided transaction and returns whether it appears suspicious,
    along with a risk score and risk level.
    """
    start_time = time.time()

    detector = get_detector()
    result = detector.detect(transaction.model_dump())

    processing_time = (time.time() - start_time) * 1000

    return DetectionResponse(
        is_suspicious=bool(result["is_suspicious"]),
        risk_score=float(result["risk_score"]),
        risk_level=result["risk_level"],
        processing_time_ms=round(processing_time, 2),
    )


@app.post(
    "/detect/batch",
    response_model=BatchResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Detection"],
)
async def detect_fraud_batch(request: BatchRequest) -> BatchResponse:
    """Detect fraud in multiple transactions.

    Processes a batch of transactions and returns detection results for each.
    Maximum batch size is 1000 transactions.
    """
    start_time = time.time()

    detector = get_detector()
    transactions = [t.model_dump() for t in request.transactions]
    results = detector.batch_detect(transactions)

    processing_time = (time.time() - start_time) * 1000

    detection_results = []
    suspicious_count = 0
    per_tx_time = processing_time / len(results) if results else 0

    for result in results:
        is_suspicious = bool(result["is_suspicious"])
        if is_suspicious:
            suspicious_count += 1

        detection_results.append(
            DetectionResponse(
                is_suspicious=is_suspicious,
                risk_score=float(result["risk_score"]),
                risk_level=result["risk_level"],
                processing_time_ms=round(per_tx_time, 2),
            )
        )

    return BatchResponse(
        results=detection_results,
        total_transactions=len(results),
        suspicious_count=suspicious_count,
        processing_time_ms=round(processing_time, 2),
    )


@app.post(
    "/analyze",
    response_model=RiskAnalysisResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Analysis"],
)
async def analyze_risk(transaction: TransactionRequest) -> RiskAnalysisResponse:
    """Perform detailed risk analysis on a transaction.

    Returns a detailed breakdown of risk factors and scores.
    """
    start_time = time.time()

    result = analyze_transaction_risk(transaction.model_dump())

    processing_time = (time.time() - start_time) * 1000

    return RiskAnalysisResponse(
        risk_score=float(result["risk_score"]),
        risk_level=result["risk_level"],
        factors=RiskFactors(**result["factors"]),
        active_factor_count=result["active_factor_count"],
        processing_time_ms=round(processing_time, 2),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info() -> ModelInfoResponse:
    """Get information about the loaded model.

    Returns details about the ML model being used for detection.
    """
    detector = get_detector()
    info = detector.get_model_info()

    return ModelInfoResponse(
        model_type=info["model_type"],
        model_path=str(info["model_path"]) if info.get("model_path") else None,
        risk_threshold=info["risk_threshold"],
        feature_names=["amount", "hour", "user_age_days", "transaction_count"],
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the API server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Enable auto-reload for development.
    """
    import uvicorn

    uvicorn.run(
        "metaguard.api.rest:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
