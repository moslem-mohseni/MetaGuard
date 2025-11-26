"""
MetaGuard API Module

REST API for fraud detection.

Author: Moslem Mohseni
"""

from __future__ import annotations

from .schemas import (
    TransactionRequest,
    DetectionResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    RiskAnalysisResponse,
    ModelInfoResponse,
)

__all__: list[str] = [
    "TransactionRequest",
    "DetectionResponse",
    "BatchRequest",
    "BatchResponse",
    "HealthResponse",
    "RiskAnalysisResponse",
    "ModelInfoResponse",
]
