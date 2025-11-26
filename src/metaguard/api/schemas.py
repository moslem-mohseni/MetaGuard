"""
API Schemas for MetaGuard

Author: Moslem Mohseni

Pydantic models for API request/response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Request schema for single transaction detection."""

    amount: float = Field(..., gt=0, description="Transaction amount (must be > 0)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    user_age_days: int = Field(..., ge=1, description="Account age in days (>= 1)")
    transaction_count: int = Field(..., ge=0, description="Number of prior transactions")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "amount": 5000,
                    "hour": 3,
                    "user_age_days": 5,
                    "transaction_count": 50,
                }
            ]
        }
    }


class DetectionResponse(BaseModel):
    """Response schema for single transaction detection."""

    is_suspicious: bool = Field(..., description="Whether transaction is flagged as suspicious")
    risk_score: float = Field(..., ge=0, le=1, description="Risk probability (0.0-1.0)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "is_suspicious": True,
                    "risk_score": 0.95,
                    "risk_level": "High",
                    "processing_time_ms": 12.5,
                }
            ]
        }
    }


class BatchRequest(BaseModel):
    """Request schema for batch transaction detection."""

    transactions: list[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of transactions to check"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transactions": [
                        {"amount": 100, "hour": 14, "user_age_days": 30, "transaction_count": 5},
                        {"amount": 5000, "hour": 3, "user_age_days": 5, "transaction_count": 50},
                    ]
                }
            ]
        }
    }


class BatchResponse(BaseModel):
    """Response schema for batch transaction detection."""

    results: list[DetectionResponse] = Field(..., description="Detection results for each transaction")
    total_transactions: int = Field(..., description="Total number of transactions processed")
    suspicious_count: int = Field(..., description="Number of suspicious transactions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {"is_suspicious": False, "risk_score": 0.1, "risk_level": "Low", "processing_time_ms": 5.0},
                        {"is_suspicious": True, "risk_score": 0.95, "risk_level": "High", "processing_time_ms": 5.0},
                    ],
                    "total_transactions": 2,
                    "suspicious_count": 1,
                    "processing_time_ms": 10.0,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service status: healthy or unhealthy")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "1.1.0",
                    "model_loaded": True,
                }
            ]
        }
    }


class RiskFactors(BaseModel):
    """Risk factors breakdown."""

    high_amount: bool = Field(..., description="Amount > $1000")
    new_account: bool = Field(..., description="Account age < 30 days")
    high_frequency: bool = Field(..., description="Transaction count > 20")
    unusual_hour: bool = Field(..., description="Hour between 0-5")


class RiskAnalysisResponse(BaseModel):
    """Response schema for detailed risk analysis."""

    risk_score: float = Field(..., description="Risk score (0-100)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    factors: RiskFactors = Field(..., description="Individual risk factors")
    active_factor_count: int = Field(..., description="Number of active risk factors")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "risk_score": 85.5,
                    "risk_level": "High",
                    "factors": {
                        "high_amount": True,
                        "new_account": True,
                        "high_frequency": True,
                        "unusual_hour": True,
                    },
                    "active_factor_count": 4,
                    "processing_time_ms": 8.2,
                }
            ]
        }
    }


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    model_type: str = Field(..., description="Type of ML model")
    model_path: str | None = Field(None, description="Path to model file")
    risk_threshold: float = Field(..., description="Risk threshold for detection")
    feature_names: list[str] = Field(..., description="Feature names used by model")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_type": "RandomForestClassifier",
                    "model_path": "/path/to/model.pkl",
                    "risk_threshold": 0.5,
                    "feature_names": ["amount", "hour", "user_age_days", "transaction_count"],
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "Invalid transaction: amount must be positive",
                    "error_type": "InvalidTransactionError",
                }
            ]
        }
    }
