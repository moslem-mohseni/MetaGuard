"""
MetaGuard Configuration Module

Provides configuration management for the MetaGuard library.

Author: Moslem Mohseni
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MetaGuardConfig(BaseModel):
    """
    Configuration settings for MetaGuard.

    This class manages all configuration options for the MetaGuard library,
    including model paths, detection thresholds, and logging settings.

    Author: Moslem Mohseni

    Attributes:
        model_path: Path to the trained model file
        model_type: Type of model to use (random_forest, xgboost)
        risk_threshold: Threshold for classifying transactions as suspicious
        log_level: Logging verbosity level
        log_file: Optional path to log file
        batch_size: Number of transactions to process in a batch
        n_jobs: Number of parallel jobs for processing (-1 for all cores)

    Example:
        >>> config = MetaGuardConfig(risk_threshold=0.7)
        >>> detector = SimpleDetector(config=config)
    """

    # Model settings
    model_path: str | None = Field(
        default=None,
        description="Path to the trained model file. If None, uses default location.",
    )
    model_type: Literal["random_forest", "xgboost", "ensemble"] = Field(
        default="random_forest",
        description="Type of model to use for detection",
    )

    # Detection settings
    risk_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for classifying transactions as suspicious (0-1)",
    )

    # Risk level thresholds
    low_risk_threshold: float = Field(
        default=40.0,
        ge=0.0,
        le=100.0,
        description="Upper bound for 'Low' risk level (0-100)",
    )
    high_risk_threshold: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Lower bound for 'High' risk level (0-100)",
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    log_file: str | None = Field(
        default=None,
        description="Path to log file. If None, logs only to console.",
    )
    log_json: bool = Field(
        default=False,
        description="If True, output logs in JSON format",
    )

    # Performance settings
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of transactions to process in a batch",
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Number of parallel jobs (-1 for all cores)",
    )

    # Feature settings
    use_feature_engineering: bool = Field(
        default=True,
        description="Whether to use advanced feature engineering",
    )

    model_config = {"extra": "forbid", "validate_default": True}

    @field_validator("risk_threshold")
    @classmethod
    def validate_risk_threshold(cls, v: float) -> float:
        """Validate risk threshold is within bounds."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("risk_threshold must be between 0 and 1")
        return v

    @field_validator("high_risk_threshold")
    @classmethod
    def validate_high_risk_threshold(cls, v: float, info: Any) -> float:
        """Validate high risk threshold is greater than low risk threshold."""
        low_threshold = info.data.get("low_risk_threshold", 40.0)
        if v <= low_threshold:
            raise ValueError("high_risk_threshold must be greater than low_risk_threshold")
        return v

    @classmethod
    def from_env(cls) -> MetaGuardConfig:
        """
        Create configuration from environment variables.

        Environment variables are prefixed with METAGUARD_.

        Author: Moslem Mohseni

        Returns:
            MetaGuardConfig instance with values from environment

        Example:
            >>> os.environ["METAGUARD_RISK_THRESHOLD"] = "0.7"
            >>> config = MetaGuardConfig.from_env()
            >>> config.risk_threshold
            0.7
        """
        env_mapping = {
            "model_path": os.getenv("METAGUARD_MODEL_PATH"),
            "model_type": os.getenv("METAGUARD_MODEL_TYPE", "random_forest"),
            "risk_threshold": float(os.getenv("METAGUARD_RISK_THRESHOLD", "0.5")),
            "log_level": os.getenv("METAGUARD_LOG_LEVEL", "INFO"),
            "log_file": os.getenv("METAGUARD_LOG_FILE"),
            "log_json": os.getenv("METAGUARD_LOG_JSON", "false").lower() == "true",
            "batch_size": int(os.getenv("METAGUARD_BATCH_SIZE", "100")),
            "n_jobs": int(os.getenv("METAGUARD_N_JOBS", "-1")),
            "use_feature_engineering": os.getenv(
                "METAGUARD_USE_FEATURE_ENGINEERING", "true"
            ).lower()
            == "true",
        }

        # Remove None values to use defaults
        env_mapping = {k: v for k, v in env_mapping.items() if v is not None}

        return cls(**env_mapping)

    def get_model_path(self) -> Path:
        """
        Get the resolved model path.

        If model_path is not set, returns the default model location.

        Author: Moslem Mohseni

        Returns:
            Path to the model file
        """
        if self.model_path:
            return Path(self.model_path)

        # Default model path relative to package
        package_dir = Path(__file__).parent.parent
        return package_dir / "models" / "model.pkl"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Author: Moslem Mohseni

        Returns:
            Dictionary representation of the configuration
        """
        return self.model_dump()


# Global configuration instance
_config: MetaGuardConfig | None = None


def get_config() -> MetaGuardConfig:
    """
    Get the global configuration instance.

    Creates a new configuration from environment variables if not already set.

    Author: Moslem Mohseni

    Returns:
        Global MetaGuardConfig instance

    Example:
        >>> config = get_config()
        >>> print(config.risk_threshold)
        0.5
    """
    global _config
    if _config is None:
        _config = MetaGuardConfig.from_env()
    return _config


def set_config(config: MetaGuardConfig) -> None:
    """
    Set the global configuration instance.

    Author: Moslem Mohseni

    Args:
        config: MetaGuardConfig instance to use globally

    Example:
        >>> custom_config = MetaGuardConfig(risk_threshold=0.8)
        >>> set_config(custom_config)
    """
    global _config
    _config = config


def reset_config() -> None:
    """
    Reset the global configuration to default.

    Author: Moslem Mohseni
    """
    global _config
    _config = None


@lru_cache(maxsize=1)
def get_default_model_path() -> Path:
    """
    Get the default model path.

    Author: Moslem Mohseni

    Returns:
        Path to the default model file
    """
    package_dir = Path(__file__).parent.parent
    return package_dir / "models" / "model.pkl"
