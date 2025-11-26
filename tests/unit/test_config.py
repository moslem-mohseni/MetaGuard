"""
Unit tests for MetaGuard Config Module

Author: Moslem Mohseni
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from metaguard.utils.config import (
    MetaGuardConfig,
    get_config,
    get_default_model_path,
    reset_config,
    set_config,
)


class TestMetaGuardConfig:
    """Tests for MetaGuardConfig class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MetaGuardConfig()
        assert config.model_path is None
        assert config.model_type == "random_forest"
        assert config.risk_threshold == 0.5
        assert config.log_level == "INFO"
        assert config.batch_size == 100
        assert config.n_jobs == -1

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = MetaGuardConfig(
            model_path="/custom/path.pkl",
            model_type="xgboost",
            risk_threshold=0.7,
            log_level="DEBUG",
            batch_size=200,
        )
        assert config.model_path == "/custom/path.pkl"
        assert config.model_type == "xgboost"
        assert config.risk_threshold == 0.7
        assert config.log_level == "DEBUG"
        assert config.batch_size == 200

    def test_risk_threshold_bounds(self) -> None:
        """Test risk_threshold must be between 0 and 1."""
        with pytest.raises(ValidationError):
            MetaGuardConfig(risk_threshold=1.5)

        with pytest.raises(ValidationError):
            MetaGuardConfig(risk_threshold=-0.1)

    def test_risk_threshold_edge_values(self) -> None:
        """Test risk_threshold edge values."""
        config = MetaGuardConfig(risk_threshold=0.0)
        assert config.risk_threshold == 0.0

        config = MetaGuardConfig(risk_threshold=1.0)
        assert config.risk_threshold == 1.0

    def test_high_risk_threshold_validation(self) -> None:
        """Test high_risk_threshold must be greater than low_risk_threshold."""
        with pytest.raises(ValidationError):
            MetaGuardConfig(low_risk_threshold=70, high_risk_threshold=40)

    def test_batch_size_bounds(self) -> None:
        """Test batch_size bounds."""
        with pytest.raises(ValidationError):
            MetaGuardConfig(batch_size=0)

        with pytest.raises(ValidationError):
            MetaGuardConfig(batch_size=10001)

    def test_valid_model_types(self) -> None:
        """Test valid model types."""
        for model_type in ["random_forest", "xgboost", "ensemble"]:
            config = MetaGuardConfig(model_type=model_type)
            assert config.model_type == model_type

    def test_invalid_model_type(self) -> None:
        """Test invalid model type raises error."""
        with pytest.raises(ValidationError):
            MetaGuardConfig(model_type="invalid_model")

    def test_valid_log_levels(self) -> None:
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = MetaGuardConfig(log_level=level)
            assert config.log_level == level

    def test_from_env(self) -> None:
        """Test configuration from environment variables."""
        env_vars = {
            "METAGUARD_RISK_THRESHOLD": "0.8",
            "METAGUARD_LOG_LEVEL": "DEBUG",
            "METAGUARD_BATCH_SIZE": "50",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = MetaGuardConfig.from_env()
            assert config.risk_threshold == 0.8
            assert config.log_level == "DEBUG"
            assert config.batch_size == 50

    def test_from_env_defaults(self) -> None:
        """Test configuration uses defaults when env vars not set."""
        # Clear any existing METAGUARD_ env vars
        env_to_clear = [k for k in os.environ if k.startswith("METAGUARD_")]
        with patch.dict(os.environ, dict.fromkeys(env_to_clear, ""), clear=False):
            config = MetaGuardConfig.from_env()
            # Should use default values
            assert config.model_type == "random_forest"

    def test_get_model_path_custom(self) -> None:
        """Test get_model_path with custom path."""
        config = MetaGuardConfig(model_path="/custom/model.pkl")
        path = config.get_model_path()
        assert path == Path("/custom/model.pkl")

    def test_get_model_path_default(self) -> None:
        """Test get_model_path with default path."""
        config = MetaGuardConfig()
        path = config.get_model_path()
        assert path.name == "model.pkl"
        assert "models" in str(path)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = MetaGuardConfig(risk_threshold=0.7)
        data = config.to_dict()
        assert isinstance(data, dict)
        assert data["risk_threshold"] == 0.7
        assert "model_type" in data
        assert "log_level" in data

    def test_extra_fields_forbidden(self) -> None:
        """Test extra fields are not allowed."""
        with pytest.raises(ValidationError):
            MetaGuardConfig(extra_field="value")


class TestGlobalConfig:
    """Tests for global configuration functions."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        reset_config()

    def test_get_config_creates_default(self) -> None:
        """Test get_config creates default config."""
        config = get_config()
        assert isinstance(config, MetaGuardConfig)

    def test_get_config_returns_same_instance(self) -> None:
        """Test get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self) -> None:
        """Test set_config sets global config."""
        custom_config = MetaGuardConfig(risk_threshold=0.9)
        set_config(custom_config)

        config = get_config()
        assert config.risk_threshold == 0.9

    def test_reset_config(self) -> None:
        """Test reset_config clears global config."""
        custom_config = MetaGuardConfig(risk_threshold=0.9)
        set_config(custom_config)

        reset_config()

        # Should create new default config
        config = get_config()
        assert config.risk_threshold == 0.5


class TestGetDefaultModelPath:
    """Tests for get_default_model_path function."""

    def test_returns_path(self) -> None:
        """Test function returns Path object."""
        path = get_default_model_path()
        assert isinstance(path, Path)

    def test_path_ends_with_model_pkl(self) -> None:
        """Test path ends with model.pkl."""
        path = get_default_model_path()
        assert path.name == "model.pkl"

    def test_path_contains_models_dir(self) -> None:
        """Test path contains models directory."""
        path = get_default_model_path()
        assert "models" in str(path)

    def test_cached(self) -> None:
        """Test function result is cached."""
        path1 = get_default_model_path()
        path2 = get_default_model_path()
        # Should return exact same object due to lru_cache
        assert path1 is path2
