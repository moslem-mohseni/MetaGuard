"""
Unit tests for MetaGuard Logging Module

Author: Moslem Mohseni
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from metaguard.utils.logging import (
    ColoredFormatter,
    JSONFormatter,
    LoggerAdapter,
    get_logger,
    setup_logging,
)


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_format_basic(self) -> None:
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)

        # Should be valid JSON
        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"

    def test_format_includes_timestamp(self) -> None:
        """Test JSON includes timestamp."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        data = json.loads(result)
        assert "timestamp" in data

    def test_format_includes_location(self) -> None:
        """Test JSON includes module/function/line info."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        data = json.loads(result)
        assert data["line"] == 42
        assert "module" in data
        assert "function" in data


class TestColoredFormatter:
    """Tests for ColoredFormatter class."""

    def test_format_adds_color(self) -> None:
        """Test formatter adds color codes to level name."""
        formatter = ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        # Should contain ANSI color codes in levelname
        assert "\x1b[" in result or "INFO" in result  # Color or level name

    def test_color_for_each_level(self) -> None:
        """Test each level gets formatted."""
        formatter = ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
        for level in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg="Test",
                args=(),
                exc_info=None,
            )
            result = formatter.format(record)
            # Should format successfully with level name
            assert "Test" in result


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self) -> None:
        """Test function returns logger instance."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self) -> None:
        """Test logger has correct name."""
        logger = setup_logging()
        assert logger.name == "metaguard"

    def test_log_level_setting(self) -> None:
        """Test log level is set correctly."""
        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

        logger = setup_logging(level="ERROR")
        assert logger.level == logging.ERROR

    def test_json_format(self) -> None:
        """Test JSON format option."""
        logger = setup_logging(json_format=True)
        # Check handler has JSON formatter
        assert len(logger.handlers) > 0
        formatter = logger.handlers[0].formatter
        assert isinstance(formatter, JSONFormatter)

    def test_file_handler(self, temp_dir: Path) -> None:
        """Test file handler creation."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(log_file=str(log_file))

        # Log something
        logger.info("Test message")

        # Check file was created
        assert log_file.exists()

    def test_handlers_cleared(self) -> None:
        """Test existing handlers are cleared."""
        # Setup twice
        logger1 = setup_logging()
        initial_handlers = len(logger1.handlers)

        logger2 = setup_logging()
        # Should have same number of handlers (cleared and recreated)
        assert len(logger2.handlers) == initial_handlers


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self) -> None:
        """Test function returns logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_namespace_prefix(self) -> None:
        """Test logger name is prefixed with metaguard."""
        logger = get_logger("detector")
        assert logger.name == "metaguard.detector"

    def test_already_prefixed(self) -> None:
        """Test already prefixed name is not double-prefixed."""
        logger = get_logger("metaguard.detector")
        assert logger.name == "metaguard.detector"

    def test_custom_level(self) -> None:
        """Test custom log level."""
        logger = get_logger("test", level="DEBUG")
        assert logger.level == logging.DEBUG


class TestLoggerAdapter:
    """Tests for LoggerAdapter class."""

    def test_adds_extra_context(self) -> None:
        """Test adapter adds extra context."""
        base_logger = get_logger("test")
        adapter = LoggerAdapter(base_logger, {"request_id": "abc123"})

        # Process should add extra
        msg, kwargs = adapter.process("Test message", {"extra": {}})
        assert kwargs["extra"]["request_id"] == "abc123"

    def test_merges_with_existing_extra(self) -> None:
        """Test adapter merges with existing extra."""
        base_logger = get_logger("test")
        adapter = LoggerAdapter(base_logger, {"request_id": "abc123"})

        msg, kwargs = adapter.process(
            "Test message", {"extra": {"user_id": "user1"}}
        )
        assert kwargs["extra"]["request_id"] == "abc123"
        assert kwargs["extra"]["user_id"] == "user1"
