"""
MetaGuard Logging Module

Provides structured logging capabilities for the MetaGuard library.

Author: Moslem Mohseni
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing and analysis.

    Author: Moslem Mohseni

    Example:
        >>> logger = get_logger("metaguard", json_format=True)
        >>> logger.info("Transaction processed", extra={"tx_id": "123"})
        {"timestamp": "2024-01-01T12:00:00Z", "level": "INFO", "message": "Transaction processed", "tx_id": "123"}
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON string representation of the log record
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra") and record.extra:
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.

    Adds ANSI color codes to log levels for better visibility.

    Author: Moslem Mohseni
    """

    # ANSI color codes
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        """Initialize the formatter with optional format string."""
        super().__init__(fmt, datefmt)
        self._fmt = fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    json_format: bool = False,
    colored: bool = True,
) -> logging.Logger:
    """
    Configure the MetaGuard logging system.

    Sets up logging for the entire MetaGuard library with the specified
    configuration options.

    Author: Moslem Mohseni

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file
        json_format: If True, output logs in JSON format
        colored: If True and not json_format, use colored console output

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="metaguard.log")
        >>> logger.info("MetaGuard initialized")
    """
    logger = logging.getLogger("metaguard")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    elif colored and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))

        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
                )
            )

        logger.addHandler(file_handler)

    return logger


def get_logger(
    name: str = "metaguard",
    level: str | None = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Creates or retrieves a logger with the specified name. If the logger
    doesn't exist, it will be created as a child of the metaguard logger.

    Author: Moslem Mohseni

    Args:
        name: Name of the logger (will be prefixed with 'metaguard.')
        level: Optional logging level override
        json_format: If True, use JSON format for this logger

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger("detector")
        >>> logger.info("Processing transaction")
    """
    # Ensure name is under metaguard namespace
    if not name.startswith("metaguard"):
        name = f"metaguard.{name}"

    logger = logging.getLogger(name)

    # Set level if specified
    if level:
        logger.setLevel(getattr(logging, level.upper()))

    # If no handlers exist on parent, set up basic logging
    parent_logger = logging.getLogger("metaguard")
    if not parent_logger.handlers:
        setup_logging(level=level or "INFO", json_format=json_format)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for adding context to log messages.

    Author: Moslem Mohseni

    Example:
        >>> logger = get_logger("detector")
        >>> ctx_logger = LoggerAdapter(logger, {"request_id": "abc123"})
        >>> ctx_logger.info("Processing request")
    """

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Process the log message and add context."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs
