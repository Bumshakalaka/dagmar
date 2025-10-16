"""Logging configuration for Dagmar.

This module provides centralized logging setup with stdout-only output
and configurable log levels.
"""

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(log_level: LogLevel = "WARNING") -> None:
    """Configure logging for the application.

    Sets up a root logger with stdout handler and custom formatter.
    Ensures no duplicate handlers are created.

    Args:
        log_level: Logging level to set (default: WARNING)

    """
    # Get the root logger
    logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set the log level
    logger.setLevel(getattr(logging, log_level))

    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level))

    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)8s] [%(name)10s]: %(message)s")
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
