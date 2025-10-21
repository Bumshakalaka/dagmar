"""Logging configuration for Dagmar.

This module provides centralized logging setup with stderr output
and configurable log levels.
"""

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(log_level: LogLevel = "WARNING", stderr: bool = False) -> None:
    """Configure logging for the application.

    Sets up a 'dagmar' logger with the desired log level, and sends
    all other loggers' levels to WARNING. Ensures no duplicate handlers.

    Args:
        log_level: Log level to set for the 'dagmar' logger.
        stderr: Whether to write logs to stderr. If False, writes to stdout.

    """
    # Set up 'dagmar' logger (and submodules)
    dagmar_logger = logging.getLogger("dagmar")
    dagmar_logger.handlers.clear()
    dagmar_logger.setLevel(getattr(logging, log_level))

    handler = logging.StreamHandler(sys.stderr if stderr else sys.stdout)
    handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter("%(asctime)s [%(levelname)8s] [%(name)10s]: %(message)s")
    handler.setFormatter(formatter)
    dagmar_logger.addHandler(handler)
    dagmar_logger.propagate = False

    # Set all root and other loggers (except dagmar) to WARNING level
    logging.getLogger().setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        if not name.startswith("dagmar"):
            logging.getLogger(name).setLevel(logging.WARNING)
