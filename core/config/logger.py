"""
logger.py — Centralized professional logging configuration for SQL-CHATBOT.

Configures colorized, metadata-rich console logs for developers and 
plain-text rotating file logs for persistent auditing.

Usage:
    from core.config.logger import get_logger
    log = get_logger(__name__)
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler


class ColorFormatter(logging.Formatter):
    """Custom logging formatter adding ANSI color codes to log levels for console output.

    Attributes:
        GREY (str): ANSI grey code string.
        CYAN (str): ANSI cyan code string.
        YELLOW (str): ANSI yellow code string.
        RED (str): ANSI red code string.
        BOLD_RED (str): ANSI bold red code string.
        RESET (str): ANSI reset code string.
    """

    GREY = "\x1b[90m"
    CYAN = "\x1b[36m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified log record with ANSI level colorization.

        Args:
            record (logging.LogRecord): Log record instance to format.

        Returns:
            str: Colorized formatted log string.
        """
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        orig_levelname = record.levelname
        record.levelname = f"{color}{orig_levelname}{self.RESET}"

        formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.DATE_FORMAT)
        result = formatter.format(record)

        record.levelname = orig_levelname
        return result


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a colorized StreamHandler and RotatingFileHandler.

    Args:
        level (int, optional): Logging level threshold int. Defaults to logging.INFO.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    for quiet_logger in ("httpx", "httpcore", "openai", "google.genai", "google", "urllib3", "asyncio"):
        logging.getLogger(quiet_logger).setLevel(logging.WARNING)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColorFormatter())
    root_logger.addHandler(console_handler)

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "chatbot.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(level)

        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.getLogger(__name__).info(
            f"Logging initialized. File logs rotating at {log_file}"
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Failed to initialize file logger: {e}. Console logging only."
        )


def get_logger(name: str) -> logging.Logger:
    """Retrieve named Logger instance.

    Args:
        name (str): Logger namespace name string.

    Returns:
        logging.Logger: Named Logger instance.
    """
    return logging.getLogger(name)
