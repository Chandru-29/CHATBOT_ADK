"""
logger.py — Centralized professional logging configuration for SQL-CHATBOT.

Configures colorized, metadata-rich console logs for developers and 
plain-text rotating file logs for persistent auditing.

Usage:
    from config.logger import get_logger
    log = get_logger(__name__)
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler


class ColorFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI colors to log levels for console output."""
    
    # ANSI escape sequences for text colors
    GREY = "\x1b[90m"      # Grey for debug
    CYAN = "\x1b[36m"      # Cyan for info
    YELLOW = "\x1b[33m"    # Yellow for warning
    RED = "\x1b[31m"       # Red for error
    BOLD_RED = "\x1b[31;1m" # Bold Red for critical
    RESET = "\x1b[0m"
    
    # Structured format: [Time] | [Level] | [Filename:Line] | [Message]
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record):
        # Colorize the level name dynamically
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        orig_levelname = record.levelname
        record.levelname = f"{color}{orig_levelname}{self.RESET}"
        
        # Format the record using parent formatting logic
        formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.DATE_FORMAT)
        result = formatter.format(record)
        
        # Restore the original level name so other handlers receive it clean
        record.levelname = orig_levelname
        return result


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a colorized StreamHandler for console
    and a plain-text RotatingFileHandler for disk logging.
    
    Clears existing handlers to prevent duplicate output lines on reloads.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to prevent duplicate logging outputs on uvicorn reload
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        
    # ── Console Handler (Colorized, stdout) ───────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColorFormatter())
    root_logger.addHandler(console_handler)
    
    # ── File Handler (Plain text, rotating) ───────────────────────────────────
    try:
        # Get project root (parent directory of config/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "chatbot.log")
        # Rotate logs: Max 5MB per file, keep up to 3 backups
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(level)
        
        # Plain text formatter (without ANSI colors)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Log successful initialization
        logging.getLogger(__name__).info(
            f"Logging initialized. File logs rotating at {log_file}"
        )
    except Exception as e:
        # Fall back gracefully if file log directory is write-protected
        logging.getLogger(__name__).warning(
            f"Failed to initialize file logger: {e}. Console logging only."
        )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Modules call this instead of logging.getLogger directly."""
    return logging.getLogger(name)
