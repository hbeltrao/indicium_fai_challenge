"""
Centralized Logging Configuration.

This module provides a structured logging system with:
- Console output with colored formatting
- Optional file logging with rotation
- Log levels configurable via environment
- JSON-structured logs for observability tools
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os


# ANSI color codes for console output
class LogColors:
    """ANSI escape codes for colored console output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors based on log level."""
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED + LogColors.BOLD,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        record.levelname = f"{color}{record.levelname:8}{LogColors.RESET}"
        
        # Add color to logger name
        record.name = f"{LogColors.CYAN}{record.name}{LogColors.RESET}"
        
        return super().format(record)


def setup_logging(
    level: Optional[int] = None,
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Args:
        level: Logging level (default: from LOG_LEVEL env var or INFO)
        log_to_file: Whether to also log to file
        log_dir: Directory for log files
        
    Returns:
        Configured root logger for the application
    """
    # Determine log level from environment or default
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    
    # Get or create the application logger
    logger = logging.getLogger("indicium_fai")
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(level)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = ColoredFormatter(
        '%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (plain text, more detailed)
    if log_to_file:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(exist_ok=True)
            
            log_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(
                log_path / log_filename,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG in file
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
            logger.debug(f"Log file created: {log_path / log_filename}")
        except Exception as e:
            logger.warning(f"Could not create log file: {e}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.
    
    Args:
        name: Module or component name (e.g., 'tools.data', 'agents.news')
        
    Returns:
        Logger instance with hierarchical naming
        
    Example:
        >>> logger = get_logger("tools.data")
        >>> logger.info("Processing dataset...")
    """
    return logging.getLogger(f"indicium_fai.{name}")


# Initialize the root logger on module import
_root_logger = setup_logging()
