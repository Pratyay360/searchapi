"""
Logging utilities for DataOrchestra.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import get_config, LogLevel


def setup_logging(
    level: Optional[LogLevel] = None,
    console: bool = True,
    file_path: Optional[Path] = None,
    format_str: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration for DataOrchestra.
    
    Args:
        level: Logging level
        console: Whether to log to console
        file_path: Path to log file (optional)
        format_str: Custom format string
        date_format: Custom date format
    
    Returns:
        Configured logger
    """
    config = get_config()
    
    # Use provided values or fall back to config
    log_level = level or config.logging.level
    console_output = console if console is not None else config.logging.console_output
    file_output = file_path or config.logging.file_output
    format_string = format_str or config.logging.format
    datefmt = date_format or config.logging.date_format
    
    # Convert LogLevel to logging level
    logging_level = getattr(logging, log_level.value.upper())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging_level)
        console_formatter = logging.Formatter(format_string, datefmt)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if file_output:
        file_output.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_output)
        file_handler.setLevel(logging_level)
        file_formatter = logging.Formatter(format_string, datefmt)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set up specific logger for DataOrchestra
    logger = logging.getLogger("DataOrchestra")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"DataOrchestra.{name}")


def log_function_call(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """Log a function call with its parameters.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Function parameters
    """
    if logger.isEnabledFor(logging.DEBUG):
        params = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        logger.debug(f"Calling {func_name}({params})")


def log_function_result(logger: logging.Logger, func_name: str, result, duration: float) -> None:
    """Log a function result with execution time.
    
    Args:
        logger: Logger instance
        func_name: Name of the function that was called
        result: Function result
        duration: Execution time in seconds
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{func_name} completed in {duration:.3f}s -> {result!r}")


class LogContext:
    """Context manager for logging function execution."""
    
    def __init__(self, logger: logging.Logger, func_name: str, **kwargs):
        self.logger = logger
        self.func_name = func_name
        self.kwargs = kwargs
        self.duration = None
    
    def __enter__(self):
        log_function_call(self.logger, self.func_name, **self.kwargs)
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.duration = time.time() - self.start_time
        
        if exc_type is None:
            log_function_result(self.logger, self.func_name, "SUCCESS", self.duration)
        else:
            self.logger.error(
                f"{self.func_name} failed after {self.duration:.3f}s: {exc_type.__name__}: {exc_val}"
            )