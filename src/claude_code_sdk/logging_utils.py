"""
Logging utilities for Claude Code SDK.

This module provides a flexible logging configuration with support for:
- Multiple log levels
- Colored console output
- File logging with rotation
- Structured logging
- Context managers for temporary log level changes
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Any, Dict
from datetime import datetime
import json
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager


# ANSI color codes for console output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to the level name
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            levelname = record.levelname
            color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
            record.levelname = f"{color}{levelname}{Colors.RESET}"
            
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname to original (in case record is used elsewhere)
        record.levelname = record.levelname.replace(Colors.RESET, '').replace(color, '')
        
        return formatted


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs logs as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


class SDKLogger:
    """
    Enhanced logger for Claude Code SDK with additional features.
    
    Features:
    - Automatic logger naming based on module
    - Built-in performance logging
    - Context tracking
    - Structured logging support
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._context: Dict[str, Any] = {}
    
    def add_context(self, **kwargs) -> None:
        """Add persistent context that will be included in all log messages."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all persistent context."""
        self._context.clear()
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Internal method to log with context."""
        extra = kwargs.get('extra', {})
        extra['extra_fields'] = {**self._context, **kwargs.get('extra_fields', {})}
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    @contextmanager
    def timer(self, operation: str):
        """Context manager to log operation duration."""
        start = datetime.now()
        self.debug(f"Starting {operation}")
        try:
            yield
        finally:
            duration = (datetime.now() - start).total_seconds()
            self.info(f"Completed {operation} in {duration:.3f}s", 
                     extra_fields={'operation': operation, 'duration_seconds': duration})


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    colored: bool = True,
    structured: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure logging for the SDK.
    
    Args:
        level: Log level (can be string like 'DEBUG' or int like logging.DEBUG)
        log_file: Path to log file (optional)
        console: Whether to log to console
        colored: Whether to use colored output for console
        structured: Whether to use structured (JSON) logging
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Convert string level to int if necessary
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        
        if structured:
            console_formatter = StructuredFormatter()
        elif colored:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> SDKLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (if None, uses calling module's name)
        level: Log level
        
    Returns:
        SDKLogger instance
    """
    if name is None:
        # Get calling module's name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'claude_code_sdk')
    
    return SDKLogger(name, level)


@contextmanager
def log_level(level: Union[str, int]):
    """
    Context manager to temporarily change log level.
    
    Example:
        with log_level('DEBUG'):
            # Debug logging enabled here
            logger.debug("This will be shown")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(old_level)


# Convenience functions
def debug(msg: str, *args, **kwargs):
    """Log a debug message."""
    logging.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log an info message."""
    logging.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log a warning message."""
    logging.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log an error message."""
    logging.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log a critical message."""
    logging.critical(msg, *args, **kwargs)


# Initialize default logging
setup_logging(level=logging.INFO, colored=True)