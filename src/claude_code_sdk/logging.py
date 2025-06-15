"""Logging and debug configuration for Claude Code SDK.

This module provides comprehensive logging capabilities:
- Configurable log levels and formats
- Structured logging with context
- Performance logging and profiling
- Debug mode with detailed traces
- Log filtering and routing
- Integration with external logging systems
"""

import functools
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TextIO, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class LogLevel(str, Enum):
    """Log levels with custom additions."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRACE = "TRACE"  # More detailed than DEBUG
    METRICS = "METRICS"  # For performance metrics


# Add custom log levels
TRACE_LEVEL = 5
METRICS_LEVEL = 25

logging.addLevelName(TRACE_LEVEL, "TRACE")
logging.addLevelName(METRICS_LEVEL, "METRICS")


@dataclass
class LogConfig:
    """Configuration for logging.
    
    Example:
        ```python
        config = LogConfig(
            level=LogLevel.INFO,
            format="json",
            enable_colors=True,
            log_file="claude.log",
            rotate_size=10_000_000,  # 10MB
            debug_mode=True
        )
        ```
    """

    level: LogLevel | str = LogLevel.INFO
    format: str = "json"  # "json", "text", "structured"
    enable_colors: bool = True
    log_file: Path | str | None = None
    log_to_console: bool = True
    rotate_size: int | None = 10_000_000  # 10MB
    rotate_count: int = 5
    debug_mode: bool = False
    trace_mode: bool = False
    metrics_enabled: bool = True
    include_timestamp: bool = True
    include_context: bool = True
    filter_sensitive: bool = True
    sensitive_keys: set[str] = field(
        default_factory=lambda: {
            "password", "token", "api_key", "secret", "auth", "authorization"
        }
    )
    context_vars: dict[str, Any] = field(default_factory=dict)


class ColorFormatter(logging.Formatter):
    """Formatter with color support for console output."""

    COLORS = {
        "TRACE": "\033[90m",  # Dark gray
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "METRICS": "\033[35m",  # Magenta
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[91m",  # Bright red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.config.include_context:
            log_data["context"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
                "process": record.process,
                "thread": record.thread,
            }

        # Add custom context variables
        if self.config.context_vars:
            log_data["custom_context"] = self.config.context_vars.copy()

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", 
                           "funcName", "levelname", "levelno", "lineno", 
                           "module", "msecs", "message", "pathname", "process",
                           "processName", "relativeCreated", "thread", "threadName"]:
                log_data[key] = value

        # Filter sensitive data if enabled
        if self.config.filter_sensitive:
            log_data = self._filter_sensitive(log_data)

        return json.dumps(log_data)

    def _filter_sensitive(self, data: Any) -> Any:
        """Recursively filter sensitive data."""
        if isinstance(data, dict):
            return {
                k: "***FILTERED***" if k.lower() in self.config.sensitive_keys else self._filter_sensitive(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._filter_sensitive(item) for item in data]
        return data


class StructuredFormatter(logging.Formatter):
    """Human-readable structured formatter."""

    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config

    def format(self, record: logging.LogRecord) -> str:
        parts = []
        
        if self.config.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            parts.append(f"[{timestamp}]")
        
        parts.append(f"[{record.levelname}]")
        parts.append(f"[{record.name}]")
        
        if self.config.include_context:
            parts.append(f"[{record.filename}:{record.lineno}]")
        
        parts.append(record.getMessage())
        
        # Add extra fields
        extras = []
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", 
                           "funcName", "levelname", "levelno", "lineno", 
                           "module", "msecs", "message", "pathname", "process",
                           "processName", "relativeCreated", "thread", "threadName"]:
                extras.append(f"{key}={value}")
        
        if extras:
            parts.append(f"[{', '.join(extras)}]")
        
        return " ".join(parts)


class PerformanceLogger:
    """Logger for performance metrics and profiling.
    
    Example:
        ```python
        perf_logger = PerformanceLogger()
        
        with perf_logger.measure("api_call"):
            # Code to measure
            pass
        
        # Or as decorator
        @perf_logger.measure_func
        def slow_function():
            pass
        ```
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, operation: str, **kwargs: Any):
        """Measure operation duration."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            
            # Log metric
            self.logger.log(
                METRICS_LEVEL,
                f"Operation '{operation}' completed",
                extra={
                    "operation": operation,
                    "duration_ms": duration * 1000,
                    "success": True,
                    **kwargs
                }
            )
            
            # Store metric
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)

    def measure_func(self, func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to measure function performance."""
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with self.measure(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def get_stats(self, operation: str | None = None) -> dict[str, Any]:
        """Get performance statistics."""
        if operation:
            durations = self.metrics.get(operation, [])
            if not durations:
                return {}
            
            return {
                "count": len(durations),
                "total_ms": sum(durations) * 1000,
                "avg_ms": (sum(durations) / len(durations)) * 1000,
                "min_ms": min(durations) * 1000,
                "max_ms": max(durations) * 1000,
            }
        
        # Return all stats
        return {
            op: self.get_stats(op) for op in self.metrics
        }


class LogManager:
    """Central log management for Claude Code SDK.
    
    Example:
        ```python
        from claude_code_sdk.logging import LogManager, LogConfig
        
        # Configure logging
        log_manager = LogManager()
        log_manager.configure(LogConfig(
            level=LogLevel.DEBUG,
            format="json",
            log_file="claude.log"
        ))
        
        # Get logger
        logger = log_manager.get_logger("my_module")
        logger.info("Hello", extra={"user_id": 123})
        
        # Performance logging
        perf = log_manager.get_performance_logger()
        with perf.measure("api_call"):
            # Your code
            pass
        ```
    """

    def __init__(self):
        self.config = LogConfig()
        self.loggers: dict[str, logging.Logger] = {}
        self.handlers: list[logging.Handler] = []
        self.performance_loggers: dict[str, PerformanceLogger] = {}
        self._configured = False

    def configure(self, config: LogConfig) -> None:
        """Configure logging system."""
        self.config = config
        
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(self._get_numeric_level())
        
        # Clear existing handlers
        for handler in self.handlers:
            root_logger.removeHandler(handler)
        self.handlers.clear()
        
        # Add console handler
        if config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._get_numeric_level())
            
            formatter = self._create_formatter()
            if config.enable_colors and config.format == "text":
                formatter = ColorFormatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            self.handlers.append(console_handler)
        
        # Add file handler
        if config.log_file:
            self._setup_file_handler(root_logger)
        
        self._configured = True

    def _get_numeric_level(self) -> int:
        """Get numeric log level."""
        if self.config.trace_mode:
            return TRACE_LEVEL
        
        level_map = {
            LogLevel.TRACE: TRACE_LEVEL,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.METRICS: METRICS_LEVEL,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        
        if isinstance(self.config.level, LogLevel):
            return level_map[self.config.level]
        
        return getattr(logging, self.config.level.upper(), logging.INFO)

    def _create_formatter(self) -> logging.Formatter:
        """Create formatter based on config."""
        if self.config.format == "json":
            return JSONFormatter(self.config)
        elif self.config.format == "structured":
            return StructuredFormatter(self.config)
        else:
            # Default text format
            return logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

    def _setup_file_handler(self, root_logger: logging.Logger) -> None:
        """Set up file handler with rotation."""
        from logging.handlers import RotatingFileHandler
        
        file_path = Path(self.config.log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.rotate_size:
            handler = RotatingFileHandler(
                str(file_path),
                maxBytes=self.config.rotate_size,
                backupCount=self.config.rotate_count,
            )
        else:
            handler = logging.FileHandler(str(file_path))
        
        handler.setLevel(self._get_numeric_level())
        handler.setFormatter(self._create_formatter())
        
        root_logger.addHandler(handler)
        self.handlers.append(handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance."""
        if not self._configured:
            self.configure(self.config)
        
        if name not in self.loggers:
            logger = logging.getLogger(name)
            
            # Add trace method
            def trace(msg: str, *args: Any, **kwargs: Any) -> None:
                logger.log(TRACE_LEVEL, msg, *args, **kwargs)
            
            logger.trace = trace  # type: ignore
            
            # Add metrics method
            def metrics(msg: str, *args: Any, **kwargs: Any) -> None:
                logger.log(METRICS_LEVEL, msg, *args, **kwargs)
            
            logger.metrics = metrics  # type: ignore
            
            self.loggers[name] = logger
        
        return self.loggers[name]

    def get_performance_logger(self, name: str = "performance") -> PerformanceLogger:
        """Get a performance logger."""
        if name not in self.performance_loggers:
            logger = self.get_logger(f"{name}.metrics")
            self.performance_loggers[name] = PerformanceLogger(logger)
        
        return self.performance_loggers[name]

    def set_context(self, **kwargs: Any) -> None:
        """Set global context variables."""
        self.config.context_vars.update(kwargs)

    def clear_context(self) -> None:
        """Clear global context variables."""
        self.config.context_vars.clear()

    @contextmanager
    def context(self, **kwargs: Any):
        """Temporary context for logging."""
        old_context = self.config.context_vars.copy()
        self.config.context_vars.update(kwargs)
        
        try:
            yield
        finally:
            self.config.context_vars = old_context

    def enable_debug(self) -> None:
        """Enable debug mode."""
        self.config.debug_mode = True
        self.config.level = LogLevel.DEBUG
        self.configure(self.config)

    def enable_trace(self) -> None:
        """Enable trace mode."""
        self.config.trace_mode = True
        self.config.level = LogLevel.TRACE
        self.configure(self.config)


# Global log manager instance
_log_manager = LogManager()


def configure_logging(
    level: LogLevel | str = LogLevel.INFO,
    format: str = "json",
    log_file: Path | str | None = None,
    debug: bool = False,
    trace: bool = False,
    **kwargs: Any
) -> LogManager:
    """Configure global logging.
    
    Args:
        level: Log level
        format: Output format ("json", "text", "structured")
        log_file: Optional log file path
        debug: Enable debug mode
        trace: Enable trace mode
        **kwargs: Additional config options
        
    Returns:
        Configured LogManager instance
        
    Example:
        ```python
        from claude_code_sdk.logging import configure_logging, LogLevel
        
        configure_logging(
            level=LogLevel.DEBUG,
            format="json",
            log_file="claude.log",
            enable_colors=True
        )
        ```
    """
    config = LogConfig(
        level=level,
        format=format,
        log_file=log_file,
        debug_mode=debug,
        trace_mode=trace,
        **kwargs
    )
    
    _log_manager.configure(config)
    return _log_manager


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
        
    Example:
        ```python
        from claude_code_sdk.logging import get_logger
        
        logger = get_logger(__name__)
        logger.info("Starting operation", extra={"user_id": 123})
        ```
    """
    return _log_manager.get_logger(name)


def get_performance_logger(name: str = "performance") -> PerformanceLogger:
    """Get a performance logger.
    
    Args:
        name: Logger name
        
    Returns:
        Performance logger instance
        
    Example:
        ```python
        from claude_code_sdk.logging import get_performance_logger
        
        perf = get_performance_logger()
        
        with perf.measure("database_query", query_type="select"):
            # Your database operation
            pass
        ```
    """
    return _log_manager.get_performance_logger(name)


def log_context(**kwargs: Any):
    """Context manager for temporary logging context.
    
    Example:
        ```python
        from claude_code_sdk.logging import log_context
        
        with log_context(request_id="123", user_id="456"):
            logger.info("Processing request")
            # All logs in this block will include the context
        ```
    """
    return _log_manager.context(**kwargs)