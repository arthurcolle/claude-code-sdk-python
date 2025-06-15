"""Enhanced subprocess transport with retry logic and better error handling."""

import asyncio
import json
import logging
import os
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from subprocess import PIPE
from typing import Any, Optional, Callable
from dataclasses import dataclass

import anyio
from anyio.abc import Process
from anyio.streams.text import TextReceiveStream

from ..._errors import CLIConnectionError, CLINotFoundError, ProcessError
from ..._errors import CLIJSONDecodeError as SDKJSONDecodeError
from ...types import ClaudeCodeOptions
from .subprocess_cli import SubprocessCLITransport

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: tuple[type[Exception], ...] = (
        CLIConnectionError,
        ProcessError,
        asyncio.TimeoutError,
    )


class EnhancedSubprocessCLITransport(SubprocessCLITransport):
    """Enhanced subprocess transport with retry logic and improved error handling."""

    def __init__(
        self,
        prompt: str,
        options: ClaudeCodeOptions,
        cli_path: str | Path | None = None,
        retry_config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
    ):
        super().__init__(prompt, options, cli_path)
        self.retry_config = retry_config or RetryConfig()
        self.on_retry = on_retry
        self._connection_attempts = 0
        self._last_error: Optional[Exception] = None
        self._start_time = time.time()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        delay = min(
            self.retry_config.initial_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            # Add random jitter (±25%)
            import random
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter
        
        return max(0, delay)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        return isinstance(error, self.retry_config.retryable_errors)

    async def connect_with_retry(self) -> None:
        """Connect with automatic retry on failure."""
        last_error = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                self._connection_attempts += 1
                await self.connect()
                logger.info(f"Successfully connected after {attempt + 1} attempt(s)")
                return
                
            except Exception as e:
                last_error = e
                self._last_error = e
                
                if not self._is_retryable_error(e) or attempt == self.retry_config.max_attempts - 1:
                    logger.error(f"Connection failed permanently: {e}")
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                
                if self.on_retry:
                    self.on_retry(attempt + 1, e)
                
                await asyncio.sleep(delay)
                
                # Clean up before retry
                await self.disconnect()
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error

    async def receive_messages_with_recovery(self) -> AsyncIterator[dict[str, Any]]:
        """Receive messages with automatic recovery on stream errors."""
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while True:
            try:
                async for message in self.receive_messages():
                    consecutive_errors = 0  # Reset on successful message
                    yield message
                
                # Normal completion
                break
                
            except (asyncio.TimeoutError, anyio.ClosedResourceError) as e:
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), giving up")
                    raise
                
                logger.warning(
                    f"Stream error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                
                # Try to recover by checking process status
                if not self.is_connected():
                    logger.info("Process disconnected, attempting reconnection...")
                    await self.connect_with_retry()
                else:
                    # Brief pause before retry
                    await asyncio.sleep(0.5)
            
            except SDKJSONDecodeError as e:
                # Log but continue - might be a partial message
                logger.warning(f"JSON decode error (continuing): {e}")
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    raise

    async def send_request_with_timeout(
        self,
        messages: list[Any],
        options: dict[str, Any],
        timeout: float = 30.0
    ) -> None:
        """Send request with timeout (not used for CLI but included for interface compatibility)."""
        # CLI transport doesn't use this method, but we can add timeout to the process
        pass

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about the transport."""
        return {
            "connected": self.is_connected(),
            "connection_attempts": self._connection_attempts,
            "last_error": str(self._last_error) if self._last_error else None,
            "uptime_seconds": time.time() - self._start_time if self.is_connected() else 0,
            "cli_path": self._cli_path,
            "process_pid": self._process.pid if self._process else None,
        }


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half_open
        self._half_open_calls = 0

    def call_succeeded(self):
        """Record a successful call."""
        self._failure_count = 0
        self._half_open_calls = 0
        self._state = "closed"

    def call_failed(self):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == "half_open":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = "open"
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"

    def can_proceed(self) -> bool:
        """Check if calls can proceed."""
        if self._state == "closed":
            return True
            
        if self._state == "open":
            if self._last_failure_time and \
               time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "half_open"
                self._half_open_calls = 0
                return True
            return False
            
        # half_open
        return self._half_open_calls < self.half_open_max_calls


class RobustSubprocessCLITransport(EnhancedSubprocessCLITransport):
    """Most robust transport with circuit breaker and advanced error handling."""
    
    def __init__(
        self,
        prompt: str,
        options: ClaudeCodeOptions,
        cli_path: str | Path | None = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        super().__init__(prompt, options, cli_path, retry_config)
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._message_buffer: list[dict[str, Any]] = []
        self._partial_line_buffer = ""

    async def connect(self) -> None:
        """Connect with circuit breaker protection."""
        if not self.circuit_breaker.can_proceed():
            raise CLIConnectionError(
                "Circuit breaker is open due to repeated failures. "
                f"Will retry after {self.circuit_breaker.recovery_timeout} seconds."
            )
        
        try:
            await super().connect()
            self.circuit_breaker.call_succeeded()
        except Exception as e:
            self.circuit_breaker.call_failed()
            raise

    async def receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Enhanced message receiving with better error recovery."""
        if not self._process or not self._stdout_stream:
            raise CLIConnectionError("Not connected")

        stderr_buffer = []
        
        async def collect_stderr() -> None:
            """Collect stderr for error diagnosis."""
            if self._stderr_stream:
                try:
                    async for line in self._stderr_stream:
                        stderr_buffer.append(line.strip())
                        # Log important stderr messages immediately
                        if any(keyword in line.lower() for keyword in ['error', 'warning', 'fatal']):
                            logger.warning(f"CLI stderr: {line.strip()}")
                except anyio.ClosedResourceError:
                    pass

        async with anyio.create_task_group() as tg:
            tg.start_soon(collect_stderr)
            
            try:
                async for line in self._stdout_stream:
                    # Handle partial lines from stream
                    if self._partial_line_buffer:
                        line = self._partial_line_buffer + line
                        self._partial_line_buffer = ""
                    
                    line_str = line.strip()
                    if not line_str:
                        continue
                    
                    # Check if line is complete JSON
                    if not (line_str.startswith('{') or line_str.startswith('[')):
                        # Not JSON, might be debug output
                        logger.debug(f"Non-JSON output: {line_str}")
                        continue
                    
                    try:
                        # Attempt to parse JSON
                        data = json.loads(line_str)
                        self._message_buffer.append(data)
                        yield data
                        
                    except json.JSONDecodeError as e:
                        # Check if it's a partial JSON line
                        if line_str.count('{') > line_str.count('}'):
                            # Incomplete JSON, buffer it
                            self._partial_line_buffer = line_str
                            logger.debug("Buffering partial JSON line")
                        else:
                            # Complete but invalid JSON
                            logger.error(f"Invalid JSON: {line_str[:100]}...")
                            if len(self._message_buffer) == 0:
                                # No valid messages yet, this might be a real problem
                                raise SDKJSONDecodeError(line_str, e) from e
                            # Otherwise, log and continue
                
            except anyio.ClosedResourceError:
                logger.info("Stream closed, checking for final messages")
                
            finally:
                tg.cancel_scope.cancel()
        
        # Check process exit status
        await self._process.wait()
        
        if self._process.returncode is not None and self._process.returncode != 0:
            stderr_output = "\n".join(stderr_buffer)
            
            # Provide helpful error messages based on common issues
            if "ENOENT" in stderr_output or "command not found" in stderr_output:
                raise CLINotFoundError(
                    "Claude Code CLI not found. Please ensure it's installed correctly."
                )
            elif "EACCES" in stderr_output or "permission denied" in stderr_output:
                raise CLIConnectionError(
                    "Permission denied. Check file permissions and try again."
                )
            elif "rate limit" in stderr_output.lower():
                raise ProcessError(
                    "Rate limit exceeded. Please wait before trying again.",
                    exit_code=self._process.returncode,
                    stderr=stderr_output
                )
            else:
                raise ProcessError(
                    f"CLI process failed with exit code {self._process.returncode}",
                    exit_code=self._process.returncode,
                    stderr=stderr_output
                )

    def get_message_statistics(self) -> dict[str, Any]:
        """Get statistics about processed messages."""
        return {
            "total_messages": len(self._message_buffer),
            "message_types": self._count_message_types(),
            "has_partial_buffer": bool(self._partial_line_buffer),
            "circuit_breaker_state": self.circuit_breaker._state,
            "failure_count": self.circuit_breaker._failure_count,
        }
    
    def _count_message_types(self) -> dict[str, int]:
        """Count messages by type."""
        type_counts = {}
        for msg in self._message_buffer:
            msg_type = msg.get("type", "unknown")
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
        return type_counts


# Export the most robust version as the default enhanced transport
EnhancedTransport = RobustSubprocessCLITransport