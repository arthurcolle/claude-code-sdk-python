"""Tests for retry transport functionality."""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import anyio
from anyio.abc import Process
from anyio.streams.text import TextReceiveStream

from claude_code_sdk._internal.transport.subprocess_cli_with_retry import (
    CircuitBreaker,
    EnhancedSubprocessCLITransport,
    RetryConfig,
    RobustSubprocessCLITransport,
)
from claude_code_sdk._errors import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)
from claude_code_sdk.types import ClaudeCodeOptions


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert CLIConnectionError in config.retryable_errors
        assert ProcessError in config.retryable_errors
        assert asyncio.TimeoutError in config.retryable_errors

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=60.0,
            exponential_base=3.0,
            jitter=False,
            retryable_errors=(CLIConnectionError,)
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 60.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_errors == (CLIConnectionError,)


class TestEnhancedSubprocessCLITransport:
    """Test enhanced subprocess CLI transport with retry."""

    @pytest.fixture
    def transport(self):
        """Create transport instance."""
        options = ClaudeCodeOptions()
        return EnhancedSubprocessCLITransport("test prompt", options)

    def test_calculate_delay(self, transport):
        """Test delay calculation for retries."""
        transport.retry_config.jitter = False
        
        # Test exponential backoff
        assert transport._calculate_delay(0) == 1.0  # 1 * 2^0
        assert transport._calculate_delay(1) == 2.0  # 1 * 2^1
        assert transport._calculate_delay(2) == 4.0  # 1 * 2^2
        assert transport._calculate_delay(3) == 8.0  # 1 * 2^3
        
        # Test max delay cap
        assert transport._calculate_delay(10) == 30.0  # Capped at max_delay

    def test_calculate_delay_with_jitter(self, transport):
        """Test delay calculation with jitter."""
        transport.retry_config.jitter = True
        
        # Test that jitter adds variability
        delays = [transport._calculate_delay(1) for _ in range(10)]
        assert len(set(delays)) > 1  # Should have different values
        assert all(1.5 <= d <= 2.5 for d in delays)  # Within ±25% of 2.0

    def test_is_retryable_error(self, transport):
        """Test retryable error detection."""
        assert transport._is_retryable_error(CLIConnectionError("test"))
        assert transport._is_retryable_error(ProcessError("test", 1))
        assert transport._is_retryable_error(asyncio.TimeoutError())
        assert not transport._is_retryable_error(ValueError("test"))
        assert not transport._is_retryable_error(CLINotFoundError("test"))

    @pytest.mark.asyncio
    async def test_connect_with_retry_success(self, transport):
        """Test successful connection with retry."""
        attempt_count = 0
        
        async def mock_connect():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise CLIConnectionError("Connection failed")
            # Success on third attempt
        
        transport.connect = mock_connect
        
        await transport.connect_with_retry()
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_connect_with_retry_failure(self, transport):
        """Test connection failure after max retries."""
        transport.retry_config.max_attempts = 2
        
        async def mock_connect():
            raise CLIConnectionError("Connection always fails")
        
        transport.connect = mock_connect
        
        with pytest.raises(CLIConnectionError) as exc_info:
            await transport.connect_with_retry()
        
        assert "Connection always fails" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connect_with_retry_non_retryable(self, transport):
        """Test non-retryable error stops retry immediately."""
        attempt_count = 0
        
        async def mock_connect():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Non-retryable error")
        
        transport.connect = mock_connect
        
        with pytest.raises(ValueError):
            await transport.connect_with_retry()
        
        assert attempt_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_connect_with_retry_callback(self, transport):
        """Test retry callback is called."""
        retry_calls = []
        
        def on_retry(attempt, error):
            retry_calls.append((attempt, str(error)))
        
        transport.on_retry = on_retry
        transport.retry_config.max_attempts = 3
        
        async def mock_connect():
            raise CLIConnectionError("Test error")
        
        transport.connect = mock_connect
        
        with pytest.raises(CLIConnectionError):
            await transport.connect_with_retry()
        
        assert len(retry_calls) == 2  # Called for attempts 1 and 2
        assert retry_calls[0] == (1, "Test error")
        assert retry_calls[1] == (2, "Test error")

    @pytest.mark.asyncio
    async def test_receive_messages_with_recovery(self, transport):
        """Test message receiving with recovery."""
        messages = [
            {"type": "user", "content": "test"},
            {"type": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ]
        error_after = 1
        attempt_count = 0
        
        async def mock_receive():
            nonlocal attempt_count
            for i, msg in enumerate(messages):
                if i == error_after and attempt_count == 0:
                    attempt_count += 1
                    raise anyio.ClosedResourceError()
                yield msg
        
        transport.receive_messages = mock_receive
        transport.is_connected = lambda: True
        
        received = []
        async for msg in transport.receive_messages_with_recovery():
            received.append(msg)
        
        assert len(received) == len(messages)

    @pytest.mark.asyncio
    async def test_get_diagnostics(self, transport):
        """Test diagnostic information."""
        transport._connection_attempts = 3
        transport._last_error = ValueError("Test error")
        transport._cli_path = "/usr/bin/claude"
        transport.is_connected = lambda: True
        
        diagnostics = transport.get_diagnostics()
        
        assert diagnostics["connected"] is True
        assert diagnostics["connection_attempts"] == 3
        assert "Test error" in diagnostics["last_error"]
        assert diagnostics["cli_path"] == "/usr/bin/claude"
        assert diagnostics["uptime_seconds"] >= 0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker()
        assert cb._state == "closed"
        assert cb.can_proceed() is True

    def test_failure_threshold(self):
        """Test circuit breaker opens after threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # First two failures don't open circuit
        cb.call_failed()
        cb.call_failed()
        assert cb._state == "closed"
        assert cb.can_proceed() is True
        
        # Third failure opens circuit
        cb.call_failed()
        assert cb._state == "open"
        assert cb.can_proceed() is False

    def test_recovery_timeout(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        cb.call_failed()
        assert cb._state == "open"
        assert cb.can_proceed() is False
        
        # Wait for recovery timeout
        time.sleep(0.15)
        assert cb.can_proceed() is True
        assert cb._state == "half_open"

    def test_half_open_success(self):
        """Test circuit breaker closes on success in half-open state."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        cb.call_failed()
        time.sleep(0.15)
        
        assert cb._state == "half_open"
        cb.call_succeeded()
        assert cb._state == "closed"
        assert cb._failure_count == 0

    def test_half_open_failure(self):
        """Test circuit breaker reopens on failure in half-open state."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=2
        )
        
        cb.call_failed()
        time.sleep(0.15)
        
        assert cb._state == "half_open"
        cb.call_failed()
        assert cb._state == "half_open"  # Still half-open after one failure
        
        cb.call_failed()
        assert cb._state == "open"  # Opens after max calls in half-open


class TestRobustSubprocessCLITransport:
    """Test robust subprocess CLI transport."""

    @pytest.fixture
    def transport(self):
        """Create transport instance."""
        options = ClaudeCodeOptions()
        return RobustSubprocessCLITransport("test prompt", options)

    @pytest.mark.asyncio
    async def test_connect_with_circuit_breaker(self, transport):
        """Test connection with circuit breaker protection."""
        # Open the circuit breaker
        transport.circuit_breaker._state = "open"
        
        with pytest.raises(CLIConnectionError) as exc_info:
            await transport.connect()
        
        assert "Circuit breaker is open" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_receive_messages_json_handling(self, transport):
        """Test robust JSON handling in message receiving."""
        mock_process = MagicMock(spec=Process)
        mock_process.returncode = 0
        mock_process.wait = AsyncMock(return_value=None)
        
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()
        
        # Mix of valid JSON, partial JSON, and non-JSON
        stdout_lines = [
            '{"type": "user", "content": "test"}',
            'Debug: Processing message',  # Non-JSON
            '{"type": "assistant",',  # Partial JSON
            '"content": [{"type": "text", "text": "Hello"}]}',  # Continuation
            '{"type": "result", "subtype": "success"}',
        ]
        
        async def stdout_iter():
            for line in stdout_lines:
                yield line
        
        async def stderr_iter():
            yield "Info: Starting process"
        
        mock_stdout.__aiter__.return_value = stdout_iter()
        mock_stderr.__aiter__.return_value = stderr_iter()
        
        transport._process = mock_process
        transport._stdout_stream = mock_stdout
        transport._stderr_stream = mock_stderr
        
        messages = []
        async for msg in transport.receive_messages():
            messages.append(msg)
        
        # Should handle partial JSON correctly
        assert len(messages) == 3
        assert messages[0]["type"] == "user"
        assert messages[1]["type"] == "assistant"
        assert messages[2]["type"] == "result"

    @pytest.mark.asyncio
    async def test_receive_messages_error_diagnosis(self, transport):
        """Test enhanced error diagnosis from stderr."""
        mock_process = MagicMock(spec=Process)
        mock_process.returncode = 1
        mock_process.wait = AsyncMock(return_value=None)
        
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()
        
        async def stdout_iter():
            yield '{"type": "user", "content": "test"}'
            raise anyio.ClosedResourceError()
        
        async def stderr_iter():
            yield "Error: ENOENT: no such file or directory"
        
        mock_stdout.__aiter__.return_value = stdout_iter()
        mock_stderr.__aiter__.return_value = stderr_iter()
        
        transport._process = mock_process
        transport._stdout_stream = mock_stdout
        transport._stderr_stream = mock_stderr
        
        with pytest.raises(CLINotFoundError) as exc_info:
            messages = []
            async for msg in transport.receive_messages():
                messages.append(msg)
        
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_receive_messages_permission_error(self, transport):
        """Test permission error handling."""
        mock_process = MagicMock(spec=Process)
        mock_process.returncode = 1
        mock_process.wait = AsyncMock(return_value=None)
        
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()
        
        async def stdout_iter():
            raise anyio.ClosedResourceError()
        
        async def stderr_iter():
            yield "Error: EACCES: permission denied, open '/protected/file'"
        
        mock_stdout.__aiter__.return_value = stdout_iter()
        mock_stderr.__aiter__.return_value = stderr_iter()
        
        transport._process = mock_process
        transport._stdout_stream = mock_stdout
        transport._stderr_stream = mock_stderr
        
        with pytest.raises(CLIConnectionError) as exc_info:
            async for _ in transport.receive_messages():
                pass
        
        assert "Permission denied" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_receive_messages_rate_limit(self, transport):
        """Test rate limit error handling."""
        mock_process = MagicMock(spec=Process)
        mock_process.returncode = 429
        mock_process.wait = AsyncMock(return_value=None)
        
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()
        
        async def stdout_iter():
            raise anyio.ClosedResourceError()
        
        async def stderr_iter():
            yield "Error: Rate limit exceeded. Please try again later."
        
        mock_stdout.__aiter__.return_value = stdout_iter()
        mock_stderr.__aiter__.return_value = stderr_iter()
        
        transport._process = mock_process
        transport._stdout_stream = mock_stdout
        transport._stderr_stream = mock_stderr
        
        with pytest.raises(ProcessError) as exc_info:
            async for _ in transport.receive_messages():
                pass
        
        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.exit_code == 429

    def test_get_message_statistics(self, transport):
        """Test message statistics."""
        transport._message_buffer = [
            {"type": "user", "content": "test"},
            {"type": "assistant", "content": []},
            {"type": "assistant", "content": []},
            {"type": "result", "subtype": "success"},
        ]
        transport._partial_line_buffer = '{"partial":'
        transport.circuit_breaker._state = "half_open"
        transport.circuit_breaker._failure_count = 2
        
        stats = transport.get_message_statistics()
        
        assert stats["total_messages"] == 4
        assert stats["message_types"]["user"] == 1
        assert stats["message_types"]["assistant"] == 2
        assert stats["message_types"]["result"] == 1
        assert stats["has_partial_buffer"] is True
        assert stats["circuit_breaker_state"] == "half_open"
        assert stats["failure_count"] == 2

    @pytest.mark.asyncio
    async def test_integration_with_retries(self, transport):
        """Test full integration with retries and circuit breaker."""
        transport.retry_config.max_attempts = 3
        transport.retry_config.initial_delay = 0.01  # Fast retries for testing
        
        connect_attempts = 0
        
        async def mock_connect():
            nonlocal connect_attempts
            connect_attempts += 1
            if connect_attempts < 2:
                raise CLIConnectionError("Connection failed")
        
        # Mock the parent's connect method
        with patch.object(
            transport.__class__.__bases__[0],
            'connect',
            mock_connect
        ):
            await transport.connect_with_retry()
            
            assert connect_attempts == 2
            assert transport.circuit_breaker._state == "closed"