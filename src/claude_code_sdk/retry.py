"""Retry mechanism with exponential backoff for Claude SDK."""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Union
from collections.abc import AsyncIterator

from ._errors import ClaudeSDKError, ProcessError, CLIConnectionError
from .types import Message


T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (
        ProcessError,
        CLIConnectionError,
        asyncio.TimeoutError,
    )
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if error should be retried."""
        if attempt >= self.max_attempts:
            return False
        return isinstance(error, self.retry_on)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        delay = min(
            self.initial_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random())
        
        return delay


class RetryableQuery:
    """Wrapper for retryable queries."""
    
    def __init__(
        self,
        query_func: Callable[..., AsyncIterator[Message]],
        config: RetryConfig
    ):
        self.query_func = query_func
        self.config = config
        self._attempt = 0
    
    async def __call__(self, *args, **kwargs) -> AsyncIterator[Message]:
        """Execute query with retry logic."""
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self._attempt = attempt
            try:
                # Attempt the query
                async for message in self.query_func(*args, **kwargs):
                    yield message
                return  # Success
                
            except Exception as e:
                last_error = e
                
                if not self.config.should_retry(e, attempt):
                    raise
                
                if attempt < self.config.max_attempts:
                    delay = self.config.get_delay(attempt)
                    
                    if self.config.on_retry:
                        self.config.on_retry(e, attempt, delay)
                    
                    await asyncio.sleep(delay)
        
        # All attempts exhausted
        if last_error:
            raise RetryExhaustedError(
                f"All {self.config.max_attempts} retry attempts failed",
                last_error=last_error,
                attempts=self.config.max_attempts
            )


class RetryExhaustedError(ClaudeSDKError):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, last_error: Exception, attempts: int):
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts


def with_retry(
    config: Optional[RetryConfig] = None
) -> Callable[[Callable[..., AsyncIterator[Message]]], RetryableQuery]:
    """
    Decorator to add retry logic to query functions.
    
    Args:
        config: Retry configuration (uses defaults if None)
        
    Returns:
        Decorated function with retry capability
        
    Example:
        ```python
        @with_retry(RetryConfig(max_attempts=5))
        async def my_query(prompt: str) -> AsyncIterator[Message]:
            async for msg in query(prompt=prompt):
                yield msg
        
        # Use with automatic retry
        async for message in my_query("Hello"):
            print(message)
        ```
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., AsyncIterator[Message]]) -> RetryableQuery:
        return RetryableQuery(func, config)
    
    return decorator


async def query_with_retry(
    *,
    prompt: str,
    options: Any = None,
    retry_config: Optional[RetryConfig] = None
) -> AsyncIterator[Message]:
    """
    Query Claude with automatic retry on failures.
    
    Args:
        prompt: The prompt to send
        options: Query options
        retry_config: Retry configuration (uses defaults if None)
        
    Yields:
        Messages from the conversation
        
    Example:
        ```python
        config = RetryConfig(
            max_attempts=5,
            on_retry=lambda e, attempt, delay: 
                print(f"Retry {attempt} after {delay}s: {e}")
        )
        
        async for message in query_with_retry(
            prompt="Hello",
            retry_config=config
        ):
            print(message)
        ```
    """
    from . import query
    
    if retry_config is None:
        retry_config = RetryConfig()
    
    retryable = RetryableQuery(query, retry_config)
    async for message in retryable(prompt=prompt, options=options):
        yield message