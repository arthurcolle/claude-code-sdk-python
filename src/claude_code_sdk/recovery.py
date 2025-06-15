"""Automatic error recovery strategies for Claude SDK."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections.abc import AsyncIterator
from enum import Enum

from .types import Message, ClaudeCodeOptions
from ._errors import ClaudeSDKError, ProcessError, CLIConnectionError
from .retry import RetryConfig, query_with_retry


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    TIMEOUT = "timeout"
    CACHE_FALLBACK = "cache_fallback"
    DEGRADE = "degrade"


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_requests: int = 3
    
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _state: str = field(default="closed", init=False)  # closed, open, half_open
    _half_open_count: int = field(default=0, init=False)
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self._state == "open":
            import time
            if self._last_failure_time and \
               time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "half_open"
                self._half_open_count = 0
                return False
            return True
        return False
    
    def record_success(self) -> None:
        """Record successful request."""
        if self._state == "half_open":
            self._half_open_count += 1
            if self._half_open_count >= self.half_open_requests:
                self._state = "closed"
                self._failure_count = 0
        elif self._state == "closed":
            self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed request."""
        import time
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == "half_open":
            self._state = "open"
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"
    
    def get_state(self) -> str:
        """Get circuit state."""
        return self._state


@dataclass
class RecoveryConfig:
    """Configuration for error recovery."""
    
    strategies: list[RecoveryStrategy] = field(
        default_factory=lambda: [RecoveryStrategy.RETRY]
    )
    
    # Retry config
    retry_config: Optional[RetryConfig] = None
    
    # Circuit breaker config
    circuit_breaker: Optional[CircuitBreaker] = None
    
    # Timeout config
    timeout_seconds: float = 120.0
    
    # Fallback config
    fallback_prompt: Optional[str] = None
    fallback_handler: Optional[Callable[..., AsyncIterator[Message]]] = None
    
    # Cache fallback
    use_cache_on_error: bool = True
    
    # Degrade config
    degrade_options: Optional[ClaudeCodeOptions] = None
    
    # Error handler
    on_error: Optional[Callable[[Exception, str], None]] = None


class RecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self._circuit_breaker = config.circuit_breaker or CircuitBreaker()
    
    async def execute_with_recovery(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        base_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Execute query with recovery strategies."""
        
        # Check circuit breaker
        if RecoveryStrategy.CIRCUIT_BREAKER in self.config.strategies:
            if self._circuit_breaker.is_open():
                if self.config.on_error:
                    self.config.on_error(
                        Exception("Circuit breaker is open"),
                        "circuit_open"
                    )
                
                # Try fallback or fail
                if self.config.fallback_handler:
                    async for msg in self.config.fallback_handler(
                        prompt=prompt,
                        options=options
                    ):
                        yield msg
                    return
                else:
                    raise ClaudeSDKError("Circuit breaker is open")
        
        try:
            # Apply timeout if configured
            if RecoveryStrategy.TIMEOUT in self.config.strategies:
                async with asyncio.timeout(self.config.timeout_seconds):
                    async for msg in self._execute_with_strategies(
                        prompt, options, base_handler
                    ):
                        yield msg
            else:
                async for msg in self._execute_with_strategies(
                    prompt, options, base_handler
                ):
                    yield msg
            
            # Record success
            if RecoveryStrategy.CIRCUIT_BREAKER in self.config.strategies:
                self._circuit_breaker.record_success()
                
        except asyncio.TimeoutError as e:
            if self.config.on_error:
                self.config.on_error(e, "timeout")
            
            # Try recovery
            async for msg in self._handle_error(e, prompt, options):
                yield msg
                
        except Exception as e:
            if self.config.on_error:
                self.config.on_error(e, "error")
            
            # Record failure
            if RecoveryStrategy.CIRCUIT_BREAKER in self.config.strategies:
                self._circuit_breaker.record_failure()
            
            # Try recovery
            async for msg in self._handle_error(e, prompt, options):
                yield msg
    
    async def _execute_with_strategies(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        base_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Execute with primary strategies."""
        
        # Apply retry if configured
        if RecoveryStrategy.RETRY in self.config.strategies:
            if self.config.retry_config:
                async for msg in query_with_retry(
                    prompt=prompt,
                    options=options,
                    retry_config=self.config.retry_config
                ):
                    yield msg
            else:
                # Default retry
                async for msg in query_with_retry(
                    prompt=prompt,
                    options=options
                ):
                    yield msg
        else:
            # No retry, execute directly
            async for msg in base_handler(prompt=prompt, options=options):
                yield msg
    
    async def _handle_error(
        self,
        error: Exception,
        prompt: str,
        options: ClaudeCodeOptions | None
    ) -> AsyncIterator[Message]:
        """Handle error with recovery strategies."""
        
        # Try cache fallback
        if RecoveryStrategy.CACHE_FALLBACK in self.config.strategies and \
           self.config.use_cache_on_error:
            try:
                from .cache import get_global_cache
                cache = get_global_cache()
                cached = await cache.get(prompt, options)
                if cached:
                    for msg in cached:
                        yield msg
                    return
            except Exception:
                pass
        
        # Try fallback
        if RecoveryStrategy.FALLBACK in self.config.strategies:
            if self.config.fallback_handler:
                try:
                    async for msg in self.config.fallback_handler(
                        prompt=prompt,
                        options=options
                    ):
                        yield msg
                    return
                except Exception:
                    pass
            
            if self.config.fallback_prompt:
                # Use fallback prompt
                from . import query
                try:
                    async for msg in query(
                        prompt=self.config.fallback_prompt,
                        options=options
                    ):
                        yield msg
                    return
                except Exception:
                    pass
        
        # Try degraded mode
        if RecoveryStrategy.DEGRADE in self.config.strategies and \
           self.config.degrade_options:
            from . import query
            try:
                async for msg in query(
                    prompt=prompt,
                    options=self.config.degrade_options
                ):
                    yield msg
                return
            except Exception:
                pass
        
        # All strategies failed, re-raise
        raise error


async def query_with_recovery(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    recovery_config: Optional[RecoveryConfig] = None,
    **recovery_kwargs
) -> AsyncIterator[Message]:
    """
    Query Claude with automatic error recovery.
    
    Args:
        prompt: The prompt to send
        options: Query options
        recovery_config: Recovery configuration
        **recovery_kwargs: Additional arguments for RecoveryConfig
        
    Yields:
        Messages from conversation
        
    Example:
        ```python
        # Simple recovery with retry
        async for msg in query_with_recovery(
            prompt="Hello",
            strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            fallback_prompt="Hi there!"
        ):
            print(msg)
        
        # Advanced recovery
        config = RecoveryConfig(
            strategies=[
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CACHE_FALLBACK,
                RecoveryStrategy.TIMEOUT
            ],
            retry_config=RetryConfig(max_attempts=3),
            circuit_breaker=CircuitBreaker(failure_threshold=3),
            timeout_seconds=60,
            on_error=lambda e, t: print(f"Error ({t}): {e}")
        )
        
        async for msg in query_with_recovery(
            prompt="Complex query",
            recovery_config=config
        ):
            print(msg)
        ```
    """
    from . import query
    
    if recovery_config is None:
        recovery_config = RecoveryConfig(**recovery_kwargs)
    
    manager = RecoveryManager(recovery_config)
    
    async for message in manager.execute_with_recovery(
        prompt, options, query
    ):
        yield message


# Convenience functions for common patterns

async def query_with_fallback(
    *,
    prompt: str,
    fallback_prompt: str,
    options: ClaudeCodeOptions | None = None
) -> AsyncIterator[Message]:
    """Query with simple fallback."""
    config = RecoveryConfig(
        strategies=[RecoveryStrategy.FALLBACK],
        fallback_prompt=fallback_prompt
    )
    async for msg in query_with_recovery(
        prompt=prompt,
        options=options,
        recovery_config=config
    ):
        yield msg


async def query_with_circuit_breaker(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
) -> AsyncIterator[Message]:
    """Query with circuit breaker protection."""
    config = RecoveryConfig(
        strategies=[RecoveryStrategy.CIRCUIT_BREAKER, RecoveryStrategy.RETRY],
        circuit_breaker=CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
    )
    async for msg in query_with_recovery(
        prompt=prompt,
        options=options,
        recovery_config=config
    ):
        yield msg