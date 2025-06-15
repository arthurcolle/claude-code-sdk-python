"""Middleware and plugin system for Claude SDK."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
from collections.abc import AsyncIterator
import inspect

from .types import Message, ClaudeCodeOptions


class Middleware(Protocol):
    """Middleware protocol."""
    
    async def process(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        next_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Process the query through middleware."""
        ...


@dataclass
class MiddlewareChain:
    """Chain of middleware handlers."""
    
    middlewares: list[Middleware] = field(default_factory=list)
    
    def add(self, middleware: Middleware) -> "MiddlewareChain":
        """Add middleware to chain."""
        self.middlewares.append(middleware)
        return self
    
    def remove(self, middleware: Middleware) -> "MiddlewareChain":
        """Remove middleware from chain."""
        self.middlewares.remove(middleware)
        return self
    
    async def execute(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        base_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Execute the middleware chain."""
        
        async def _create_handler(index: int) -> AsyncIterator[Message]:
            if index >= len(self.middlewares):
                # Base case: call the original handler
                async for msg in base_handler(prompt=prompt, options=options):
                    yield msg
            else:
                # Recursive case: call next middleware
                middleware = self.middlewares[index]
                next_handler = lambda **kwargs: _create_handler(index + 1)
                async for msg in middleware.process(prompt, options, next_handler):
                    yield msg
        
        async for msg in _create_handler(0):
            yield msg


# Built-in middleware implementations

@dataclass
class LoggingMiddleware:
    """Logs all queries and responses."""
    
    logger: Any = None  # Can be any logger with info/error methods
    log_prompt: bool = True
    log_messages: bool = True
    log_errors: bool = True
    
    async def process(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        next_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Process with logging."""
        if self.logger and self.log_prompt:
            self.logger.info(f"Query prompt: {prompt[:100]}...")
        
        try:
            message_count = 0
            async for message in next_handler(prompt=prompt, options=options):
                message_count += 1
                if self.logger and self.log_messages:
                    self.logger.info(f"Message {message_count}: {type(message).__name__}")
                yield message
                
        except Exception as e:
            if self.logger and self.log_errors:
                self.logger.error(f"Query error: {e}")
            raise


@dataclass
class MetricsMiddleware:
    """Collects metrics about queries."""
    
    metrics: dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.metrics = {
            "total_queries": 0,
            "total_messages": 0,
            "total_errors": 0,
            "query_times": [],
        }
    
    async def process(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        next_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Process with metrics collection."""
        import time
        
        start_time = time.time()
        self.metrics["total_queries"] += 1
        message_count = 0
        
        try:
            async for message in next_handler(prompt=prompt, options=options):
                message_count += 1
                yield message
                
        except Exception as e:
            self.metrics["total_errors"] += 1
            raise
        finally:
            duration = time.time() - start_time
            self.metrics["query_times"].append(duration)
            self.metrics["total_messages"] += message_count
    
    def get_stats(self) -> dict[str, Any]:
        """Get collected metrics."""
        times = self.metrics["query_times"]
        avg_time = sum(times) / len(times) if times else 0
        
        return {
            **self.metrics,
            "average_query_time": avg_time,
            "success_rate": 1 - (self.metrics["total_errors"] / self.metrics["total_queries"])
            if self.metrics["total_queries"] > 0 else 0
        }


@dataclass
class AuthenticationMiddleware:
    """Adds authentication to queries."""
    
    auth_token: Optional[str] = None
    auth_header: str = "Authorization"
    
    async def process(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        next_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Process with authentication."""
        # Modify options to include auth
        if options is None:
            options = ClaudeCodeOptions()
        
        # Note: This is an example - actual auth implementation would depend on transport
        if self.auth_token:
            # Could modify options or environment here
            pass
        
        async for message in next_handler(prompt=prompt, options=options):
            yield message


@dataclass
class RateLimitMiddleware:
    """Rate limits queries."""
    
    max_queries_per_minute: int = 60
    _timestamps: list[float] = field(default_factory=list, init=False)
    
    async def process(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        next_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Process with rate limiting."""
        import time
        import asyncio
        
        now = time.time()
        minute_ago = now - 60
        
        # Clean old timestamps
        self._timestamps = [t for t in self._timestamps if t > minute_ago]
        
        # Check rate limit
        if len(self._timestamps) >= self.max_queries_per_minute:
            # Calculate wait time
            oldest = self._timestamps[0]
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self._timestamps.append(time.time())
        
        async for message in next_handler(prompt=prompt, options=options):
            yield message


@dataclass
class TransformMiddleware:
    """Transforms prompts and messages."""
    
    prompt_transformer: Optional[Callable[[str], str]] = None
    message_transformer: Optional[Callable[[Message], Message]] = None
    
    async def process(
        self,
        prompt: str,
        options: ClaudeCodeOptions | None,
        next_handler: Callable[..., AsyncIterator[Message]]
    ) -> AsyncIterator[Message]:
        """Process with transformations."""
        # Transform prompt
        if self.prompt_transformer:
            prompt = self.prompt_transformer(prompt)
        
        # Process messages
        async for message in next_handler(prompt=prompt, options=options):
            if self.message_transformer:
                message = self.message_transformer(message)
            yield message


# Plugin system

class Plugin(Protocol):
    """Plugin protocol."""
    
    def get_middlewares(self) -> list[Middleware]:
        """Get middlewares provided by plugin."""
        ...
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        ...
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        ...


@dataclass
class PluginManager:
    """Manages plugins and their lifecycles."""
    
    plugins: list[Plugin] = field(default_factory=list)
    middleware_chain: MiddlewareChain = field(default_factory=MiddlewareChain)
    _initialized: bool = field(default=False, init=False)
    
    async def register(self, plugin: Plugin) -> None:
        """Register a plugin."""
        self.plugins.append(plugin)
        
        if self._initialized:
            await plugin.initialize()
            
        # Add plugin middlewares
        for middleware in plugin.get_middlewares():
            self.middleware_chain.add(middleware)
    
    async def unregister(self, plugin: Plugin) -> None:
        """Unregister a plugin."""
        if plugin in self.plugins:
            # Remove plugin middlewares
            for middleware in plugin.get_middlewares():
                self.middleware_chain.remove(middleware)
            
            if self._initialized:
                await plugin.cleanup()
                
            self.plugins.remove(plugin)
    
    async def initialize_all(self) -> None:
        """Initialize all plugins."""
        if not self._initialized:
            for plugin in self.plugins:
                await plugin.initialize()
            self._initialized = True
    
    async def cleanup_all(self) -> None:
        """Clean up all plugins."""
        if self._initialized:
            for plugin in self.plugins:
                await plugin.cleanup()
            self._initialized = False
    
    def get_middleware_chain(self) -> MiddlewareChain:
        """Get the combined middleware chain."""
        return self.middleware_chain


# Convenience function for queries with middleware

async def query_with_middleware(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    middlewares: list[Middleware] | None = None,
    chain: Optional[MiddlewareChain] = None
) -> AsyncIterator[Message]:
    """
    Query Claude with middleware processing.
    
    Args:
        prompt: The prompt to send
        options: Query options
        middlewares: List of middleware to apply
        chain: Pre-built middleware chain (overrides middlewares)
        
    Yields:
        Messages processed through middleware
        
    Example:
        ```python
        # With individual middleware
        logging = LoggingMiddleware(logger=logger)
        metrics = MetricsMiddleware()
        
        async for msg in query_with_middleware(
            prompt="Hello",
            middlewares=[logging, metrics]
        ):
            print(msg)
        
        print(f"Stats: {metrics.get_stats()}")
        
        # With middleware chain
        chain = MiddlewareChain()
        chain.add(RateLimitMiddleware(max_queries_per_minute=30))
        chain.add(LoggingMiddleware())
        
        async for msg in query_with_middleware(
            prompt="Hello",
            chain=chain
        ):
            print(msg)
        ```
    """
    from . import query
    
    if chain is None:
        chain = MiddlewareChain()
        if middlewares:
            for mw in middlewares:
                chain.add(mw)
    
    async for message in chain.execute(prompt, options, query):
        yield message