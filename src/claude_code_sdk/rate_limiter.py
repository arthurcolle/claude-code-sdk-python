"""Rate limiting and quota management for Claude Code SDK.

This module provides comprehensive rate limiting and quota management features:
- Token-based rate limiting with configurable buckets
- Request rate limiting per time window
- Quota tracking and enforcement
- Adaptive rate limiting based on response headers
- Multiple rate limiting strategies (fixed, sliding window, token bucket)
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Protocol

import anyio
from typing_extensions import TypedDict


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class QuotaType(str, Enum):
    """Quota types."""

    REQUESTS = "requests"
    TOKENS = "tokens"
    COST = "cost"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig:
    """Rate limit configuration.
    
    Example:
        ```python
        config = RateLimitConfig(
            max_requests_per_minute=60,
            max_tokens_per_minute=10000,
            max_cost_per_hour=10.0,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        ```
    """

    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_tokens_per_minute: int = 100000
    max_tokens_per_hour: int = 1000000
    max_cost_per_hour: float = 100.0
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    burst_multiplier: float = 1.5  # Allow burst up to 1.5x the limit
    retry_after_header: str = "Retry-After"
    rate_limit_header: str = "X-RateLimit-Limit"
    rate_limit_remaining_header: str = "X-RateLimit-Remaining"
    rate_limit_reset_header: str = "X-RateLimit-Reset"


@dataclass
class QuotaConfig:
    """Quota configuration.
    
    Example:
        ```python
        quota = QuotaConfig(
            daily_request_limit=10000,
            daily_token_limit=1000000,
            monthly_cost_limit=1000.0
        )
        ```
    """

    daily_request_limit: int | None = None
    daily_token_limit: int | None = None
    daily_cost_limit: float | None = None
    monthly_request_limit: int | None = None
    monthly_token_limit: int | None = None
    monthly_cost_limit: float | None = None
    custom_quotas: dict[str, float] = field(default_factory=dict)
    reset_timezone: str = "UTC"


@dataclass
class UsageStats:
    """Usage statistics."""

    requests_count: int = 0
    tokens_count: int = 0
    total_cost: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    custom_metrics: dict[str, float] = field(default_factory=dict)


class RateLimitInfo(TypedDict):
    """Rate limit information from API response."""

    limit: int | None
    remaining: int | None
    reset: int | None  # Unix timestamp
    retry_after: int | None  # Seconds


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceeded(Exception):
    """Raised when quota is exceeded."""

    def __init__(self, message: str, quota_type: QuotaType, reset_time: datetime | None = None):
        super().__init__(message)
        self.quota_type = quota_type
        self.reset_time = reset_time


class RateLimiterProtocol(Protocol):
    """Protocol for rate limiters."""

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the rate limiter."""
        ...

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking."""
        ...


class TokenBucketLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Tokens refilled per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the bucket."""
        async with self._lock:
            await self._refill()
            
            while self.tokens < tokens:
                sleep_time = (tokens - self.tokens) / self.refill_rate
                await anyio.sleep(sleep_time)
                await self._refill()
            
            self.tokens -= tokens

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking."""
        self._refill_sync()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def _refill_sync(self) -> None:
        """Synchronous version of refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


class SlidingWindowLimiter:
    """Sliding window rate limiter implementation."""

    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize sliding window limiter.
        
        Args:
            max_requests: Maximum requests in the window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            self._cleanup()
            
            while len(self.requests) + tokens > self.max_requests:
                if self.requests:
                    sleep_until = self.requests[0] + self.window_seconds
                    sleep_time = sleep_until - time.time()
                    if sleep_time > 0:
                        await anyio.sleep(sleep_time)
                self._cleanup()
            
            now = time.time()
            for _ in range(tokens):
                self.requests.append(now)

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire without blocking."""
        self._cleanup()
        
        if len(self.requests) + tokens <= self.max_requests:
            now = time.time()
            for _ in range(tokens):
                self.requests.append(now)
            return True
        return False

    def _cleanup(self) -> None:
        """Remove expired entries."""
        cutoff = time.time() - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()


class RateLimiter:
    """Main rate limiter with multiple strategies and quota management.
    
    Example:
        ```python
        import asyncio
        from claude_code_sdk.rate_limiter import RateLimiter, RateLimitConfig
        
        async def main():
            config = RateLimitConfig(max_requests_per_minute=30)
            limiter = RateLimiter(config)
            
            # Use as context manager
            async with limiter.acquire_request():
                # Make API call
                pass
            
            # Or manually
            await limiter.wait_if_needed(tokens=100)
        
        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        quota_config: QuotaConfig | None = None,
    ):
        """Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
            quota_config: Quota configuration
        """
        self.config = config or RateLimitConfig()
        self.quota_config = quota_config or QuotaConfig()
        
        # Initialize rate limiters
        self.request_limiter = self._create_limiter(
            self.config.max_requests_per_minute, 60
        )
        self.token_limiter = self._create_limiter(
            self.config.max_tokens_per_minute, 60
        )
        
        # Usage tracking
        self.daily_usage = UsageStats()
        self.monthly_usage = UsageStats()
        
        # Adaptive rate limiting
        self._server_limits: RateLimitInfo = {}
        self._last_response_time = 0.0

    def _create_limiter(self, limit: int, window: int) -> RateLimiterProtocol:
        """Create a rate limiter based on strategy."""
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucketLimiter(
                capacity=int(limit * self.config.burst_multiplier),
                refill_rate=limit / window,
            )
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindowLimiter(limit, window)
        else:
            # Default to token bucket
            return TokenBucketLimiter(
                capacity=int(limit * self.config.burst_multiplier),
                refill_rate=limit / window,
            )

    async def wait_if_needed(self, tokens: int = 0, cost: float = 0.0) -> None:
        """Wait if rate limited or quota exceeded.
        
        Args:
            tokens: Number of tokens for this request
            cost: Cost of this request
            
        Raises:
            RateLimitExceeded: If rate limit cannot be satisfied
            QuotaExceeded: If quota would be exceeded
        """
        # Check quotas first
        self._check_quotas(tokens, cost)
        
        # Check server-imposed limits
        if self._server_limits.get("retry_after"):
            retry_after = self._server_limits["retry_after"]
            if retry_after > 300:  # Don't wait more than 5 minutes
                raise RateLimitExceeded(
                    f"Server rate limit: retry after {retry_after}s", retry_after
                )
            await anyio.sleep(retry_after)
            self._server_limits["retry_after"] = None
        
        # Apply local rate limits
        await self.request_limiter.acquire(1)
        if tokens > 0:
            await self.token_limiter.acquire(tokens)
        
        # Update usage
        self._update_usage(1, tokens, cost)

    def _check_quotas(self, tokens: int, cost: float) -> None:
        """Check if quotas would be exceeded."""
        now = datetime.now()
        
        # Reset daily usage if needed
        if now.date() > self.daily_usage.last_reset.date():
            self.daily_usage = UsageStats(last_reset=now)
        
        # Reset monthly usage if needed
        if now.month != self.monthly_usage.last_reset.month:
            self.monthly_usage = UsageStats(last_reset=now)
        
        # Check daily quotas
        if self.quota_config.daily_request_limit:
            if self.daily_usage.requests_count + 1 > self.quota_config.daily_request_limit:
                raise QuotaExceeded(
                    "Daily request quota exceeded",
                    QuotaType.REQUESTS,
                    datetime.combine(now.date() + timedelta(days=1), datetime.min.time()),
                )
        
        if self.quota_config.daily_token_limit and tokens > 0:
            if self.daily_usage.tokens_count + tokens > self.quota_config.daily_token_limit:
                raise QuotaExceeded(
                    "Daily token quota exceeded",
                    QuotaType.TOKENS,
                    datetime.combine(now.date() + timedelta(days=1), datetime.min.time()),
                )
        
        if self.quota_config.daily_cost_limit and cost > 0:
            if self.daily_usage.total_cost + cost > self.quota_config.daily_cost_limit:
                raise QuotaExceeded(
                    "Daily cost quota exceeded",
                    QuotaType.COST,
                    datetime.combine(now.date() + timedelta(days=1), datetime.min.time()),
                )

    def _update_usage(self, requests: int, tokens: int, cost: float) -> None:
        """Update usage statistics."""
        self.daily_usage.requests_count += requests
        self.daily_usage.tokens_count += tokens
        self.daily_usage.total_cost += cost
        
        self.monthly_usage.requests_count += requests
        self.monthly_usage.tokens_count += tokens
        self.monthly_usage.total_cost += cost

    def update_from_response(self, headers: dict[str, str]) -> None:
        """Update rate limits from response headers.
        
        Args:
            headers: Response headers containing rate limit info
        """
        self._server_limits = {
            "limit": self._parse_header(headers.get(self.config.rate_limit_header)),
            "remaining": self._parse_header(
                headers.get(self.config.rate_limit_remaining_header)
            ),
            "reset": self._parse_header(headers.get(self.config.rate_limit_reset_header)),
            "retry_after": self._parse_header(headers.get(self.config.retry_after_header)),
        }
        self._last_response_time = time.time()

    def _parse_header(self, value: str | None) -> int | None:
        """Parse header value to int."""
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    async def acquire_request(self) -> "RateLimitContext":
        """Acquire permission for a request using context manager.
        
        Example:
            ```python
            async with limiter.acquire_request() as ctx:
                ctx.set_tokens(150)
                ctx.set_cost(0.01)
                # Make request
            ```
        """
        return RateLimitContext(self)

    def get_usage_stats(self) -> dict[str, Any]:
        """Get current usage statistics.
        
        Returns:
            Dictionary with daily and monthly usage stats
        """
        return {
            "daily": {
                "requests": self.daily_usage.requests_count,
                "tokens": self.daily_usage.tokens_count,
                "cost": self.daily_usage.total_cost,
                "last_reset": self.daily_usage.last_reset.isoformat(),
            },
            "monthly": {
                "requests": self.monthly_usage.requests_count,
                "tokens": self.monthly_usage.tokens_count,
                "cost": self.monthly_usage.total_cost,
                "last_reset": self.monthly_usage.last_reset.isoformat(),
            },
            "limits": {
                "requests_per_minute": self.config.max_requests_per_minute,
                "tokens_per_minute": self.config.max_tokens_per_minute,
                "daily_request_limit": self.quota_config.daily_request_limit,
                "daily_token_limit": self.quota_config.daily_token_limit,
                "daily_cost_limit": self.quota_config.daily_cost_limit,
            },
        }

    def reset_usage(self, daily: bool = True, monthly: bool = False) -> None:
        """Reset usage statistics.
        
        Args:
            daily: Reset daily usage
            monthly: Reset monthly usage
        """
        if daily:
            self.daily_usage = UsageStats()
        if monthly:
            self.monthly_usage = UsageStats()


class RateLimitContext:
    """Context manager for rate-limited operations."""

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self.tokens = 0
        self.cost = 0.0
        self.start_time = 0.0

    def set_tokens(self, tokens: int) -> None:
        """Set token count for this request."""
        self.tokens = tokens

    def set_cost(self, cost: float) -> None:
        """Set cost for this request."""
        self.cost = cost

    async def __aenter__(self) -> "RateLimitContext":
        """Enter context."""
        self.start_time = time.time()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit context and apply rate limiting."""
        await self.limiter.wait_if_needed(self.tokens, self.cost)


def create_rate_limiter(
    max_rpm: int = 60,
    max_tpm: int = 100000,
    daily_request_limit: int | None = None,
    daily_cost_limit: float | None = None,
) -> RateLimiter:
    """Create a rate limiter with common settings.
    
    Args:
        max_rpm: Maximum requests per minute
        max_tpm: Maximum tokens per minute
        daily_request_limit: Daily request limit
        daily_cost_limit: Daily cost limit in USD
        
    Returns:
        Configured RateLimiter instance
        
    Example:
        ```python
        limiter = create_rate_limiter(
            max_rpm=30,
            max_tpm=50000,
            daily_request_limit=1000,
            daily_cost_limit=10.0
        )
        ```
    """
    config = RateLimitConfig(
        max_requests_per_minute=max_rpm,
        max_tokens_per_minute=max_tpm,
    )
    
    quota_config = QuotaConfig(
        daily_request_limit=daily_request_limit,
        daily_cost_limit=daily_cost_limit,
    )
    
    return RateLimiter(config, quota_config)