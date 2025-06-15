"""Rate limiting and quota management for Claude SDK."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from enum import Enum

from .types import Message, ClaudeCodeOptions, ResultMessage
from ._errors import ClaudeSDKError


class QuotaExceededError(ClaudeSDKError):
    """Raised when quota is exceeded."""
    pass


class RateLimitError(ClaudeSDKError):
    """Raised when rate limit is exceeded."""
    pass


class QuotaPeriod(Enum):
    """Quota period types."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class QuotaLimit:
    """Defines a quota limit."""
    
    limit: int
    period: QuotaPeriod
    current: int = 0
    reset_at: datetime = field(default_factory=datetime.now)
    
    def get_period_seconds(self) -> int:
        """Get period duration in seconds."""
        if self.period == QuotaPeriod.MINUTE:
            return 60
        elif self.period == QuotaPeriod.HOUR:
            return 3600
        elif self.period == QuotaPeriod.DAY:
            return 86400
        elif self.period == QuotaPeriod.WEEK:
            return 604800
        elif self.period == QuotaPeriod.MONTH:
            return 2592000  # 30 days
        return 60
    
    def check_and_reset(self) -> None:
        """Check if quota should be reset."""
        now = datetime.now()
        period_delta = timedelta(seconds=self.get_period_seconds())
        
        if now >= self.reset_at + period_delta:
            self.current = 0
            self.reset_at = now
    
    def has_quota(self) -> bool:
        """Check if quota is available."""
        self.check_and_reset()
        return self.current < self.limit
    
    def consume(self, amount: int = 1) -> None:
        """Consume quota."""
        self.check_and_reset()
        self.current += amount
        if self.current > self.limit:
            raise QuotaExceededError(
                f"Quota exceeded: {self.current}/{self.limit} per {self.period.value}"
            )
    
    def get_remaining(self) -> int:
        """Get remaining quota."""
        self.check_and_reset()
        return max(0, self.limit - self.current)
    
    def get_reset_time(self) -> datetime:
        """Get next reset time."""
        period_delta = timedelta(seconds=self.get_period_seconds())
        return self.reset_at + period_delta


@dataclass
class RateLimiter:
    """Token bucket rate limiter."""
    
    rate: float  # Tokens per second
    burst: int  # Maximum burst size
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    
    def __post_init__(self):
        self._tokens = float(self.burst)
        self._last_update = time.time()
    
    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary. Returns wait time."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            
            # Add tokens based on elapsed time
            self._tokens = min(
                self.burst,
                self._tokens + elapsed * self.rate
            )
            self._last_update = now
            
            # Wait if not enough tokens
            wait_time = 0.0
            if self._tokens < tokens:
                wait_time = (tokens - self._tokens) / self.rate
                await asyncio.sleep(wait_time)
                self._tokens = tokens
            
            self._tokens -= tokens
            return wait_time
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting."""
        now = time.time()
        elapsed = now - self._last_update
        
        # Add tokens based on elapsed time
        self._tokens = min(
            self.burst,
            self._tokens + elapsed * self.rate
        )
        self._last_update = now
        
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


@dataclass
class QuotaManager:
    """Manages quotas and rate limits."""
    
    # Rate limits
    queries_per_second: float = 1.0
    tokens_per_minute: int = 100000
    cost_per_day_usd: float = 100.0
    
    # Quotas
    queries_per_hour: Optional[int] = None
    queries_per_day: Optional[int] = None
    total_cost_limit_usd: Optional[float] = None
    
    # Internal state
    _rate_limiter: RateLimiter = field(init=False)
    _quotas: dict[str, QuotaLimit] = field(default_factory=dict, init=False)
    _total_cost: float = field(default=0.0, init=False)
    _stats: dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self._rate_limiter = RateLimiter(
            rate=self.queries_per_second,
            burst=int(self.queries_per_second * 10)  # 10 second burst
        )
        
        # Initialize quotas
        if self.queries_per_hour:
            self._quotas["queries_hour"] = QuotaLimit(
                limit=self.queries_per_hour,
                period=QuotaPeriod.HOUR
            )
        
        if self.queries_per_day:
            self._quotas["queries_day"] = QuotaLimit(
                limit=self.queries_per_day,
                period=QuotaPeriod.DAY
            )
        
        if self.tokens_per_minute:
            self._quotas["tokens_minute"] = QuotaLimit(
                limit=self.tokens_per_minute,
                period=QuotaPeriod.MINUTE
            )
        
        if self.cost_per_day_usd:
            self._quotas["cost_day"] = QuotaLimit(
                limit=int(self.cost_per_day_usd * 10000),  # Store as cents * 100
                period=QuotaPeriod.DAY
            )
        
        self._stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "rate_limited_count": 0,
            "quota_exceeded_count": 0,
        }
    
    async def check_and_wait(self) -> None:
        """Check rate limits and quotas, waiting if necessary."""
        # Check rate limit
        wait_time = await self._rate_limiter.acquire()
        if wait_time > 0:
            self._stats["rate_limited_count"] += 1
        
        # Check quotas
        for quota in self._quotas.values():
            if not quota.has_quota():
                self._stats["quota_exceeded_count"] += 1
                raise QuotaExceededError(
                    f"Quota exceeded for {quota.period.value}: "
                    f"{quota.current}/{quota.limit}"
                )
    
    def update_usage(self, message: Message) -> None:
        """Update usage from message."""
        if isinstance(message, ResultMessage):
            # Update cost
            if message.cost_usd:
                self._total_cost += message.cost_usd
                self._stats["total_cost_usd"] = self._total_cost
                
                # Check total cost limit
                if self.total_cost_limit_usd and self._total_cost > self.total_cost_limit_usd:
                    raise QuotaExceededError(
                        f"Total cost limit exceeded: ${self._total_cost:.2f} > "
                        f"${self.total_cost_limit_usd:.2f}"
                    )
                
                # Update cost quota
                if "cost_day" in self._quotas:
                    cost_cents = int(message.cost_usd * 10000)
                    self._quotas["cost_day"].consume(cost_cents)
            
            # Update token usage
            if message.usage:
                total_tokens = message.usage.get("total_tokens", 0)
                self._stats["total_tokens"] += total_tokens
                
                if "tokens_minute" in self._quotas:
                    self._quotas["tokens_minute"].consume(total_tokens)
        
        # Update query count
        self._stats["total_queries"] += 1
        for key in ["queries_hour", "queries_day"]:
            if key in self._quotas:
                self._quotas[key].consume(1)
    
    def get_status(self) -> dict[str, Any]:
        """Get current quota status."""
        status = {
            "stats": self._stats,
            "quotas": {},
            "rate_limit": {
                "rate": self.queries_per_second,
                "available": self._rate_limiter._tokens,
            }
        }
        
        for name, quota in self._quotas.items():
            status["quotas"][name] = {
                "limit": quota.limit,
                "current": quota.current,
                "remaining": quota.get_remaining(),
                "reset_at": quota.get_reset_time().isoformat(),
                "period": quota.period.value,
            }
        
        return status
    
    def reset_quotas(self) -> None:
        """Reset all quotas."""
        for quota in self._quotas.values():
            quota.current = 0
            quota.reset_at = datetime.now()


# Global quota manager
_global_quota_manager: Optional[QuotaManager] = None


def get_global_quota_manager(
    queries_per_second: float = 1.0,
    **kwargs
) -> QuotaManager:
    """Get or create global quota manager."""
    global _global_quota_manager
    if _global_quota_manager is None:
        _global_quota_manager = QuotaManager(
            queries_per_second=queries_per_second,
            **kwargs
        )
    return _global_quota_manager


async def query_with_quota(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    quota_manager: Optional[QuotaManager] = None,
    wait_for_quota: bool = True
) -> AsyncIterator[Message]:
    """
    Query Claude with quota management.
    
    Args:
        prompt: The prompt to send
        options: Query options
        quota_manager: Quota manager (uses global if None)
        wait_for_quota: Wait for quota if rate limited
        
    Yields:
        Messages from conversation
        
    Example:
        ```python
        # Configure quota manager
        manager = QuotaManager(
            queries_per_second=0.5,  # 1 query every 2 seconds
            queries_per_hour=100,
            cost_per_day_usd=10.0
        )
        
        # Query with quota management
        try:
            async for msg in query_with_quota(
                prompt="Hello",
                quota_manager=manager
            ):
                print(msg)
        except QuotaExceededError as e:
            print(f"Quota exceeded: {e}")
        
        # Check status
        print(manager.get_status())
        ```
    """
    from . import query
    
    if quota_manager is None:
        quota_manager = get_global_quota_manager()
    
    # Check quotas before query
    if wait_for_quota:
        await quota_manager.check_and_wait()
    else:
        # Just check without waiting
        for quota in quota_manager._quotas.values():
            if not quota.has_quota():
                raise QuotaExceededError(
                    f"Quota exceeded for {quota.period.value}"
                )
    
    # Execute query
    async for message in query(prompt=prompt, options=options):
        # Update usage
        quota_manager.update_usage(message)
        yield message