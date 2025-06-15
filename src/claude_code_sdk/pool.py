"""Connection pooling for concurrent queries."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Set
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from ._internal.client import InternalClient
from .types import Message, ClaudeCodeOptions


@dataclass
class PooledConnection:
    """A pooled connection wrapper."""
    
    client: InternalClient
    pool: "ConnectionPool"
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    query_count: int = 0
    in_use: bool = False
    
    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used = datetime.now()
        self.query_count += 1
    
    def is_stale(self, max_idle_seconds: float) -> bool:
        """Check if connection is stale."""
        idle_time = datetime.now() - self.last_used
        return idle_time > timedelta(seconds=max_idle_seconds)


@dataclass 
class ConnectionPool:
    """Manages a pool of Claude SDK connections for concurrent queries."""
    
    max_size: int = 10
    max_idle_seconds: float = 300.0  # 5 minutes
    max_queries_per_connection: int = 100
    
    _connections: list[PooledConnection] = field(default_factory=list, init=False)
    _available: asyncio.Queue[PooledConnection] = field(init=False)
    _stats: dict[str, int] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _closed: bool = field(default=False, init=False)
    
    def __post_init__(self):
        self._available = asyncio.Queue(maxsize=self.max_size)
        self._stats = {
            "created": 0,
            "reused": 0,
            "evicted": 0,
            "queries": 0,
        }
    
    async def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        client = InternalClient()
        conn = PooledConnection(client=client, pool=self)
        self._connections.append(conn)
        self._stats["created"] += 1
        return conn
    
    async def _evict_stale(self) -> None:
        """Remove stale connections."""
        async with self._lock:
            active_conns = []
            for conn in self._connections:
                if not conn.in_use and conn.is_stale(self.max_idle_seconds):
                    self._stats["evicted"] += 1
                else:
                    active_conns.append(conn)
            self._connections = active_conns
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        # Try to get available connection
        conn = None
        try:
            conn = self._available.get_nowait()
            if conn.is_stale(self.max_idle_seconds) or \
               conn.query_count >= self.max_queries_per_connection:
                # Connection is stale or overused, create new one
                self._stats["evicted"] += 1
                conn = None
        except asyncio.QueueEmpty:
            pass
        
        # Create new connection if needed
        if conn is None:
            async with self._lock:
                if len(self._connections) < self.max_size:
                    conn = await self._create_connection()
                else:
                    # Wait for available connection
                    conn = await self._available.get()
        
        if conn is None:
            raise RuntimeError("Failed to acquire connection")
        
        conn.in_use = True
        conn.touch()
        self._stats["reused"] += 1
        
        try:
            yield conn
        finally:
            conn.in_use = False
            if not self._closed:
                await self._available.put(conn)
    
    async def query(
        self,
        *,
        prompt: str,
        options: ClaudeCodeOptions | None = None
    ) -> AsyncIterator[Message]:
        """Execute a query using a pooled connection."""
        async with self.acquire() as conn:
            self._stats["queries"] += 1
            async for message in conn.client.process_query(
                prompt=prompt,
                options=options
            ):
                yield message
    
    async def execute_batch(
        self,
        queries: list[tuple[str, ClaudeCodeOptions | None]]
    ) -> list[list[Message]]:
        """Execute multiple queries concurrently."""
        async def _execute_single(
            prompt: str,
            options: ClaudeCodeOptions | None
        ) -> list[Message]:
            messages = []
            async for msg in self.query(prompt=prompt, options=options):
                messages.append(msg)
            return messages
        
        tasks = [
            _execute_single(prompt, options)
            for prompt, options in queries
        ]
        return await asyncio.gather(*tasks)
    
    async def close(self) -> None:
        """Close the connection pool."""
        self._closed = True
        # Clean up connections
        async with self._lock:
            self._connections.clear()
        # Clear queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    def get_stats(self) -> dict[str, int]:
        """Get pool statistics."""
        return {
            **self._stats,
            "active": len(self._connections),
            "available": self._available.qsize(),
        }
    
    async def health_check(self) -> dict[str, Any]:
        """Perform health check on the pool."""
        await self._evict_stale()
        return {
            "healthy": not self._closed,
            "stats": self.get_stats(),
            "max_size": self.max_size,
        }


# Global pool instance
_global_pool: Optional[ConnectionPool] = None


def get_global_pool(
    max_size: int = 10,
    max_idle_seconds: float = 300.0
) -> ConnectionPool:
    """
    Get or create the global connection pool.
    
    Args:
        max_size: Maximum pool size
        max_idle_seconds: Maximum idle time before eviction
        
    Returns:
        Global connection pool instance
        
    Example:
        ```python
        pool = get_global_pool()
        
        # Use pool for queries
        async for message in pool.query(prompt="Hello"):
            print(message)
        
        # Execute batch queries
        results = await pool.execute_batch([
            ("Query 1", None),
            ("Query 2", options),
            ("Query 3", None),
        ])
        ```
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = ConnectionPool(
            max_size=max_size,
            max_idle_seconds=max_idle_seconds
        )
    return _global_pool


async def pooled_query(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    pool: Optional[ConnectionPool] = None
) -> AsyncIterator[Message]:
    """
    Execute a query using connection pooling.
    
    Args:
        prompt: The prompt to send
        options: Query options
        pool: Connection pool (uses global if None)
        
    Yields:
        Messages from the conversation
        
    Example:
        ```python
        # Using global pool
        async for message in pooled_query(prompt="Hello"):
            print(message)
        
        # Using custom pool
        my_pool = ConnectionPool(max_size=20)
        async for message in pooled_query(
            prompt="Hello",
            pool=my_pool
        ):
            print(message)
        ```
    """
    if pool is None:
        pool = get_global_pool()
    
    async for message in pool.query(prompt=prompt, options=options):
        yield message