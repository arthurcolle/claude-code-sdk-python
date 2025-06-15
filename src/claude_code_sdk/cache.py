"""Caching layer for repeated queries."""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from collections.abc import AsyncIterator
from pathlib import Path
import pickle
from datetime import datetime, timedelta

from .types import Message, ClaudeCodeOptions


@dataclass
class CacheEntry:
    """A single cache entry."""
    
    key: str
    messages: list[Message]
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 1
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheBackend:
    """Base class for cache backends."""
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        raise NotImplementedError
    
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache."""
        raise NotImplementedError
    
    async def delete(self, key: str) -> None:
        """Delete entry from cache."""
        raise NotImplementedError
    
    async def clear(self) -> None:
        """Clear all entries."""
        raise NotImplementedError
    
    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        raise NotImplementedError


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                if entry.is_expired():
                    del self._cache[key]
                    self._misses += 1
                    return None
                entry.touch()
                self._hits += 1
                return entry
            self._misses += 1
            return None
    
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache."""
        async with self._lock:
            # Evict old entries if needed
            if len(self._cache) >= self.max_size:
                # Evict least recently accessed
                lru_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].accessed_at
                )
                del self._cache[lru_key]
            
            self._cache[key] = entry
    
    async def delete(self, key: str) -> None:
        """Delete entry from cache."""
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class FileCacheBackend(CacheBackend):
    """File-based cache backend."""
    
    def __init__(self, cache_dir: Union[str, Path] = ".claude_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._stats_file = self.cache_dir / "stats.json"
        self._load_stats()
    
    def _load_stats(self) -> None:
        """Load statistics."""
        if self._stats_file.exists():
            with open(self._stats_file, "r") as f:
                self._stats = json.load(f)
        else:
            self._stats = {"hits": 0, "misses": 0}
    
    def _save_stats(self) -> None:
        """Save statistics."""
        with open(self._stats_file, "w") as f:
            json.dump(self._stats, f)
    
    def _get_path(self, key: str) -> Path:
        """Get file path for key."""
        return self.cache_dir / f"{key}.pkl"
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        path = self._get_path(key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    entry = pickle.load(f)
                if entry.is_expired():
                    path.unlink()
                    self._stats["misses"] += 1
                    self._save_stats()
                    return None
                entry.touch()
                # Save updated entry
                with open(path, "wb") as f:
                    pickle.dump(entry, f)
                self._stats["hits"] += 1
                self._save_stats()
                return entry
            except Exception:
                # Corrupted cache file
                path.unlink()
        
        self._stats["misses"] += 1
        self._save_stats()
        return None
    
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache."""
        path = self._get_path(key)
        with open(path, "wb") as f:
            pickle.dump(entry, f)
    
    async def delete(self, key: str) -> None:
        """Delete entry from cache."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
    
    async def clear(self) -> None:
        """Clear all entries."""
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
        self._stats = {"hits": 0, "misses": 0}
        self._save_stats()
    
    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        entries = len(list(self.cache_dir.glob("*.pkl")))
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            "entries": entries,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
        }


@dataclass
class CacheConfig:
    """Cache configuration."""
    
    backend: CacheBackend = field(default_factory=MemoryCacheBackend)
    ttl_seconds: Optional[float] = 3600  # 1 hour default
    key_prefix: str = "claude_"
    include_options: bool = True
    hash_prompt: bool = True


class QueryCache:
    """Cache for Claude queries."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
    
    def _generate_key(self, prompt: str, options: Optional[ClaudeCodeOptions]) -> str:
        """Generate cache key."""
        # Create key components
        components = [self.config.key_prefix]
        
        if self.config.hash_prompt:
            # Hash the prompt for shorter keys
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            components.append(prompt_hash)
        else:
            # Use sanitized prompt
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in " -_")
            components.append(safe_prompt)
        
        if self.config.include_options and options:
            # Include relevant options in key
            opt_dict = {
                "model": options.model,
                "system_prompt": options.system_prompt,
                "max_thinking_tokens": options.max_thinking_tokens,
            }
            opt_str = json.dumps(opt_dict, sort_keys=True)
            opt_hash = hashlib.sha256(opt_str.encode()).hexdigest()[:8]
            components.append(opt_hash)
        
        return "_".join(components)
    
    async def get(
        self,
        prompt: str,
        options: Optional[ClaudeCodeOptions]
    ) -> Optional[list[Message]]:
        """Get cached messages if available."""
        key = self._generate_key(prompt, options)
        entry = await self.config.backend.get(key)
        if entry:
            return entry.messages
        return None
    
    async def set(
        self,
        prompt: str,
        options: Optional[ClaudeCodeOptions],
        messages: list[Message]
    ) -> None:
        """Cache messages."""
        key = self._generate_key(prompt, options)
        entry = CacheEntry(
            key=key,
            messages=messages,
            ttl_seconds=self.config.ttl_seconds
        )
        await self.config.backend.set(key, entry)
    
    async def invalidate(
        self,
        prompt: str,
        options: Optional[ClaudeCodeOptions]
    ) -> None:
        """Invalidate cached entry."""
        key = self._generate_key(prompt, options)
        await self.config.backend.delete(key)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        await self.config.backend.clear()
    
    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self.config.backend.get_stats()


# Global cache instance
_global_cache: Optional[QueryCache] = None


def get_global_cache(config: Optional[CacheConfig] = None) -> QueryCache:
    """Get or create global cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = QueryCache(config)
    return _global_cache


async def cached_query(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    cache: Optional[QueryCache] = None,
    force_refresh: bool = False
) -> AsyncIterator[Message]:
    """
    Query Claude with caching.
    
    Args:
        prompt: The prompt to send
        options: Query options
        cache: Cache instance (uses global if None)
        force_refresh: Force cache refresh
        
    Yields:
        Messages from conversation (cached or fresh)
        
    Example:
        ```python
        # First query hits Claude
        async for msg in cached_query(prompt="Hello"):
            print(msg)
        
        # Second identical query uses cache
        async for msg in cached_query(prompt="Hello"):
            print(msg)  # Much faster!
        
        # Force refresh
        async for msg in cached_query(
            prompt="Hello",
            force_refresh=True
        ):
            print(msg)
        ```
    """
    from . import query
    
    if cache is None:
        cache = get_global_cache()
    
    # Check cache unless forcing refresh
    if not force_refresh:
        cached_messages = await cache.get(prompt, options)
        if cached_messages:
            for message in cached_messages:
                yield message
            return
    
    # Execute query and collect messages
    messages = []
    async for message in query(prompt=prompt, options=options):
        messages.append(message)
        yield message
    
    # Cache the results
    await cache.set(prompt, options, messages)