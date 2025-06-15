"""Tests for caching functionality."""

import pytest
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import json
import sqlite3
import time

from claude_code_sdk import ClaudeCodeOptions
from claude_code_sdk.types import (
    AssistantMessage, UserMessage, ResultMessage, SystemMessage,
    TextBlock, ToolUseBlock, ToolResultBlock
)
from claude_code_sdk.cache import (
    CacheManager, CacheConfig, CacheBackend, CacheStrategy,
    MemoryCacheBackend, FileCacheBackend, SQLiteCacheBackend,
    query_with_cache, get_global_cache
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file-based tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_messages():
    """Create mock messages for testing."""
    return [
        AssistantMessage(content=[TextBlock(text="Test response")]),
        ResultMessage(
            subtype="success",
            cost_usd=0.01,
            duration_ms=500,
            session_id="test-session",
            total_cost_usd=0.01
        )
    ]


@pytest.fixture
def mock_query(mock_messages):
    """Create a mock query function."""
    call_count = 0
    
    async def _mock_query(prompt: str, options: ClaudeCodeOptions | None = None):
        nonlocal call_count
        call_count += 1
        for msg in mock_messages:
            yield msg
    
    _mock_query.call_count = lambda: call_count
    return _mock_query


class TestMemoryCacheBackend:
    """Test memory cache backend."""
    
    @pytest.mark.asyncio
    async def test_get_set(self):
        """Test basic get/set operations."""
        backend = MemoryCacheBackend()
        
        # Test set and get
        await backend.set("key1", "value1", ttl=60)
        assert await backend.get("key1") == "value1"
        
        # Test missing key
        assert await backend.get("missing") is None
    
    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete operation."""
        backend = MemoryCacheBackend()
        
        await backend.set("key1", "value1")
        assert await backend.get("key1") == "value1"
        
        await backend.delete("key1")
        assert await backend.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clear operation."""
        backend = MemoryCacheBackend()
        
        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        
        await backend.clear()
        assert await backend.get("key1") is None
        assert await backend.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        backend = MemoryCacheBackend()
        
        # Set with very short TTL
        await backend.set("key1", "value1", ttl=0.01)
        assert await backend.get("key1") == "value1"
        
        # Wait for expiration
        await asyncio.sleep(0.02)
        assert await backend.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_max_size(self):
        """Test max size limit."""
        backend = MemoryCacheBackend(max_size=2)
        
        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        await backend.set("key3", "value3")  # Should evict key1
        
        assert await backend.get("key1") is None
        assert await backend.get("key2") == "value2"
        assert await backend.get("key3") == "value3"


class TestFileCacheBackend:
    """Test file cache backend."""
    
    @pytest.mark.asyncio
    async def test_get_set(self, temp_dir):
        """Test basic get/set operations."""
        backend = FileCacheBackend(cache_dir=temp_dir)
        
        await backend.set("key1", {"data": "value1"})
        assert await backend.get("key1") == {"data": "value1"}
    
    @pytest.mark.asyncio
    async def test_delete(self, temp_dir):
        """Test delete operation."""
        backend = FileCacheBackend(cache_dir=temp_dir)
        
        await backend.set("key1", "value1")
        await backend.delete("key1")
        assert await backend.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_clear(self, temp_dir):
        """Test clear operation."""
        backend = FileCacheBackend(cache_dir=temp_dir)
        
        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        
        await backend.clear()
        assert await backend.get("key1") is None
        assert await backend.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, temp_dir):
        """Test TTL expiration."""
        backend = FileCacheBackend(cache_dir=temp_dir)
        
        await backend.set("key1", "value1", ttl=0.01)
        assert await backend.get("key1") == "value1"
        
        await asyncio.sleep(0.02)
        assert await backend.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_safe_key_conversion(self, temp_dir):
        """Test safe key conversion for filesystem."""
        backend = FileCacheBackend(cache_dir=temp_dir)
        
        # Test key with special characters
        key = "test/key:with*special|chars"
        await backend.set(key, "value")
        assert await backend.get(key) == "value"


class TestSQLiteCacheBackend:
    """Test SQLite cache backend."""
    
    @pytest.mark.asyncio
    async def test_get_set(self, temp_dir):
        """Test basic get/set operations."""
        db_path = Path(temp_dir) / "cache.db"
        backend = SQLiteCacheBackend(db_path=str(db_path))
        
        await backend.set("key1", {"data": "value1"})
        assert await backend.get("key1") == {"data": "value1"}
    
    @pytest.mark.asyncio
    async def test_delete(self, temp_dir):
        """Test delete operation."""
        db_path = Path(temp_dir) / "cache.db"
        backend = SQLiteCacheBackend(db_path=str(db_path))
        
        await backend.set("key1", "value1")
        await backend.delete("key1")
        assert await backend.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_clear(self, temp_dir):
        """Test clear operation."""
        db_path = Path(temp_dir) / "cache.db"
        backend = SQLiteCacheBackend(db_path=str(db_path))
        
        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        
        await backend.clear()
        assert await backend.get("key1") is None
        assert await backend.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, temp_dir):
        """Test TTL expiration and cleanup."""
        db_path = Path(temp_dir) / "cache.db"
        backend = SQLiteCacheBackend(db_path=str(db_path))
        
        await backend.set("key1", "value1", ttl=0.01)
        assert await backend.get("key1") == "value1"
        
        await asyncio.sleep(0.02)
        assert await backend.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_max_size(self, temp_dir):
        """Test max size limit."""
        db_path = Path(temp_dir) / "cache.db"
        backend = SQLiteCacheBackend(db_path=str(db_path), max_size=2)
        
        await backend.set("key1", "value1")
        await backend.set("key2", "value2")
        await backend.set("key3", "value3")  # Should trigger cleanup
        
        # One of the first two should be evicted
        remaining_count = 0
        if await backend.get("key1"):
            remaining_count += 1
        if await backend.get("key2"):
            remaining_count += 1
        assert remaining_count == 1
        assert await backend.get("key3") == "value3"


class TestCacheManager:
    """Test cache manager."""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_query, mock_messages):
        """Test cache hit scenario."""
        backend = MemoryCacheBackend()
        manager = CacheManager(CacheConfig(backend=backend))
        
        # First query - cache miss
        messages1 = []
        async for msg in manager.query_with_cache("test prompt", query_func=mock_query):
            messages1.append(msg)
        
        assert len(messages1) == len(mock_messages)
        assert mock_query.call_count() == 1
        
        # Second query - cache hit
        messages2 = []
        async for msg in manager.query_with_cache("test prompt", query_func=mock_query):
            messages2.append(msg)
        
        assert len(messages2) == len(mock_messages)
        assert mock_query.call_count() == 1  # Should not increase
    
    @pytest.mark.asyncio
    async def test_cache_key_with_options(self, mock_query):
        """Test cache key generation with options."""
        backend = MemoryCacheBackend()
        manager = CacheManager(CacheConfig(backend=backend))
        
        options1 = ClaudeCodeOptions(allowed_tools=["Read"])
        options2 = ClaudeCodeOptions(allowed_tools=["Write"])
        
        # Query with options1
        async for _ in manager.query_with_cache("test", options=options1, query_func=mock_query):
            pass
        
        assert mock_query.call_count() == 1
        
        # Query with options2 - should be cache miss
        async for _ in manager.query_with_cache("test", options=options2, query_func=mock_query):
            pass
        
        assert mock_query.call_count() == 2
    
    @pytest.mark.asyncio
    async def test_ttl_configuration(self, mock_query):
        """Test TTL configuration."""
        backend = MemoryCacheBackend()
        manager = CacheManager(CacheConfig(
            backend=backend,
            ttl_seconds=0.01
        ))
        
        # First query
        async for _ in manager.query_with_cache("test", query_func=mock_query):
            pass
        
        assert mock_query.call_count() == 1
        
        # Wait for TTL to expire
        await asyncio.sleep(0.02)
        
        # Second query - should be cache miss
        async for _ in manager.query_with_cache("test", query_func=mock_query):
            pass
        
        assert mock_query.call_count() == 2
    
    @pytest.mark.asyncio
    async def test_disabled_cache(self, mock_query):
        """Test disabled cache."""
        manager = CacheManager(CacheConfig(enabled=False))
        
        # Multiple queries should all hit the actual function
        for _ in range(3):
            async for _ in manager.query_with_cache("test", query_func=mock_query):
                pass
        
        assert mock_query.call_count() == 3
    
    @pytest.mark.asyncio
    async def test_cache_strategies(self, mock_query):
        """Test different cache strategies."""
        # Test ALWAYS strategy (default)
        backend = MemoryCacheBackend()
        manager = CacheManager(CacheConfig(
            backend=backend,
            strategy=CacheStrategy.ALWAYS
        ))
        
        async for _ in manager.query_with_cache("test", query_func=mock_query):
            pass
        async for _ in manager.query_with_cache("test", query_func=mock_query):
            pass
        
        assert mock_query.call_count() == 1
        
        # Test NEVER strategy
        manager = CacheManager(CacheConfig(
            backend=backend,
            strategy=CacheStrategy.NEVER
        ))
        
        async for _ in manager.query_with_cache("test", query_func=mock_query):
            pass
        
        assert mock_query.call_count() == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in cache operations."""
        # Create a faulty backend
        class FaultyBackend:
            async def get(self, key: str):
                raise Exception("Backend error")
            
            async def set(self, key: str, value, ttl=None):
                raise Exception("Backend error")
        
        manager = CacheManager(CacheConfig(backend=FaultyBackend()))
        
        async def simple_query(prompt, options=None):
            yield AssistantMessage(content=[TextBlock(text="Response")])
        
        # Should still work despite backend errors
        messages = []
        async for msg in manager.query_with_cache("test", query_func=simple_query):
            messages.append(msg)
        
        assert len(messages) == 1
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, mock_query):
        """Test cache statistics."""
        backend = MemoryCacheBackend()
        manager = CacheManager(CacheConfig(backend=backend))
        
        # Generate some hits and misses
        async for _ in manager.query_with_cache("prompt1", query_func=mock_query):
            pass
        async for _ in manager.query_with_cache("prompt1", query_func=mock_query):
            pass
        async for _ in manager.query_with_cache("prompt2", query_func=mock_query):
            pass
        
        stats = manager.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 1/3
    
    @pytest.mark.asyncio
    async def test_invalidate_cache(self, mock_query):
        """Test cache invalidation."""
        backend = MemoryCacheBackend()
        manager = CacheManager(CacheConfig(backend=backend))
        
        # Cache a query
        async for _ in manager.query_with_cache("test", query_func=mock_query):
            pass
        
        assert mock_query.call_count() == 1
        
        # Invalidate specific key
        await manager.invalidate("test")
        
        # Should be cache miss
        async for _ in manager.query_with_cache("test", query_func=mock_query):
            pass
        
        assert mock_query.call_count() == 2
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_query):
        """Test clearing entire cache."""
        backend = MemoryCacheBackend()
        manager = CacheManager(CacheConfig(backend=backend))
        
        # Cache multiple queries
        async for _ in manager.query_with_cache("test1", query_func=mock_query):
            pass
        async for _ in manager.query_with_cache("test2", query_func=mock_query):
            pass
        
        # Clear cache
        await manager.clear()
        
        # Both should be cache misses
        call_count_before = mock_query.call_count()
        async for _ in manager.query_with_cache("test1", query_func=mock_query):
            pass
        async for _ in manager.query_with_cache("test2", query_func=mock_query):
            pass
        
        assert mock_query.call_count() == call_count_before + 2


@pytest.mark.asyncio
async def test_query_with_cache_function(mock_query):
    """Test the convenience query_with_cache function."""
    # Set up global cache
    cache = get_global_cache()
    cache._config.backend = MemoryCacheBackend()
    cache._config.enabled = True
    
    with patch("claude_code_sdk.cache.query", mock_query):
        # First call - cache miss
        messages1 = []
        async for msg in query_with_cache(prompt="test"):
            messages1.append(msg)
        
        # Second call - cache hit
        messages2 = []
        async for msg in query_with_cache(prompt="test"):
            messages2.append(msg)
    
    assert len(messages1) == len(messages2)
    assert mock_query.call_count() == 1


@pytest.mark.asyncio
async def test_global_cache_singleton():
    """Test global cache singleton pattern."""
    cache1 = get_global_cache()
    cache2 = get_global_cache()
    
    assert cache1 is cache2


@pytest.mark.asyncio
async def test_cache_with_complex_messages():
    """Test caching with complex message types."""
    backend = MemoryCacheBackend()
    manager = CacheManager(CacheConfig(backend=backend))
    
    complex_messages = [
        AssistantMessage(content=[
            TextBlock(text="Hello"),
            ToolUseBlock(id="tool1", name="Read", input={"file": "test.py"}),
            ToolResultBlock(tool_use_id="tool1", content="File contents"),
        ]),
        SystemMessage(subtype="info", data={"info": "Processing"}),
        ResultMessage(
            subtype="success",
            cost_usd=0.02,
            duration_ms=1000,
            session_id="complex-session",
            total_cost_usd=0.02,
            usage={"input_tokens": 50, "output_tokens": 100}
        )
    ]
    
    async def complex_query(prompt, options=None):
        for msg in complex_messages:
            yield msg
    
    # Cache the complex messages
    messages1 = []
    async for msg in manager.query_with_cache("complex", query_func=complex_query):
        messages1.append(msg)
    
    # Retrieve from cache
    messages2 = []
    async for msg in manager.query_with_cache("complex", query_func=complex_query):
        messages2.append(msg)
    
    # Verify all message types are properly cached
    assert len(messages1) == len(messages2) == len(complex_messages)
    for m1, m2 in zip(messages1, messages2):
        assert type(m1) == type(m2)
        assert m1 == m2


import asyncio  # Add missing import