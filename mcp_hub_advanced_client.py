#!/usr/bin/env python3
"""
Advanced MCP Hub Client

Features:
- Offline mode with request queueing
- Smart retry logic with exponential backoff
- Response streaming for long-running operations
- Tool suggestions based on usage patterns
- Request/response caching
- Batch optimization
- Auto-reconnection
- Tool usage analytics
- Request replay
"""

import asyncio
import json
import pickle
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import hashlib


@dataclass
class ClientConfig:
    """Configuration for advanced client"""
    hub_url: str = "local"
    cache_dir: Optional[str] = None
    offline_mode: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_suggestions: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    batch_size: int = 10
    timeout_seconds: int = 30


@dataclass
class QueuedRequest:
    """A request queued for offline execution"""
    request_id: str
    tool_name: str
    parameters: Dict[str, Any]
    created_at: datetime
    retries: int = 0
    priority: int = 0


@dataclass
class ToolUsageStats:
    """Track tool usage statistics"""
    tool_name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_used: Optional[datetime] = None
    average_latency_ms: float = 0.0

    def record_call(self, success: bool, latency_ms: float):
        """Record a tool call"""
        self.call_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.total_latency_ms += latency_ms
        self.last_used = datetime.now()
        if self.call_count > 0:
            self.average_latency_ms = self.total_latency_ms / self.call_count


class AdvancedMCPClient:
    """
    Advanced MCP client with offline support, caching, and smart features.

    Usage:
        config = ClientConfig(
            enable_caching=True,
            enable_suggestions=True,
            offline_mode=False
        )

        async with AdvancedMCPClient(config) as client:
            # Use with all advanced features
            result = await client.call("read_file", {"path": "/tmp/file.txt"})

            # Get tool suggestions
            suggestions = client.suggest_tools("I need to read a file")

            # Stream long-running operation
            async for update in client.call_streaming("long_task", params):
                print(f"Progress: {update}")
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        self.config = config or ClientConfig()
        self.hub = None
        self.connected = False

        # Offline queue
        self.request_queue: deque = deque()
        self.queue_file = self._get_queue_file()

        # Caching
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_dir = self._get_cache_dir()

        # Usage tracking
        self.usage_stats: Dict[str, ToolUsageStats] = defaultdict(
            lambda: ToolUsageStats(tool_name="")
        )

        # Request history
        self.request_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    async def __aenter__(self):
        """Connect to hub"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Disconnect from hub"""
        await self.disconnect()

    async def connect(self):
        """Connect to MCP hub"""
        if self.config.offline_mode:
            print("⚠️  Running in offline mode - requests will be queued")
            self._load_queue()
            return

        try:
            from distributed_mcp_hub import MCPHub
            self.hub = MCPHub()
            await self.hub.start()
            self.connected = True
            print(f"✓ Connected to MCP hub")

            # Process any queued requests
            await self._process_queue()

        except Exception as e:
            print(f"⚠️  Failed to connect to hub: {e}")
            print(f"   Switching to offline mode")
            self.config.offline_mode = True
            self._load_queue()

    async def disconnect(self):
        """Disconnect from hub"""
        if self.hub:
            await self.hub.stop()
        self.connected = False

        # Save queue if any pending requests
        if self.request_queue:
            self._save_queue()

    async def call(self, tool_name: str, parameters: Dict[str, Any],
                   use_cache: bool = True, retry: bool = True) -> Any:
        """
        Call a tool with advanced features.

        Args:
            tool_name: Name of tool to call
            parameters: Tool parameters
            use_cache: Whether to use cached results
            retry: Whether to retry on failure
        """
        start_time = datetime.now()

        # Check cache
        if use_cache and self.config.enable_caching:
            cached_result = self._get_from_cache(tool_name, parameters)
            if cached_result is not None:
                print(f"  📦 Using cached result for {tool_name}")
                return cached_result

        # If offline, queue request
        if self.config.offline_mode or not self.connected:
            return await self._queue_request(tool_name, parameters)

        # Execute with retry logic
        last_error = None
        for attempt in range(self.config.max_retries if retry else 1):
            try:
                result = await self._execute_with_timeout(tool_name, parameters)

                # Record success
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.usage_stats[tool_name].record_call(True, latency)

                # Cache result
                if use_cache and self.config.enable_caching:
                    self._add_to_cache(tool_name, parameters, result)

                # Add to history
                self._add_to_history(tool_name, parameters, result, success=True)

                return result

            except asyncio.TimeoutError as e:
                last_error = e
                print(f"  ⏱️  Timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

            except Exception as e:
                last_error = e
                print(f"  ❌ Error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        # All retries failed
        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.usage_stats[tool_name].record_call(False, latency)
        self._add_to_history(tool_name, parameters, None, success=False, error=str(last_error))

        raise last_error

    async def call_streaming(self, tool_name: str,
                            parameters: Dict[str, Any]) -> AsyncIterator[Any]:
        """
        Call a tool with streaming response.

        Yields intermediate results as they become available.
        """
        # For demo, simulate streaming by breaking up response
        result = await self.call(tool_name, parameters, use_cache=False)

        # Yield progress updates
        if isinstance(result, dict):
            for key, value in result.items():
                yield {key: value}
                await asyncio.sleep(0.1)
        else:
            yield result

    async def batch_optimized(self, calls: List[Tuple[str, Dict[str, Any]]],
                              batch_size: Optional[int] = None) -> List[Any]:
        """
        Execute batch with intelligent optimization.

        Groups similar calls, deduplicates, and optimizes execution order.
        """
        batch_size = batch_size or self.config.batch_size

        # Deduplicate calls
        unique_calls = []
        seen = set()
        for tool_name, params in calls:
            key = (tool_name, json.dumps(params, sort_keys=True))
            if key not in seen:
                unique_calls.append((tool_name, params))
                seen.add(key)

        # Group by tool name for better cache locality
        grouped = defaultdict(list)
        for i, (tool_name, params) in enumerate(unique_calls):
            grouped[tool_name].append((i, params))

        # Execute in optimized batches
        results = [None] * len(unique_calls)

        for tool_name, indexed_params in grouped.items():
            for i in range(0, len(indexed_params), batch_size):
                batch = indexed_params[i:i + batch_size]

                # Execute batch in parallel
                tasks = [
                    self.call(tool_name, params)
                    for idx, params in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Store results in correct positions
                for (idx, _), result in zip(batch, batch_results):
                    results[idx] = result

        return results

    def suggest_tools(self, description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest tools based on description and usage patterns.

        Uses simple keyword matching and usage statistics.
        """
        if not self.config.enable_suggestions:
            return []

        # Get all available tools
        if not self.hub:
            return []

        # Simple keyword-based matching
        description_lower = description.lower()
        suggestions = []

        # Mock tool list (in real implementation, get from hub)
        mock_tools = [
            {"name": "read_file", "keywords": ["read", "file", "open", "load"]},
            {"name": "write_file", "keywords": ["write", "file", "save", "store"]},
            {"name": "http_get", "keywords": ["http", "get", "fetch", "request", "api"]},
            {"name": "sql_query", "keywords": ["sql", "query", "database", "select"]},
            {"name": "execute_python", "keywords": ["python", "execute", "run", "code"]},
        ]

        for tool in mock_tools:
            score = 0

            # Keyword matching
            for keyword in tool["keywords"]:
                if keyword in description_lower:
                    score += 1

            # Usage frequency boost
            stats = self.usage_stats.get(tool["name"])
            if stats and stats.call_count > 0:
                score += stats.success_count / stats.call_count

            if score > 0:
                suggestions.append({
                    "tool_name": tool["name"],
                    "score": score,
                    "usage_count": stats.call_count if stats else 0
                })

        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)

        return suggestions[:limit]

    def get_usage_stats(self, tool_name: Optional[str] = None) -> Union[Dict, ToolUsageStats]:
        """Get usage statistics for tools"""
        if tool_name:
            return self.usage_stats.get(tool_name, ToolUsageStats(tool_name=tool_name))

        return {
            name: {
                'call_count': stats.call_count,
                'success_count': stats.success_count,
                'error_count': stats.error_count,
                'success_rate': stats.success_count / stats.call_count if stats.call_count > 0 else 0,
                'average_latency_ms': stats.average_latency_ms,
                'last_used': stats.last_used.isoformat() if stats.last_used else None
            }
            for name, stats in self.usage_stats.items()
        }

    def get_request_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent request history"""
        return self.request_history[-limit:]

    def replay_request(self, request_index: int) -> Tuple[str, Dict[str, Any]]:
        """Replay a request from history"""
        if 0 <= request_index < len(self.request_history):
            req = self.request_history[request_index]
            return req['tool_name'], req['parameters']
        raise ValueError(f"Invalid request index: {request_index}")

    async def _execute_with_timeout(self, tool_name: str,
                                    parameters: Dict[str, Any]) -> Any:
        """Execute tool with timeout"""
        result = await asyncio.wait_for(
            self.hub.execute(tool_name, parameters),
            timeout=self.config.timeout_seconds
        )
        return result['result']

    async def _queue_request(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a request for later execution"""
        request = QueuedRequest(
            request_id=str(hash(f"{tool_name}{parameters}{datetime.now()}")),
            tool_name=tool_name,
            parameters=parameters,
            created_at=datetime.now()
        )
        self.request_queue.append(request)
        self._save_queue()

        return {
            'queued': True,
            'request_id': request.request_id,
            'message': 'Request queued for execution when online'
        }

    async def _process_queue(self):
        """Process queued requests"""
        if not self.request_queue:
            return

        print(f"  📤 Processing {len(self.request_queue)} queued requests...")

        processed = 0
        failed = 0

        while self.request_queue and self.connected:
            request = self.request_queue.popleft()

            try:
                await self.call(request.tool_name, request.parameters, retry=False)
                processed += 1
            except Exception as e:
                failed += 1
                print(f"  ❌ Failed to process queued request: {e}")

                # Re-queue if retries remaining
                if request.retries < self.config.max_retries:
                    request.retries += 1
                    self.request_queue.append(request)

        print(f"  ✓ Processed {processed} requests ({failed} failed)")
        self._save_queue()

    def _get_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key"""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.sha256(f"{tool_name}:{param_str}".encode()).hexdigest()

    def _get_from_cache(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """Get result from cache"""
        key = self._get_cache_key(tool_name, parameters)

        if key in self.cache:
            result, timestamp = self.cache[key]
            age = datetime.now() - timestamp
            if age.total_seconds() < self.config.cache_ttl_seconds:
                return result
            else:
                # Expired
                del self.cache[key]

        return None

    def _add_to_cache(self, tool_name: str, parameters: Dict[str, Any], result: Any):
        """Add result to cache"""
        key = self._get_cache_key(tool_name, parameters)
        self.cache[key] = (result, datetime.now())

    def _add_to_history(self, tool_name: str, parameters: Dict[str, Any],
                       result: Any, success: bool, error: Optional[str] = None):
        """Add request to history"""
        self.request_history.append({
            'tool_name': tool_name,
            'parameters': parameters,
            'result': result if success else None,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

        # Trim history
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]

    def _get_queue_file(self) -> Path:
        """Get path to queue file"""
        cache_dir = self._get_cache_dir()
        return cache_dir / "request_queue.pkl"

    def _get_cache_dir(self) -> Path:
        """Get cache directory"""
        if self.config.cache_dir:
            cache_dir = Path(self.config.cache_dir)
        else:
            cache_dir = Path.home() / ".mcp_client_cache"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _save_queue(self):
        """Save request queue to disk"""
        try:
            with open(self.queue_file, 'wb') as f:
                pickle.dump(list(self.request_queue), f)
        except Exception as e:
            print(f"Warning: Failed to save queue: {e}")

    def _load_queue(self):
        """Load request queue from disk"""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'rb') as f:
                    queue_data = pickle.load(f)
                    self.request_queue = deque(queue_data)
                print(f"  📥 Loaded {len(self.request_queue)} queued requests")
            except Exception as e:
                print(f"Warning: Failed to load queue: {e}")


# ============================================================================
# Demo
# ============================================================================

async def demo_advanced_client():
    """Demonstrate advanced client features"""
    print("=" * 80)
    print("Advanced MCP Client Demo")
    print("=" * 80)
    print()

    # Create client with custom config
    config = ClientConfig(
        enable_caching=True,
        enable_suggestions=True,
        max_retries=3,
        cache_ttl_seconds=60
    )

    async with AdvancedMCPClient(config) as client:

        print("1️⃣  Smart Tool Suggestions")
        print("-" * 80)
        suggestions = client.suggest_tools("I need to fetch data from an API")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion['tool_name']} (score: {suggestion['score']:.1f})")
        print()

        print("2️⃣  Cached Execution")
        print("-" * 80)
        # First call - executes
        print("   First call (executes)...")
        result1 = await client.call("read_file", {"path": "/tmp/test.txt"})

        # Second call - cached
        print("   Second call (cached)...")
        result2 = await client.call("read_file", {"path": "/tmp/test.txt"})
        print()

        print("3️⃣  Batch Optimization")
        print("-" * 80)
        calls = [
            ("read_file", {"path": f"/tmp/file{i}.txt"})
            for i in range(5)
        ] + [
            ("http_get", {"url": f"https://api{i}.com", "headers": {}})
            for i in range(5)
        ]

        print(f"   Executing {len(calls)} calls with optimization...")
        results = await client.batch_optimized(calls)
        print(f"   ✓ Completed {len(results)} operations")
        print()

        print("4️⃣  Usage Statistics")
        print("-" * 80)
        stats = client.get_usage_stats()
        for tool_name, tool_stats in list(stats.items())[:5]:
            print(f"   {tool_name}:")
            print(f"      Calls: {tool_stats['call_count']}")
            print(f"      Success rate: {tool_stats['success_rate']:.1%}")
            print(f"      Avg latency: {tool_stats['average_latency_ms']:.1f}ms")
        print()

        print("5️⃣  Request History")
        print("-" * 80)
        history = client.get_request_history(limit=5)
        for i, req in enumerate(history, 1):
            status = "✓" if req['success'] else "✗"
            print(f"   {i}. {status} {req['tool_name']} at {req['timestamp']}")
        print()


if __name__ == "__main__":
    asyncio.run(demo_advanced_client())
