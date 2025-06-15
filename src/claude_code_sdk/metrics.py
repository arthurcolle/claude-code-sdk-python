"""Performance metrics and analytics for Claude Code SDK.

This module provides comprehensive metrics collection and analysis:
- Real-time performance monitoring
- Request/response metrics
- Token usage analytics
- Cost tracking and projections
- Custom metric collectors
- Export to various monitoring systems
"""

import asyncio
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Optional, Protocol

import anyio
from typing_extensions import TypedDict

from .types import Message, ResultMessage


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AggregationType(str, Enum):
    """Aggregation types for metrics."""

    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"
    RATE = "rate"


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    name: str
    count: int
    sum: float
    min: float
    max: float
    average: float
    stddev: float
    p50: float  # median
    p90: float
    p95: float
    p99: float
    rate_per_second: float | None = None
    rate_per_minute: float | None = None


class MetricExporter(Protocol):
    """Protocol for metric exporters."""

    async def export(self, metrics: list[MetricPoint]) -> None:
        """Export metrics to external system."""
        ...


class ConsoleExporter:
    """Export metrics to console for debugging."""

    async def export(self, metrics: list[MetricPoint]) -> None:
        """Print metrics to console."""
        for metric in metrics:
            labels_str = ", ".join(f"{k}={v}" for k, v in metric.labels.items())
            print(
                f"[{metric.timestamp.isoformat()}] "
                f"{metric.name}{{{labels_str}}} = {metric.value}"
            )


class PrometheusExporter:
    """Export metrics in Prometheus format."""

    def __init__(self, prefix: str = "claude_code"):
        self.prefix = prefix

    async def export(self, metrics: list[MetricPoint]) -> None:
        """Export metrics in Prometheus format."""
        # Group by metric name
        grouped: dict[str, list[MetricPoint]] = defaultdict(list)
        for metric in metrics:
            grouped[metric.name].append(metric)
        
        lines = []
        for name, points in grouped.items():
            metric_name = f"{self.prefix}_{name}"
            
            # Add TYPE comment
            if points:
                lines.append(f"# TYPE {metric_name} {points[0].metric_type.value}")
            
            # Add metric lines
            for point in points:
                labels_str = ",".join(
                    f'{k}="{v}"' for k, v in point.labels.items()
                )
                if labels_str:
                    lines.append(f"{metric_name}{{{labels_str}}} {point.value}")
                else:
                    lines.append(f"{metric_name} {point.value}")
        
        # In real implementation, would send to Prometheus pushgateway
        # For now, just return the formatted output
        return "\n".join(lines)


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation."""

    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    tool_uses: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    response_times: list[float] = field(default_factory=list)
    
    def add_message(self, message: Message) -> None:
        """Add message to metrics."""
        self.total_messages += 1
        
        if hasattr(message, "__class__"):
            if message.__class__.__name__ == "UserMessage":
                self.user_messages += 1
            elif message.__class__.__name__ == "AssistantMessage":
                self.assistant_messages += 1
    
    def add_result(self, result: ResultMessage) -> None:
        """Add result message to metrics."""
        if result.cost_usd:
            self.total_cost += result.cost_usd
        
        if result.usage:
            self.total_tokens += result.usage.get("total_tokens", 0)
            self.prompt_tokens += result.usage.get("prompt_tokens", 0)
            self.completion_tokens += result.usage.get("completion_tokens", 0)
        
        if result.duration_ms:
            self.response_times.append(result.duration_ms / 1000.0)
    
    def get_summary(self) -> dict[str, Any]:
        """Get conversation summary."""
        duration = (self.end_time or datetime.now()) - self.start_time
        
        summary = {
            "session_id": self.session_id,
            "duration_seconds": duration.total_seconds(),
            "messages": {
                "total": self.total_messages,
                "user": self.user_messages,
                "assistant": self.assistant_messages,
            },
            "tokens": {
                "total": self.total_tokens,
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            },
            "cost_usd": self.total_cost,
            "tool_uses": self.tool_uses,
            "errors": len(self.errors),
        }
        
        if self.response_times:
            summary["response_times"] = {
                "count": len(self.response_times),
                "avg_seconds": statistics.mean(self.response_times),
                "min_seconds": min(self.response_times),
                "max_seconds": max(self.response_times),
                "p95_seconds": statistics.quantiles(
                    self.response_times, n=20
                )[18] if len(self.response_times) > 1 else self.response_times[0],
            }
        
        return summary


class MetricsCollector:
    """Collects and aggregates metrics.
    
    Example:
        ```python
        collector = MetricsCollector()
        
        # Record metrics
        collector.increment("api_calls", labels={"endpoint": "query"})
        collector.record_value("response_time", 0.234, labels={"model": "claude-3"})
        
        # Get summaries
        summary = collector.get_summary("response_time")
        ```
    """

    def __init__(self, window_size: int = 3600, max_points: int = 10000):
        """Initialize collector.
        
        Args:
            window_size: Time window in seconds for rate calculations
            max_points: Maximum data points to keep per metric
        """
        self.window_size = window_size
        self.max_points = max_points
        self._metrics: dict[str, Deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=max_points)
        )
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        async with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            
            point = MetricPoint(
                name=name,
                value=self._counters[key],
                labels=labels or {},
                metric_type=MetricType.COUNTER,
            )
            self._metrics[key].append(point)

    async def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric."""
        async with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            
            point = MetricPoint(
                name=name,
                value=value,
                labels=labels or {},
                metric_type=MetricType.GAUGE,
            )
            self._metrics[key].append(point)

    async def record_value(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a value for histogram/summary metrics."""
        async with self._lock:
            key = self._make_key(name, labels)
            
            point = MetricPoint(
                name=name,
                value=value,
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM,
            )
            self._metrics[key].append(point)

    async def time_operation(self, name: str, labels: dict[str, str] | None = None):
        """Context manager to time an operation."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            await self.record_value(name, duration, labels)

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create unique key for metric."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name},{label_str}"

    async def get_summary(
        self, name: str, labels: dict[str, str] | None = None
    ) -> MetricSummary | None:
        """Get summary statistics for a metric."""
        async with self._lock:
            key = self._make_key(name, labels)
            points = list(self._metrics.get(key, []))
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        # Calculate rate
        rate_per_second = None
        rate_per_minute = None
        if len(points) > 1:
            time_span = (points[-1].timestamp - points[0].timestamp).total_seconds()
            if time_span > 0:
                if points[0].metric_type == MetricType.COUNTER:
                    value_diff = points[-1].value - points[0].value
                    rate_per_second = value_diff / time_span
                    rate_per_minute = rate_per_second * 60
        
        return MetricSummary(
            name=name,
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            average=statistics.mean(values),
            stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
            p50=statistics.median(values),
            p90=statistics.quantiles(values, n=10)[8] if len(values) > 1 else values[0],
            p95=statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0],
            p99=statistics.quantiles(values, n=100)[98] if len(values) > 1 else values[0],
            rate_per_second=rate_per_second,
            rate_per_minute=rate_per_minute,
        )

    async def get_all_metrics(self) -> list[MetricPoint]:
        """Get all current metric points."""
        async with self._lock:
            all_points = []
            for points in self._metrics.values():
                if points:
                    all_points.append(points[-1])  # Latest point
            return all_points

    async def clear_old_metrics(self, max_age_seconds: int = 3600) -> int:
        """Clear metrics older than specified age."""
        async with self._lock:
            cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)
            cleared = 0
            
            for key, points in list(self._metrics.items()):
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                    cleared += 1
                
                # Remove empty metric
                if not points:
                    del self._metrics[key]
            
            return cleared


class MetricsManager:
    """Main metrics management system.
    
    Example:
        ```python
        import asyncio
        from claude_code_sdk.metrics import MetricsManager
        
        async def main():
            manager = MetricsManager()
            
            # Start a conversation
            conv_metrics = manager.start_conversation("session-123")
            
            # Record metrics
            await manager.record_api_call("query", duration=0.5, success=True)
            await manager.record_token_usage(prompt=100, completion=50)
            
            # Get analytics
            analytics = await manager.get_analytics()
            print(analytics)
            
            # Export metrics
            await manager.export_metrics()
        
        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        collector: MetricsCollector | None = None,
        exporters: list[MetricExporter] | None = None,
        export_interval: float = 60.0,
    ):
        """Initialize metrics manager.
        
        Args:
            collector: Metrics collector instance
            exporters: List of metric exporters
            export_interval: Interval for automatic export
        """
        self.collector = collector or MetricsCollector()
        self.exporters = exporters or [ConsoleExporter()]
        self.export_interval = export_interval
        self.conversations: dict[str, ConversationMetrics] = {}
        self._export_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start metrics collection and export."""
        if self._running:
            return
        
        self._running = True
        self._export_task = asyncio.create_task(self._export_loop())

    async def stop(self) -> None:
        """Stop metrics collection and export."""
        self._running = False
        
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

    async def _export_loop(self) -> None:
        """Periodic export of metrics."""
        while self._running:
            try:
                await anyio.sleep(self.export_interval)
                await self.export_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Error exporting metrics: {e}")

    def start_conversation(self, session_id: str) -> ConversationMetrics:
        """Start tracking a conversation."""
        conv_metrics = ConversationMetrics(session_id=session_id)
        self.conversations[session_id] = conv_metrics
        return conv_metrics

    def end_conversation(self, session_id: str) -> ConversationMetrics | None:
        """End tracking a conversation."""
        if session_id in self.conversations:
            conv_metrics = self.conversations[session_id]
            conv_metrics.end_time = datetime.now()
            
            # Record final metrics
            asyncio.create_task(self._record_conversation_metrics(conv_metrics))
            
            return conv_metrics
        return None

    async def _record_conversation_metrics(self, conv: ConversationMetrics) -> None:
        """Record conversation metrics to collector."""
        summary = conv.get_summary()
        
        # Record various metrics
        await self.collector.increment("conversations_total")
        await self.collector.record_value("conversation_duration", summary["duration_seconds"])
        await self.collector.record_value("conversation_messages", summary["messages"]["total"])
        await self.collector.record_value("conversation_tokens", summary["tokens"]["total"])
        await self.collector.record_value("conversation_cost", summary["cost_usd"])
        
        if "response_times" in summary:
            for time_val in conv.response_times:
                await self.collector.record_value("response_time", time_val)

    async def record_api_call(
        self,
        endpoint: str,
        duration: float,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record an API call metric."""
        labels = {"endpoint": endpoint, "status": "success" if success else "error"}
        
        await self.collector.increment("api_calls", labels=labels)
        await self.collector.record_value("api_call_duration", duration, labels=labels)
        
        if error:
            await self.collector.increment(
                "api_errors", labels={"endpoint": endpoint, "error": error}
            )

    async def record_token_usage(
        self,
        prompt: int = 0,
        completion: int = 0,
        model: str = "claude-3",
    ) -> None:
        """Record token usage metrics."""
        total = prompt + completion
        
        await self.collector.increment("tokens_total", value=total, labels={"model": model})
        await self.collector.increment("tokens_prompt", value=prompt, labels={"model": model})
        await self.collector.increment("tokens_completion", value=completion, labels={"model": model})

    async def record_cost(self, amount: float, model: str = "claude-3") -> None:
        """Record cost metrics."""
        await self.collector.increment("cost_total", value=amount, labels={"model": model})

    async def record_tool_use(self, tool_name: str, success: bool = True) -> None:
        """Record tool usage metrics."""
        labels = {"tool": tool_name, "status": "success" if success else "error"}
        await self.collector.increment("tool_uses", labels=labels)

    async def get_analytics(self) -> dict[str, Any]:
        """Get comprehensive analytics."""
        analytics = {
            "conversations": {
                "active": len([c for c in self.conversations.values() if c.end_time is None]),
                "completed": len([c for c in self.conversations.values() if c.end_time is not None]),
            },
            "metrics": {},
        }
        
        # Get summaries for key metrics
        for metric_name in [
            "api_calls",
            "tokens_total",
            "cost_total",
            "response_time",
            "conversation_duration",
        ]:
            summary = await self.collector.get_summary(metric_name)
            if summary:
                analytics["metrics"][metric_name] = {
                    "count": summary.count,
                    "total": summary.sum,
                    "average": summary.average,
                    "min": summary.min,
                    "max": summary.max,
                    "p95": summary.p95,
                    "rate_per_minute": summary.rate_per_minute,
                }
        
        return analytics

    async def export_metrics(self) -> None:
        """Export metrics to all configured exporters."""
        metrics = await self.collector.get_all_metrics()
        
        for exporter in self.exporters:
            try:
                await exporter.export(metrics)
            except Exception as e:
                # Log error but continue with other exporters
                print(f"Error in exporter {type(exporter).__name__}: {e}")

    async def clear_old_data(self, max_age_hours: int = 24) -> int:
        """Clear old metrics and conversation data."""
        # Clear old metrics
        cleared = await self.collector.clear_old_metrics(max_age_hours * 3600)
        
        # Clear old conversations
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        old_sessions = []
        
        for session_id, conv in self.conversations.items():
            if conv.end_time and conv.end_time < cutoff_time:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            del self.conversations[session_id]
            cleared += 1
        
        return cleared


def create_metrics_manager(
    export_interval: float = 60.0,
    prometheus_enabled: bool = False,
) -> MetricsManager:
    """Create a configured metrics manager.
    
    Args:
        export_interval: Interval for metric export in seconds
        prometheus_enabled: Enable Prometheus exporter
        
    Returns:
        Configured MetricsManager instance
        
    Example:
        ```python
        manager = create_metrics_manager(
            export_interval=30.0,
            prometheus_enabled=True
        )
        
        await manager.start()
        ```
    """
    exporters: list[MetricExporter] = []
    
    if prometheus_enabled:
        exporters.append(PrometheusExporter())
    else:
        exporters.append(ConsoleExporter())
    
    return MetricsManager(exporters=exporters, export_interval=export_interval)