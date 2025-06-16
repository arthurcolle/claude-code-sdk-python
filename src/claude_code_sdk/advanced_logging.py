"""
Advanced Logging System for Claude Max Gaming Platform.

This module provides enterprise-grade logging with:
- Real-time metrics and analytics
- Game session tracking
- Performance profiling
- Distributed logging support
- Event streaming
- Anomaly detection
- Player behavior analytics
"""

import logging
import sys
import time
import json
import asyncio
import threading
import queue
import hashlib
import uuid
from pathlib import Path
from typing import Optional, Union, Any, Dict, List, Callable, TypedDict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps
import pickle
import zlib
import struct
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import psutil
import traceback


class EventType(Enum):
    """Game event types for Claude Max platform."""
    GAME_START = "game_start"
    GAME_END = "game_end"
    PLAYER_ACTION = "player_action"
    ACHIEVEMENT = "achievement"
    ERROR = "error"
    PERFORMANCE = "performance"
    TRANSACTION = "transaction"
    CHAT = "chat"
    MATCHMAKING = "matchmaking"
    LEVEL_UP = "level_up"
    ITEM_ACQUIRED = "item_acquired"
    QUEST_COMPLETE = "quest_complete"
    PVP_MATCH = "pvp_match"
    GUILD_EVENT = "guild_event"
    SYSTEM_EVENT = "system_event"


class MetricType(Enum):
    """Performance metric types."""
    FPS = "fps"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    NETWORK_THROUGHPUT = "network_throughput"
    RENDER_TIME = "render_time"
    PHYSICS_TIME = "physics_time"
    AI_COMPUTE_TIME = "ai_compute_time"


@dataclass
class GameSession:
    """Represents a game session."""
    session_id: str
    player_id: str
    game_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    metrics: Dict[str, List[float]] = None
    events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.metrics is None:
            self.metrics = defaultdict(list)
        if self.events is None:
            self.events = []


class PerformanceTracker:
    """Tracks performance metrics with statistical analysis."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.anomaly_thresholds: Dict[str, float] = {}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record(self, metric: str, value: float) -> None:
        """Record a metric value."""
        with self._lock:
            self.metrics[metric].append((time.time(), value))
            self._check_anomalies(metric, value)
    
    def get_stats(self, metric: str) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        with self._lock:
            values = [v for _, v in self.metrics.get(metric, [])]
            if not values:
                return {}
            
            sorted_values = sorted(values)
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'median': sorted_values[len(sorted_values) // 2],
                'p95': sorted_values[int(len(sorted_values) * 0.95)] if len(values) > 20 else sorted_values[-1],
                'p99': sorted_values[int(len(sorted_values) * 0.99)] if len(values) > 100 else sorted_values[-1],
            }
    
    def set_anomaly_threshold(self, metric: str, threshold: float) -> None:
        """Set anomaly detection threshold for a metric."""
        self.anomaly_thresholds[metric] = threshold
    
    def add_anomaly_callback(self, metric: str, callback: Callable) -> None:
        """Add callback for anomaly detection."""
        self.callbacks[metric].append(callback)
    
    def _check_anomalies(self, metric: str, value: float) -> None:
        """Check for anomalies in metrics."""
        if metric in self.anomaly_thresholds:
            stats = self.get_stats(metric)
            if stats and 'mean' in stats:
                deviation = abs(value - stats['mean'])
                if deviation > self.anomaly_thresholds[metric] * stats.get('mean', 1):
                    for callback in self.callbacks[metric]:
                        callback(metric, value, stats)


class EventStreamProcessor:
    """Processes game events in real-time with buffering and batching."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.event_queue: queue.Queue = queue.Queue()
        self.processors: List[Callable] = []
        self.running = False
        self._thread = None
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def start(self) -> None:
        """Start the event processor."""
        self.running = True
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the event processor."""
        self.running = False
        if self._thread:
            self._thread.join()
        self._executor.shutdown(wait=True)
    
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add an event to the processing queue."""
        event['timestamp'] = datetime.utcnow().isoformat()
        event['event_id'] = str(uuid.uuid4())
        self.event_queue.put(event)
    
    def add_processor(self, processor: Callable) -> None:
        """Add an event processor function."""
        self.processors.append(processor)
    
    def _process_events(self) -> None:
        """Process events in batches."""
        batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Try to get an event with timeout
                timeout = max(0.1, self.flush_interval - (time.time() - last_flush))
                event = self.event_queue.get(timeout=timeout)
                batch.append(event)
                
                # Check if we should flush
                should_flush = (
                    len(batch) >= self.batch_size or
                    time.time() - last_flush >= self.flush_interval
                )
                
                if should_flush and batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except queue.Empty:
                # Flush on timeout if we have events
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
    
    def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of events."""
        for processor in self.processors:
            self._executor.submit(processor, batch.copy())


class DistributedLogger:
    """Distributed logger with sharding and replication support."""
    
    def __init__(self, node_id: str, shard_count: int = 4):
        self.node_id = node_id
        self.shard_count = shard_count
        self.shards: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.replicas: Dict[str, 'DistributedLogger'] = {}
        self._lock = threading.Lock()
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message with sharding."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'node_id': self.node_id,
            'level': level,
            'message': message,
            'data': kwargs
        }
        
        # Determine shard based on hash
        shard_key = hashlib.md5(message.encode()).hexdigest()
        shard_id = int(shard_key, 16) % self.shard_count
        
        with self._lock:
            self.shards[shard_id].append(log_entry)
            
            # Replicate to other nodes
            for replica in self.replicas.values():
                replica._receive_replica(shard_id, log_entry)
    
    def _receive_replica(self, shard_id: int, log_entry: Dict[str, Any]) -> None:
        """Receive replicated log entry."""
        with self._lock:
            self.shards[shard_id].append(log_entry)
    
    def add_replica(self, node_id: str, logger: 'DistributedLogger') -> None:
        """Add a replica node."""
        self.replicas[node_id] = logger
    
    def query(self, start_time: datetime, end_time: datetime, 
              level: Optional[str] = None, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query logs across all shards."""
        results = []
        
        with self._lock:
            for shard_logs in self.shards.values():
                for log in shard_logs:
                    log_time = datetime.fromisoformat(log['timestamp'])
                    if start_time <= log_time <= end_time:
                        if level and log['level'] != level:
                            continue
                        if pattern and pattern not in log['message']:
                            continue
                        results.append(log)
        
        return sorted(results, key=lambda x: x['timestamp'])


class ClaudeMaxLogger:
    """
    Advanced logger for Claude Max gaming platform.
    
    Features:
    - Game session management
    - Real-time performance tracking
    - Event streaming and processing
    - Player analytics
    - Distributed logging
    - Automatic crash reporting
    """
    
    def __init__(self, game_id: str, player_id: str, config: Optional[Dict[str, Any]] = None):
        self.game_id = game_id
        self.player_id = player_id
        self.config = config or {}
        
        # Core components
        self.session: Optional[GameSession] = None
        self.performance_tracker = PerformanceTracker()
        self.event_processor = EventStreamProcessor()
        self.distributed_logger = DistributedLogger(f"{game_id}:{player_id}")
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Analytics
        self.player_stats: Dict[str, Any] = defaultdict(int)
        self.achievement_tracker: List[str] = []
        
        # Start background services
        self.event_processor.start()
        self._setup_event_processors()
        self._setup_performance_monitoring()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup the core logger."""
        logger = logging.getLogger(f"claude_max.{self.game_id}.{self.player_id}")
        logger.setLevel(logging.DEBUG)
        
        # Console handler with custom formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            GameLogFormatter(
                '%(asctime)s [%(levelname)s] [%(game_session)s] %(message)s'
            )
        )
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_event_processors(self) -> None:
        """Setup event processing pipelines."""
        # Analytics processor
        def analytics_processor(events: List[Dict[str, Any]]) -> None:
            for event in events:
                if event.get('type') == EventType.PLAYER_ACTION.value:
                    self.player_stats[event.get('action', 'unknown')] += 1
                elif event.get('type') == EventType.ACHIEVEMENT.value:
                    self.achievement_tracker.append(event.get('achievement_id'))
        
        # Performance processor
        def performance_processor(events: List[Dict[str, Any]]) -> None:
            for event in events:
                if event.get('type') == EventType.PERFORMANCE.value:
                    metric = event.get('metric')
                    value = event.get('value')
                    if metric and value is not None:
                        self.performance_tracker.record(metric, value)
        
        self.event_processor.add_processor(analytics_processor)
        self.event_processor.add_processor(performance_processor)
    
    def _setup_performance_monitoring(self) -> None:
        """Setup automatic performance monitoring."""
        def monitor_system():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.track_metric(MetricType.CPU_USAGE, cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.track_metric(MetricType.MEMORY_USAGE, memory.percent)
                    
                    time.sleep(5)  # Monitor every 5 seconds
                except Exception:
                    break
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def start_session(self, **metadata) -> GameSession:
        """Start a new game session."""
        self.session = GameSession(
            session_id=str(uuid.uuid4()),
            player_id=self.player_id,
            game_id=self.game_id,
            start_time=datetime.utcnow(),
            metadata=metadata
        )
        
        self.log_event(EventType.GAME_START, session_id=self.session.session_id, **metadata)
        self.logger.info(f"Game session started: {self.session.session_id}")
        
        return self.session
    
    def end_session(self) -> Optional[GameSession]:
        """End the current game session."""
        if not self.session:
            return None
        
        self.session.end_time = datetime.utcnow()
        duration = (self.session.end_time - self.session.start_time).total_seconds()
        
        # Calculate session statistics
        session_stats = {
            'duration_seconds': duration,
            'total_events': len(self.session.events),
            'achievements_earned': len([e for e in self.session.events if e.get('type') == EventType.ACHIEVEMENT.value]),
            'performance_metrics': {
                metric: self.performance_tracker.get_stats(metric)
                for metric in self.performance_tracker.metrics
            }
        }
        
        self.log_event(EventType.GAME_END, 
                      session_id=self.session.session_id,
                      stats=session_stats)
        
        self.logger.info(f"Game session ended: {self.session.session_id} (duration: {duration:.2f}s)")
        
        session = self.session
        self.session = None
        return session
    
    def log_event(self, event_type: EventType, **data) -> None:
        """Log a game event."""
        event = {
            'type': event_type.value,
            'game_id': self.game_id,
            'player_id': self.player_id,
            'session_id': self.session.session_id if self.session else None,
            **data
        }
        
        # Add to session if active
        if self.session:
            self.session.events.append(event)
        
        # Process through event stream
        self.event_processor.add_event(event)
        
        # Distributed logging
        self.distributed_logger.log('EVENT', f"{event_type.value}", **event)
    
    def track_metric(self, metric_type: MetricType, value: float) -> None:
        """Track a performance metric."""
        self.performance_tracker.record(metric_type.value, value)
        
        if self.session:
            self.session.metrics[metric_type.value].append(value)
        
        # Log high-level metrics
        if metric_type in [MetricType.FPS, MetricType.LATENCY]:
            self.logger.debug(f"{metric_type.value}: {value:.2f}")
    
    @contextmanager
    def measure_performance(self, operation: str):
        """Context manager to measure operation performance."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.track_metric(MetricType.RENDER_TIME, duration)
            self.log_event(EventType.PERFORMANCE, 
                          operation=operation,
                          duration_ms=duration)
    
    def log_player_action(self, action: str, **details) -> None:
        """Log a player action."""
        self.log_event(EventType.PLAYER_ACTION, action=action, **details)
        self.logger.info(f"Player action: {action}")
    
    def log_achievement(self, achievement_id: str, **details) -> None:
        """Log an achievement unlock."""
        self.log_event(EventType.ACHIEVEMENT, 
                      achievement_id=achievement_id,
                      **details)
        self.logger.info(f"Achievement unlocked: {achievement_id}")
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error with full context."""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.log_event(EventType.ERROR, **error_data)
        self.logger.error(f"Error occurred: {error}", exc_info=True)
    
    def get_session_replay(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get replay data for a session."""
        # This would typically query a database
        # For now, return current session if it matches
        if self.session and self.session.session_id == session_id:
            return asdict(self.session)
        return None
    
    def export_analytics(self) -> Dict[str, Any]:
        """Export analytics data."""
        return {
            'player_id': self.player_id,
            'game_id': self.game_id,
            'total_sessions': 1 if self.session else 0,
            'player_stats': dict(self.player_stats),
            'achievements': self.achievement_tracker,
            'performance_summary': {
                metric: self.performance_tracker.get_stats(metric)
                for metric in self.performance_tracker.metrics
            }
        }


class GameLogFormatter(logging.Formatter):
    """Custom formatter for game logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add game session info if available
        if hasattr(record, 'game_session'):
            record.game_session = record.game_session
        else:
            record.game_session = 'NO_SESSION'
        
        return super().format(record)


# Decorators for automatic logging
def log_game_action(action_name: str):
    """Decorator to automatically log game actions."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'logger') and isinstance(self.logger, ClaudeMaxLogger):
                self.logger.log_player_action(action_name, 
                                            function=func.__name__,
                                            args=str(args)[:100])
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def measure_game_performance(metric_type: MetricType):
    """Decorator to measure function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'logger') and isinstance(self.logger, ClaudeMaxLogger):
                with self.logger.measure_performance(func.__name__):
                    return func(self, *args, **kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# Async support
class AsyncClaudeMaxLogger(ClaudeMaxLogger):
    """Async version of ClaudeMaxLogger for async game engines."""
    
    async def log_event_async(self, event_type: EventType, **data) -> None:
        """Async version of log_event."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.log_event, event_type, **data
        )
    
    async def track_metric_async(self, metric_type: MetricType, value: float) -> None:
        """Async version of track_metric."""
        await asyncio.get_event_loop().run_in_executor(
            None, self.track_metric, metric_type, value
        )
    
    @contextmanager
    async def measure_performance_async(self, operation: str):
        """Async context manager for performance measurement."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration = (time.perf_counter() - start_time) * 1000
            await self.track_metric_async(MetricType.RENDER_TIME, duration)
            await self.log_event_async(EventType.PERFORMANCE,
                                     operation=operation,
                                     duration_ms=duration)