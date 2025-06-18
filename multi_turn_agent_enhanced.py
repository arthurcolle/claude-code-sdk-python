"""
Enhanced Multi-turn Function Calling Agent with Reflection, Task Queue, and Dynamic Environment
==============================================================================================
A highly dynamic agent with self-reflection, task queue management, worker pool execution,
and real-time environment monitoring capabilities.

New Features:
- Self-reflection system for performance analysis and improvement
- Priority-based task queue with dependency management
- Worker pool with 4 concurrent executors
- Dynamic environment monitoring and adaptation
- Real-time performance metrics and insights
- Adaptive behavior based on reflection insights
- Visualization for task queue and worker status
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sqlite3
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    Union,
)
import psutil
import platform

import duckdb
import openai
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, PrivateAttr
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import numpy as np
from sentence_transformers import SentenceTransformer

# Import base components from original
from multi_turn_agent import (
    Environment,
    ToolRegistry,
    make_message,
    R,
    C,
    U,
    A,
    S,
    Message,
    _py_to_json_type,
)

# ————————————————————————————————————————————————————————————————
# Logging & Configuration
# ————————————————————————————————————————————————————————————————
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ————————————————————————————————————————————————————————————————
# Task System
# ————————————————————————————————————————————————————————————————

class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class Task:
    """Represents a task to be executed by the agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None

    def __lt__(self, other):
        """Enable priority queue comparison."""
        return self.priority.value < other.priority.value

# ————————————————————————————————————————————————————————————————
# Reflection System
# ————————————————————————————————————————————————————————————————

@dataclass
class Reflection:
    """Agent's self-reflection on a task or interaction."""
    task_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    observation: str = ""
    analysis: str = ""
    improvements: List[str] = field(default_factory=list)
    confidence: float = 0.5
    impact: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReflectionSystem:
    """Manages agent's self-reflection and improvement."""
    
    def __init__(self, max_reflections: int = 100):
        self.reflections: deque[Reflection] = deque(maxlen=max_reflections)
        self.patterns: Dict[str, int] = defaultdict(int)
        self.improvements_applied: List[str] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    async def reflect_on_task(self, task: Task, context: Dict[str, Any]) -> Reflection:
        """Reflect on task execution."""
        duration = None
        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()
        
        observation = self._observe_task(task, duration)
        analysis = await self._analyze_performance(task, context)
        improvements = self._suggest_improvements(task, analysis)
        
        reflection = Reflection(
            task_id=task.id,
            observation=observation,
            analysis=analysis,
            improvements=improvements,
            confidence=self._calculate_confidence(task),
            impact=self._assess_impact(task),
            metadata={
                "duration": duration,
                "status": task.status.value,
                "error": task.error
            }
        )
        
        self.reflections.append(reflection)
        self._update_patterns(reflection)
        
        return reflection
    
    def _observe_task(self, task: Task, duration: Optional[float]) -> str:
        """Create observation about task execution."""
        obs_parts = [f"Task '{task.name}' (ID: {task.id[:8]})"]
        
        if task.status == TaskStatus.COMPLETED:
            obs_parts.append(f"completed successfully")
            if duration:
                obs_parts.append(f"in {duration:.2f}s")
        elif task.status == TaskStatus.FAILED:
            obs_parts.append(f"failed with error: {task.error}")
        
        return " ".join(obs_parts)
    
    async def _analyze_performance(self, task: Task, context: Dict[str, Any]) -> str:
        """Analyze task performance and context."""
        analysis_points = []
        
        # Analyze execution time
        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()
            avg_duration = np.mean(self.performance_metrics.get("task_duration", [1.0]))
            if duration > avg_duration * 1.5:
                analysis_points.append(f"Task took {duration/avg_duration:.1f}x longer than average")
        
        # Analyze failure patterns
        if task.status == TaskStatus.FAILED and task.error:
            if "timeout" in task.error.lower():
                analysis_points.append("Task failed due to timeout - may need optimization")
            elif "permission" in task.error.lower():
                analysis_points.append("Task failed due to permissions - need better access control")
        
        # Analyze dependencies
        if task.dependencies:
            analysis_points.append(f"Task had {len(task.dependencies)} dependencies")
        
        return "; ".join(analysis_points) if analysis_points else "Task executed within normal parameters"
    
    def _suggest_improvements(self, task: Task, analysis: str) -> List[str]:
        """Suggest improvements based on task execution."""
        improvements = []
        
        if "timeout" in analysis:
            improvements.append("Consider breaking down long-running tasks into smaller subtasks")
            improvements.append("Implement progressive timeout strategies")
        
        if "permission" in analysis:
            improvements.append("Pre-check permissions before task execution")
            improvements.append("Implement fallback strategies for permission-denied scenarios")
        
        if task.status == TaskStatus.FAILED:
            improvements.append(f"Add retry logic for tasks that fail with: {task.error}")
        
        return improvements
    
    def _calculate_confidence(self, task: Task) -> float:
        """Calculate confidence score for the task execution."""
        if task.status == TaskStatus.COMPLETED:
            return 0.9
        elif task.status == TaskStatus.FAILED:
            return 0.3
        else:
            return 0.5
    
    def _assess_impact(self, task: Task) -> str:
        """Assess the impact of the task."""
        if task.priority == TaskPriority.CRITICAL:
            return "High impact - critical task"
        elif task.status == TaskStatus.FAILED:
            return "Negative impact - task failure"
        else:
            return "Normal impact"
    
    def _update_patterns(self, reflection: Reflection):
        """Update pattern recognition from reflections."""
        # Track error patterns
        if reflection.metadata.get("error"):
            error_type = reflection.metadata["error"].split(":")[0]
            self.patterns[f"error_{error_type}"] += 1
        
        # Track performance patterns
        if reflection.metadata.get("duration"):
            self.performance_metrics["task_duration"].append(
                reflection.metadata["duration"]
            )
    
    def get_insights(self) -> Dict[str, Any]:
        """Get insights from accumulated reflections."""
        recent_reflections = list(self.reflections)[-10:]
        
        return {
            "total_reflections": len(self.reflections),
            "common_errors": dict(sorted(
                ((k, v) for k, v in self.patterns.items() if k.startswith("error_")),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            "avg_confidence": np.mean([r.confidence for r in recent_reflections]) if recent_reflections else 0.5,
            "recent_improvements": list(set(
                imp for r in recent_reflections for imp in r.improvements
            ))[:5],
            "performance_trends": {
                "avg_task_duration": np.mean(self.performance_metrics["task_duration"][-20:])
                if self.performance_metrics["task_duration"] else 0
            }
        }

# ————————————————————————————————————————————————————————————————
# Environment Monitor
# ————————————————————————————————————————————————————————————————

class EnvironmentMonitor:
    """Monitors and tracks dynamic environment changes."""
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.change_history: deque[Dict[str, Any]] = deque(maxlen=1000)
        self.subscribers: List[Callable] = []
        self._last_check = datetime.now()
    
    async def update(self):
        """Update environment state."""
        current_state = await self._gather_environment_data()
        changes = self._detect_changes(current_state)
        
        if changes:
            self.change_history.append({
                "timestamp": datetime.now(),
                "changes": changes
            })
            await self._notify_subscribers(changes)
        
        self.state = current_state
        self._last_check = datetime.now()
    
    async def _gather_environment_data(self) -> Dict[str, Any]:
        """Gather current environment data."""
        return {
            "system": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "free": psutil.disk_usage("/").free,
                    "percent": psutil.disk_usage("/").percent
                }
            },
            "process": {
                "pid": os.getpid(),
                "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.Process().cpu_percent(interval=0.1)
            },
            "time": {
                "current": datetime.now().isoformat(),
                "timezone": datetime.now().astimezone().tzname()
            },
            "network": {
                "interfaces": [
                    {"name": iface, "up": stats.isup}
                    for iface, stats in psutil.net_if_stats().items()
                ]
            }
        }
    
    def _detect_changes(self, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes between states."""
        if not self.state:
            return {"initial_state": True}
        
        changes = {}
        
        # Check memory changes
        old_mem = self.state.get("system", {}).get("memory", {}).get("percent", 0)
        new_mem = new_state.get("system", {}).get("memory", {}).get("percent", 0)
        if abs(new_mem - old_mem) > 10:
            changes["memory_change"] = {
                "old": old_mem,
                "new": new_mem,
                "delta": new_mem - old_mem
            }
        
        # Check disk changes
        old_disk = self.state.get("system", {}).get("disk", {}).get("percent", 0)
        new_disk = new_state.get("system", {}).get("disk", {}).get("percent", 0)
        if abs(new_disk - old_disk) > 5:
            changes["disk_change"] = {
                "old": old_disk,
                "new": new_disk,
                "delta": new_disk - old_disk
            }
        
        return changes
    
    async def _notify_subscribers(self, changes: Dict[str, Any]):
        """Notify subscribers of environment changes."""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(changes)
                else:
                    subscriber(changes)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback: Callable):
        """Subscribe to environment changes."""
        self.subscribers.append(callback)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get environment summary."""
        return {
            "current_state": self.state,
            "last_check": self._last_check.isoformat(),
            "recent_changes": list(self.change_history)[-5:],
            "alerts": self._generate_alerts()
        }
    
    def _generate_alerts(self) -> List[str]:
        """Generate alerts based on environment state."""
        alerts = []
        
        if self.state.get("system", {}).get("memory", {}).get("percent", 0) > 90:
            alerts.append("High memory usage detected (>90%)")
        
        if self.state.get("system", {}).get("disk", {}).get("percent", 0) > 85:
            alerts.append("Low disk space warning (<15% free)")
        
        if self.state.get("process", {}).get("cpu_percent", 0) > 80:
            alerts.append("High CPU usage by agent process")
        
        return alerts

# ————————————————————————————————————————————————————————————————
# Task Queue and Worker Pool
# ————————————————————————————————————————————————————————————————

class TaskQueue:
    """Priority queue for task management."""
    
    def __init__(self):
        self.queue: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: deque[Task] = deque(maxlen=100)
        self._lock = asyncio.Lock()
    
    async def add_task(self, task: Task) -> str:
        """Add task to queue."""
        async with self._lock:
            # Check dependencies
            for dep_id in task.dependencies:
                if dep_id not in self.tasks or self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.PENDING
                    self.tasks[task.id] = task
                    return task.id
            
            # Queue task if dependencies are met
            task.status = TaskStatus.QUEUED
            await self.queue.put(task)
            self.tasks[task.id] = task
            
        return task.id
    
    async def get_task(self) -> Optional[Task]:
        """Get next task from queue."""
        try:
            task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            return task
        except asyncio.TimeoutError:
            return None
    
    async def complete_task(self, task_id: str, result: Any = None, error: Optional[str] = None):
        """Mark task as completed."""
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.COMPLETED if error is None else TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.result = result
                task.error = error
                
                self.completed_tasks.append(task)
                
                # Check and queue dependent tasks
                for tid, t in self.tasks.items():
                    if task_id in t.dependencies and t.status == TaskStatus.PENDING:
                        if all(
                            self.tasks[dep].status == TaskStatus.COMPLETED
                            for dep in t.dependencies
                        ):
                            t.status = TaskStatus.QUEUED
                            await self.queue.put(t)
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status."""
        status_counts = defaultdict(int)
        for task in self.tasks.values():
            status_counts[task.status.value] += 1
        
        return {
            "total_tasks": len(self.tasks),
            "queued": self.queue.qsize(),
            "by_status": dict(status_counts),
            "completed_recently": len(self.completed_tasks)
        }

class WorkerPool:
    """Manages a pool of workers for task execution."""
    
    def __init__(self, num_workers: int = 4, task_executor: Optional[Callable] = None):
        self.num_workers = num_workers
        self.workers: List[asyncio.Task] = []
        self.task_executor = task_executor or self._default_executor
        self.active_tasks: Dict[str, str] = {}  # worker_id -> task_id
        self._running = False
        self._stats = defaultdict(int)
    
    async def start(self, task_queue: TaskQueue):
        """Start worker pool."""
        self._running = True
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}", task_queue))
            self.workers.append(worker)
        logger.info(f"Started {self.num_workers} workers")
    
    async def stop(self):
        """Stop worker pool."""
        self._running = False
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Worker pool stopped")
    
    async def _worker(self, worker_id: str, task_queue: TaskQueue):
        """Worker coroutine."""
        logger.info(f"{worker_id} started")
        
        while self._running:
            try:
                task = await task_queue.get_task()
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                logger.info(f"{worker_id} executing task {task.id[:8]}: {task.name}")
                
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                self.active_tasks[worker_id] = task.id
                self._stats["tasks_started"] += 1
                
                try:
                    result = await self.task_executor(task)
                    await task_queue.complete_task(task.id, result=result)
                    self._stats["tasks_completed"] += 1
                    logger.info(f"{worker_id} completed task {task.id[:8]}")
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    await task_queue.complete_task(task.id, error=error_msg)
                    self._stats["tasks_failed"] += 1
                    logger.error(f"{worker_id} task {task.id[:8]} failed: {error_msg}")
                finally:
                    if worker_id in self.active_tasks:
                        del self.active_tasks[worker_id]
                
            except Exception as e:
                logger.error(f"{worker_id} error: {e}")
                self._stats["worker_errors"] += 1
                await asyncio.sleep(1)
        
        logger.info(f"{worker_id} stopped")
    
    async def _default_executor(self, task: Task) -> Any:
        """Default task executor."""
        # Simulate task execution
        await asyncio.sleep(0.5)
        return f"Executed task: {task.name}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker pool status."""
        return {
            "num_workers": self.num_workers,
            "active_workers": len(self.active_tasks),
            "active_tasks": dict(self.active_tasks),
            "stats": dict(self._stats)
        }

# ————————————————————————————————————————————————————————————————
# Direction & State Management
# ————————————————————————————————————————————————————————————————
class ConversationDirection(Enum):
    """Represents the current direction/mode of conversation."""

    EXPLORING = "exploring"
    FOCUSED = "focused"
    DEBUGGING = "debugging"
    CREATING = "creating"
    REVIEWING = "reviewing"
    PIVOTING = "pivoting"


class AgentState(BaseModel):
    """Persistent agent state."""

    session_id: str
    direction: ConversationDirection = ConversationDirection.EXPLORING
    context_summary: str = ""
    active_goals: list[str] = Field(default_factory=list)
    completed_goals: list[str] = Field(default_factory=list)
    key_decisions: list[dict] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {"use_enum_values": True, "json_encoders": {datetime: lambda v: v.isoformat()}}


# ————————————————————————————————————————————————————————————————
# Persistent Storage Backend
# ————————————————————————————————————————————————————————————————
class PersistentStorage:
    """
    Handles persistent storage using SQLite for metadata and DuckDB for analytics.
    """

    def __init__(self, db_path: str = "agent_state.db", use_duckdb: bool = True):
        self.db_path = Path(db_path)
        self.use_duckdb = use_duckdb
        self._init_sqlite()
        if use_duckdb:
            self._init_duckdb()

        # Initialize embedding model for retrieval
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _init_sqlite(self):
        """Initialize SQLite database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                direction TEXT,
                context_summary TEXT,
                active_goals TEXT,
                completed_goals TEXT,
                key_decisions TEXT,
                metadata TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                tool_calls TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding BLOB,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                checkpoint_name TEXT,
                state_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
        """)
        self.conn.commit()

    def _init_duckdb(self):
        """Initialize DuckDB for analytics queries."""
        self.duck_conn = duckdb.connect(f"{self.db_path}.duckdb")

        # Create analytics tables
        self.duck_conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_analytics (
                session_id VARCHAR,
                turn_count INTEGER,
                tool_usage_count INTEGER,
                avg_response_length DOUBLE,
                direction_changes INTEGER,
                timestamp TIMESTAMP
            )
        """)

    async def save_state(self, state: AgentState):
        """Save agent state to database."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO sessions 
            (session_id, direction, context_summary, active_goals, 
             completed_goals, key_decisions, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                state.session_id,
                state.direction if isinstance(state.direction, str) else state.direction.value,
                state.context_summary,
                json.dumps(state.active_goals),
                json.dumps(state.completed_goals),
                json.dumps(state.key_decisions),
                json.dumps(state.metadata),
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
            ),
        )
        self.conn.commit()

    async def load_state(self, session_id: str) -> AgentState | None:
        """Load agent state from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return AgentState(
            session_id=row["session_id"],
            direction=ConversationDirection(row["direction"]),
            context_summary=row["context_summary"],
            active_goals=json.loads(row["active_goals"]),
            completed_goals=json.loads(row["completed_goals"]),
            key_decisions=json.loads(row["key_decisions"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def save_message(self, session_id: str, message: Message):
        """Save message with embedding for retrieval."""
        content = str(message.get(C, ""))

        # Generate embedding
        embedding = None
        if content:
            embedding = self.embedder.encode(content).tobytes()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (session_id, role, content, tool_calls, embedding)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                session_id,
                message.get(R),
                content,
                json.dumps(message.get("tool_calls", [])),
                embedding,
            ),
        )
        self.conn.commit()

    async def search_messages(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        role_filter: str | None = None,
    ) -> list[Message]:
        """Search messages using semantic similarity."""
        # Generate query embedding
        query_embedding = self.embedder.encode(query)

        cursor = self.conn.cursor()

        # Get all messages with embeddings
        where_clause = "WHERE session_id = ? AND embedding IS NOT NULL"
        params = [session_id]

        if role_filter:
            where_clause += " AND role = ?"
            params.append(role_filter)

        cursor.execute(
            f"""
            SELECT role, content, tool_calls, embedding
            FROM messages
            {where_clause}
        """,
            params,
        )

        # Calculate similarities
        results = []
        for row in cursor.fetchall():
            if row["embedding"]:
                msg_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                similarity = np.dot(query_embedding, msg_embedding)
                results.append(
                    (
                        similarity,
                        {
                            R: row["role"],
                            C: row["content"],
                            "tool_calls": json.loads(row["tool_calls"])
                            if row["tool_calls"]
                            else None,
                        },
                    )
                )

        # Sort by similarity and return top k
        results.sort(key=lambda x: x[0], reverse=True)
        return [msg for _, msg in results[:top_k]]

    async def create_checkpoint(self, session_id: str, name: str, state_data: dict):
        """Create a checkpoint for recovery."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO checkpoints (session_id, checkpoint_name, state_data)
            VALUES (?, ?, ?)
        """,
            (session_id, name, json.dumps(state_data)),
        )
        self.conn.commit()

    async def load_checkpoint(self, session_id: str, name: str) -> dict | None:
        """Load a checkpoint."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT state_data FROM checkpoints 
            WHERE session_id = ? AND checkpoint_name = ?
            ORDER BY timestamp DESC LIMIT 1
        """,
            (session_id, name),
        )

        row = cursor.fetchone()
        return json.loads(row["state_data"]) if row else None

    async def update_analytics(self, session_id: str, metrics: dict):
        """Update conversation analytics in DuckDB."""
        if not self.use_duckdb:
            return

        self.duck_conn.execute(
            """
            INSERT INTO conversation_analytics 
            (session_id, turn_count, tool_usage_count, avg_response_length, 
             direction_changes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                metrics.get("turn_count", 0),
                metrics.get("tool_usage_count", 0),
                metrics.get("avg_response_length", 0.0),
                metrics.get("direction_changes", 0),
                datetime.now(),
            ),
        )


# ————————————————————————————————————————————————————————————————
# Enhanced Memory with Persistence
# ————————————————————————————————————————————————————————————————
class PersistentConversationMemory:
    """Memory with persistent storage and advanced retrieval."""

    def __init__(
        self,
        session_id: str,
        storage: PersistentStorage,
        max_tokens: int,
        threshold_words: int,
    ):
        self.session_id = session_id
        self.storage = storage
        self.history: list[Message] = []
        self.max_tokens = max_tokens
        self.threshold_words = threshold_words
        self._client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Load existing messages if any
        asyncio.create_task(self._load_history())

    async def _load_history(self):
        """Load conversation history from storage."""
        cursor = self.storage.conn.cursor()
        cursor.execute(
            """
            SELECT role, content, tool_calls 
            FROM messages 
            WHERE session_id = ?
            ORDER BY timestamp
        """,
            (self.session_id,),
        )

        for row in cursor.fetchall():
            msg = {R: row["role"], C: row["content"]}
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            self.history.append(msg)

    async def append(self, role: str, content: Any):
        """Add message and save to storage."""
        msg = make_message(role, content)
        self.history.append(msg)

        # Save to persistent storage
        await self.storage.save_message(self.session_id, msg)

        # Check if summarization needed
        if self._word_count() > self.threshold_words:
            await self._summarize()

    def _word_count(self) -> int:
        """Count total words in history."""
        return sum(len(str(m[C]).split()) for m in self.history)

    async def _summarize(self):
        """Summarize conversation to reduce context size."""
        logger.info(f"Context exceeded {self.threshold_words} words, summarizing...")

        # Preserve system message if exists
        system_msgs = [m for m in self.history if m[R] == S]
        other_msgs = [m for m in self.history if m[R] != S]

        prompt = "\n".join(f"[{m[R]}] {m[C]}" for m in other_msgs)

        summary_resp = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                make_message(S, "Summarize the following conversation concisely:"),
                make_message(U, prompt),
            ],
            temperature=0,
            stream=False,
        )

        summary_text = summary_resp.choices[0].message.content

        # Create checkpoint before summarization
        await self.storage.create_checkpoint(
            self.session_id,
            f"pre_summary_{datetime.now().isoformat()}",
            {"history": self.history},
        )

        self.history = system_msgs + [
            make_message(S, f"Previous conversation summary: {summary_text}")
        ]

    async def search(self, query: str, top_k: int = 3) -> list[Message]:
        """Search conversation history using persistent storage."""
        return await self.storage.search_messages(self.session_id, query, top_k)

    async def get_relevant_context(
        self, query: str, max_messages: int = 5
    ) -> list[Message]:
        """Get relevant historical context for a query."""
        return await self.storage.search_messages(self.session_id, query, max_messages)


# ————————————————————————————————————————————————————————————————
# Enhanced Multi-turn Agent with Reflection and Task Management
# ————————————————————————————————————————————————————————————————
class EnhancedStatefulAgent:
    """
    Enhanced multi-turn agent with reflection, task queue, worker pool, and environment monitoring.
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        system_prompt: str | None = None,
        tools_registry: ToolRegistry | None = None,
        stream: bool = True,
        storage_path: str = "agent_state.db",
        enable_pivot: bool = False,
        num_workers: int = 4,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.storage = PersistentStorage(storage_path)
        self.tools_registry = tools_registry or ToolRegistry(Environment())
        self.stream = stream
        self.enable_pivot = enable_pivot

        # Initialize state
        self.state = AgentState(session_id=self.session_id)
        # Ensure direction is string for consistency
        if isinstance(self.state.direction, ConversationDirection):
            self.state.direction = self.state.direction.value

        # Initialize core components
        self.reflection_system = ReflectionSystem()
        self.environment_monitor = EnvironmentMonitor()
        self.task_queue = TaskQueue()
        self.worker_pool = WorkerPool(num_workers=num_workers, task_executor=self._execute_task)
        
        # Initialize memory with persistence
        self.memory = PersistentConversationMemory(
            session_id=self.session_id,
            storage=self.storage,
            max_tokens=200_000,
            threshold_words=3_000,
        )

        # Metrics tracking
        self.metrics = {
            "turn_count": 0,
            "tool_usage_count": 0,
            "total_response_length": 0,
            "direction_changes": 0,
            "tasks_created": 0,
            "reflections_generated": 0,
        }
        
        # Register enhanced tools
        self._register_enhanced_tools()

        # Enhanced system prompt
        enhanced_prompt = system_prompt or """You are an enhanced AI assistant with advanced capabilities:
1. Self-reflection: You analyze your own performance and learn from interactions
2. Task management: You can create and manage tasks with priorities and dependencies
3. Concurrent execution: You have 4 workers that can execute tasks in parallel
4. Environment awareness: You monitor system resources and adapt to changes
5. Performance optimization: You use insights from reflections to improve over time

Use these capabilities to provide more efficient and thoughtful assistance."""

        if enhanced_prompt:
            asyncio.create_task(self.memory.append(S, enhanced_prompt))

        # Start background services
        asyncio.create_task(self._start_services())
        
        # Load existing state if resuming
        asyncio.create_task(self._load_state())

    async def _load_state(self):
        """Load existing state if available."""
        existing_state = await self.storage.load_state(self.session_id)
        if existing_state:
            self.state = existing_state
            dir_val = self.state.direction if isinstance(self.state.direction, str) else self.state.direction.value
            logger.info(
                f"Resumed session {self.session_id} in {dir_val} mode"
            )
    
    async def _start_services(self):
        """Start background services."""
        # Start worker pool
        await self.worker_pool.start(self.task_queue)
        
        # Start environment monitoring
        self.environment_monitor.subscribe(self._on_environment_change)
        asyncio.create_task(self._monitor_environment())
        
        logger.info(f"Enhanced services started for session {self.session_id[:8]}")
    
    async def _monitor_environment(self):
        """Monitor environment periodically."""
        while True:
            try:
                await self.environment_monitor.update()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Environment monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _on_environment_change(self, changes: Dict[str, Any]):
        """Handle environment changes."""
        if "memory_change" in changes and changes["memory_change"]["delta"] > 20:
            logger.warning(f"High memory increase detected: {changes['memory_change']['delta']:.1f}%")
            # Create a task to optimize memory if needed
            task = Task(
                name="Memory optimization",
                description=f"Optimize memory usage due to {changes['memory_change']['delta']:.1f}% increase",
                priority=TaskPriority.HIGH
            )
            await self.task_queue.add_task(task)
        
        if "disk_change" in changes and changes["disk_change"]["new"] > 90:
            logger.error(f"Critical disk usage: {changes['disk_change']['new']:.1f}%")
            # Create urgent cleanup task
            task = Task(
                name="Disk cleanup",
                description=f"Clean up disk space - usage at {changes['disk_change']['new']:.1f}%",
                priority=TaskPriority.CRITICAL
            )
            await self.task_queue.add_task(task)
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a task with reflection."""
        context = {
            "environment": self.environment_monitor.state,
            "queue_status": self.task_queue.get_status(),
            "session_id": self.session_id,
            "direction": self.state.direction
        }
        
        try:
            # Execute task logic
            if task.callback:
                if asyncio.iscoroutinefunction(task.callback):
                    result = await task.callback(task)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, task.callback, task
                    )
            else:
                # Default execution based on task name
                if "memory" in task.name.lower():
                    # Simulate memory optimization
                    await asyncio.sleep(2)
                    result = "Memory optimization completed"
                elif "disk" in task.name.lower():
                    # Simulate disk cleanup
                    await asyncio.sleep(3)
                    result = "Disk cleanup completed"
                else:
                    # Generic task execution
                    await asyncio.sleep(1)
                    result = f"Completed: {task.name}"
            
            # Reflect on successful execution
            reflection = await self.reflection_system.reflect_on_task(task, context)
            self.metrics["reflections_generated"] += 1
            
            # Apply improvements if confidence is high
            if reflection.confidence > 0.8 and reflection.improvements:
                for improvement in reflection.improvements[:1]:  # Apply first improvement
                    if improvement not in self.reflection_system.improvements_applied:
                        logger.info(f"Applying improvement: {improvement}")
                        self.reflection_system.improvements_applied.append(improvement)
                        # Could trigger additional tasks based on improvement
            
            return result
            
        except Exception as e:
            # Reflect on failure
            task.error = str(e)
            reflection = await self.reflection_system.reflect_on_task(task, context)
            self.metrics["reflections_generated"] += 1
            
            # Learn from failure
            if "retry" in reflection.improvements[0].lower() if reflection.improvements else False:
                # Create retry task with lower priority
                retry_task = Task(
                    name=f"Retry: {task.name}",
                    description=f"Retry of failed task: {task.description}",
                    priority=TaskPriority(min(task.priority.value + 1, TaskPriority.LOW.value)),
                    metadata={"original_task_id": task.id, "retry_count": task.metadata.get("retry_count", 0) + 1}
                )
                if retry_task.metadata["retry_count"] < 3:  # Max 3 retries
                    await self.task_queue.add_task(retry_task)
            
            raise
    
    def _register_enhanced_tools(self):
        """Register enhanced tools for the agent."""
        
        @self.tools_registry.register(description="Create a new task")
        async def create_task(
            name: str,
            description: str = "",
            priority: str = "medium",
            dependencies: List[str] = []
        ) -> str:
            """Create a new task in the task queue."""
            task = Task(
                name=name,
                description=description,
                priority=TaskPriority[priority.upper()],
                dependencies=dependencies
            )
            task_id = await self.task_queue.add_task(task)
            self.metrics["tasks_created"] += 1
            return f"Created task {task_id[:8]}: {name} (priority: {priority})"
        
        @self.tools_registry.register(description="Get task status")
        async def get_task_status(task_id: str = "") -> Dict[str, Any]:
            """Get status of a specific task or all tasks."""
            if task_id:
                task = self.task_queue.tasks.get(task_id)
                if task:
                    return {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status.value,
                        "priority": task.priority.name,
                        "created": task.created_at.isoformat(),
                        "error": task.error,
                        "result": str(task.result) if task.result else None
                    }
                return {"error": "Task not found"}
            return self.task_queue.get_status()
        
        @self.tools_registry.register(description="Get reflection insights")
        async def get_insights() -> Dict[str, Any]:
            """Get agent's self-reflection insights."""
            insights = self.reflection_system.get_insights()
            insights["total_tasks_reflected"] = self.metrics["reflections_generated"]
            return insights
        
        @self.tools_registry.register(description="Get environment status")
        async def get_environment() -> Dict[str, Any]:
            """Get current environment state and alerts."""
            return self.environment_monitor.get_summary()
        
        @self.tools_registry.register(description="Get worker pool status")
        async def get_worker_status() -> Dict[str, Any]:
            """Get worker pool status."""
            return self.worker_pool.get_status()
        
        @self.tools_registry.register(description="Search conversation history")
        async def search_history(query: str, max_results: int = 5) -> List[str]:
            """Search through conversation history using semantic search."""
            messages = await self.memory.search(query, max_results)
            return [f"[{msg[R]}] {msg[C][:100]}..." for msg in messages]
        
        @self.tools_registry.register(description="Get comprehensive agent status")
        async def get_agent_status() -> Dict[str, Any]:
            """Get comprehensive status of the agent including all subsystems."""
            return {
                "session": await self.get_session_summary(),
                "task_queue": self.task_queue.get_status(),
                "worker_pool": self.worker_pool.get_status(),
                "reflections": self.reflection_system.get_insights(),
                "environment": self.environment_monitor.get_summary()
            }

    async def change_direction(
        self, new_direction: ConversationDirection | str, reason: str = ""
    ):
        """Change conversation direction with tracking."""
        old_direction = self.state.direction
        # Handle both enum and string inputs
        new_dir_value = new_direction.value if isinstance(new_direction, ConversationDirection) else new_direction
        self.state.direction = new_dir_value
        
        old_dir_value = old_direction if isinstance(old_direction, str) else old_direction.value
        
        self.state.key_decisions.append(
            {
                "type": "direction_change",
                "from": old_dir_value,
                "to": new_dir_value,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.metrics["direction_changes"] += 1

        # Add system message about direction change
        await self.memory.append(
            S,
            f"[Direction changed from {old_dir_value} to {new_dir_value}. Reason: {reason}]",
        )

        # Save state
        await self.storage.save_state(self.state)

        logger.info(
            f"Direction changed: {old_dir_value} -> {new_dir_value}"
        )

    async def send_user(
        self,
        content: str,
        *,
        auto_execute_tools: bool = True,
        max_tool_rounds: int = 10,
        use_retrieval: bool = True,
        pivot_on_request: bool | None = None,
        **chat_kwargs: Any,
    ) -> str:
        """
        Send user message with enhanced state management.

        Args:
            content: User message
            auto_execute_tools: Whether to auto-execute tool calls
            max_tool_rounds: Maximum tool execution rounds
            use_retrieval: Whether to use semantic retrieval
            pivot_on_request: Override enable_pivot setting
            **chat_kwargs: Additional chat parameters

        Returns:
            Final assistant response
        """
        # Update metrics
        self.metrics["turn_count"] += 1

        # Check for pivot request
        if pivot_on_request if pivot_on_request is not None else self.enable_pivot:
            if any(
                keyword in content.lower()
                for keyword in ["pivot", "change direction", "switch focus"]
            ):
                await self.change_direction(
                    "pivoting",
                    f"User requested pivot: {content[:50]}...",
                )

        # Get relevant context if using retrieval
        relevant_context = []
        if use_retrieval and self.metrics["turn_count"] > 1:
            relevant_context = await self.memory.get_relevant_context(
                content, max_messages=3
            )
            if relevant_context:
                # Inject relevant context as a system message
                context_summary = "\n".join(
                    [f"[{msg[R]}]: {msg[C][:100]}..." for msg in relevant_context]
                )
                await self.memory.append(
                    S, f"[Relevant context from earlier:\n{context_summary}]"
                )

        # Add user message
        await self.memory.append(U, content)

        # Prepare messages for API call
        messages = self.memory.history.copy()

        # Add state context if direction is not exploring
        if self.state.direction != "exploring":
            state_context = f"[Current mode: {self.state.direction}. "
            if self.state.active_goals:
                state_context += (
                    f"Active goals: {', '.join(self.state.active_goals[:3])}. "
                )
            state_context += "]"
            messages.insert(0, make_message(S, state_context))

        # Get tools
        user_tools = chat_kwargs.pop("tools_param", None) or []
        tools_combined = (
            [*user_tools, *self.tools_registry.schemas]
            if (user_tools or self.tools_registry.schemas)
            else None
        )

        rounds = 0
        final_response = ""

        while rounds < max_tool_rounds:
            rounds += 1
            assistant_content = ""
            tool_calls = []

            # Make API call
            from multi_turn_agent import chat

            async for msg in chat(
                messages,
                tools_param=tools_combined,
                tool_choice="auto" if tools_combined and rounds == 1 else None,
                stream=self.stream,
                **chat_kwargs,
            ):
                if "token" in msg:  # Streaming token
                    if self.stream:
                        print(msg["token"], end="", flush=True)
                else:  # Complete message
                    if C in msg:
                        assistant_content = msg[C]
                    if "tool_calls" in msg:
                        tool_calls = msg["tool_calls"]

            # Update metrics
            if assistant_content:
                self.metrics["total_response_length"] += len(assistant_content)
                final_response = assistant_content

            # Add assistant message
            assistant_msg = {R: A}
            if assistant_content:
                assistant_msg[C] = assistant_content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
                self.metrics["tool_usage_count"] += len(tool_calls)

            await self.memory.append(A, assistant_msg)

            # Execute tools if needed
            if tool_calls and auto_execute_tools:
                if self.stream:
                    print()  # New line after streaming

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])

                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                    try:
                        result = await self.tools_registry.call(tool_name, **tool_args)
                        result_str = str(result)
                    except Exception as e:
                        result_str = f"Error: {str(e)}"
                        logger.error(f"Tool execution failed: {e}")

                    # Add tool result
                    tool_msg = {
                        R: "tool",
                        "tool_call_id": tool_call["id"],
                        C: result_str,
                    }
                    await self.memory.append("tool", tool_msg)

                # Continue conversation after tools
                messages = self.memory.history.copy()
                continue
            else:
                # No tool calls, we're done
                break

        # Update state and save
        self.state.updated_at = datetime.now()
        await self.storage.save_state(self.state)

        # Update analytics
        await self.storage.update_analytics(
            self.session_id,
            {
                **self.metrics,
                "avg_response_length": self.metrics["total_response_length"]
                / self.metrics["turn_count"],
            },
        )

        return final_response

    async def add_goal(self, goal: str):
        """Add an active goal."""
        self.state.active_goals.append(goal)
        await self.storage.save_state(self.state)

    async def complete_goal(self, goal: str):
        """Mark a goal as completed."""
        if goal in self.state.active_goals:
            self.state.active_goals.remove(goal)
            self.state.completed_goals.append(goal)
            await self.storage.save_state(self.state)

    async def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        return {
            "session_id": self.session_id,
            "direction": self.state.direction if isinstance(self.state.direction, str) else self.state.direction.value,
            "turn_count": self.metrics["turn_count"],
            "tool_usage_count": self.metrics["tool_usage_count"],
            "tasks_created": self.metrics["tasks_created"],
            "reflections_generated": self.metrics["reflections_generated"],
            "active_goals": self.state.active_goals,
            "completed_goals": self.state.completed_goals,
            "direction_changes": self.metrics["direction_changes"],
            "key_decisions": self.state.key_decisions[-5:],  # Last 5 decisions
            "created_at": self.state.created_at.isoformat(),
            "updated_at": self.state.updated_at.isoformat(),
        }


# ————————————————————————————————————————————————————————————————
# FastAPI State Management Server
# ————————————————————————————————————————————————————————————————
app = FastAPI(title="Multi-turn Agent State API")

# Global storage instance
global_storage = PersistentStorage()


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session state."""
    state = await global_storage.load_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.model_dump(mode='json')


@app.get("/sessions/{session_id}/messages")
async def search_messages(
    session_id: str,
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    role: str | None = Query(None, description="Filter by role"),
):
    """Search session messages."""
    messages = await global_storage.search_messages(session_id, query, top_k, role)
    return {"query": query, "results": messages}


@app.post("/sessions/{session_id}/checkpoint")
async def create_checkpoint(
    session_id: str, name: str = Query(..., description="Checkpoint name")
):
    """Create a checkpoint."""
    # Load current state
    state = await global_storage.load_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    # Create checkpoint
    await global_storage.create_checkpoint(
        session_id,
        name,
        {"state": state.model_dump(mode='json'), "timestamp": datetime.now().isoformat()},
    )

    return {"message": f"Checkpoint '{name}' created"}


@app.get("/sessions/{session_id}/analytics")
async def get_analytics(session_id: str):
    """Get session analytics from DuckDB."""
    if not global_storage.use_duckdb:
        raise HTTPException(status_code=501, detail="DuckDB not enabled")

    result = global_storage.duck_conn.execute(
        """
        SELECT * FROM conversation_analytics 
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT 10
    """,
        (session_id,),
    ).fetchall()

    return {"session_id": session_id, "analytics": result}


# ————————————————————————————————————————————————————————————————
# Visualization
# ————————————————————————————————————————————————————————————————

def visualize_agent_status(agent: EnhancedStatefulAgent):
    """Create a comprehensive visualization of agent status."""
    print("\n" + "="*80)
    print(f"🤖 ENHANCED AGENT STATUS - Session: {agent.session_id[:8]}")
    print("="*80)
    
    # Task Queue Status
    queue_status = agent.task_queue.get_status()
    print(f"\n📋 TASK QUEUE:")
    print(f"   Total Tasks: {queue_status['total_tasks']} | Queued: {queue_status['queued']}")
    print(f"   By Status: ", end="")
    for status, count in queue_status["by_status"].items():
        if count > 0:
            emoji = {"completed": "✅", "failed": "❌", "running": "🔄", "queued": "⏳", "pending": "📍"}.get(status, "❓")
            print(f"{emoji} {status}:{count} ", end="")
    print()
    
    # Worker Pool Status
    worker_status = agent.worker_pool.get_status()
    print(f"\n👷 WORKER POOL (4 workers):")
    print(f"   Active Workers: {worker_status['active_workers']}/{worker_status['num_workers']}")
    if worker_status["active_tasks"]:
        print("   Active Tasks:")
        for worker, task_id in worker_status["active_tasks"].items():
            task = agent.task_queue.tasks.get(task_id)
            if task:
                print(f"     - {worker}: {task.name[:30]}... ({task.priority.name})")
    stats = worker_status["stats"]
    print(f"   Statistics: Started={stats.get('tasks_started', 0)} | "
          f"Completed={stats.get('tasks_completed', 0)} | "
          f"Failed={stats.get('tasks_failed', 0)}")
    
    # Recent Tasks
    if agent.task_queue.completed_tasks:
        print(f"\n📝 RECENT COMPLETED TASKS:")
        for task in list(agent.task_queue.completed_tasks)[-5:]:
            status_icon = "✅" if task.status == TaskStatus.COMPLETED else "❌"
            duration = ""
            if task.started_at and task.completed_at:
                duration = f" ({(task.completed_at - task.started_at).total_seconds():.1f}s)"
            print(f"   {status_icon} {task.name[:40]}...{duration}")
    
    # Reflection Insights
    insights = agent.reflection_system.get_insights()
    print(f"\n🤔 REFLECTION INSIGHTS:")
    print(f"   Total Reflections: {insights['total_reflections']}")
    print(f"   Average Confidence: {insights['avg_confidence']:.2f}")
    if insights["common_errors"]:
        print("   Common Errors:")
        for error_type, count in list(insights["common_errors"].items())[:3]:
            print(f"     - {error_type}: {count} occurrences")
    if insights["recent_improvements"]:
        print("   Recent Improvements:")
        for imp in insights["recent_improvements"][:3]:
            print(f"     - {imp[:60]}...")
    if insights["performance_trends"]["avg_task_duration"] > 0:
        print(f"   Avg Task Duration: {insights['performance_trends']['avg_task_duration']:.2f}s")
    
    # Environment Status
    env_summary = agent.environment_monitor.get_summary()
    if env_summary["current_state"]:
        system = env_summary["current_state"].get("system", {})
        process = env_summary["current_state"].get("process", {})
        print(f"\n🖥️  ENVIRONMENT:")
        print(f"   System: {system.get('platform', 'Unknown')} | "
              f"CPU: {system.get('cpu_count', 0)} cores | "
              f"Python: {system.get('python_version', 'Unknown')}")
        print(f"   Memory: {system.get('memory', {}).get('percent', 0):.1f}% used | "
              f"Disk: {system.get('disk', {}).get('percent', 0):.1f}% used")
        print(f"   Process: PID {process.get('pid', 0)} | "
              f"Memory: {process.get('memory_mb', 0):.1f}MB | "
              f"CPU: {process.get('cpu_percent', 0):.1f}%")
    
    if env_summary["alerts"]:
        print(f"\n⚠️  ENVIRONMENT ALERTS:")
        for alert in env_summary["alerts"]:
            print(f"   - {alert}")
    
    # Agent Metrics
    print(f"\n📊 AGENT METRICS:")
    print(f"   Session Duration: {(datetime.now() - agent.state.created_at).total_seconds() / 60:.1f} minutes")
    print(f"   Conversation Turns: {agent.metrics['turn_count']}")
    print(f"   Tools Used: {agent.metrics['tool_usage_count']}")
    print(f"   Tasks Created: {agent.metrics['tasks_created']}")
    print(f"   Reflections Generated: {agent.metrics['reflections_generated']}")
    print(f"   Direction Changes: {agent.metrics['direction_changes']}")
    
    # Current State
    print(f"\n🎯 CURRENT STATE:")
    print(f"   Direction: {agent.state.direction}")
    if agent.state.active_goals:
        print(f"   Active Goals: {', '.join(agent.state.active_goals[:3])}")
    if agent.state.completed_goals:
        print(f"   Completed Goals: {len(agent.state.completed_goals)}")
    
    print("="*80 + "\n")


# ————————————————————————————————————————————————————————————————
# CLI Integration
# ————————————————————————————————————————————————————————————————
async def interactive_demo_enhanced():
    """Enhanced interactive demo with reflection, task queue, and environment monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Multi-turn Agent with Reflection")
    parser.add_argument("--session", help="Resume session ID")
    parser.add_argument(
        "--pivot", action="store_true", help="Enable direction pivoting"
    )
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    args = parser.parse_args()

    if args.api:
        # Start FastAPI server
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    print("🚀 Enhanced Multi-turn Agent with Reflection & Task Management")
    print("=" * 80)
    print("Features:")
    print("  • Self-reflection and performance analysis")
    print("  • Task queue with priority management")
    print("  • Worker pool for concurrent execution")
    print("  • Dynamic environment monitoring")
    print("  • Semantic conversation search")
    print("\nCommands: quit, status, history, goals, summary, pivot <direction>, clear")
    print("=" * 80)

    # Create or resume enhanced agent
    agent = EnhancedStatefulAgent(
        session_id=args.session,
        stream=True,
        enable_pivot=args.pivot,
        num_workers=args.workers
    )
    
    # Register additional demo tools
    register_enhanced_demo_tools(agent)

    if args.session:
        summary = await agent.get_session_summary()
        print(f"\nResumed session: {args.session}")
        print(f"Current direction: {summary['direction']}")
        print(f"Turn count: {summary['turn_count']}")
        if summary["active_goals"]:
            print(f"Active goals: {', '.join(summary['active_goals'])}")
    else:
        print(f"\nNew session: {agent.session_id}")

    # Periodically show status
    last_status_time = datetime.now()
    status_interval = timedelta(minutes=2)  # Show status every 2 minutes
    
    while True:
        try:
            # Auto-show status periodically
            if datetime.now() - last_status_time > status_interval:
                visualize_agent_status(agent)
                last_status_time = datetime.now()
            
            user_input = input("\n💬 You: ").strip()

            if user_input.lower() == "quit":
                print("👋 Shutting down agent services...")
                await agent.worker_pool.stop()
                print("Goodbye!")
                break
            elif user_input.lower() == "status":
                visualize_agent_status(agent)
                continue
            elif user_input.lower() == "history":
                history = agent.memory.history
                print("\n📜 CONVERSATION HISTORY (last 10 messages)")
                print("-" * 60)
                for msg in history[-10:]:
                    role = msg.get(R, "unknown")
                    content = str(msg.get(C, ""))[:200]
                    role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️", "tool": "🔧"}.get(role, "❓")
                    print(f"{role_emoji} [{role}] {content}...")
                continue
            elif user_input.lower().startswith("goals"):
                parts = user_input.split(maxsplit=2)
                if len(parts) == 3 and parts[1] == "add":
                    await agent.add_goal(parts[2])
                    print(f"✅ Goal added: {parts[2]}")
                elif len(parts) == 3 and parts[1] == "complete":
                    await agent.complete_goal(parts[2])
                    print(f"🎯 Goal completed: {parts[2]}")
                else:
                    summary = await agent.get_session_summary()
                    print(f"🎯 Active goals: {summary['active_goals']}")
                    print(f"✅ Completed goals: {summary['completed_goals']}")
                continue
            elif user_input.lower() == "summary":
                summary = await agent.get_session_summary()
                print(f"\n📊 SESSION SUMMARY")
                print("-" * 60)
                for key, value in summary.items():
                    print(f"{key}: {value}")
                continue
            elif user_input.lower().startswith("pivot"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2:
                    try:
                        new_dir = ConversationDirection(parts[1].lower())
                        await agent.change_direction(new_dir, "User request")
                        print(f"🔄 Direction changed to: {new_dir.value}")
                    except ValueError:
                        print(
                            f"❌ Invalid direction. Choose from: {[d.value for d in ConversationDirection]}"
                        )
                continue
            elif user_input.lower() == "clear":
                # Clear history but keep state
                agent.memory.history = []
                print("🧹 History cleared (state preserved)")
                continue
            elif user_input.lower() == "help":
                print("\n📖 AVAILABLE COMMANDS:")
                print("  • status    - Show comprehensive agent status")
                print("  • history   - Show conversation history")
                print("  • goals     - Manage goals (add/complete/list)")
                print("  • summary   - Show session summary")
                print("  • pivot     - Change conversation direction")
                print("  • clear     - Clear conversation history")
                print("  • help      - Show this help message")
                print("  • quit      - Exit the program")
                continue

            print("🤖 Assistant: ", end="", flush=True)
            response = await agent.send_user(user_input)
            
            # Show brief status after tool-heavy interactions
            if agent.metrics["tool_usage_count"] > 0 and agent.metrics["tool_usage_count"] % 5 == 0:
                print("\n")
                visualize_agent_status(agent)

        except KeyboardInterrupt:
            print("\n👋 Shutting down agent services...")
            await agent.worker_pool.stop()
            print("Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Error in interactive demo")


# ————————————————————————————————————————————————————————————————
# Example Tools with State Awareness
# ————————————————————————————————————————————————————————————————
def register_enhanced_demo_tools(agent: EnhancedStatefulAgent):
    """Register additional demo tools for the enhanced agent."""
    
    @agent.tools_registry.register(description="Simulate a long-running analysis task")
    async def analyze_performance(
        target: str = "system",
        duration: int = 5
    ) -> str:
        """Simulate a performance analysis task."""
        task = Task(
            name=f"Performance analysis: {target}",
            description=f"Analyze {target} performance for {duration} seconds",
            priority=TaskPriority.MEDIUM,
            callback=lambda t: asyncio.sleep(duration)
        )
        task_id = await agent.task_queue.add_task(task)
        return f"Started performance analysis task {task_id[:8]}"
    
    @agent.tools_registry.register(description="Create multiple related tasks")
    async def create_workflow(
        workflow_name: str,
        steps: List[str]
    ) -> str:
        """Create a workflow with multiple dependent tasks."""
        task_ids = []
        prev_id = None
        
        for i, step in enumerate(steps):
            task = Task(
                name=f"{workflow_name} - Step {i+1}: {step}",
                description=step,
                priority=TaskPriority.HIGH if i == 0 else TaskPriority.MEDIUM,
                dependencies=[prev_id] if prev_id else []
            )
            task_id = await agent.task_queue.add_task(task)
            task_ids.append(task_id)
            prev_id = task_id
        
        return f"Created workflow '{workflow_name}' with {len(steps)} steps: {[id[:8] for id in task_ids]}"
    
    @agent.tools_registry.register(description="Trigger memory optimization")
    async def optimize_memory() -> str:
        """Trigger a memory optimization task."""
        task = Task(
            name="Memory optimization",
            description="Optimize agent memory usage",
            priority=TaskPriority.HIGH
        )
        task_id = await agent.task_queue.add_task(task)
        return f"Triggered memory optimization task {task_id[:8]}"


if __name__ == "__main__":
    # Run enhanced interactive demo
    asyncio.run(interactive_demo_enhanced())
