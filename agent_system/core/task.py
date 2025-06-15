"""
Task management system for agent coordination.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
import uuid


class TaskStatus(str, Enum):
    """Status of a task in the system."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of tasks that agents can perform."""
    TOOL_CREATION = "tool_creation"
    TOOL_EXECUTION = "tool_execution"
    TOOL_VALIDATION = "tool_validation"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    WEB_SEARCH = "web_search"
    MONITORING = "monitoring"
    COMPOSITE = "composite"  # Task that requires multiple subtasks


class TaskResult(BaseModel):
    """Result of a completed task."""
    
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None  # in seconds


class Task(BaseModel):
    """Task representation in the agent system."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=5, ge=1, le=10)  # 1-10, higher is more important
    created_at: datetime = Field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None  # Agent ID
    created_by: str  # Agent ID or system
    parent_task_id: Optional[str] = None  # For subtasks
    dependencies: List[str] = Field(default_factory=list)  # Task IDs
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[TaskResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[int] = None  # Timeout in seconds
    retry_count: int = 0
    max_retries: int = 3
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def assign_to(self, agent_id: str):
        """Assign task to an agent."""
        self.assigned_to = agent_id
        self.assigned_at = datetime.now()
        self.status = TaskStatus.ASSIGNED
    
    def start(self):
        """Mark task as started."""
        self.started_at = datetime.now()
        self.status = TaskStatus.IN_PROGRESS
    
    def complete(self, result: TaskResult):
        """Mark task as completed with result."""
        self.completed_at = datetime.now()
        self.status = TaskStatus.COMPLETED
        self.output_data = result
        if self.started_at:
            result.execution_time = (self.completed_at - self.started_at).total_seconds()
    
    def fail(self, error: str):
        """Mark task as failed."""
        self.completed_at = datetime.now()
        self.status = TaskStatus.FAILED
        self.output_data = TaskResult(success=False, output=None, error=error)
    
    def cancel(self):
        """Cancel the task."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    def retry(self):
        """Retry the task."""
        if self.can_retry():
            self.retry_count += 1
            self.status = TaskStatus.PENDING
            self.assigned_to = None
            self.assigned_at = None
            self.started_at = None
            self.completed_at = None
            self.output_data = None
            return True
        return False
    
    def is_ready(self, completed_tasks: List[str]) -> bool:
        """Check if task is ready to be executed (all dependencies met)."""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def is_timeout(self) -> bool:
        """Check if task has timed out."""
        if self.timeout is None or self.started_at is None:
            return False
        elapsed = (datetime.now() - self.started_at).total_seconds()
        return elapsed > self.timeout


class TaskQueue:
    """Priority queue for task management."""
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._pending_tasks: List[Task] = []
        self._completed_task_ids: List[str] = []
    
    def add(self, task: Task):
        """Add a task to the queue."""
        self._tasks[task.id] = task
        if task.status == TaskStatus.PENDING:
            self._pending_tasks.append(task)
            self._pending_tasks.sort(key=lambda t: t.priority, reverse=True)
    
    def get_next_ready_task(self) -> Optional[Task]:
        """Get the next task that's ready to be executed."""
        for i, task in enumerate(self._pending_tasks):
            if task.is_ready(self._completed_task_ids):
                return self._pending_tasks.pop(i)
        return None
    
    def mark_completed(self, task_id: str):
        """Mark a task as completed."""
        if task_id not in self._completed_task_ids:
            self._completed_task_ids.append(task_id)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with a specific status."""
        return [task for task in self._tasks.values() if task.status == status]
    
    def get_agent_tasks(self, agent_id: str) -> List[Task]:
        """Get all tasks assigned to a specific agent."""
        return [task for task in self._tasks.values() if task.assigned_to == agent_id]