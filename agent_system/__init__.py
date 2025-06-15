"""
Complex Agent System for Tool Registry Integration.

This system provides a sophisticated multi-agent architecture that integrates
with the evalscompany tool registry on port 2016.
"""

from .config import AgentConfig, AgentRole, AgentCapability
from .core.base_agent import BaseAgent, AgentState
from .core.message import Message, MessageType, MessagePriority
from .core.task import Task, TaskStatus, TaskResult

__all__ = [
    "AgentConfig",
    "AgentRole", 
    "AgentCapability",
    "BaseAgent",
    "AgentState",
    "Message",
    "MessageType",
    "MessagePriority",
    "Task",
    "TaskStatus",
    "TaskResult"
]

__version__ = "0.1.0"