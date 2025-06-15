"""
Base agent class for the complex agent system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
import uuid

from ..config import AgentRole, AgentCapability, config
from .message import Message, MessageType, MessagePriority, MessageQueue
from .task import Task, TaskStatus, TaskResult, TaskType


class AgentState(str, Enum):
    """States an agent can be in."""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentMetrics(BaseModel):
    """Metrics tracked for each agent."""
    
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    last_heartbeat: datetime = Field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    error_count: int = 0
    tools_created: int = 0
    tools_executed: int = 0


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        role: AgentRole = AgentRole.EXECUTOR,
        capabilities: Optional[Set[AgentCapability]] = None,
        message_queue: Optional[MessageQueue] = None
    ):
        self.id = agent_id or str(uuid.uuid4())
        self.name = name or f"{role.value}_{self.id[:8]}"
        self.role = role
        self.capabilities = capabilities or set()
        self.state = AgentState.IDLE
        self.current_task: Optional[Task] = None
        self.message_queue = message_queue or MessageQueue()
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"agent.{self.name}")
        self._running = False
        self._tasks = asyncio.Queue()
        self._start_time = datetime.now()
        
        # Knowledge base - stores information learned/discovered
        self.knowledge_base: Dict[str, Any] = {}
        
        # Relationships with other agents
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        
        # Tool registry client will be initialized when needed
        self._tool_registry_client = None
    
    @abstractmethod
    async def process_task(self, task: Task) -> TaskResult:
        """Process a specific task. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def initialize(self):
        """Initialize the agent. Override for custom initialization."""
        pass
    
    async def start(self):
        """Start the agent's main loop."""
        self._running = True
        await self.initialize()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._message_loop())
        
        self.logger.info(f"Agent {self.name} started with role {self.role.value}")
        
        # Main task processing loop
        while self._running:
            try:
                # Check for new tasks
                task = await asyncio.wait_for(self._tasks.get(), timeout=1.0)
                await self._execute_task(task)
            except asyncio.TimeoutError:
                # No tasks, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.state = AgentState.ERROR
                self.metrics.error_count += 1
    
    async def stop(self):
        """Stop the agent gracefully."""
        self._running = False
        self.state = AgentState.SHUTDOWN
        self.logger.info(f"Agent {self.name} shutting down")
    
    async def _execute_task(self, task: Task):
        """Execute a task with proper state management."""
        self.state = AgentState.BUSY
        self.current_task = task
        task.start()
        
        start_time = datetime.now()
        
        try:
            # Process the task
            result = await self.process_task(task)
            task.complete(result)
            
            # Update metrics
            self.metrics.tasks_completed += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics.total_execution_time += execution_time
            
            # Send completion message
            await self.send_message(
                Message(
                    type=MessageType.TASK_RESULT,
                    sender_id=self.id,
                    recipient_id=task.created_by,
                    payload={
                        "task_id": task.id,
                        "result": result.dict()
                    }
                )
            )
            
        except Exception as e:
            self.logger.error(f"Task {task.id} failed: {e}")
            task.fail(str(e))
            self.metrics.tasks_failed += 1
            
            # Send failure message
            await self.send_message(
                Message(
                    type=MessageType.ERROR,
                    priority=MessagePriority.HIGH,
                    sender_id=self.id,
                    recipient_id=task.created_by,
                    payload={
                        "task_id": task.id,
                        "error": str(e)
                    }
                )
            )
        
        finally:
            self.current_task = None
            self.state = AgentState.IDLE
    
    async def assign_task(self, task: Task):
        """Assign a task to this agent."""
        task.assign_to(self.id)
        await self._tasks.put(task)
        self.logger.info(f"Task {task.id} assigned to agent {self.name}")
    
    async def send_message(self, message: Message):
        """Send a message to another agent or broadcast."""
        self.message_queue.send(message)
        self.logger.debug(f"Message sent: {message.type} to {message.recipient_id or 'broadcast'}")
    
    async def receive_messages(self) -> List[Message]:
        """Receive messages addressed to this agent."""
        return self.message_queue.receive(self.id)
    
    async def _message_loop(self):
        """Background loop to process incoming messages."""
        while self._running:
            try:
                messages = await self.receive_messages()
                for message in messages:
                    await self._handle_message(message)
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                self.logger.error(f"Error in message loop: {e}")
    
    async def _handle_message(self, message: Message):
        """Handle an incoming message."""
        self.logger.debug(f"Received message: {message.type} from {message.sender_id}")
        
        if message.type == MessageType.TASK_ASSIGNMENT:
            task_data = message.payload.get("task")
            if task_data:
                task = Task(**task_data)
                await self.assign_task(task)
        
        elif message.type == MessageType.QUERY:
            # Handle queries about agent state/capabilities
            response_payload = {
                "agent_id": self.id,
                "name": self.name,
                "role": self.role.value,
                "state": self.state.value,
                "capabilities": [cap.value for cap in self.capabilities],
                "metrics": self.metrics.dict()
            }
            
            reply = message.create_reply(self.id, response_payload)
            await self.send_message(reply)
        
        elif message.type == MessageType.STATUS_UPDATE:
            # Update knowledge about other agents
            agent_id = message.sender_id
            self.known_agents[agent_id] = message.payload
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            try:
                self.metrics.last_heartbeat = datetime.now()
                self.metrics.uptime_seconds = (datetime.now() - self._start_time).total_seconds()
                
                # Send heartbeat
                await self.send_message(
                    Message(
                        type=MessageType.HEARTBEAT,
                        sender_id=self.id,
                        priority=MessagePriority.LOW,
                        payload={
                            "state": self.state.value,
                            "metrics": self.metrics.dict()
                        }
                    )
                )
                
                await asyncio.sleep(config.AGENT_HEARTBEAT_INTERVAL)
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
    
    def add_capability(self, capability: AgentCapability):
        """Add a new capability to the agent."""
        self.capabilities.add(capability)
        self.logger.info(f"Added capability: {capability.value}")
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities
    
    def update_knowledge(self, key: str, value: Any):
        """Update the agent's knowledge base."""
        self.knowledge_base[key] = value
        self.logger.debug(f"Knowledge updated: {key}")
    
    def get_knowledge(self, key: str, default: Any = None) -> Any:
        """Get information from the knowledge base."""
        return self.knowledge_base.get(key, default)