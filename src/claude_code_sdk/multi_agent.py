"""Multi-agent system implementation for Claude SDK."""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .types import (
    AssistantMessage,
    ClaudeCodeOptions,
    Message,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from . import query


logger = logging.getLogger(__name__)


# Type definitions
T = TypeVar("T")
AgentType = TypeVar("AgentType", bound="BaseAgent")


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CommunicationProtocol(Enum):
    """Communication protocols between agents."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"


@dataclass
class AgentMessage:
    """Message exchanged between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcasts
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # For request-response patterns
    reply_to: Optional[str] = None  # ID of message being replied to
    protocol: CommunicationProtocol = CommunicationProtocol.DIRECT


@dataclass
class Task:
    """Task that can be assigned to agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    assigned_to: Optional[str] = None
    created_by: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (all dependencies met)."""
        return all(dep in completed_tasks for dep in self.dependencies)


class AgentCapability(Protocol):
    """Protocol for agent capabilities."""
    
    @property
    def name(self) -> str:
        """Capability name."""
        ...
        
    def can_handle(self, task: Task) -> bool:
        """Check if this capability can handle a task."""
        ...
        
    async def execute(self, agent: "BaseAgent", task: Task) -> Any:
        """Execute the capability."""
        ...


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent_{self.id[:8]}"
        self.capabilities = capabilities or []
        self.system_prompt = system_prompt
        self.session_id: Optional[str] = None
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._subscriptions: Set[str] = set()
        self._message_handlers: Dict[str, Callable] = {}
        self._running = False
        self._tasks: List[Task] = []
        
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process an incoming message."""
        pass
        
    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a task."""
        pass
        
    async def query_claude(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> List[Message]:
        """Query Claude with agent configuration."""
        options = ClaudeCodeOptions(
            system_prompt=self.system_prompt,
            resume=self.session_id,
            allowed_tools=tools or [],
            **kwargs
        )
        
        messages = []
        async for message in query(prompt=prompt, options=options):
            messages.append(message)
            if isinstance(message, ResultMessage):
                self.session_id = message.session_id
                
        return messages
        
    async def send_message(
        self,
        recipient: Union[str, "BaseAgent"],
        content: Any,
        **kwargs
    ) -> AgentMessage:
        """Send a message to another agent."""
        recipient_id = recipient.id if isinstance(recipient, BaseAgent) else recipient
        
        message = AgentMessage(
            sender_id=self.id,
            recipient_id=recipient_id,
            content=content,
            **kwargs
        )
        
        # This will be handled by the coordinator
        return message
        
    async def broadcast_message(self, content: Any, **kwargs) -> AgentMessage:
        """Broadcast a message to all agents."""
        message = AgentMessage(
            sender_id=self.id,
            recipient_id=None,
            content=content,
            protocol=CommunicationProtocol.BROADCAST,
            **kwargs
        )
        
        return message
        
    async def receive_message(self, timeout: Optional[float] = None) -> AgentMessage:
        """Receive a message from the queue."""
        if timeout:
            return await asyncio.wait_for(self._message_queue.get(), timeout)
        return await self._message_queue.get()
        
    async def start(self):
        """Start the agent's message processing loop."""
        self._running = True
        logger.info(f"Agent {self.name} started")
        
        while self._running:
            try:
                message = await self.receive_message(timeout=1.0)
                response = await self.process_message(message)
                
                if response:
                    # Response will be routed by coordinator
                    await self._handle_response(response)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in agent {self.name}: {e}", exc_info=True)
                
    async def stop(self):
        """Stop the agent."""
        self._running = False
        logger.info(f"Agent {self.name} stopped")
        
    async def _handle_response(self, response: AgentMessage):
        """Handle response message (to be overridden by coordinator)."""
        pass
        
    def subscribe(self, topic: str):
        """Subscribe to a topic for pub/sub messaging."""
        self._subscriptions.add(topic)
        
    def unsubscribe(self, topic: str):
        """Unsubscribe from a topic."""
        self._subscriptions.discard(topic)
        
    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message types."""
        self._message_handlers[message_type] = handler


class ClaudeAgent(BaseAgent):
    """Agent that uses Claude for processing."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        tools: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(agent_id, name, **kwargs)
        self.role = role or "assistant"
        self.tools = tools or []
        
        if not self.system_prompt:
            self.system_prompt = f"You are {self.name}, a {self.role}. You collaborate with other agents to complete tasks."
            
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process message using Claude."""
        # Check for registered handlers
        message_type = message.metadata.get("type", "default")
        if message_type in self._message_handlers:
            return await self._message_handlers[message_type](message)
            
        # Default Claude processing
        prompt = f"""
        You received a message from {message.sender_id}:
        
        {json.dumps(message.content, indent=2)}
        
        Context: {json.dumps(message.metadata, indent=2)}
        
        Please process this message and provide an appropriate response.
        """
        
        messages = await self.query_claude(prompt, tools=self.tools)
        
        response_content = ""
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response_content += block.text
                        
        if response_content:
            return AgentMessage(
                sender_id=self.id,
                recipient_id=message.sender_id,
                content=response_content,
                reply_to=message.id,
                correlation_id=message.correlation_id
            )
            
        return None
        
    async def execute_task(self, task: Task) -> Any:
        """Execute a task using Claude."""
        prompt = f"""
        Execute the following task:
        
        Name: {task.name}
        Description: {task.description}
        Context: {json.dumps(task.metadata, indent=2)}
        
        Please complete this task and provide the result.
        """
        
        messages = await self.query_claude(prompt, tools=self.tools)
        
        result = {
            "messages": [],
            "tools_used": [],
            "files_created": [],
            "files_modified": []
        }
        
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        result["messages"].append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        result["tools_used"].append({
                            "tool": block.name,
                            "input": block.input
                        })
                        
                        # Track file operations
                        if block.name in ["Write", "Create"]:
                            result["files_created"].append(
                                block.input.get("file_path", "unknown")
                            )
                        elif block.name in ["Edit", "Modify"]:
                            result["files_modified"].append(
                                block.input.get("file_path", "unknown")
                            )
                            
        return result


class AgentCoordinator:
    """Coordinates communication and task execution between agents."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self._message_router: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self._running = False
        self._topics: Dict[str, Set[str]] = {}  # topic -> agent_ids
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator."""
        self.agents[agent.id] = agent
        
        # Override agent's response handler
        original_handler = agent._handle_response
        
        async def coordinated_handler(response: AgentMessage):
            await self._message_router.put(response)
            
        agent._handle_response = coordinated_handler
        
        logger.info(f"Registered agent: {agent.name} ({agent.id})")
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            logger.info(f"Unregistered agent: {agent.name}")
            
    async def send_message(
        self,
        sender: Union[str, BaseAgent],
        recipient: Union[str, BaseAgent],
        content: Any,
        **kwargs
    ) -> AgentMessage:
        """Send a message between agents."""
        sender_id = sender.id if isinstance(sender, BaseAgent) else sender
        recipient_id = recipient.id if isinstance(recipient, BaseAgent) else recipient
        
        message = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            **kwargs
        )
        
        await self._message_router.put(message)
        return message
        
    async def broadcast(
        self,
        sender: Union[str, BaseAgent],
        content: Any,
        exclude: Optional[Set[str]] = None,
        **kwargs
    ) -> List[AgentMessage]:
        """Broadcast a message to all agents."""
        sender_id = sender.id if isinstance(sender, BaseAgent) else sender
        exclude = exclude or set()
        exclude.add(sender_id)  # Don't send to self
        
        messages = []
        for agent_id in self.agents:
            if agent_id not in exclude:
                message = await self.send_message(
                    sender_id,
                    agent_id,
                    content,
                    protocol=CommunicationProtocol.BROADCAST,
                    **kwargs
                )
                messages.append(message)
                
        return messages
        
    async def publish(
        self,
        topic: str,
        sender: Union[str, BaseAgent],
        content: Any,
        **kwargs
    ) -> List[AgentMessage]:
        """Publish a message to a topic."""
        sender_id = sender.id if isinstance(sender, BaseAgent) else sender
        subscribers = self._topics.get(topic, set())
        
        messages = []
        for agent_id in subscribers:
            if agent_id != sender_id:  # Don't send to self
                message = await self.send_message(
                    sender_id,
                    agent_id,
                    content,
                    protocol=CommunicationProtocol.PUBLISH_SUBSCRIBE,
                    metadata={**kwargs.get("metadata", {}), "topic": topic},
                    **kwargs
                )
                messages.append(message)
                
        return messages
        
    def subscribe_agent(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic."""
        if topic not in self._topics:
            self._topics[topic] = set()
        self._topics[topic].add(agent_id)
        
        if agent_id in self.agents:
            self.agents[agent_id].subscribe(topic)
            
    def unsubscribe_agent(self, agent_id: str, topic: str):
        """Unsubscribe an agent from a topic."""
        if topic in self._topics:
            self._topics[topic].discard(agent_id)
            
        if agent_id in self.agents:
            self.agents[agent_id].unsubscribe(topic)
            
    def create_task(
        self,
        name: str,
        description: str,
        assigned_to: Optional[Union[str, BaseAgent]] = None,
        **kwargs
    ) -> Task:
        """Create a new task."""
        task = Task(
            name=name,
            description=description,
            assigned_to=assigned_to.id if isinstance(assigned_to, BaseAgent) else assigned_to,
            **kwargs
        )
        
        self.tasks[task.id] = task
        return task
        
    async def assign_task(self, task: Union[str, Task], agent: Union[str, BaseAgent]):
        """Assign a task to an agent."""
        task_id = task.id if isinstance(task, Task) else task
        agent_id = agent.id if isinstance(agent, BaseAgent) else agent
        
        if task_id in self.tasks and agent_id in self.agents:
            task_obj = self.tasks[task_id]
            task_obj.assigned_to = agent_id
            task_obj.status = TaskStatus.ASSIGNED
            
            await self._task_queue.put(task_obj)
            logger.info(f"Assigned task {task_obj.name} to {self.agents[agent_id].name}")
            
    async def start(self):
        """Start the coordinator."""
        self._running = True
        
        # Start all agents
        agent_tasks = [
            asyncio.create_task(agent.start())
            for agent in self.agents.values()
        ]
        
        # Start routing and task execution
        router_task = asyncio.create_task(self._route_messages())
        executor_task = asyncio.create_task(self._execute_tasks())
        
        logger.info("Coordinator started")
        
        try:
            await asyncio.gather(
                router_task,
                executor_task,
                *agent_tasks
            )
        except Exception as e:
            logger.error(f"Coordinator error: {e}", exc_info=True)
            
    async def stop(self):
        """Stop the coordinator."""
        self._running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
            
        logger.info("Coordinator stopped")
        
    async def _route_messages(self):
        """Route messages between agents."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_router.get(),
                    timeout=1.0
                )
                
                # Route based on protocol
                if message.protocol == CommunicationProtocol.BROADCAST:
                    # Already handled by broadcast method
                    pass
                elif message.protocol == CommunicationProtocol.PUBLISH_SUBSCRIBE:
                    # Already handled by publish method
                    pass
                else:
                    # Direct message
                    if message.recipient_id and message.recipient_id in self.agents:
                        recipient = self.agents[message.recipient_id]
                        await recipient._message_queue.put(message)
                        logger.debug(
                            f"Routed message from {message.sender_id} to {message.recipient_id}"
                        )
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message routing error: {e}", exc_info=True)
                
    async def _execute_tasks(self):
        """Execute tasks assigned to agents."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._task_queue.get(),
                    timeout=1.0
                )
                
                if task.assigned_to and task.assigned_to in self.agents:
                    agent = self.agents[task.assigned_to]
                    
                    # Check dependencies
                    if not task.is_ready(self.completed_tasks):
                        # Put back in queue
                        await self._task_queue.put(task)
                        await asyncio.sleep(1)  # Wait before retry
                        continue
                        
                    # Execute task
                    task.status = TaskStatus.IN_PROGRESS
                    task.started_at = datetime.utcnow()
                    
                    try:
                        logger.info(f"Agent {agent.name} executing task {task.name}")
                        result = await agent.execute_task(task)
                        
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.utcnow()
                        self.completed_tasks.add(task.id)
                        
                        logger.info(f"Task {task.name} completed successfully")
                        
                    except Exception as e:
                        task.error = str(e)
                        task.status = TaskStatus.FAILED
                        task.completed_at = datetime.utcnow()
                        
                        logger.error(f"Task {task.name} failed: {e}", exc_info=True)
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Task execution error: {e}", exc_info=True)
                
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
        
    def get_agent_tasks(self, agent_id: str) -> List[Task]:
        """Get all tasks assigned to an agent."""
        return [
            task for task in self.tasks.values()
            if task.assigned_to == agent_id
        ]
        
    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.COMPLETED
        ]
        
    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks."""
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.FAILED
        ]


# Specialized agent classes
class DeveloperAgent(ClaudeAgent):
    """Agent specialized in software development."""
    
    def __init__(self, name: str = "Developer", **kwargs):
        super().__init__(
            name=name,
            role="software developer",
            tools=["Read", "Write", "Edit", "Bash", "Grep"],
            **kwargs
        )
        
        self.system_prompt = """
        You are a skilled software developer. You write clean, efficient, and well-documented code.
        You follow best practices and design patterns. You collaborate with other agents to build software.
        """


class ReviewerAgent(ClaudeAgent):
    """Agent specialized in code review."""
    
    def __init__(self, name: str = "Reviewer", **kwargs):
        super().__init__(
            name=name,
            role="code reviewer",
            tools=["Read", "Grep"],
            **kwargs
        )
        
        self.system_prompt = """
        You are an experienced code reviewer. You look for bugs, security issues, performance problems,
        and code quality issues. You provide constructive feedback and suggestions for improvement.
        """


class TesterAgent(ClaudeAgent):
    """Agent specialized in testing."""
    
    def __init__(self, name: str = "Tester", **kwargs):
        super().__init__(
            name=name,
            role="QA engineer",
            tools=["Read", "Write", "Bash"],
            **kwargs
        )
        
        self.system_prompt = """
        You are a thorough QA engineer. You write comprehensive tests, find edge cases,
        and ensure code quality through testing. You create unit tests, integration tests,
        and end-to-end tests as needed.
        """


class ArchitectAgent(ClaudeAgent):
    """Agent specialized in system architecture."""
    
    def __init__(self, name: str = "Architect", **kwargs):
        super().__init__(
            name=name,
            role="software architect",
            tools=["Write"],
            **kwargs
        )
        
        self.system_prompt = """
        You are a software architect. You design system architectures, choose appropriate
        technologies, define interfaces, and ensure scalability and maintainability.
        You create architectural diagrams and documentation.
        """


class SecurityAgent(ClaudeAgent):
    """Agent specialized in security."""
    
    def __init__(self, name: str = "Security", **kwargs):
        super().__init__(
            name=name,
            role="security expert",
            tools=["Read", "Grep"],
            **kwargs
        )
        
        self.system_prompt = """
        You are a security expert. You identify security vulnerabilities, suggest fixes,
        and ensure best security practices are followed. You look for common vulnerabilities
        like SQL injection, XSS, authentication issues, and data exposure.
        """


# Helper functions
async def create_development_team() -> Tuple[AgentCoordinator, Dict[str, BaseAgent]]:
    """Create a standard software development team."""
    coordinator = AgentCoordinator()
    
    agents = {
        "architect": ArchitectAgent("Alice_Architect"),
        "developer": DeveloperAgent("Bob_Developer"),
        "reviewer": ReviewerAgent("Charlie_Reviewer"),
        "tester": TesterAgent("Diana_Tester"),
        "security": SecurityAgent("Eve_Security"),
    }
    
    for agent in agents.values():
        coordinator.register_agent(agent)
        
    return coordinator, agents


async def create_microservices_team() -> Tuple[AgentCoordinator, Dict[str, BaseAgent]]:
    """Create a team for microservices development."""
    coordinator = AgentCoordinator()
    
    agents = {
        "architect": ArchitectAgent("Architect"),
        "auth_dev": DeveloperAgent("Auth_Service_Dev"),
        "user_dev": DeveloperAgent("User_Service_Dev"),
        "api_dev": DeveloperAgent("API_Gateway_Dev"),
        "db_dev": DeveloperAgent("Database_Service_Dev"),
        "tester": TesterAgent("Integration_Tester"),
    }
    
    for agent in agents.values():
        coordinator.register_agent(agent)
        
    # Set up pub/sub topics
    for topic in ["architecture", "integration", "deployment"]:
        for agent in agents.values():
            coordinator.subscribe_agent(agent.id, topic)
            
    return coordinator, agents